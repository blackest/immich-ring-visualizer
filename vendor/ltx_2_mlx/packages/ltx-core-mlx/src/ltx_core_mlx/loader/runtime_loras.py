"""Unfused, run-time LoRA application — ``y = base(x) + scale * B(A x)``.

Why this exists
---------------
:mod:`ltx_core_mlx.loader.fuse_loras` merges a LoRA into the weight tensor:
``dequantize(W) + B@A`` then ``quantize(...)`` again. On a **quantized** pack
that round-trip is lossy in exactly the wrong place. Measured on the real
LTX packs with ``bizarrotrn_v2`` (rank 32, ``‖B@A‖/‖W‖ ≈ 0.08``), sampling 48
attention modules across blocks 0/10/20/30/40/47:

===============================  ======  =====================================
pack                             bits    LoRA delta LOST ``‖eff−D‖/‖D‖``
===============================  ======  =====================================
``ltx-2.3-mlx-q4``               4       **94.9 %**
``ltx-2.5-mlx-q4``               4       **92.7 %**
``ltx-2.3-mlx-q8``               8       10.2 %
``ltx-2.5-mlx-q8``               8       9.8 %
===============================  ======  =====================================

The base weight is already exactly on the quantization grid (it came *from*
dequantizing), and at ``bits=4, group_size=64`` one step is large enough that
most of an 8 %-magnitude delta falls below half a step — so ``quantize(W + D)``
rounds those elements straight back to ``W``. Not a no-op: severe attenuation.
A character LoRA fused at q4 runs at roughly a twentieth of the strength it was
trained at, which is enough to change the picture and not enough to carry an
identity. Two investigations have now been spent on that failure.

Keeping the low-rank branch **out** of the weight removes the round-trip
entirely: the base matmul still runs against the packed weight (no memory
cost, no dequantization), and the rank-``r`` correction is applied to the
*activations* in float. The result is exact at any bit width, and costs
``2r/out_features`` extra FLOPs — ~1.5 % at rank 32 on a 4096-wide projection.

The same trick is proven in the H3 MLX runner (``minimax_h3_mlx/lora.py``),
where a bf16 fold destroys ~87 % of a much smaller turbo delta. This module is
its LTX analogue, with one deliberate design difference: **the adapters keep
the base module's state-dict paths.**

State-dict path preservation (load-bearing)
-------------------------------------------
H3 wraps the base layer in a new module, so ``…to_q.weight`` becomes
``…to_q.base.weight``. That is fine there because nothing else in the runner
reads the DiT's flattened parameters afterwards. In this codebase two later
stages do:

* ``TI2VidTwoStagesPipeline._fuse_distilled_lora`` flattens the live DiT and
  fuses the rank-384 distilled LoRA into it before stage 2;
* ``ICLoraPipeline._fuse_loras`` does the same for control LoRAs.

Both match a LoRA key ``X.lora_A.weight`` against a model key ``X.weight``.
Had the character adapter renamed the target to ``X.base.weight``, the
distilled LoRA would have found no target on every attention projection and
been **silently dropped** — the exact class of bug ``#52`` and the June 2026
mosaic came from. So the adapters here *subclass* the layer they replace and
adopt its parameters by reference: ``weight`` / ``scales`` / ``biases`` /
``bias`` stay at their original paths, and only ``lora_a`` / ``lora_b`` are
added. A later fuse pass lands on the base weight, underneath the runtime
branch, and both apply.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength

__all__ = [
    "LORA_MODES",
    "LoRALinear",
    "LoRAQuantizedLinear",
    "RuntimeLoraReport",
    "attach_loras",
    "load_and_attach_loras",
    "model_has_quantized_linears",
    "resolve_lora_mode",
]

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"

#: Selectable LoRA application modes.
#:
#: - ``"unfused"`` — runtime low-rank branch. Exact at any bit width.
#: - ``"fuse"``    — merge into the weight tensor. Exact on bf16, lossy on
#:   quantized (see the module docstring); kept because it is free on bf16 and
#:   because it is the honest control arm for A/B work.
#: - ``"auto"``    — ``unfused`` when the target model holds quantized linears,
#:   ``fuse`` otherwise.
LORA_MODES = ("auto", "unfused", "fuse")


# ---------------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------------


class LoRAQuantizedLinear(nn.QuantizedLinear):
    """A :class:`~mlx.nn.QuantizedLinear` with an unfused low-rank branch.

    ``y = quantized_matmul(x, W) + bias + scale * (x @ Aᵀ) @ Bᵀ``

    Constructed from an already-loaded base layer whose parameters are adopted
    **by reference** — no copy, no dequantization, no extra weight memory. The
    base's quantization config (``group_size`` / ``bits`` / ``mode``) travels
    with it, so this works on any pack the loader can read.

    ``nn.Module.__init__`` is called directly rather than
    ``super().__init__()``: :class:`~mlx.nn.QuantizedLinear`'s initializer
    allocates and quantizes a fresh random weight from ``(in_dims, out_dims)``,
    which we would immediately throw away.
    """

    def __init__(self, base: nn.QuantizedLinear, a: mx.array, b: mx.array, scale: float = 1.0):
        nn.Module.__init__(self)
        self.group_size = base.group_size
        self.bits = base.bits
        # `or "affine"` rather than a getattr default: MLX's Module.__getattr__
        # returns None for a missing key instead of raising, so the default
        # would never be reached and `mode=None` would reach quantized_matmul.
        self.mode = getattr(base, "mode", None) or "affine"
        self.weight = base["weight"]
        self.scales = base["scales"]
        biases = base.get("biases")
        if biases is not None:
            self.biases = biases
        if "bias" in base:
            self.bias = base["bias"]
        self.lora_a = a
        self.lora_b = b
        self.lora_scale = float(scale)
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        y = super().__call__(x)
        if self.lora_scale == 0.0:
            return y
        return _add_lora_delta(y, x, self.lora_a, self.lora_b, self.lora_scale)


class LoRALinear(nn.Linear):
    """A :class:`~mlx.nn.Linear` with an unfused low-rank branch.

    The float-weight counterpart of :class:`LoRAQuantizedLinear`. On bf16
    weights fusion is *nearly* free — LTX's deltas are ~8 % of ‖W‖, far above
    bfloat16's 3.9e-3 relative ULP — so ``auto`` mode fuses there. This class
    exists so ``--lora-mode unfused`` means the same thing on every pack, and
    so a bf16 run can serve as the exactness reference for a quantized one.
    """

    def __init__(self, base: nn.Linear, a: mx.array, b: mx.array, scale: float = 1.0):
        nn.Module.__init__(self)
        self.weight = base["weight"]
        if "bias" in base:
            self.bias = base["bias"]
        self.lora_a = a
        self.lora_b = b
        self.lora_scale = float(scale)

    def __call__(self, x: mx.array) -> mx.array:
        y = super().__call__(x)
        if self.lora_scale == 0.0:
            return y
        return _add_lora_delta(y, x, self.lora_a, self.lora_b, self.lora_scale)


def _add_lora_delta(y: mx.array, x: mx.array, a: mx.array, b: mx.array, scale: float) -> mx.array:
    """``y + scale * (x @ Aᵀ) @ Bᵀ``, in two hops and one fused epilogue.

    Two things make this cheap, and both were measured (``scripts/bench_runtime_lora.py``):

    * **Rank-first association.** ``(x @ Aᵀ) @ Bᵀ`` costs ``2·N·(in+out)·r``
      FLOPs — 1.6 % of the base matmul at rank 32 on a 4096-wide projection.
      Materialising ``B@A`` first would cost a full ``in×out`` matmul *and* the
      weight-sized temporary this whole module exists to avoid.
    * **A fused add.** At these widths the layer is bound by the output write,
      not by arithmetic: a separate ``y + delta`` writes the ``N×out`` delta,
      reads it back, and writes the sum — tripling the epilogue traffic for a
      1.6 % FLOP increase, which showed up as ~17 % wall clock.
      :func:`mlx.core.addmm` folds the add into the second matmul's epilogue
      and brings that to ~3 %.

    ``addmm`` needs one dtype across all three operands; mixed-precision callers
    fall back to the explicit form rather than being silently up-cast.
    """
    mid = x.astype(a.dtype) @ a.T
    if mid.dtype == b.dtype == y.dtype:
        return mx.addmm(y, mid, b.T, scale, 1.0)
    delta = mid @ b.T
    if scale != 1.0:
        delta = delta * scale
    return y + delta.astype(y.dtype)


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


@dataclass
class ModuleReport:
    """One LoRA module's fate."""

    name: str
    reason: str = ""
    rank: int = 0
    shape: tuple[int, ...] = ()


@dataclass
class RuntimeLoraReport:
    """What :func:`attach_loras` did, in enough detail to debug a silent no-op.

    Issue #52 (a prefix mismatch that dropped every delta while the render
    still "worked") is the reason this is a report and not a bare count.
    """

    mode: str = "unfused"
    applied: list[ModuleReport] = field(default_factory=list)
    skipped: list[ModuleReport] = field(default_factory=list)
    quantized_targets: int = 0
    float_targets: int = 0

    def render(self) -> str:
        by_reason: dict[str, int] = {}
        for module in self.skipped:
            by_reason[module.reason] = by_reason.get(module.reason, 0) + 1
        lines = [
            f"LoRA mode {self.mode}: {len(self.applied)} modules attached "
            f"({self.quantized_targets} quantized, {self.float_targets} float), "
            f"{len(self.skipped)} skipped"
        ]
        for reason, count in sorted(by_reason.items()):
            lines.append(f"  skipped {count}: {reason}")
        return "\n".join(lines)


def model_has_quantized_linears(model: nn.Module) -> bool:
    """True when any submodule is a :class:`~mlx.nn.QuantizedLinear`.

    The signal ``auto`` mode dispatches on. Reads the *built* model rather
    than the checkpoint name, so a pack that quantizes only some modules is
    classified by what it actually holds.
    """
    return any(isinstance(module, nn.QuantizedLinear) for _, module in model.named_modules())


def resolve_lora_mode(requested: str, model: nn.Module) -> str:
    """Turn ``auto`` into a concrete mode for ``model``; validate the rest.

    Raises:
        ValueError: on an unknown mode name.
    """
    if requested not in LORA_MODES:
        raise ValueError(f"lora mode must be one of {LORA_MODES}, got {requested!r}")
    if requested != "auto":
        return requested
    return "unfused" if model_has_quantized_linears(model) else "fuse"


def _split_lora_pairs(sd: dict[str, mx.array]) -> dict[str, tuple[mx.array, mx.array]]:
    """``{module_path: (A, B)}`` from an already key-remapped LoRA state dict."""
    pairs: dict[str, tuple[mx.array, mx.array]] = {}
    for key in sd:
        if not key.endswith(LORA_A_SUFFIX):
            continue
        name = key[: -len(LORA_A_SUFFIX)]
        b_key = name + LORA_B_SUFFIX
        if b_key in sd:
            pairs[name] = (sd[key], sd[b_key])
    return pairs


def _resolve_module(root: nn.Module, path: str) -> tuple[object, str | int, nn.Module] | None:
    """Walk a dotted path to ``(container, key, module)``; ``None`` if absent.

    ``container`` is the parent module or list, and ``key`` the attribute name
    or list index to assign the replacement into. List steps are supported
    because ``transformer_blocks`` is a plain Python list.
    """
    parts = path.split(".")
    container: object = root
    node: object = root
    key: str | int = ""
    for part in parts:
        container = node
        if isinstance(node, (list, tuple)):
            try:
                index = int(part)
            except ValueError:
                return None
            if index >= len(node):
                return None
            key = index
            node = node[index]
        else:
            if not isinstance(node, nn.Module) or part not in node:
                return None
            key = part
            node = node[part]
    if not isinstance(node, nn.Module):
        return None
    return container, key, node


def _logical_in_features(module: nn.Module) -> int:
    """The layer's true ``in_features``, unpacked.

    A ``QuantizedLinear``'s ``weight`` holds **packed** columns
    (``in_features * bits / 32``); comparing a LoRA's logical width against
    that skipped every module on a Q8 DiT in the H3 runner — silently, with the
    render still "working". ``scales`` carries the unpacked truth:
    ``scales.shape[-1] * group_size``.
    """
    if isinstance(module, nn.QuantizedLinear):
        return int(module["scales"].shape[-1]) * int(module.group_size)
    return int(module["weight"].shape[-1])


def _compute_dtype(base: nn.Module) -> mx.Dtype:
    """The dtype the layer computes in.

    A ``QuantizedLinear``'s ``weight`` is packed ``uint32``; its ``scales`` carry
    the real precision (bfloat16 on every LTX pack we ship).
    """
    if isinstance(base, nn.QuantizedLinear):
        return base["scales"].dtype
    return base["weight"].dtype


def _match_dtype(a: mx.array, b: mx.array, base: nn.Module) -> tuple[mx.array, mx.array]:
    """Narrow the adapters to the layer's compute dtype. Never widen.

    LoRA files are often float32 — ``bizarrotrn_v2`` is, 856 MB of it — while the
    forward runs in bfloat16. Holding fp32 adapters against a bf16 layer buys
    nothing and costs three things: double the resident memory, fp32 matmuls for
    the low-rank hops, and the mixed-dtype fallback instead of the fused
    :func:`mlx.core.addmm` epilogue. Measured at 4096 wide, rank 32, 9216 tokens:
    **+9.7 % wall clock as fp32 vs +2.7 % narrowed**, and 856 MB vs 428 MB.

    What it costs in accuracy: the delta's own relative error goes from 0.17 %
    (fp32 hops, rounded once on the add) to 0.33 % — which is 0.02 % of the
    layer's output, and still **6x better than fusing into bf16** (2.1 %) and
    ~280x better than fusing into int4 (94 %). Widening is never done: a bf16
    adapter on an fp32 layer would only waste memory.
    """
    dtype = _compute_dtype(base)
    if a.dtype.size > dtype.size:
        a = a.astype(dtype)
    if b.dtype.size > dtype.size:
        b = b.astype(dtype)
    return a, b


def _wrap(base: nn.Module, a: mx.array, b: mx.array, scale: float) -> nn.Module:
    a, b = _match_dtype(a, b, base)
    if isinstance(base, nn.QuantizedLinear):
        return LoRAQuantizedLinear(base, a, b, scale)
    return LoRALinear(base, a, b, scale)


def attach_loras(
    model: nn.Module,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    *,
    verbose: bool = True,
) -> RuntimeLoraReport:
    """Attach LoRAs to ``model`` as unfused runtime branches. Mutates in place.

    Deltas from multiple LoRAs targeting the same module are concatenated along
    the rank axis rather than summed into a weight — ``[A₁; A₂]`` and
    ``[s₁B₁ | s₂B₂]`` reproduce ``s₁B₁A₁ + s₂B₂A₂`` exactly, at rank
    ``r₁ + r₂``, with no weight-sized temporary.

    Args:
        model: A built, weight-loaded model. Attach **after** quantization and
            ``load_weights``; the adapters adopt live parameters by reference.
        lora_sd_and_strengths: ``(state_dict, strength)`` pairs whose keys have
            already been through ``LTXV_LORA_COMFY_RENAMING_MAP``.
        verbose: Print the applied/skipped summary to stderr.

    Returns:
        A :class:`RuntimeLoraReport`. A LoRA key whose module is absent, or
        whose shape disagrees with the target, is recorded as skipped with a
        reason — never dropped in silence.
    """
    report = RuntimeLoraReport(mode="unfused")

    # module path -> list of (A, B, strength), in the order the LoRAs were given
    merged: dict[str, list[tuple[mx.array, mx.array, float]]] = {}
    for lora_sd, strength in lora_sd_and_strengths:
        for name, (a, b) in _split_lora_pairs(lora_sd.sd).items():
            merged.setdefault(name, []).append((a, b, float(strength)))

    for name, deltas in merged.items():
        rank = sum(int(a.shape[0]) for a, _, _ in deltas)
        resolved = _resolve_module(model, name)
        if resolved is None:
            report.skipped.append(ModuleReport(name, "no matching module in the model tree", rank))
            continue
        container, key, base = resolved
        if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
            report.skipped.append(
                ModuleReport(name, f"target is {type(base).__name__}, not a linear layer", rank)
            )
            continue

        if isinstance(base, (LoRAQuantizedLinear, LoRALinear)):
            # Attaching twice must not silently discard the first LoRA. Wrapping
            # an adapter in another adapter would: the outer one adopts weight /
            # scales / biases by reference but knows nothing of the inner one's
            # branch. Fold the existing branch in as another delta instead — the
            # same rank-axis concatenation used for multiple LoRAs in one call.
            deltas = [(base["lora_a"], base["lora_b"] * base.lora_scale, 1.0), *deltas]
            rank += int(base["lora_a"].shape[0])

        in_features = _logical_in_features(base)
        out_features = int(base["weight"].shape[0])
        bad = next(
            (
                (a, b)
                for a, b, _ in deltas
                if int(a.shape[-1]) != in_features or int(b.shape[0]) != out_features
            ),
            None,
        )
        if bad is not None:
            a, b = bad
            report.skipped.append(
                ModuleReport(
                    name,
                    f"shape mismatch: base [{out_features}, {in_features}] vs "
                    f"B[{b.shape[0]}] A[{a.shape[-1]}]",
                    rank,
                )
            )
            continue

        if len(deltas) == 1:
            a, b, strength = deltas[0]
        else:
            # Stack on the rank axis: A = [A₁; A₂], B = [s₁B₁ | s₂B₂], scale 1.
            dtype = deltas[0][0].dtype
            a = mx.concatenate([d[0].astype(dtype) for d in deltas], axis=0)
            b = mx.concatenate([d[1].astype(dtype) * d[2] for d in deltas], axis=1)
            strength = 1.0

        adapter = _wrap(base, a, b, strength)
        if isinstance(container, list):
            container[key] = adapter  # type: ignore[index]
        else:
            setattr(container, key, adapter)

        if isinstance(base, nn.QuantizedLinear):
            report.quantized_targets += 1
        else:
            report.float_targets += 1
        report.applied.append(ModuleReport(name, "", rank, (out_features, in_features)))

    if merged and not report.applied:
        # A LoRA was supplied and nothing matched. Never silent: this is #52,
        # where a prefix mismatch made every render a LoRA-free render that
        # still "worked".
        print(
            f"WARNING: {len(lora_sd_and_strengths)} LoRA(s) supplied but 0 modules attached — "
            "the render will be LoRA-free. Check the key remapping (sd_ops).",
            file=sys.stderr,
            flush=True,
        )
    if verbose:
        print(report.render(), file=sys.stderr, flush=True)
        for module in report.skipped[:2]:
            print(f"  e.g. {module.name}: {module.reason}", file=sys.stderr, flush=True)
    return report


def load_and_attach_loras(
    model: nn.Module,
    lora_paths: list[tuple[str, float]],
    *,
    verbose: bool = True,
) -> RuntimeLoraReport:
    """Read LoRA files, remap their keys, and attach them unfused to ``model``.

    The one-call entry point for callers that hold paths rather than state
    dicts (the CLI, the pipelines, and the Phosphene panel helper). Paths may
    be local ``.safetensors`` files or already-resolved absolute paths; HF repo
    resolution belongs to the caller.
    """
    from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core_mlx.loader.sft_loader import SafetensorsStateDictLoader

    loader = SafetensorsStateDictLoader()
    specs: list[LoraStateDictWithStrength] = []
    for path, strength in lora_paths:
        specs.append(
            LoraStateDictWithStrength(
                state_dict=loader.load(str(Path(path)), sd_ops=LTXV_LORA_COMFY_RENAMING_MAP),
                strength=float(strength),
            )
        )
        if verbose:
            print(
                f"  Attaching LoRA (unfused): {path} (strength={float(strength):.2f})",
                file=sys.stderr,
                flush=True,
            )
    return attach_loras(model, specs, verbose=verbose)
