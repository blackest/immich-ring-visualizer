"""LoRA weight fusion into model state dicts.

Ported from ltx-core/src/ltx_core/loader/fuse_loras.py

Adapted for MLX:
- Uses mx.array instead of torch.Tensor
- Handles MLX quantized weights (int4/int8 with scales/biases) instead of FP8
- No CUDA-specific paths

.. warning::

   **Fusing into a quantized weight is lossy, and on q4 it is catastrophic.**
   ``quantize(dequantize(W) + B@A)`` rounds most of a rank-32 delta straight
   back onto the grid it came from: measured on the real LTX packs, **94.9 %**
   (2.3 q4) / **92.7 %** (2.5 q4) of the delta is destroyed, versus ~10 % at q8.
   That is what made two rounds of character-LoRA renders look like the LoRA
   "wasn't triggering".

   Prefer :mod:`ltx_core_mlx.loader.runtime_loras`, which applies the delta as
   an unfused runtime branch — exact at any bit width for ~1.5 % extra FLOPs.
   This module now warns, loudly and once, whenever it fuses into quantized
   weights; the warning can be acknowledged with ``quantized_ok=True`` by
   callers that genuinely want the fused arm (an A/B control, or a bf16 pack
   with a few quantized stragglers).
"""

from __future__ import annotations

import sys

import mlx.core as mx

from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.utils.weights import derive_quant_params

#: How many quantized fusions to measure before trusting the average. Each probe
#: is one extra dequantize + two norms on an already-materialised weight, so a
#: handful is free next to the fusion itself.
QUANTIZED_FUSION_PROBE_LIMIT = 8

#: Fraction of the delta that may be lost before the warning shouts. Set between
#: the two measured regimes: ~10 % at int8 (tolerable, and what upstream ships)
#: and ~93-95 % at int4 (identity-destroying).
QUANTIZED_FUSION_WARN_THRESHOLD = 0.25


def apply_loras(
    model_sd: StateDict,
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    dtype: mx.Dtype | None = None,
    destination_sd: StateDict | None = None,
    *,
    quantized_ok: bool = False,
    verbose: bool = True,
) -> StateDict:
    """Fuse one or more LoRA weight deltas into a model state dict.

    For each weight key in the model, finds matching lora_A/lora_B pairs
    in the LoRA state dicts, computes delta = B @ A * strength, and adds
    it to the original weight.

    Handles MLX quantized weights (int4/int8): dequantizes, fuses the
    LoRA delta, then re-quantizes — **lossily**; see the module warning and
    ``quantized_ok``.

    Args:
        model_sd: Base model state dict.
        lora_sd_and_strengths: List of (lora_state_dict, strength) pairs.
        dtype: Target dtype for fused weights. If None, uses source dtype.
        destination_sd: Optional existing dict to merge into.
        quantized_ok: Acknowledge quantized fusion and suppress the warning.
            Set it only where the loss is understood and wanted.
        verbose: Emit the applied-module count and the quantized-fusion
            warning. A count is printed unconditionally when a LoRA was
            supplied but matched **nothing** — issue #52's silent no-op.

    Returns:
        New StateDict with fused weights.
    """
    sd: dict[str, mx.array] = {}
    if destination_sd is not None:
        sd = dict(destination_sd.sd)

    size = 0
    dtypes: set[mx.Dtype] = set()
    fused_float = 0
    fused_quantized = 0
    # (bits, fraction of the delta destroyed) for the first few quantized fusions
    probe: list[tuple[int, float]] = []

    for key, weight in model_sd.sd.items():
        if weight is None:
            continue
        # Skip quantization metadata keys — handled with their weight
        if key.endswith(".scales") or key.endswith(".biases"):
            continue

        target_dtype = dtype if dtype is not None else weight.dtype

        # Check if this weight is quantized (has scales)
        scales_key = f"{key[: -len('.weight')]}.scales" if key.endswith(".weight") else None
        biases_key = f"{key[: -len('.weight')]}.biases" if key.endswith(".weight") else None
        is_quantized = scales_key is not None and scales_key in model_sd.sd

        deltas = _prepare_deltas(lora_sd_and_strengths, key)
        fused = _fuse_deltas(
            deltas,
            weight,
            key,
            target_dtype,
            is_quantized,
            scales_key,
            biases_key,
            model_sd,
            probe,
        )
        if deltas is not None:
            if is_quantized:
                fused_quantized += 1
            else:
                fused_float += 1

        sd.update(fused)
        for tensor in fused.values():
            dtypes.add(tensor.dtype)
            size += tensor.nbytes

    _report_fusion(
        lora_sd_and_strengths,
        fused_float,
        fused_quantized,
        probe,
        quantized_ok=quantized_ok,
        verbose=verbose,
    )

    if destination_sd is not None:
        return StateDict(sd=sd, size=size, dtype=dtypes)
    return StateDict(sd=sd, size=size, dtype=dtypes)


def _report_fusion(
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    fused_float: int,
    fused_quantized: int,
    probe: list[tuple[int, float]],
    *,
    quantized_ok: bool,
    verbose: bool,
) -> None:
    """Print the applied count, and shout if the quantized fusion just ate the delta."""
    if not lora_sd_and_strengths:
        return
    total = fused_float + fused_quantized

    if total == 0:
        # A LoRA was supplied and nothing matched. Never silent: this is #52,
        # where a prefix mismatch made every render a LoRA-free render that
        # still "worked".
        print(
            f"WARNING: {len(lora_sd_and_strengths)} LoRA(s) supplied but 0 modules matched — "
            "the render will be LoRA-free. Check the key remapping (sd_ops) and the "
            "block prefix.",
            file=sys.stderr,
            flush=True,
        )
        return

    if verbose:
        print(
            f"  LoRA fused into {total} modules ({fused_quantized} quantized, {fused_float} float)",
            file=sys.stderr,
            flush=True,
        )

    if not fused_quantized or quantized_ok or not probe:
        return

    bits = min(b for b, _ in probe)
    lost = sum(loss for _, loss in probe) / len(probe)
    if lost < QUANTIZED_FUSION_WARN_THRESHOLD:
        if verbose:
            print(
                f"  (int{bits} re-quantization cost {lost:.1%} of the LoRA delta, "
                f"measured on {len(probe)} modules)",
                file=sys.stderr,
                flush=True,
            )
        return

    print(
        "\n"
        + "=" * 72
        + "\n"
        f"WARNING: fused a LoRA into {fused_quantized} QUANTIZED modules (int{bits}) and\n"
        f"  MEASURED {lost:.1%} of the LoRA delta destroyed by re-quantization\n"
        f"  (mean over the first {len(probe)} fused modules, this exact pack + LoRA).\n"
        "  Re-quantizing W + B@A rounds most of the delta back onto the grid it came\n"
        "  from. Character identity does not survive this at int4.\n"
        "  Use the unfused runtime branch instead — exact at any bit width:\n"
        "      --lora-mode unfused                                    (CLI)\n"
        "      ltx_core_mlx.loader.runtime_loras.attach_loras(...)    (library)\n"
        "  Pass quantized_ok=True to acknowledge this and silence the warning.\n"
        + "=" * 72
        + "\n",
        file=sys.stderr,
        flush=True,
    )


def _prepare_deltas(
    lora_sd_and_strengths: list[LoraStateDictWithStrength],
    key: str,
) -> mx.array | None:
    """Compute the combined LoRA delta for a given weight key.

    Looks for matching lora_A.weight and lora_B.weight keys and computes
    delta = sum(B_i @ A_i * strength_i) for all matching LoRAs.

    Args:
        lora_sd_and_strengths: List of (lora_state_dict, strength) pairs.
        key: The model weight key to find LoRA deltas for.

    Returns:
        Combined delta array, or None if no LoRA matches this key.
    """
    deltas = []
    prefix = key[: -len(".weight")] if key.endswith(".weight") else key
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"

    for lsd, coef in lora_sd_and_strengths:
        if key_a not in lsd.sd or key_b not in lsd.sd:
            continue
        a = lsd.sd[key_a].astype(mx.float32)
        b = lsd.sd[key_b].astype(mx.float32)
        product = mx.matmul(b * coef, a)
        deltas.append(product)

    if len(deltas) == 0:
        return None
    if len(deltas) == 1:
        return deltas[0]
    return mx.sum(mx.stack(deltas, axis=0), axis=0)


def _fuse_deltas(
    deltas: mx.array | None,
    weight: mx.array,
    key: str,
    target_dtype: mx.Dtype,
    is_quantized: bool,
    scales_key: str | None,
    biases_key: str | None,
    model_sd: StateDict,
    probe: list[tuple[int, float]] | None = None,
) -> dict[str, mx.array]:
    """Fuse LoRA deltas into a weight, handling quantized and non-quantized cases.

    Args:
        deltas: Combined LoRA delta, or None if no LoRA applies.
        weight: Original model weight.
        key: Weight key name.
        target_dtype: Target dtype for the fused weight.
        is_quantized: Whether this weight is int4/int8 quantized.
        scales_key: Key for quantization scales (if quantized).
        biases_key: Key for quantization biases (if quantized).
        model_sd: Full model state dict (for accessing scales/biases).
        probe: Optional accumulator for ``(bits, fraction of delta destroyed)``
            samples; see :data:`QUANTIZED_FUSION_PROBE_LIMIT`.

    Returns:
        Dict of fused weight entries (may include scales/biases).
    """
    if deltas is None:
        # No LoRA for this key — copy original weight
        result = {key: weight.astype(target_dtype)}
        if is_quantized and scales_key:
            result[scales_key] = model_sd.sd[scales_key]
            if biases_key and biases_key in model_sd.sd:
                result[biases_key] = model_sd.sd[biases_key]
        return result

    if is_quantized:
        return _fuse_delta_with_quantized(deltas, weight, key, scales_key, biases_key, model_sd, probe)
    return _fuse_delta_with_float(deltas, weight, key, target_dtype)


def _fuse_delta_with_quantized(
    deltas: mx.array,
    weight: mx.array,
    key: str,
    scales_key: str | None,
    biases_key: str | None,
    model_sd: StateDict,
    probe: list[tuple[int, float]] | None = None,
) -> dict[str, mx.array]:
    """Fuse LoRA delta with a quantized weight.

    Dequantizes the weight, adds the LoRA delta, then re-quantizes.
    MLX quantized weights are stored as (out_features, in_features_packed)
    with separate scales and optional biases per group.

    Args:
        deltas: LoRA delta in float32.
        weight: Quantized weight array.
        key: Weight key name.
        scales_key: Key for quantization scales.
        biases_key: Key for quantization biases.
        model_sd: Full model state dict.
        probe: Optional accumulator for the measured delta loss.

    Returns:
        Dict with re-quantized weight, scales, and biases.
    """
    scales = model_sd.sd[scales_key] if scales_key else None
    biases = model_sd.sd.get(biases_key) if biases_key else None

    # Infer quantization parameters BEFORE dequantizing. The LoRA delta carries
    # the true (out, in) weight shape, so the real in_features disambiguates
    # (bits, group_size) for any group size (32/64/128) and bit width (4/8) —
    # see derive_quant_params, the single home shared with load-time quantization.
    in_features_packed = weight.shape[-1]
    if scales is not None and scales.ndim > 1:
        num_groups = scales.shape[1]
        in_features_real = deltas.shape[-1]
        bits, group_size = derive_quant_params(in_features_real, in_features_packed, num_groups)
    else:
        group_size = 64
        bits = 8

    # Dequantize: reconstruct the float weight from quantized representation
    original = mx.dequantize(
        weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
    ).astype(mx.float32)

    # Fuse
    new_weight = original + deltas.astype(mx.float32)

    new_quantized, new_scales, new_biases = mx.quantize(new_weight, group_size=group_size, bits=bits)

    if probe is not None and len(probe) < QUANTIZED_FUSION_PROBE_LIMIT:
        loss = _measure_delta_loss(
            original, deltas, new_quantized, new_scales, new_biases, group_size, bits
        )
        probe.append((bits, loss))

    result = {key: new_quantized}
    if scales_key:
        result[scales_key] = new_scales
    if biases_key:
        result[biases_key] = new_biases
    return result


def _measure_delta_loss(
    original: mx.array,
    deltas: mx.array,
    new_quantized: mx.array,
    new_scales: mx.array,
    new_biases: mx.array | None,
    group_size: int,
    bits: int,
) -> float:
    """``‖effective_delta − intended_delta‖ / ‖intended_delta‖`` for one module.

    The *effective* delta is what the model will actually run:
    ``dequantize(quantize(W + D)) − W``. Comparing it against ``D`` turns "we
    re-quantized" into a number the caller can act on — and makes the warning a
    measurement of this exact pack + LoRA rather than a remembered constant.
    """
    effective = (
        mx.dequantize(new_quantized, scales=new_scales, biases=new_biases, group_size=group_size, bits=bits).astype(
            mx.float32
        )
        - original
    )
    intended_norm = float(mx.sqrt(mx.sum(deltas.astype(mx.float32) ** 2)).item())
    if intended_norm == 0.0:
        return 0.0
    residual = float(mx.sqrt(mx.sum((effective - deltas.astype(mx.float32)) ** 2)).item())
    return residual / intended_norm


def _fuse_delta_with_float(
    deltas: mx.array,
    weight: mx.array,
    key: str,
    target_dtype: mx.Dtype,
) -> dict[str, mx.array]:
    """Fuse LoRA delta with a float (non-quantized) weight.

    Args:
        deltas: LoRA delta.
        weight: Original float weight.
        key: Weight key name.
        target_dtype: Target dtype for the fused weight.

    Returns:
        Dict with the fused weight.
    """
    fused = (weight.astype(mx.float32) + deltas.astype(mx.float32)).astype(target_dtype)
    return {key: fused}
