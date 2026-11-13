"""Gemma 4 **text tower** — vendored, text-only, no vision, no audio.

LTX-2.5 encodes prompts with a custom Gemma 4 12B fine-tune. This module builds
that tower directly instead of routing through ``mlx_lm.models.gemma4``, and the
reason is a version trap with no way out:

* ``mlx_lm.models.gemma4`` **first ships in mlx-lm 0.31.2**. No earlier release
  has it (verified file-by-file against the v0.29.0 … v0.31.3 tags).
* Phosphene pins ``mlx==0.31.1``. ``mlx`` **0.31.2** attenuates the LTX vocoder
  by ~22 dB (measured: −9 dB peak → −42 dB peak, same model, prompt and seed).
  That pin is a ship-blocker fix, not a preference.
* ``mlx-lm`` 0.31.2 declares ``mlx>=0.30.4``, so it *would* co-install with the
  pinned ``mlx`` — but its gemma4 is the first cut, before **#1158** (KV-shared
  layers build ``k_proj``/``v_proj``/``k_norm`` that the checkpoint does not
  contain) and **#1240** (``sanitize`` does not strip them either).
* ``mlx-lm`` 0.31.3 carries #1158 but declares ``mlx>=0.31.2`` — it *drags the
  vocoder regression in as a dependency*.

So every available upgrade costs either correct audio or a correct tower. The
tower is ~600 lines of arithmetic that runs **once per render**; the vocoder
regression is unfixable from our side. Vendoring is the cheaper risk, and it
permanently decouples the text encoder from mlx-lm's release cadence.

Derived from two independent references that agree throughout:

* ``mlx_lm/models/gemma4_text.py`` @ ``main`` (Apple, MIT) — with #1158 and
  #1240 already applied, which is precisely what no *release* gives us;
* ``comfy/text_encoders/gemma4.py`` (ComfyUI) — the PyTorch reference the LTX-2.5
  ComfyUI support was merged against.

What Gemma 4 changes versus Gemma 3, all of which fail *silently* if missed:

===========================  =============================  ==========================
Feature                      Gemma 3                        Gemma 4
===========================  =============================  ==========================
Head dim                     one, all layers                ``head_dim`` on sliding
                                                            layers, ``global_head_dim``
                                                            on full-attention layers
RoPE                         full rotary, both layer types  full rotary on sliding;
                                                            **partial** ("proportional")
                                                            on full-attention layers
Value projection             always present                 absent on full-attention
                                                            layers when ``k_eq_v``
                                                            (V *is* the raw K projection)
Value norm                   none                           scale-free RMSNorm on V
Attention scale              ``query_pre_attn_scalar**-.5``  **1.0**
Per-block output scale       none                           learned ``layer_scalar``
Layer pattern                5 sliding : 1 full             ``layer_types`` from config
KV sharing                   none                           late layers may reuse an
                                                            earlier layer's K/V
FFN                          single MLP                     optional dense+MoE pair;
                                                            optional double-wide MLP
Per-layer input embeddings   none                           optional (small variants)
===========================  =============================  ==========================

Only the text tower is built. ``gemma4_unified`` — what LTX-2.5's encoder
declares — is the encoder-free multimodal packaging; its vision and audio towers
are dropped at sanitize time and never allocated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#: Reference values for Gemma 4 12B, the tower LTX-2.5 fine-tunes. Kept as
#: documentation and as the fixture the structural tests assert against — never
#: as a default that could paper over a checkpoint that says something else.
GEMMA4_12B_REFERENCE: dict[str, Any] = {
    "hidden_size": 3840,
    "num_hidden_layers": 48,
    "intermediate_size": 15360,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "head_dim": 256,
    "global_head_dim": 512,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "hidden_size_per_layer_input": 0,
    "num_kv_shared_layers": 0,
    "use_double_wide_mlp": False,
    "enable_moe_block": False,
    "final_logit_softcapping": 30.0,
}

_DEFAULT_ROPE_PARAMETERS: dict[str, dict[str, Any]] = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {
        "partial_rotary_factor": 1.0,
        "rope_theta": 10000.0,
        "rope_type": "default",
    },
}

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


@dataclass
class Gemma4TextConfig:
    """Architecture of a Gemma 4 text tower, read from a checkpoint's config.

    Every field is read, never assumed: the defaults exist so a synthetic tiny
    config in a test is writable in three lines, not so a real checkpoint can
    quietly fall back to them.
    """

    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    num_global_key_value_heads: int | None = None
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    hidden_size_per_layer_input: int = 256
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = True
    enable_moe_block: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    layer_types: list[str] | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    rope_traditional: bool = False
    max_position_embeddings: int = 131072
    final_logit_softcapping: float | None = 30.0
    model_type: str = "gemma4_text"

    def __post_init__(self) -> None:
        if self.rope_parameters is None:
            self.rope_parameters = {
                key: dict(value) for key, value in _DEFAULT_ROPE_PARAMETERS.items()
            }
        if self.layer_types is None:
            # Upstream's convention: the LAST layer of every window of
            # ``sliding_window_pattern`` is the full-attention one. Comfy spells
            # the same thing as ``sliding_attention=[1024, ..., False]``.
            pattern = [SLIDING_ATTENTION] * (self.sliding_window_pattern - 1) + [FULL_ATTENTION]
            repeats = self.num_hidden_layers // len(pattern) + 1
            self.layer_types = (pattern * repeats)[: self.num_hidden_layers]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but the config "
                f"declares {self.num_hidden_layers} layers"
            )
        unknown = set(self.layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")

    # -- derived ----------------------------------------------------------

    @property
    def first_kv_shared_layer(self) -> int:
        """Index of the first layer that reuses an earlier layer's K/V."""
        if self.num_kv_shared_layers <= 0:
            return self.num_hidden_layers
        return self.num_hidden_layers - self.num_kv_shared_layers

    def head_dim_for(self, layer_type: str) -> int:
        """Full-attention layers are wider per head than sliding ones."""
        if layer_type == FULL_ATTENTION and self.global_head_dim:
            return self.global_head_dim
        return self.head_dim

    def uses_k_eq_v(self, layer_type: str) -> bool:
        """``V`` reuses the ``K`` projection — full-attention layers only."""
        return bool(self.attention_k_eq_v) and layer_type == FULL_ATTENTION

    def kv_heads_for(self, layer_type: str) -> int:
        if self.uses_k_eq_v(layer_type) and self.num_global_key_value_heads is not None:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    def window_for(self, layer_type: str) -> int | None:
        return self.sliding_window if layer_type == SLIDING_ATTENTION else None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Gemma4TextConfig:
        """Build from a checkpoint config, unwrapping a nested text config.

        A ``gemma4_unified`` config carries the text tower under ``text_config``
        with vision/audio configs beside it as decoys. Reading ``hidden_size``
        off the top level of such a file yields the multimodal width and sizes
        every projection wrong — with no error at load time.
        """
        source = _unwrap_text_config(config)
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        kwargs = {key: value for key, value in source.items() if key in known}

        # Comfy spells the layer pattern as a per-index list of window sizes
        # (``False`` meaning full attention). Translate it if that is what we get.
        sliding = source.get("sliding_attention")
        if kwargs.get("layer_types") is None and isinstance(sliding, (list, tuple)) and sliding:
            num_layers = int(kwargs.get("num_hidden_layers", cls.num_hidden_layers))
            kwargs["layer_types"] = [
                SLIDING_ATTENTION if sliding[i % len(sliding)] else FULL_ATTENTION
                for i in range(num_layers)
            ]
            windows = [w for w in sliding if w]
            if windows and "sliding_window" not in kwargs:
                kwargs["sliding_window"] = int(windows[0])

        # HF nests rope settings under a few different spellings.
        if kwargs.get("rope_parameters") is None:
            rope = source.get("rope_scaling") or source.get("rope_config")
            if isinstance(rope, dict) and rope:
                kwargs["rope_parameters"] = rope

        return cls(**kwargs)


def _unwrap_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the text tower's own config sub-tree, however it is nested."""
    for key in ("text_config", "language_model_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and "hidden_size" in nested:
            merged = dict(nested)
            # ``vocab_size`` sometimes lives only at the top level.
            for inherited in ("vocab_size", "model_type"):
                if inherited not in merged and inherited in config:
                    merged[inherited] = config[inherited]
            return merged
    return config


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNormNoScale(nn.Module):
    """RMSNorm with no learnable gain — Gemma 4 normalises V with this.

    Carries no parameters *on purpose*: giving it a weight would add a tensor the
    checkpoint does not have, which loads as random noise rather than failing.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class ProportionalRoPE(nn.Module):
    """Partial rotary embedding, vendored because mlx-lm 0.31.1 has no idea.

    Gemma 4's full-attention layers rotate only the first
    ``partial_rotary_factor`` of each head and leave the rest untouched (NoPE) —
    for the 12B tower that is 128 of 512 dims. Two details are load-bearing and
    both are silent when wrong:

    1. The frequency exponent is divided by the **full** head dim, not by the
       rotated width. That is what "proportional" names, and it is why the
       rotated dims here do *not* match a plain RoPE of width ``rotated_dims``.
    2. The unrotated tail is padded with **infinite** period (angle 0 → cos 1,
       sin 0), i.e. exact identity. Omitting the pad silently rotates dims that
       must not move.

    ``mlx_lm.models.rope_utils`` in 0.31.1 raises ``ValueError: Unsupported RoPE
    type proportional`` — loud, which is the one mercy here. Anything that
    "helpfully" fell back to a default RoPE would produce a tower that loads,
    runs, and encodes every prompt slightly wrong.
    """

    def __init__(
        self,
        dims: int,
        rotated_dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        factor: float = 1.0,
    ):
        super().__init__()
        if rotated_dims > dims:
            raise ValueError(f"rotated_dims ({rotated_dims}) must not exceed dims ({dims})")
        if rotated_dims % 2 or dims % 2:
            raise ValueError(f"dims ({dims}) and rotated_dims ({rotated_dims}) must both be even")
        self.dims = dims
        self.rotated_dims = rotated_dims
        self.traditional = traditional
        exponents = mx.arange(0, rotated_dims, 2, dtype=mx.float32) / dims
        self._freqs = mx.concatenate(
            [factor * (base**exponents), mx.full(((dims - rotated_dims) // 2,), mx.inf)]
        )

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def build_rope(config: Gemma4TextConfig, layer_type: str) -> nn.Module:
    """Build the rotary embedding a given layer type wants."""
    params = (config.rope_parameters or {}).get(layer_type, {})
    rope_type = params.get("rope_type") or params.get("type") or "default"
    base = float(params.get("rope_theta", 10000.0))
    dims = config.head_dim_for(layer_type)
    factor = float(params.get("factor", 1.0))
    partial = float(params.get("partial_rotary_factor", 1.0))

    if rope_type == "proportional" or partial < 1.0:
        return ProportionalRoPE(
            dims=dims,
            rotated_dims=int(dims * partial),
            traditional=config.rope_traditional,
            base=base,
            factor=factor,
        )
    if rope_type in ("default", "linear"):
        scale = 1.0 / factor if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=config.rope_traditional, base=base, scale=scale)
    raise ValueError(
        f"Unsupported RoPE type {rope_type!r} for a Gemma 4 {layer_type} layer. "
        "Supported: 'proportional', 'default', 'linear'."
    )


def gelu_gate(gate: mx.array, x: mx.array) -> mx.array:
    """GeGLU — tanh-approximate GELU on the gate, matching both references."""
    return nn.gelu_approx(gate) * x


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class Gemma4MLP(nn.Module):
    """GeGLU feed-forward. Widened 2x on KV-shared layers when configured."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int = 0, intermediate_size: int | None = None):
        super().__init__()
        if intermediate_size is None:
            is_kv_shared = layer_idx >= config.first_kv_shared_layer and config.num_kv_shared_layers > 0
            double = config.use_double_wide_mlp and is_kv_shared
            intermediate_size = config.intermediate_size * (2 if double else 1)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(gelu_gate(self.gate_proj(x), self.up_proj(x)))


class Gemma4Router(nn.Module):
    """MoE router: scaled RMSNorm -> project -> top-k -> softmax -> rescale."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        if not config.num_experts or not config.top_k_experts:
            raise ValueError("MoE block requested without num_experts / top_k_experts")
        self.eps = config.rms_norm_eps
        self.top_k = config.top_k_experts
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
        scores = self.proj(x)
        idx = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.softmax(mx.take_along_axis(scores, idx, axis=-1), axis=-1)
        return idx, weights * self.per_expert_scale[idx]


class Gemma4Experts(nn.Module):
    """Sparse experts over ``SwitchGLU`` (present in mlx-lm 0.31.1)."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchGLU

        class _GeGLU(nn.Module):
            def __call__(self, x, gate):
                return gelu_gate(gate, x)

        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=_GeGLU(),
            bias=False,
        )

    def __call__(self, x: mx.array, idx: mx.array, weights: mx.array) -> mx.array:
        return (mx.expand_dims(weights, -1) * self.switch_glu(x, idx)).sum(-2)


class Gemma4Attention(nn.Module):
    """Self-attention for one layer, shaped by that layer's type.

    Three things differ from Gemma 3 and each is silent when wrong:

    * ``scale = 1.0``. Gemma 4 folds the softmax temperature into ``q_norm``;
      applying the usual ``1/sqrt(head_dim)`` on top rescales every logit.
    * On ``k_eq_v`` layers there is **no** ``v_proj``, and V is the **raw** K
      projection — taken *before* ``k_norm`` and *before* RoPE, then passed
      through a scale-free RMSNorm. Norming or rotating V here is wrong, and
      both produce output that looks fine and decodes badly.
    * KV-shared layers own no K/V parameters at all. Building them would create
      tensors absent from the checkpoint, which is the mosaic failure mode:
      random weights, no load error.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == SLIDING_ATTENTION
        self.window = config.window_for(self.layer_type)
        self.has_kv = layer_idx < config.first_kv_shared_layer
        self.use_k_eq_v = config.uses_k_eq_v(self.layer_type)

        self.head_dim = config.head_dim_for(self.layer_type)
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.kv_heads_for(self.layer_type)
        self.scale = 1.0

        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.has_kv:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = RMSNormNoScale(eps=config.rms_norm_eps)

        self.rope = build_rope(config, self.layer_type)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        shared_kv: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        elif not self.has_kv:
            raise ValueError(
                f"Layer {self.layer_idx} is a KV-shared layer but no shared K/V was supplied. "
                "The tower must feed it the K/V of the last non-shared layer of the same type."
            )
        else:
            raw_k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            # V is the RAW k projection on k_eq_v layers — pre-norm, pre-RoPE.
            raw_v = raw_k if self.use_k_eq_v else self.v_proj(x).reshape(
                B, L, self.n_kv_heads, self.head_dim
            )
            keys = self.rope(self.k_norm(raw_k).transpose(0, 2, 1, 3))
            values = self.v_norm(raw_v).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class Gemma4DecoderLayer(nn.Module):
    """One Gemma 4 block: four norms, optional MoE, and a learned output scale."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)

        eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(config.hidden_size, eps=eps)

        self.enable_moe = bool(config.enable_moe_block)
        if self.enable_moe:
            self.router = Gemma4Router(config)
            self.experts = Gemma4Experts(config)
            self.pre_feedforward_layernorm_2 = nn.RMSNorm(config.hidden_size, eps=eps)
            self.post_feedforward_layernorm_1 = nn.RMSNorm(config.hidden_size, eps=eps)
            self.post_feedforward_layernorm_2 = nn.RMSNorm(config.hidden_size, eps=eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(config.hidden_size, eps=eps)

        # Present on every Gemma 4 variant, independent of everything above.
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        shared_kv: tuple[mx.array, mx.array] | None = None,
        per_layer_input: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        residual = x
        h, kv = self.self_attn(self.input_layernorm(x), mask=mask, shared_kv=shared_kv)
        h = residual + self.post_attention_layernorm(h)

        residual = h
        if self.enable_moe:
            dense = self.post_feedforward_layernorm_1(self.mlp(self.pre_feedforward_layernorm(h)))
            idx, weights = self.router(h)
            sparse = self.post_feedforward_layernorm_2(
                self.experts(self.pre_feedforward_layernorm_2(h), idx, weights)
            )
            h = dense + sparse
        else:
            h = self.mlp(self.pre_feedforward_layernorm(h))
        h = residual + self.post_feedforward_layernorm(h)

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = h
            gate = nn.gelu_approx(self.per_layer_input_gate(h)) * per_layer_input
            h = residual + self.post_per_layer_input_norm(self.per_layer_projection(gate))

        return h * self.layer_scalar, kv


# ---------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------


class Gemma4TextTower(nn.Module):
    """The text tower alone — what LTX uses, and nothing else.

    The only forward this needs is a single full-sequence pass with no KV cache:
    LTX encodes the prompt **once per render**. That removes the cache
    bookkeeping the mlx-lm implementation carries for generation and leaves the
    arithmetic, which is the part that has to be right.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )

        # Which earlier layer each layer borrows K/V from (itself, when it owns them).
        self.kv_source = list(range(config.num_hidden_layers))
        if config.num_kv_shared_layers > 0:
            boundary = config.first_kv_shared_layer
            last_of_type: dict[str, int] = {}
            for i in range(boundary):
                last_of_type[config.layer_types[i]] = i
            for j in range(boundary, config.num_hidden_layers):
                layer_type = config.layer_types[j]
                if layer_type not in last_of_type:
                    raise ValueError(
                        f"layer {j} ({layer_type}) shares K/V but no non-shared "
                        f"{layer_type} layer exists before index {boundary}"
                    )
                self.kv_source[j] = last_of_type[layer_type]

    # -- masks ------------------------------------------------------------

    def _build_masks(
        self,
        seq_len: int,
        attention_mask: mx.array | None,
        dtype: mx.Dtype,
    ) -> dict[str, mx.array]:
        """One additive mask per layer type: causal + padding + sliding window.

        ``-1e9`` rather than ``finfo.min`` deliberately, matching the Gemma 3
        encoder beside this one. The masks are **added**, and two ``finfo.min``
        terms saturate to ``-inf``; a fully-masked row (every left-padding query
        attends only to padding) then softmaxes to ``NaN`` and poisons the whole
        batch. At ``-1e9`` such a row degenerates to a uniform average instead,
        and its output is discarded downstream anyway.
        """
        neg = -1e9
        causal = mx.triu(mx.full((seq_len, seq_len), neg, dtype=dtype), k=1)

        pad = None
        if attention_mask is not None:
            # (B, T) with 1 = real token, 0 = padding -> (B, 1, 1, T)
            pad = (1 - attention_mask[:, None, None, :].astype(dtype)) * neg

        masks: dict[str, mx.array] = {}
        for layer_type in set(self.config.layer_types):
            base = causal
            window = self.config.window_for(layer_type)
            if window is not None and seq_len > window:
                # Forbid attending further back than ``window`` tokens.
                rows = mx.arange(seq_len)[:, None]
                cols = mx.arange(seq_len)[None, :]
                too_old = (rows - cols) >= window
                base = base + mx.where(too_old, mx.array(neg, dtype=dtype), mx.array(0, dtype=dtype))
            combined = base[None, None, :, :]
            if pad is not None:
                combined = combined + pad
            masks[layer_type] = combined.astype(dtype)
        return masks

    def _per_layer_inputs(self, token_ids: mx.array, h: mx.array) -> list[mx.array | None]:
        config = self.config
        if not config.hidden_size_per_layer_input:
            return [None] * config.num_hidden_layers
        shape = (*h.shape[:-1], config.num_hidden_layers, config.hidden_size_per_layer_input)
        projection = self.per_layer_model_projection(h) * (config.hidden_size**-0.5)
        projection = self.per_layer_projection_norm(projection.reshape(shape))
        if token_ids is not None:
            embedded = self.embed_tokens_per_layer(token_ids).reshape(shape)
            embedded = embedded * (config.hidden_size_per_layer_input**0.5)
            projection = (projection + embedded) * (0.5**0.5)
        return [projection[:, :, i, :] for i in range(config.num_hidden_layers)]

    # -- forward ----------------------------------------------------------

    def embed(self, token_ids: mx.array) -> mx.array:
        """Token embeddings scaled by ``sqrt(hidden_size)``.

        The scalar is rounded through bfloat16 before it multiplies, matching
        both references (and the Gemma 3 path already in this package) — the
        rounding is observable, so doing it in fp32 is a quiet divergence.
        """
        h = self.embed_tokens(token_ids)
        scale = mx.array(self.config.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)
        return h * scale

    def all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
        eval_every: int | None = None,
    ) -> list[mx.array]:
        """Embedding output plus every layer output — ``num_hidden_layers + 1``.

        This is what the LTX feature extractor consumes: it concatenates all of
        them, which is why ``projection_input_dim`` is
        ``hidden_size * (num_hidden_layers + 1)``.

        The final RMS norm is deliberately **not** applied to the last entry —
        that mirrors the Gemma 3 path this package already ships and renders
        with. Changing the convention for Gemma 4 alone would silently shift the
        last of 49 feature planes.
        """
        h = self.embed(token_ids)
        states = [h]

        masks = self._build_masks(token_ids.shape[1], attention_mask, h.dtype)
        per_layer = self._per_layer_inputs(token_ids, h)

        if eval_every is None:
            eval_every = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1"))

        cached_kv: dict[int, tuple[mx.array, mx.array]] = {}
        for i, layer in enumerate(self.layers):
            source = self.kv_source[i]
            shared = cached_kv.get(source) if source != i else None
            h, kv = layer(
                h,
                mask=masks[layer.layer_type],
                shared_kv=shared,
                per_layer_input=per_layer[i],
            )
            if source == i and i in set(self.kv_source[i + 1 :]):
                cached_kv[i] = kv
            states.append(h)
            # Per-layer eval keeps each Metal command buffer inside the macOS GPU
            # watchdog window; see the Gemma 3 encoder for the full rationale.
            if eval_every and (i + 1) % eval_every == 0:
                mx.eval(h)

        return states

    def __call__(self, token_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """Final normalised hidden state."""
        return self.norm(self.all_hidden_states(token_ids, attention_mask)[-1])


# ---------------------------------------------------------------------------
# Weight sanitizing
# ---------------------------------------------------------------------------

#: Prefixes belonging to towers we never build. ``gemma4_unified`` packs its
#: encoder-free vision and audio embedders alongside the text tower; loading
#: them would cost gigabytes to compute nothing.
NON_TEXT_PREFIXES: tuple[str, ...] = (
    "vision_tower",
    "audio_tower",
    "vision_model",
    "audio_model",
    "multi_modal_projector",
    # LTX-2.5's encoder ships this beside multi_modal_projector — the audio
    # half of the same unified-multimodal packaging. Confirmed present in the
    # real checkpoint header; without it a strict load fails on a tensor this
    # tower correctly never builds.
    "audio_projector",
    "embed_vision",
    "embed_audio",
    "vision_embedder",
    "audio_embedder",
)

#: Buffers some exporters emit that are not parameters of anything.
_DROPPED_SUBSTRINGS: tuple[str, ...] = (
    "self_attn.rotary_emb",
    "input_max",
    "input_min",
    "output_max",
    "output_min",
)


def sanitize_weights(
    weights: dict[str, mx.array],
    config: Gemma4TextConfig,
) -> dict[str, mx.array]:
    """Map checkpoint keys onto this module's names, dropping what we never build.

    Handles the three nestings a Gemma 4 checkpoint shows up in
    (``model.``, ``language_model.``, ``model.language_model.``), drops the
    vision/audio towers, drops K/V parameters belonging to KV-shared layers
    (which this module does not allocate), and splits packed MoE expert weights.

    Anything left unrecognised is **kept**, so a load-time key mismatch surfaces
    as a loud failure rather than a silently skipped tensor.
    """
    out: dict[str, mx.array] = {}
    boundary = config.first_kv_shared_layer

    for key, value in weights.items():
        name = key
        for prefix in ("model.language_model.", "language_model.model.", "language_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break

        if name.startswith(NON_TEXT_PREFIXES) or key.startswith(NON_TEXT_PREFIXES):
            continue
        if any(dropped in name for dropped in _DROPPED_SUBSTRINGS):
            continue
        if name.startswith("lm_head"):
            # Text-encoder use never produces logits.
            continue

        if config.num_kv_shared_layers > 0 and any(
            marker in name
            for marker in (".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.k_norm", ".self_attn.v_norm")
        ):
            index = _layer_index(name)
            if index is not None and index >= boundary:
                continue

        if config.attention_k_eq_v and ".self_attn.v_proj" in name:
            index = _layer_index(name)
            if index is not None and config.layer_types[index] == FULL_ATTENTION:
                continue

        if name.endswith(".experts.gate_up_proj"):
            base = name.removesuffix(".gate_up_proj")
            gate, up = map(mx.contiguous, mx.split(value, 2, axis=-2))
            out[f"{base}.switch_glu.gate_proj.weight"] = gate
            out[f"{base}.switch_glu.up_proj.weight"] = up
            continue
        if name.endswith(".experts.down_proj"):
            out[f"{name.removesuffix('.down_proj')}.switch_glu.down_proj.weight"] = value
            continue

        out[name] = value

    return out


def _layer_index(key: str) -> int | None:
    """Extract ``N`` from ``...layers.N....``; ``None`` when the key has none."""
    marker = "layers."
    position = key.find(marker)
    if position < 0:
        return None
    tail = key[position + len(marker) :]
    digits = tail.split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


__all__ = [
    "FULL_ATTENTION",
    "GEMMA4_12B_REFERENCE",
    "NON_TEXT_PREFIXES",
    "SLIDING_ATTENTION",
    "Gemma4Attention",
    "Gemma4DecoderLayer",
    "Gemma4MLP",
    "Gemma4TextConfig",
    "Gemma4TextTower",
    "ProportionalRoPE",
    "RMSNormNoScale",
    "build_rope",
    "sanitize_weights",
]
