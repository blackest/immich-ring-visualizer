"""DurationHead — predict a shot's natural length from the prompt alone.

Ported from ``comfy/ldm/lightricks/duration_head.py`` (ComfyUI commit
57ce8e1a, "Add support for LTX 2.5") and cross-checked against the official
``ltx-core`` v1.2.0 implementation.

This is the cheapest real feature in the 2.5 release: a 4 MB optional
component that reads the *connector output tokens* — the same embeddings the
DiT cross-attends to — and returns a duration in seconds, without running a
single diffusion step. "Let the model pick the length" costs one small
forward pass, not a render.

Weight keys (after :func:`normalize_state_dict` strips the prefix):

    video_input_proj.{weight,bias}          (pooler_hidden, 4096)
    video_modality_emb                      (pooler_hidden,)
    audio_input_proj.{weight,bias}          (pooler_hidden, 2048)
    audio_modality_emb                      (pooler_hidden,)
    attention_pooler.query_tokens           (num_queries, pooler_hidden)
    attention_pooler.cross_attn.{in_proj_weight,in_proj_bias}
    attention_pooler.cross_attn.out_proj.{weight,bias}
    mlp_hidden.{weight,bias}                (mlp_hidden, pooler_hidden*num_queries)
    mlp_out.{weight,bias}                   (1, mlp_hidden)

The upstream checkpoint stores the pooler's attention as a single packed
``in_proj_weight`` of shape ``(3 * hidden, hidden)`` — PyTorch's
``nn.MultiheadAttention`` layout — so this port keeps that packing rather
than splitting into three Linears. Splitting would read the same bytes and
still be correct, but the key names would stop matching the file and every
future re-pin would need a remap table.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``.

    Mirrors ``torch.nn.MultiheadAttention(batch_first=True)`` with
    ``need_weights=False``: packed QKV projection, scaled dot-product
    attention over ``num_heads`` heads, then the output projection.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        # Packed (3*H, H) / (3*H,) to match torch's MultiheadAttention keys.
        self.cross_attn = _PackedMHA(hidden_dim, num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Args: ``tokens`` (B, T, H). Returns pooled (B, num_queries, H)."""
        queries = mx.broadcast_to(self.query_tokens[None], (tokens.shape[0], self.num_queries, self.hidden_dim))
        return self.cross_attn(queries, tokens)


class _PackedMHA(nn.Module):
    """Multi-head attention with torch's packed ``in_proj_weight`` layout."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.in_proj_weight = mx.zeros((3 * hidden_dim, hidden_dim))
        self.in_proj_bias = mx.zeros((3 * hidden_dim,))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        h = self.in_proj_weight.shape[1]
        wq, wk, wv = self.in_proj_weight[:h], self.in_proj_weight[h : 2 * h], self.in_proj_weight[2 * h :]
        bq, bk, bv = self.in_proj_bias[:h], self.in_proj_bias[h : 2 * h], self.in_proj_bias[2 * h :]

        q = query @ wq.T + bq
        k = key_value @ wk.T + bk
        v = key_value @ wv.T + bv

        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = q.reshape(B, Tq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class DurationHead(nn.Module):
    """Predict duration in seconds from one or both connector outputs.

    Both modalities are projected into a shared pooler space, tagged with a
    learned per-modality embedding, concatenated along the token axis, pooled
    by cross-attention into ``num_queries`` vectors, and mapped through a
    GELU MLP. The final ``exp()`` is what makes the output a duration: the
    head regresses in log-seconds, so it cannot predict a negative length.
    """

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim, num_queries=num_queries, num_heads=num_pooler_heads
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        """Args: ``video_tokens`` (B, T_v, 4096), ``audio_tokens`` (B, T_a, 2048).

        At least one is required. Returns duration in seconds, shape (B,).
        """
        groups = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        if not groups:
            raise ValueError("DurationHead requires at least one of video_tokens / audio_tokens")

        pooled = self.attention_pooler(mx.concatenate(groups, axis=1))
        pooled = pooled.reshape(pooled.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled))
        return mx.exp(self.mlp_out(hidden).squeeze(-1))


def normalize_state_dict(sd: dict) -> dict:
    """Strip whichever checkpoint prefix the duration head arrived under.

    The 4 MB component ships both standalone (``duration_head.*``) and packed
    inside a monolithic checkpoint (``model.diffusion_model.duration_head.*``).
    """
    for prefix in ("model.diffusion_model.duration_head.", "duration_head."):
        stripped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        if stripped:
            return stripped
    return sd


def load_duration_head(path, **kwargs) -> DurationHead:
    """Build a :class:`DurationHead` and load a safetensors file into it.

    Dims are taken from the file's own shapes rather than assumed, so a
    checkpoint with a different pooler width loads without a code change.
    """
    weights = dict(mx.load(str(path)))
    sd = normalize_state_dict(weights)

    if "video_input_proj.weight" in sd and "video_cross_attention_dim" not in kwargs:
        pooler_hidden, video_dim = sd["video_input_proj.weight"].shape
        kwargs.setdefault("video_cross_attention_dim", video_dim)
        kwargs.setdefault("pooler_hidden_dim", pooler_hidden)
    if "audio_input_proj.weight" in sd and "audio_cross_attention_dim" not in kwargs:
        kwargs.setdefault("audio_cross_attention_dim", sd["audio_input_proj.weight"].shape[1])
    if "attention_pooler.query_tokens" in sd:
        kwargs.setdefault("num_queries", sd["attention_pooler.query_tokens"].shape[0])
    if "mlp_out.weight" in sd:
        kwargs.setdefault("mlp_hidden", sd["mlp_out.weight"].shape[1])

    head = DurationHead(**kwargs)
    head.load_weights(list(sd.items()))
    return head


def seconds_to_num_frames(
    seconds: float,
    frame_rate: float,
    min_seconds: float,
    max_seconds: float,
    time_scale: int = 8,
) -> int:
    """Convert seconds to a frame count on the VAE's causal ``8k + 1`` grid.

    Clamps to ``[min_seconds, max_seconds]``, then floors onto the grid. A
    floor that undershoots the minimum bumps up to the next grid point
    instead — otherwise a short prediction could round to a frame count the
    pipeline rejects.
    """
    min_frames = max(1, round(min_seconds * frame_rate))
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = (raw_frames - 1) // time_scale * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


__all__ = [
    "AttentionPooler",
    "DurationHead",
    "load_duration_head",
    "normalize_state_dict",
    "seconds_to_num_frames",
]
