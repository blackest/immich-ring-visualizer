"""LTX-2.5 diffusion video VAE decoder (``NADiffusionDecoder``) in MLX.

This is the *other* video VAE. LTX-2.5 ships two decoders behind two files:

    ltx-2.5-video-vae-conv-bf16.safetensors   170 tensors, the legacy conv decoder
    ltx-2.5-video-vae-bf16.safetensors        396 tensors, this one

Lightricks' description of this one is "sharper faces, legible text, fewer smears in
fast motion", and the vendor's own reference workflows load it, not the conv file.
It is detected by exactly one key, ``decoder.conv_in_x_t.weight``.

Shape contract is identical to the conv decoder -- 8x temporal, 32x spatial -- so it
is a drop-in at the pipeline boundary. ``decode()`` takes and returns the same
layouts as :class:`~ltx_core_mlx.model.video_vae.video_vae.VideoDecoder`.

Architecture (all of it channels-last, all of it ``Linear`` -- there is not one
convolution in this decoder, so none of the PyTorch->MLX conv permutation applies):

    stages 1-4   deterministic: NA transformer blocks + linear pixel-shuffle
                 upsample, widths (2048, 1024, 512, 512), depths (4, 6, 4, 2),
                 kernels (3,7,7) (3,7,7) (3,5,5) (3,5,5). Produces a *context*
                 volume at (T_out, H_out/4, W_out/4, 256).
    stage 5      diffusion: 8 ``DiffusionNABlock`` with kernel (11,11,11) that
                 denoise patchified noise ``x_t`` guided by that context through
                 shared AdaLN-Zero scale/shift.

``default_num_inference_steps = 1`` and ``model_output_type = "x0"``: one forward
pass yields the pixels. There is no Euler loop, so the runtime cost is one decoder
pass -- the expense is the neighborhood attention at stage 5, which runs on ~2.6M
tokens with 1331 keys each at 768x448x121.

Ported from ``comfy/ldm/lightricks/vae/na_diffusion_decoder.py`` @ Comfy-Org/ComfyUI
`57ce8e1a`, cross-checked tensor-for-tensor against the vendor's own
``ltx_core/model/video_vae/diffusion_video_decoder.py`` @ Lightricks/LTX-2 v1.2.0.
Both were read in full; where they differ in spelling (fused ``attn.qkv`` vs split
``to_q/to_k/to_v``; ``t_embedder.mlp.{0,2}`` vs a diffusers wrapper) this port follows
the spelling the **checkpoint** uses, which is ComfyUI's.

``decoder.type_emb`` -- one (128,) tensor -- is carried into the pack and held on the
module but never read. Neither reference implementation references it anywhere (both
repos grepped, including GitHub code search); ComfyUI loads the checkpoint
non-strictly and drops it silently. We keep it so that a strict load still passes and
so that nothing is dropped without a record. :func:`load_diffusion_decoder` asserts
strictness, which is the same discipline as the converter's unmapped-key abort.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from ltx_core_mlx.model.video_vae.na3d import (
    apply_abs_rope,
    default_rope_dim_split,
    na3d,
    rope_inv_freqs,
)
from ltx_core_mlx.model.video_vae.ops import PerChannelStatistics

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "LTX_25_DIFFUSION_VAE_CONFIG",
    "AdaLNZero",
    "DiffusionNABlock",
    "DiffusionVideoDecoder",
    "NABlock",
    "NADiffusionDecoder",
    "NeighborhoodAttention3D",
    "is_diffusion_decoder_weights",
    "load_diffusion_decoder",
]


#: Token budget for one chunk of a channel-wise Linear / SwiGLU application. Bounds
#: the ``[chunk, hidden]`` workspace; ``hidden`` is 4x the model width at stage 5.
#: Override with ``LTX2_DIFFVAE_TOKEN_CHUNK``.
DEFAULT_TOKEN_CHUNK = 1 << 18


#: The decoder half of the config carried in the checkpoint's own ``__metadata__``
#: (``ltx-2.5-video-vae-bf16.safetensors``, ``model_version`` 2.5.0). Kept here as the
#: fallback only -- :func:`load_diffusion_decoder` prefers the file's own copy, because
#: a decoder built from a stale constant against moved weights is the mosaic failure.
LTX_25_DIFFUSION_VAE_CONFIG: dict[str, Any] = {
    "in_channels": 128,
    "out_channels": 3,
    "patch_size": 4,
    "head_dim": 64,
    "stage_channels": [2048, 1024, 512, 512, 256],
    "stage_depths": [4, 6, 4, 2, 8],
    "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
    "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
    "stage5_kernel": [11, 11, 11],
    "t_emb_dim": 384,
    "timestep_scale_multiplier": 1000.0,
    "default_num_inference_steps": 1,
    "model_output_type": "x0",
}


def _token_chunk() -> int:
    return int(os.environ.get("LTX2_DIFFVAE_TOKEN_CHUNK", DEFAULT_TOKEN_CHUNK))


def _frame_chunk(h: int, w: int) -> int:
    """How many frames of a ``(B, T, H, W, C)`` tensor fit one token chunk."""
    return max(1, _token_chunk() // max(1, h * w))


def _apply_chunked(fn, x: mx.array, out_channels: int) -> mx.array:
    """Apply a channel-wise callable frame-slab by frame-slab into a preallocated out.

    MLX is lazy, so the point of this is not the Python loop -- it is the ``mx.eval``
    per slab, which lets the intermediate (the SwiGLU hidden, 4x wide) be freed before
    the next slab allocates its own. Without it, peak memory is the whole hidden
    tensor: 4 GB at stage 5 / 121 frames / 768x448.
    """
    b, t, h, w, _ = x.shape
    step = _frame_chunk(h, w)
    if step >= t:
        return fn(x)
    out = mx.zeros((b, t, h, w, out_channels), dtype=x.dtype)
    for t0 in range(0, t, step):
        t1 = min(t0 + step, t)
        out[:, t0:t1] = fn(x[:, t0:t1])
        mx.eval(out)
    return out


class RMSNorm(nn.Module):
    """Scale-carrying RMSNorm. Separate from ``nn.RMSNorm`` only to pin ``eps=1e-6``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self._eps)


class SwiGLU(nn.Module):
    """``w_down(silu(w_gate(x)) * w_up(x))`` -- no biases, matching the checkpoint."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self._dim = dim

    def _body(self, x: mx.array) -> mx.array:
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))

    def __call__(self, x: mx.array, pre=None) -> mx.array:
        fn = self._body if pre is None else (lambda s: self._body(pre(s)))
        return _apply_chunked(fn, x, self._dim)


class NeighborhoodAttention3D(nn.Module):
    """Fused QKV + q/k RMSNorm + absolute RoPE + 3-D neighborhood attention.

    The scale is folded into ``q_norm``'s weight rather than applied to Q, exactly as
    ComfyUI does: a positive scalar commutes with both the rotation and the softmax,
    and folding it saves a full-volume multiply at stage-5 sizes.
    """

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, rope_base: float = 10000.0):
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim={dim} not divisible by head_dim={head_dim}")
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        self._dim = dim
        self._num_heads = dim // head_dim
        self._head_dim = head_dim
        self._kernel_size = tuple(kernel_size)
        self._scale = head_dim**-0.5
        self._rope_split = default_rope_dim_split(head_dim)
        self._inv_freqs = tuple(rope_inv_freqs(d, rope_base) for d in self._rope_split)

    def __call__(self, x: mx.array, pre=None) -> mx.array:
        b, t, h, w, _ = x.shape
        src = x if pre is None else pre(x)
        qkv = self.qkv(src)
        shape = (b, t, h, w, self._num_heads, self._head_dim)
        q = qkv[..., : self._dim].reshape(shape)
        k = qkv[..., self._dim : 2 * self._dim].reshape(shape)
        v = qkv[..., 2 * self._dim :].reshape(shape)
        del qkv, src

        q = mx.fast.rms_norm(q, (self.q_norm.weight * self._scale).astype(q.dtype), 1e-6)
        k = mx.fast.rms_norm(k, self.k_norm.weight.astype(k.dtype), 1e-6)
        q = apply_abs_rope(q, self._rope_split, self._inv_freqs)
        k = apply_abs_rope(k, self._rope_split, self._inv_freqs)
        mx.eval(q, k, v)

        out = na3d(q, k, v, self._kernel_size, scale=1.0)
        del q, k, v
        out = out.reshape(b, t, h, w, self._dim)
        return _apply_chunked(self.proj, out, self._dim)


def _mlp_hidden(dim: int, mlp_ratio: float = 4.0) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU, both residual."""

    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim, mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(x, pre=self.norm1)
        return x + self.mlp(x, pre=self.norm2)


def modulate(x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    return x * (1.0 + scale) + shift


class AdaLNZero(nn.Module):
    """``t_emb`` -> 7 modulation chunks. Slots 2, 5, 6 (the gates) are unused here:
    the trained checkpoint folded them away, and both references drop them."""

    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        self._dim = dim

    def __call__(self, t_emb: mx.array) -> list[mx.array]:
        h = self.proj(nn.silu(t_emb))
        return [h[:, i * self._dim : (i + 1) * self._dim][:, None, None, None, :] for i in range(self.NUM_CHUNKS)]


class DiffusionNABlock(nn.Module):
    """NA + SwiGLU with shared AdaLN-Zero scale/shift and a per-block table offset."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = mx.zeros((AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, _mlp_hidden(dim, mlp_ratio))
        self._dim = dim

    def __call__(self, x: mx.array, latent_context: mx.array, modulation: list[mx.array]) -> mx.array:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].reshape(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        x = x + _apply_chunked(self.context_proj, latent_context, self._dim)
        mx.eval(x)
        x = x + self.attn(x, pre=lambda s: modulate(self.norm1(s), scale_msa, shift_msa))
        mx.eval(x)
        x = x + self.mlp(x, pre=lambda s: modulate(self.norm2(s), scale_mlp, shift_mlp))
        mx.eval(x)
        return x


class LinearPixelShuffleUpsample(nn.Module):
    """Linear channel-expand, then a channels-last 3-D pixel shuffle."""

    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(self.stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def __call__(self, x: mx.array, drop_leading_frame: bool = True) -> mx.array:
        b, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        c = self.out_channels
        out = mx.zeros((b, t * p1, h * p2, w * p3, c), dtype=x.dtype)
        step = _frame_chunk(h, w)
        for t0 in range(0, t, step):
            t1 = min(t0 + step, t)
            # "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c"
            y = self.proj(x[:, t0:t1]).reshape(b, t1 - t0, h, w, c, p1, p2, p3)
            y = y.transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, (t1 - t0) * p1, h * p2, w * p3, c)
            out[:, t0 * p1 : t1 * p1] = y
            mx.eval(out)
        if p1 == 2 and drop_leading_frame:
            # The causal temporal pixel-shuffle duplicates the leading frame.
            out = out[:, 1:]
        return out


class TimestepEmbedder(nn.Module):
    """Sinusoidal(256) -> MLP. ``mlp.{0,2}`` naming matches the checkpoint."""

    def __init__(self, t_emb_dim: int = 384, freq_dim: int = 256):
        super().__init__()
        self.mlp = [nn.Linear(freq_dim, t_emb_dim, bias=True), nn.SiLU(), nn.Linear(t_emb_dim, t_emb_dim, bias=True)]
        self._freq_dim = freq_dim

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        emb = timestep_embedding(timestep.reshape(-1), self._freq_dim).astype(dtype)
        for layer in self.mlp:
            emb = layer(emb)
        return emb


def timestep_embedding(timesteps: mx.array, embedding_dim: int, max_period: float = 10000.0) -> mx.array:
    """DDPM sinusoidal embedding with ``flip_sin_to_cos=True``, ``downscale_freq_shift=0``.

    Those two are not defaults -- they are what the decoder's call site passes, and
    getting either wrong silently shifts every timestep embedding.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32) / half_dim
    emb = timesteps.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def patchify_spatial_cl(x: mx.array, patch: int) -> mx.array:
    """``(B, T, H, W, C)`` -> ``(B, T, H/p, W/p, C*p*p)``.

    Channel order is ``(c, w_sub, h_sub)`` -- ``c`` outermost, then the W subdivision,
    then the H subdivision. That ordering comes straight from the reference einops
    pattern ``b c (f p) (h q) (w r) -> b (c p r q) f h w``, where ``q`` indexes H and
    ``r`` indexes W yet ``r`` is written first. Swapping them still produces a tensor
    of the right shape and a picture of scrambled 4x4 tiles.
    """
    if patch == 1:
        return x
    b, t, h, w, c = x.shape
    y = x.reshape(b, t, h // patch, patch, w // patch, patch, c)
    return y.transpose(0, 1, 2, 4, 6, 5, 3).reshape(b, t, h // patch, w // patch, c * patch * patch)


def unpatchify_spatial_cl(x: mx.array, patch: int, out_channels: int) -> mx.array:
    """Inverse of :func:`patchify_spatial_cl`."""
    if patch == 1:
        return x
    b, t, h, w, _ = x.shape
    y = x.reshape(b, t, h, w, out_channels, patch, patch)  # (c, r=w_sub, q=h_sub)
    y = y.transpose(0, 1, 2, 6, 3, 5, 4)  # b t h q w r c
    return y.reshape(b, t, h * patch, w * patch, out_channels)


class NADiffusionDecoder(nn.Module):
    """Stages 1-4 (deterministic NA upsample) + stage-5 diffusion blocks."""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
            ((1, 2, 2), 2),
            ((2, 1, 1), 2),
            ((2, 2, 2), 1),
            ((2, 2, 2), 2),
        ),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 1,
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.num_inference_steps = default_num_inference_steps
        self.temporal_upscale = math.prod(s[0] for s, _ in upsamples)
        self.spatial_upscale = math.prod(s[1] for s, _ in upsamples) * patch_size
        self.stage5_kernel = tuple(stage5_kernel)
        # NATTEN's inward window shift distorts the last frames; the vendor's
        # mitigation is to replicate the last latent frame through stages 1-4 and crop
        # the appendix off the context afterwards.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)
        self.det_stages = [
            [NABlock(stage_channels[i], stage_kernels[i], head_dim=head_dim) for _ in range(stage_depths[i])]
            for i in range(len(stage_channels) - 1)
        ]
        self.upsamples = [
            LinearPixelShuffleUpsample(
                stage_channels[i], upsamples[i][0], out_channels_reduction_factor=upsamples[i][1]
            )
            for i in range(len(stage_channels) - 1)
        ]
        self.t_embedder = TimestepEmbedder(t_emb_dim=t_emb_dim)

        c5 = stage_channels[-1]
        self.context_channels = c5
        noised_pixel_channels = out_channels * (patch_size**2)
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim)
        self.diff_blocks = [
            DiffusionNABlock(c5, self.stage5_kernel, context_channels=c5, head_dim=head_dim)
            for _ in range(stage_depths[-1])
        ]
        self.norm_out = RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)
        # Present in the checkpoint, referenced by neither reference implementation.
        # Carried so a strict load succeeds and nothing is dropped unrecorded.
        self.type_emb = mx.zeros((in_channels,))

    def forward_pre_diffusion(
        self, z_cl: mx.array, drop_leading_frame: bool = True, pad_trailing: bool = True
    ) -> mx.array:
        """Stages 1-4: un-normalized latent ``(B,T,H,W,C)`` -> stage-5 context.

        ``drop_leading_frame`` must be True only when ``z_cl`` contains the latent's
        true temporal origin; a tiled caller decoding a later chunk passes False.
        ``pad_trailing`` only for the chunk holding the latent's last frame.
        """
        n = self.trailing_pad_latent_frames if pad_trailing else 0
        if n > 0:
            z_cl = mx.concatenate([z_cl, mx.repeat(z_cl[:, -1:], n, axis=1)], axis=1)
        x = _apply_chunked(self.conv_in, z_cl, self.conv_in.weight.shape[0])
        for stage_i, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
                mx.eval(x)
            x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
            mx.eval(x)
        if n > 0:
            x = x[:, : -(n * self.temporal_upscale)]
        return x

    def forward_diff_step(self, context: mx.array, x_t: mx.array, t: mx.array) -> mx.array:
        """One stage-5 step. ``x_t`` and the return are pixels ``(B,T,H,W,3)``."""
        x = patchify_spatial_cl(x_t, self.patch_size)
        x = _apply_chunked(self.conv_in_x_t, x, self.context_channels)
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, x.dtype)
        modulation = self.shared_adaln(t_emb)
        for i, block in enumerate(self.diff_blocks):
            x = block(x, context, modulation)
            logger.debug("diffusion decoder stage-5 block %d/%d", i + 1, len(self.diff_blocks))
        x = _apply_chunked(lambda s: self.conv_out(self.norm_out(s)), x, self.conv_out.weight.shape[0])
        return unpatchify_spatial_cl(x, self.patch_size, self.out_channels)

    def __call__(
        self,
        z_cl: mx.array,
        key: mx.array | None = None,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
        x_t: mx.array | None = None,
    ) -> mx.array:
        """``x_t`` lets a tiled caller supply the noise instead of drawing it here.

        That matters more than it looks: this is a one-step x0 prediction *from* noise,
        so the noise is an input the output depends on. Two overlapping tiles drawing
        their own noise would predict two different realisations of the same frames, and
        cross-fading them softens exactly the detail this decoder exists to add.
        """
        context = self.forward_pre_diffusion(z_cl, drop_leading_frame, pad_trailing)
        mx.eval(context)
        b, t5, h5, w5, _ = context.shape
        pixel_shape = (b, t5, h5 * self.patch_size, w5 * self.patch_size, self.out_channels)
        if x_t is None:
            x_t = mx.random.normal(pixel_shape, dtype=mx.float32, key=key).astype(z_cl.dtype)
        elif tuple(x_t.shape) != pixel_shape:
            raise ValueError(f"supplied x_t {tuple(x_t.shape)} does not match the decoder's {pixel_shape}")

        steps = self.num_inference_steps
        timesteps = [1.0 - i * (1.0 - 1.0 / steps) / max(1, steps - 1) for i in range(steps)] if steps > 1 else [1.0]
        for i, t_val in enumerate(timesteps):
            t_now = mx.full((b,), t_val, dtype=mx.float32)
            model_out = self.forward_diff_step(context, x_t, t_now)
            if self.model_output_type == "x0":
                if i == steps - 1:
                    return model_out
                velocity = (x_t.astype(mx.float32) - model_out.astype(mx.float32)) / t_val
            else:
                velocity = model_out.astype(mx.float32)
                if i == steps - 1:
                    return (x_t.astype(mx.float32) - t_val * velocity).astype(z_cl.dtype)
            t_next = timesteps[i + 1] if i + 1 < steps else 0.0
            x_t = (x_t.astype(mx.float32) - (t_val - t_next) * velocity).astype(z_cl.dtype)
        return x_t


class DiffusionVideoDecoder(nn.Module):
    """Pipeline-facing wrapper: same ``decode()`` contract as the conv ``VideoDecoder``.

    Input ``(B, C, F, H, W)`` normalized latent, output ``(B, 3, F, H, W)`` pixels.

    **Memory, and why tiling is the primary path rather than a fallback.** ComfyUI's
    own estimator puts an untiled 960x544x121 decode near 14 GB, and the vendor
    corrected its reference workflows ten hours after release -- temporal tile 4096 ->
    64, spatial 768 -> 512 -- because the untiled setting hangs 10-second clips. So
    ``decode()`` tiles by default: it splits the latent along time whenever the
    estimated stage-5 working set exceeds ``LTX2_DIFFVAE_BUDGET_GB`` (default 8), with
    ``LTX2_DIFFVAE_TEMPORAL_OVERLAP`` latent frames of overlap cross-faded in pixel
    space. Raise the budget to force a single pass; lower it to tile harder.

    Only the **temporal** axis is tiled. The stage-5 working set is linear in
    ``T x H x W`` and temporal tiling already drives it to any budget (two latent
    frames at 1024x576 is well under 2 GB), whereas spatial tiling would need halos of
    ``depth x kernel//2`` = 8 x 5 = 40 tokens on every side of every tile through the
    diffusion stage -- more recomputation than the memory it buys at these canvases.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = dict(LTX_25_DIFFUSION_VAE_CONFIG if config is None else config)
        defaults = LTX_25_DIFFUSION_VAE_CONFIG
        self.decoder = NADiffusionDecoder(
            in_channels=cfg.get("in_channels", 128),
            out_channels=cfg.get("out_channels", 3),
            patch_size=cfg.get("patch_size", 4),
            head_dim=cfg.get("head_dim", 64),
            stage_channels=tuple(cfg.get("stage_channels", defaults["stage_channels"])),
            stage_depths=tuple(cfg.get("stage_depths", defaults["stage_depths"])),
            stage_kernels=tuple(tuple(k) for k in cfg.get("stage_kernels", defaults["stage_kernels"])),
            upsamples=tuple((tuple(s), r) for s, r in cfg.get("upsamples", defaults["upsamples"])),
            stage5_kernel=tuple(cfg.get("stage5_kernel", defaults["stage5_kernel"])),
            t_emb_dim=cfg.get("t_emb_dim", 384),
            default_num_inference_steps=cfg.get("default_num_inference_steps", 1),
            timestep_scale_multiplier=cfg.get("timestep_scale_multiplier", 1000.0),
            model_output_type=cfg.get("model_output_type", "x0"),
        )
        self.per_channel_statistics = PerChannelStatistics(cfg.get("in_channels", 128))
        self._config = cfg

    # -- geometry -----------------------------------------------------------

    @property
    def temporal_upscale(self) -> int:
        return self.decoder.temporal_upscale

    @property
    def spatial_upscale(self) -> int:
        return self.decoder.spatial_upscale

    def denormalize_latent(self, latent_cl: mx.array) -> mx.array:
        mean = self.per_channel_statistics.mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std.reshape(1, 1, 1, 1, -1)
        return latent_cl * std.astype(latent_cl.dtype) + mean.astype(latent_cl.dtype)

    # -- memory planning ----------------------------------------------------

    #: Multiple of one stage-5 tensor that peak memory actually reaches. Counted by
    #: hand you get ~6 (context, x, Q, K, V, out); **measured** on an M4 Max it is 15.
    #: The gap is MLX holding freed-but-cached buffers plus the NA gather's stacked K/V
    #: and the det-stage intermediates that are still resident. Calibrated against a
    #: real untiled decode: 768x448x121 (16 latent frames) peaks at 21.05 GiB, i.e.
    #: 1.32 GiB per latent frame, against 0.53 GiB predicted by the hand count. Using
    #: the hand count would make ``LTX2_DIFFVAE_BUDGET_GB`` mean nothing.
    PEAK_TENSOR_MULTIPLE = 15

    def stage5_bytes_per_latent_frame(self, h_lat: int, w_lat: int, itemsize: int = 2) -> int:
        """Peak driver: the stage-5 tensors scale linearly in latent frames.

        Per latent frame the stage-5 grid is ``temporal_upscale`` frames of
        ``(H_lat*8, W_lat*8)`` tokens at 256 channels.

        On top of this sits a per-clip fixed cost the tiling cannot reduce: the shared
        noise volume and the fp32 blend accumulator, together ~0.75 GiB at 768x448x121.
        """
        c5 = self.decoder.context_channels
        tokens = self.temporal_upscale * (h_lat * self.spatial_upscale // 4) * (w_lat * self.spatial_upscale // 4)
        return int(tokens * c5 * itemsize * self.PEAK_TENSOR_MULTIPLE)

    def plan_temporal_tiles(self, f_lat: int, h_lat: int, w_lat: int) -> list[tuple[int, int]]:
        """``[(t0, t1), ...]`` latent slices, overlapping, covering ``[0, f_lat)``."""
        budget = float(os.environ.get("LTX2_DIFFVAE_BUDGET_GB", "8")) * 1024**3
        per_frame = self.stage5_bytes_per_latent_frame(h_lat, w_lat)
        if per_frame * f_lat <= budget:
            return [(0, f_lat)]
        overlap = int(os.environ.get("LTX2_DIFFVAE_TEMPORAL_OVERLAP", "2"))
        # NA needs at least kernel_t stage-5 frames; kernel_t=11 over 8x upscale is
        # comfortably under any tile >= 2 latent frames, but keep a floor anyway.
        tile = max(overlap + 2, int(budget // max(1, per_frame)))
        tiles: list[tuple[int, int]] = []
        t0 = 0
        while t0 < f_lat:
            t1 = min(t0 + tile, f_lat)
            tiles.append((t0, t1))
            if t1 >= f_lat:
                break
            t0 = t1 - overlap
        return tiles

    # -- decode -------------------------------------------------------------

    def decode(self, latent: mx.array, *, seed: int = 0) -> mx.array:
        """Decode ``(B, C, F, H, W)`` normalized latent to ``(B, 3, F, H, W)`` pixels."""
        output_dtype = latent.dtype
        flat = mlx.utils.tree_flatten(self.parameters())
        weights_dtype = next((v.dtype for _, v in flat if v.ndim == 2), output_dtype)
        latent = latent.astype(weights_dtype)

        x = latent.transpose(0, 2, 3, 4, 1)  # BCFHW -> BFHWC
        x = self.denormalize_latent(x)
        b, f_lat, h_lat, w_lat, _ = x.shape

        tiles = self.plan_temporal_tiles(f_lat, h_lat, w_lat)
        key = mx.random.key(seed)
        if len(tiles) == 1:
            pixels = self.decoder(x, key=key, drop_leading_frame=True, pad_trailing=True)
        else:
            logger.info("diffusion VAE decode: %d temporal tiles over %d latent frames", len(tiles), f_lat)
            pixels = self._tiled_decode(x, tiles, key)
        return pixels.transpose(0, 4, 1, 2, 3).astype(output_dtype)

    def _tiled_decode(self, x: mx.array, tiles: list[tuple[int, int]], key: mx.array) -> mx.array:
        """Decode overlapping temporal tiles and cross-fade them in pixel space.

        Frame mapping: latent frame 0 yields 1 output frame and every later latent
        frame yields ``temporal_upscale``, so latent ``[t0, t1)`` with ``t0 > 0`` covers
        output ``[8*t0-7, 8*t1-7)``. ``drop_leading_frame`` is True only for the origin
        tile -- the duplicate frame the causal pixel-shuffle creates belongs to it alone.
        """
        up = self.temporal_upscale
        f_lat = x.shape[1]
        total = f_lat * up - (up - 1)
        # ONE noise volume for the whole clip, sliced per tile: overlapping frames must
        # get the same noise or the cross-fade averages two different predictions.
        noise = mx.random.normal(
            (x.shape[0], total, x.shape[2] * self.spatial_upscale, x.shape[3] * self.spatial_upscale, 3),
            dtype=mx.float32,
            key=key,
        ).astype(x.dtype)
        mx.eval(noise)
        acc: mx.array | None = None
        wsum: mx.array | None = None
        for idx, (t0, t1) in enumerate(tiles):
            start = 0 if t0 == 0 else t0 * up - (up - 1)
            n_expect = (t1 - t0) * up - (up - 1 if t0 == 0 else 0)
            sub = self.decoder(
                x[:, t0:t1],
                drop_leading_frame=(t0 == 0),
                pad_trailing=(t1 == f_lat),
                x_t=noise[:, start : start + n_expect],
            )
            mx.eval(sub)
            n = sub.shape[1]
            if acc is None:
                acc = mx.zeros((sub.shape[0], total, sub.shape[2], sub.shape[3], sub.shape[4]), dtype=mx.float32)
                wsum = mx.zeros((total,), dtype=mx.float32)
            ramp = _blend_ramp(n, fade_in=idx > 0, fade_out=idx < len(tiles) - 1)
            acc[:, start : start + n] += sub.astype(mx.float32) * ramp.reshape(1, n, 1, 1, 1)
            wsum[start : start + n] += ramp
            mx.eval(acc, wsum)
        assert acc is not None and wsum is not None
        return (acc / mx.maximum(wsum, 1e-6).reshape(1, total, 1, 1, 1)).astype(x.dtype)


def _blend_ramp(n: int, fade_in: bool, fade_out: bool, width: int = 8) -> mx.array:
    w = mx.ones((n,), dtype=mx.float32)
    span = min(width, n // 2)
    if span > 0 and fade_in:
        w[:span] = mx.arange(1, span + 1, dtype=mx.float32) / (span + 1)
    if span > 0 and fade_out:
        w[n - span :] = mx.arange(span, 0, -1, dtype=mx.float32) / (span + 1)
    return w


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


#: Filename the converter emits for this component (stem ``vae_decoder_diffusion``).
DIFFUSION_DECODER_FILENAME = "vae_decoder_diffusion.safetensors"

#: Opt-in switch. Default is the conv decoder and stays the conv decoder: the
#: community's verdict after 30 hours is that this decoder makes decode take longer
#: than generation, and the vendor's own labels are "higher quality, heavier" vs
#: "faster, lighter". Flipping the default is a product decision, not a code one.
DECODER_ENV_VAR = "LTX2_VIDEO_DECODER"


def is_diffusion_decoder_weights(keys) -> bool:
    """The vendor's own detection rule: one key decides which video VAE this is."""
    return any(k.endswith("decoder.conv_in_x_t.weight") or k == "conv_in_x_t.weight" for k in keys)


def diffusion_decode_requested() -> bool:
    return os.environ.get(DECODER_ENV_VAR, "conv").strip().lower() in {"diffusion", "diff", "na"}


def decoder_config_from_pack(model_dir) -> dict[str, Any] | None:
    """Read the decoder config the converter carried out of the checkpoint header.

    Preferred over :data:`LTX_25_DIFFUSION_VAE_CONFIG`, which is a transcription and
    can go stale. A decoder built at the wrong widths against real weights fails to
    load loudly, which is the good case; built at the wrong *kernels* it loads fine
    and decodes wrong, which is not.
    """
    import json
    from pathlib import Path

    for name in ("config.json", "embedded_config.json"):
        path = Path(model_dir) / name
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        vae = blob.get("vae", blob)
        dec = vae.get("decoder")
        if not isinstance(dec, dict) or "stage_channels" not in dec:
            continue
        cfg = dict(LTX_25_DIFFUSION_VAE_CONFIG)
        cfg.update({k: v for k, v in dec.items() if not k.startswith("_")})
        cfg["model_output_type"] = vae.get("model_output_type", cfg["model_output_type"])
        return cfg
    return None


def load_diffusion_decoder_from_pack(model_dir) -> DiffusionVideoDecoder:
    """Load ``vae_decoder_diffusion.safetensors`` out of a converted pack directory."""
    from pathlib import Path

    from ltx_core_mlx.utils.weights import load_split_safetensors

    path = Path(model_dir) / DIFFUSION_DECODER_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{DECODER_ENV_VAR} asked for the diffusion video decoder but {path} is not there. "
            "Convert it with `scripts/convert_ltx_mlx.py --input ltx-2.5-video-vae-bf16.safetensors` "
            "and drop the file into the pack. Refusing to fall back to the conv decoder: a quality "
            "path that silently downgrades is worse than one that stops."
        )
    weights = load_split_safetensors(path)
    return load_diffusion_decoder(weights, decoder_config_from_pack(model_dir))


def resolve_video_decoder_override(model_dir) -> DiffusionVideoDecoder | None:
    """``None`` for the default conv path; a loaded diffusion decoder when opted in.

    The single seam a caller needs: build the conv decoder as today, and use this
    instead when it returns something.
    """
    if not diffusion_decode_requested():
        return None
    logger.info("%s=diffusion — loading the NA diffusion video decoder", DECODER_ENV_VAR)
    return load_diffusion_decoder_from_pack(model_dir)


def load_diffusion_decoder(
    weights: dict[str, mx.array],
    config: dict[str, Any] | None = None,
    prefix: str = "vae_decoder_diffusion.",
) -> DiffusionVideoDecoder:
    """Build and strictly load a :class:`DiffusionVideoDecoder` from pack weights.

    ``strict`` is not optional. A non-strict load is how the checkpoint's extra
    ``type_emb`` gets dropped without a record, and -- far worse -- how a module the
    file has no weights for stays randomly initialised.
    """
    model = DiffusionVideoDecoder(config)
    stripped = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in weights.items()}
    model.load_weights(list(stripped.items()), strict=True)
    model.eval()
    return model
