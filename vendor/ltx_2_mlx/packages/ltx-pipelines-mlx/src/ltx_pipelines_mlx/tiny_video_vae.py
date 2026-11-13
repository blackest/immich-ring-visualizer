"""Decode LTX-2 latents with madebyollin's tiny autoencoder (``taeltx``).

The shipped conv video VAE is the delivery decoder. This is the *preview* decoder: a 22 MB
causal-convolutional model that turns a 128-channel LTX latent into display-ready RGB in a
fraction of a second, so a render can be watched while it is still running.

Provenance and licence
----------------------
Decode-only MLX port of ``madebyollin/taehv``'s ``taeltx2_3`` decoder
(<https://github.com/madebyollin/taehv>, MIT, (c) 2025 Ollin Boer Bohan). Only the decoder
tensors are read; the checkpoint's encoder half is deliberately ignored. The architecture
constants below are the ones ``TAEHV.__init__`` selects when the checkpoint name contains
``taeltx``::

    patch_size, latent_channels = 4, 128
    decoder_time_upscale = decoder_space_upscale = (True, True, True)

which gives **32x spatial** (2*2*2 upsample x patch 4) and **8x temporal** upscale — exactly
LTX-2's VAE compression, so its latent grid maps one-to-one onto the real decoder's.

Why the same file is valid on LTX-2.5
-------------------------------------
``taeltx2_3`` was trained against LTX-**2.3** latents. LTX-2.5 ships the *same* conv VAE:
``ltx-2.3-mlx-q8/vae_decoder.safetensors`` and ``ltx-2.5-mlx-q8/vae_decoder.safetensors``
agree on all 86 tensors byte for byte (likewise the encoders), so the two generations share
a latent space and a decoder trained on one is valid on the other. Re-run
``scripts/ltx_pack_diff.py`` if a pack is ever rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

#: Pixel frames a single latent token expands to.
TEMPORAL_UPSCALE = 8

#: Leading raw frames the decoder drops (``t_upscale - 1`` upstream), which is what makes
#: ``T`` latent tokens decode to ``T * 8 - 7`` pixel frames — LTX's own ``(F - 1) * 8 + 1``.
FRAMES_TO_TRIM = TEMPORAL_UPSCALE - 1


class FrameConv2d(nn.Conv2d):
    """Apply an MLX 2D convolution independently to every video frame."""

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 5:
            raise ValueError(f"frame convolution expects (B, T, H, W, C), got {x.shape}")
        batch, frames, height, width, channels = x.shape
        out = super().__call__(x.reshape(batch * frames, height, width, channels))
        return out.reshape(batch, frames, out.shape[1], out.shape[2], out.shape[3])


class Clamp(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x / 3.0) * 3.0


class SpatialUpsample(nn.Module):
    """Nearest-neighbour 2x upsample, matching ``torch.nn.Upsample``'s default mode."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)


class MemBlock(nn.Module):
    """Causal residual block whose memory is the previous frame's block input.

    This is the reason a preview needs *context*: decoding one frame in isolation feeds every
    MemBlock a zero memory, i.e. decodes the frame as if it opened the clip.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = [
            FrameConv2d(in_channels * 2, out_channels, 3, padding=1),
            nn.ReLU(),
            FrameConv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            FrameConv2d(out_channels, out_channels, 3, padding=1),
        ]
        self.skip = (
            FrameConv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        past = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        h = mx.concatenate([x, past], axis=-1)
        for layer in self.conv:
            h = layer(h)
        return nn.relu(h + self.skip(x))


class TemporalGrow(nn.Module):
    """Learned 1x1 projection followed by channel-to-time rearrangement.

    Upstream's ``TGrow`` produces ``(NT, stride * C, H, W)`` and reshapes to ``(-1, C, H, W)``,
    so the checkpoint's output-channel axis is ordered ``(stride, C)``. Channels-last MLX must
    split the trailing axis in that same order or the sub-frames come out interleaved wrongly.
    """

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = FrameConv2d(channels, channels * stride, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.stride == 1:
            return self.conv(x)
        batch, frames, height, width, channels = x.shape
        x = self.conv(x).reshape(batch, frames, height, width, self.stride, channels)
        return x.transpose(0, 1, 4, 2, 3, 5).reshape(batch, frames * self.stride, height, width, channels)


def pixel_shuffle(x: mx.array, factor: int) -> mx.array:
    """Channels-last equivalent of ``torch.nn.functional.pixel_shuffle``."""
    batch, frames, height, width, channels = x.shape
    output_channels = channels // (factor * factor)
    if output_channels * factor * factor != channels:
        raise ValueError(f"{channels} channels cannot be pixel-shuffled by {factor}")
    x = x.reshape(batch, frames, height, width, output_channels, factor, factor)
    return x.transpose(0, 1, 2, 5, 3, 6, 4).reshape(batch, frames, height * factor, width * factor, output_channels)


class TinyLTXVideoDecoder(nn.Module):
    """LTX-specific TAE decoder: normalized 128-channel latent to RGB in ``[0, 1]``."""

    latent_channels = 128
    patch_size = 4
    temporal_upscale = TEMPORAL_UPSCALE
    frames_to_trim = FRAMES_TO_TRIM

    def __init__(self):
        super().__init__()
        # Positional list indices intentionally reproduce the published checkpoint's
        # ``decoder.<index>`` names so loading stays a strict 1:1 mapping.
        self.decoder = [
            Clamp(),  # 0
            FrameConv2d(128, 256, 3, padding=1),  # 1
            nn.ReLU(),  # 2
            MemBlock(256, 256),  # 3
            MemBlock(256, 256),  # 4
            MemBlock(256, 256),  # 5
            SpatialUpsample(),  # 6
            TemporalGrow(256, 2),  # 7
            FrameConv2d(256, 128, 3, padding=1, bias=False),  # 8
            MemBlock(128, 128),  # 9
            MemBlock(128, 128),  # 10
            MemBlock(128, 128),  # 11
            SpatialUpsample(),  # 12
            TemporalGrow(128, 2),  # 13
            FrameConv2d(128, 64, 3, padding=1, bias=False),  # 14
            MemBlock(64, 64),  # 15
            MemBlock(64, 64),  # 16
            MemBlock(64, 64),  # 17
            SpatialUpsample(),  # 18
            TemporalGrow(64, 2),  # 19
            FrameConv2d(64, 64, 3, padding=1, bias=False),  # 20
            nn.ReLU(),  # 21
            FrameConv2d(64, 48, 3, padding=1),  # 22
        ]

    def decode(self, latents: mx.array, num_frames: int | None = None) -> mx.array:
        """Decode ``(B, 128, T, H, W)`` to ``(B, 3, T * 8 - 7, H * 32, W * 32)``.

        Args:
            latents: Normalized latents, exactly as the denoiser holds them.
            num_frames: Optional trim of the returned frame axis. ``None`` keeps all
                ``T * 8 - 7`` frames.
        """
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(f"tiny LTX decoder expects (B, 128, T, H, W), got {latents.shape}")

        x = latents.transpose(0, 2, 3, 4, 1)
        for layer in self.decoder:
            x = layer(x)
        x = pixel_shuffle(x, self.patch_size)[:, self.frames_to_trim :]
        if num_frames is not None:
            if x.shape[1] < num_frames:
                raise ValueError(
                    f"{latents.shape[2]} latent frames decode to {x.shape[1]} RGB frames; "
                    f"cannot deliver {num_frames}"
                )
            x = x[:, :num_frames]
        return mx.clip(x, 0.0, 1.0).transpose(0, 4, 1, 2, 3)


def load_tiny_ltx_video_decoder(path: str | Path) -> TinyLTXVideoDecoder:
    """Load the decoder half of a madebyollin ``taeltx2_3.safetensors`` checkpoint.

    Strict: any missing or unexpected ``decoder.*`` key raises rather than silently loading a
    partly-initialised preview model that would look plausible and be wrong.
    """
    model = TinyLTXVideoDecoder()
    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    for key, tensor in mx.load(str(path)).items():
        if not key.startswith("decoder."):
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        if tensor.ndim == 4:
            # torch OIHW -> MLX OHWI. Materialize on the CPU stream so loading a preview
            # decoder never submits incidental work to the shared GPU.
            with mx.stream(mx.cpu):
                tensor = mx.contiguous(tensor.transpose(0, 2, 3, 1))
                mx.eval(tensor)
        weights[key] = tensor.astype(mx.float32)

    missing = sorted(expected - weights.keys())
    if missing or unexpected:
        raise KeyError(
            f"TAE checkpoint/module mismatch: {len(missing)} missing ({missing[:4]}), "
            f"{len(unexpected)} unexpected ({unexpected[:4]})"
        )
    model.update(tree_unflatten(list(weights.items())))
    return model


__all__ = [
    "FRAMES_TO_TRIM",
    "TEMPORAL_UPSCALE",
    "TinyLTXVideoDecoder",
    "load_tiny_ltx_video_decoder",
]
