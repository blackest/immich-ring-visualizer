"""Conditioning rows built from MEDIA rather than from a single still.

The released FL2VA graph conditions on one or two keyframes. Everything here extends that from
the packing primitives the reference already defines — video conditioning rows held at the
keyframe noise level, audio conditioning rows held clean, target rows that may be held clean —
to the media a creator actually has:

* **carried context** (``encode_video_rows(..., sample=True)`` + ``encode_audio_rows``): the tail
  of the clip a window continues, as history rows placed just before the target on the rotary
  clock. This is what makes a chained window continue the previous one's motion and voice
  instead of restarting from one still and a fresh seed.
* **a frozen soundtrack** (``encode_audio_rows`` into the target audio rows): the picture is
  generated to match audio the user supplied (lip-sync, a song).
* **a frozen picture** (``encode_video_rows(..., sample=False)`` into the target video rows): the
  sound is generated to match a silent clip.

Pixel and latent conventions follow the reference exactly (ImageNet-normalised [0, 1] pixels, the
VAE's own latent mean/std, the seed-42 posterior sample and float16 round trip for visual
conditioning rows); see :func:`minimax_h3_mlx.pipeline.encode_keyframe_rows` for the single-still
version this generalises.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .packing import (
    AUDIO_CHANNELS,
    FRAMES_PER_CHUNK,
    KEYFRAME_ENCODE_SEED,
    LATENTS_PER_CHUNK,
    PIXEL_MEAN,
    PIXEL_STD,
    patchify_video_latents,
)


def video_latents_for_frames(num_frames: int) -> int:
    """Latent frames ``encode_video_rows`` produces for ``num_frames`` pixel frames.

    Whole 17-frame clips keep all five of their latents (``17 m -> 5 m``, the shape carried
    history needs, since it ends exactly where the next window's target starts); a ``17 n + 5``
    count follows the VAE's own convention (``-> 5 n + 2``), the shape of a target window.
    """
    if num_frames > 0 and num_frames % FRAMES_PER_CHUNK == 0:
        return num_frames // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK
    if num_frames % FRAMES_PER_CHUNK == LATENTS_PER_CHUNK:
        return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    raise ValueError(
        f"encode_video_rows takes 17*m frames (history) or 17*n+5 frames (a window), "
        f"got {num_frames}."
    )


def _normalize_pixels(frames: np.ndarray) -> np.ndarray:
    """``(F, H, W, 3)`` uint8 -> ``(1, 3, F, H, W)`` ImageNet-normalised float32."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (F, H, W, 3) uint8, got {frames.shape}")
    pixels = np.asarray(frames, dtype=np.float32).transpose(3, 0, 1, 2)[None] / 255.0
    mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    return (pixels - mean) / std


def encode_video_rows(
    video_vae,
    frames: np.ndarray,
    patch_size: tuple[int, int, int],
    *,
    sample: bool,
) -> mx.array:
    """Encode canvas-sized pixel frames into packed, normalised video rows.

    Args:
        frames: ``(F, H, W, 3)`` uint8, already on the render canvas. ``F`` is either a whole
            number of 17-frame clips (every latent kept) or ``17 n + 5`` (the VAE's own window
            convention, trailing latents of the padded last clip dropped).
        sample: draw the posterior sample under the fixed keyframe seed and round it through
            float16, exactly like a keyframe (conditioning rows that are noised to the keyframe
            level). ``False`` takes the posterior mean: a clean *target* that is held, not noised.

    Returns ``(latent_frames * rows_per_frame, C * prod(patch))`` float32 rows, frame-major.
    """
    num_frames = int(frames.shape[0])
    latent_frames = video_latents_for_frames(num_frames)
    cfg = video_vae.config
    channels = cfg.latent_channels
    pixels = mx.array(_normalize_pixels(frames))

    if num_frames % FRAMES_PER_CHUNK == 0:
        # Every 17-frame clip encodes independently (that is how the VAE chunks a video anyway), so
        # encoding them one by one and keeping every latent is exactly the chunked encode minus
        # its trailing drop — and holds one clip's activations at a time.
        clips = []
        channels_last = pixels.transpose(0, 2, 3, 4, 1)
        for start in range(0, num_frames, FRAMES_PER_CHUNK):
            moments = video_vae._encode_clip(channels_last[:, start : start + FRAMES_PER_CHUNK])
            mx.eval(moments)
            clips.append(moments)
        moments = mx.concatenate(clips, axis=1).transpose(0, 4, 1, 2, 3)
    else:
        moments = video_vae.encode(pixels)
    mx.eval(moments)
    if moments.shape[2] != latent_frames:
        raise ValueError(
            f"the video VAE returned {moments.shape[2]} latent frames for {num_frames} pixel "
            f"frames, expected {latent_frames}"
        )

    mean, logvar = moments[:, :channels], moments[:, channels:]
    if sample:
        mx.random.seed(KEYFRAME_ENCODE_SEED)
        logvar = mx.clip(logvar, -30.0, 20.0)
        latent = mean + mx.exp(0.5 * logvar) * mx.random.normal(mean.shape)
        latent = latent.astype(mx.float16).astype(mx.float32)
    else:
        latent = mean.astype(mx.float32)
    latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    rows = patchify_video_latents((latent - latents_mean) / latents_std, patch_size)
    mx.eval(rows)
    return rows


def encode_audio_rows(audio_vae, waveform: np.ndarray, num_latents: int) -> mx.array:
    """Encode a stereo waveform into ``num_latents`` packed, normalised audio rows per channel.

    Args:
        waveform: ``(2, samples)`` float32 at the audio VAE's rate (32 kHz). It is zero-padded or
            cut to exactly ``num_latents`` hops first, so a short soundtrack becomes silence at
            the end rather than a shape error, and a long one is cut where the window ends.

    Returns ``(2 * num_latents, latent_channels)`` float32 rows, channel-major — the layout of the
    target audio rows. The posterior MEAN is used: MiniMax-H3 never samples audio latents.
    """
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim == 1:
        wave = np.stack([wave, wave])
    if wave.shape[0] == 1:
        wave = np.concatenate([wave, wave])
    if wave.shape[0] != AUDIO_CHANNELS:
        raise ValueError(f"waveform must be (2, samples), got {wave.shape}")
    hop = audio_vae.config.hop_length
    want = int(num_latents) * hop
    if wave.shape[1] < want:
        wave = np.pad(wave, ((0, 0), (0, want - wave.shape[1])))
    wave = np.clip(wave[:, :want], -1.0, 1.0)

    mean, _ = audio_vae.encode(mx.array(wave[:, None, :]))  # (2, C, L)
    mx.eval(mean)
    if mean.shape[-1] < num_latents:
        raise ValueError(f"audio VAE returned {mean.shape[-1]} latents, expected {num_latents}")
    mean = mean[..., :num_latents].astype(mx.float32)
    cfg = audio_vae.config
    lmean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
    lstd = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
    normalized = (mean - lmean) / lstd
    rows = normalized.transpose(0, 2, 1).reshape(AUDIO_CHANNELS * num_latents, -1)
    mx.eval(rows)
    return rows


def frozen_row_mask(num_rows: int, frozen: int) -> np.ndarray:
    """A boolean mask holding the first ``frozen`` of ``num_rows`` target rows."""
    mask = np.zeros(num_rows, dtype=bool)
    mask[: max(0, min(num_rows, int(frozen)))] = True
    return mask


def gate_ranked_skip_blocks(
    modulation_cache, num_blocks: int, count: int, protect: tuple[int, ...] = (0, 1)
) -> list[int]:
    """The ``count`` DiT blocks whose residual gates are smallest, to skip.

    A block adds ``gate_attn * attn(...) + gate_mlp * mlp(...)`` to the residual stream. Ranking by
    the mean absolute gate over every (timestep, modality) row the run will use is a cheap proxy
    for how little a block moves the stream. The first two blocks and the last one are never
    skipped (they set up and finish the representation). Returns sorted block indices.
    """
    if count <= 0:
        return []
    protected = set(protect) | {num_blocks - 1}
    scores = []
    for index in range(num_blocks):
        if index in protected:
            continue
        mod = modulation_cache.get(index)
        gate_attn, gate_mlp = mod[2], mod[5]
        score = float(mx.mean(mx.abs(gate_attn.astype(mx.float32))).item()) + float(
            mx.mean(mx.abs(gate_mlp.astype(mx.float32))).item()
        )
        scores.append((score, index))
    scores.sort()
    return sorted(index for _, index in scores[: min(count, len(scores))])
