"""Make RefMod files: encode a set of pictures into a saved Ref2VA reference latent.

The inverse of :mod:`refmod_io`. Each picture goes through the video VAE's spatial encoder on its
own (the same path a keyframe takes), is normalised by the VAE's latent mean/std, and the pictures
are stacked as latent frames: ``(1, 24, T, H/16, W/16)`` float16 in a safetensors file whose header
carries a JSON ``refmod_meta`` -- the layout the library files use (``mode: encode``,
``source: stack``).

Open questions are settled by :mod:`scripts/refmod_roundtrip`, not assumed here: whether the
library encodes the posterior mean or a sample (``sample`` below, default mean), and that the
latents are stored normalised (``refmod_io`` already treats them so).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .conditioning import encode_video_rows, video_latents_for_frames
from .media import probe_media, read_video_frames
from .packing import PIXEL_MEAN, PIXEL_STD, patchify_video_latents, prepare_keyframe_image
from .ref2va_packing import RefGeometry

LATENT_RATIO = 16  # pixels per latent cell, per side
CANVAS_MULTIPLE = 32
REFERENCE_SHORT_EDGE = 2048  # upstream's reference image size ("max")
REFERENCE_FPS = 24.0
TEXT_SAMPLE_FPS = 2.0  # the text encoder reads a clip at 2 fps
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(sources) -> list[Path]:
    """Files and/or folders -> image paths, folders sorted by name."""
    out: list[Path] = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no such image or folder: {p}")
    if not out:
        raise ValueError("no images found")
    return out


def encode_images(video_vae, images, width: int, height: int, *, sample: bool = False, seed: int = 42) -> np.ndarray:
    """PIL images -> ``(1, 24, T, height/16, width/16)`` float32 normalised latent, one frame each.

    Pictures are cover-cropped onto the ``width`` x ``height`` canvas (aspect preserved, centre crop).
    """
    if width % (2 * LATENT_RATIO) or height % (2 * LATENT_RATIO):
        raise ValueError(f"width and height must be multiples of {2 * LATENT_RATIO} (the (1, 2, 2) patch)")
    cfg = video_vae.config
    ch = cfg.latent_channels
    lat_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    lat_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    pix_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pix_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    mx.random.seed(seed)
    frames = []
    for image in images:
        prepared = prepare_keyframe_image(image.convert("RGB"), height, width, stretch=False)
        pixels = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
        pixels = (pixels / 255.0 - pix_mean) / pix_std
        moments = video_vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))
        mean, logvar = moments[..., :ch], moments[..., ch:]
        if sample:
            logvar = mx.clip(logvar, -30.0, 20.0)
            mean = mean + mx.exp(0.5 * logvar) * mx.random.normal(mean.shape)
        latent = mean.transpose(0, 4, 1, 2, 3).astype(mx.float16).astype(mx.float32)  # (1, C, 1, H', W')
        z = (latent - lat_mean) / lat_std
        mx.eval(z)
        frames.append(np.array(z))
    return np.concatenate(frames, axis=2)


@dataclass
class Reference:
    """One ``--ref2va-reference``: a picture, or a clip as 24 fps frames already trimmed to 17n+5."""

    kind: str  # "image" | "video"
    image: object = None  # PIL image, kind == "image"
    frames: np.ndarray | None = None  # (F, H, W, 3) uint8, kind == "video"

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` this reference is encoded at."""
        if self.kind == "image":
            return self.image.size
        return self.frames.shape[2], self.frames.shape[1]


def reference_size(w: int, h: int, width: int, height: int, size: str, name: str) -> tuple[int, int]:
    """``(width, height)`` for a reference of ``w`` x ``h``, aspect kept, both sides a multiple of 32.

    ``match`` scales down to the render's pixel area (never up); ``max`` is upstream's 2048 px
    short edge, upscaling included. A reference never takes the render canvas, so nothing is cropped.
    """
    if w > 4 * h or h > 4 * w:
        raise ValueError(f"{name}: a reference must be within 1:4 and 4:1, got {w}x{h}")
    scale = REFERENCE_SHORT_EDGE / min(w, h) if size == "max" else min(1.0, math.sqrt(width * height / (w * h)))
    return (max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def trim_reference_frames(count: int) -> int:
    """Largest ``17 n + 5`` not above ``count``: what the video VAE encodes without padding."""
    if count < 5:
        raise ValueError(f"a reference clip needs at least 5 frames, got {count}")
    return (count - 5) // 17 * 17 + 5


def load_references(paths, width: int, height: int, num_frames: int, size: str = "match") -> list[Reference]:
    """``--ref2va-reference`` files -> prepared references, in order. A clip is cut to the window's length."""
    from PIL import Image, ImageOps

    refs = []
    for path in paths:
        name = Path(path).name
        if Path(path).suffix.lower() in IMAGE_SUFFIXES:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            target = reference_size(*image.size, width, height, size, name)
            refs.append(Reference("image", image=image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)))
        else:
            info = probe_media(path)
            target = reference_size(info["width"], info["height"], width, height, size, name)
            frames = read_video_frames(path, *target, max_frames=num_frames)
            refs.append(Reference("video", frames=frames[:trim_reference_frames(frames.shape[0])]))
    return refs


def sample_text_blocks(frames: np.ndarray) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """What the text encoder sees of a clip: 2 fps frames merged in pairs, each pair labelled with
    the mean of its two timestamps. An odd count repeats the last frame."""
    stride = REFERENCE_FPS / TEXT_SAMPLE_FPS
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    stamps = [i / TEXT_SAMPLE_FPS for i in range(len(indices))]
    sampled = [frames[i] for i in indices]
    if len(sampled) % 2:
        sampled.append(sampled[-1])
        stamps.append(stamps[-1])
    return [((stamps[i] + stamps[i + 1]) / 2, sampled[i], sampled[i + 1]) for i in range(0, len(sampled), 2)]


def encode_reference_rows(video_vae, refs, patch_size: tuple[int, int, int]):
    """Prepared references -> ``(packed rows, [RefGeometry])``, one block per reference in order.

    Pictures and clips take upstream's recipe: the posterior is sampled under the fixed keyframe
    seed and rounded through float16 before it is normalised.
    """
    rows, geoms = [], []
    for ref in refs:
        if ref.kind == "image":
            latent = encode_images(video_vae, [ref.image], ref.image.width, ref.image.height, sample=True)
            rows.append(patchify_video_latents(mx.array(latent), patch_size))
            geoms.append(RefGeometry("image", 1, ref.image.height // LATENT_RATIO, ref.image.width // LATENT_RATIO))
        else:
            rows.append(encode_video_rows(video_vae, ref.frames, patch_size, sample=True))
            geoms.append(RefGeometry("video", video_latents_for_frames(ref.frames.shape[0]),
                                     ref.frames.shape[1] // LATENT_RATIO, ref.frames.shape[2] // LATENT_RATIO))
    return mx.concatenate(rows), geoms


def refmod_meta(name: str, latent: np.ndarray, size: tuple[int, int], description: str = "", concept_type: str = "identity") -> dict:
    t, h, w = latent.shape[2:]
    return {
        "name": name,
        "kind": "video",
        "latent_h": int(h),
        "latent_w": int(w),
        "latent_t": int(t),
        "mode": "encode",
        "source": "stack",
        "source_shape": " +".join(f"1x{h}x{w}" for _ in range(t)),
        "pool": f"full-res {size[0]}x{size[1]}px",
        "optimize_steps": 0,
        "tags": [f"{t} img, 0 vid"],
        "description": description or f"RefMod dataset {name}",
        "concept_type": concept_type,
        "_format_version": 2,
    }


def save_refmod(path: str | Path, latent: np.ndarray, meta: dict) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_name(path.name + ".safetensors")
    mx.save_safetensors(
        str(path),
        {"latent": mx.array(latent.astype(np.float16))},
        metadata={"refmod_meta": json.dumps(meta)},
    )
    return path
