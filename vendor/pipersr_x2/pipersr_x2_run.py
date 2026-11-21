#!/usr/bin/env python3
"""2x upscale of a finished clip or a still picture with PiperSR on the Apple Neural Engine, keeping a clip's sound.

Run with this folder's own env (built by build_venv.sh):

    vendor/pipersr_x2/.venv/bin/python vendor/pipersr_x2/pipersr_x2_run.py CLIP.mp4 [-o OUT.mp4] [--strength 1]
    vendor/pipersr_x2/.venv/bin/python vendor/pipersr_x2/pipersr_x2_run.py PICTURE.png [-o OUT.png] [--strength 1]

A picture (png, jpg, jpeg, webp) comes out as a PNG; anything else is read as a clip.

PiperSR is a small image model (128x128 tiles in, 256x256 out), so each frame is cut into overlapping tiles and only the
middle of every tile is kept; the plain edge-to-edge tiling of a first try leaves faint grid seams. It sharpens what is there
and does not redraw the picture (LTX's upscaler does that). No GPU is used: the model runs on the Neural Engine.

Halo limit (always on): PiperSR's raw output roughly doubles the edge contrast but overshoots it, which shows as halos and brings
out grain in dark areas. Every output pixel is held between the darkest and brightest value of a plain Lanczos 2x resize of
the same frame within 2 pixels (a 5x5 window). On a dark AI picture that removed about 88% of the overshoot and 40% of the
grain and kept about 60% of the edge gain; judged by eye (the user, 2026-11-21) it blends in best, the wings clearly sharper.

--strength (0 to 1, default 1) then mixes that picture with the plain Lanczos resize: 1 is the halo-limited PiperSR picture, 0 is
the plain resize. Half and half keeps about a third of the edge gain with far less grain.

Licence: the pipersr code is AGPL-3.0, the model weights are free for personal and academic use with attribution; commercial
use needs a separate licence from the author. Attribution: "Powered by PiperSR from ModelPiper - https://modelpiper.com".
This file is ours and only calls PiperSR as a separate program; nothing of it is copied here.

Protocol on stdout, one line each: `[pipersr] frame i/n` (clips only), `[pipersr] done <seconds> <output>`; the attribution
line comes first.
Exit codes: 0 ok, 2 bad input, 1 anything else.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ATTRIBUTION = "Powered by PiperSR from ModelPiper - https://modelpiper.com"
DEFAULT_STRENGTH = 1.0
HALO_WINDOW = 5   # size of the window whose brightest and darkest plain-resize values bound each output pixel
TILE = 128      # the model's fixed input size
MARGIN = 16     # input pixels thrown away on each interior side of a tile (their 2x output is 32 px)
STRIDE = TILE - 2 * MARGIN


def tile_segments(length: int) -> list[tuple[int, int, int]]:
    """(origin, start, end) along one axis of an image already padded by MARGIN on both sides.

    Each tile at `origin` supplies the stretch [start, end) of the padded image; together they cover
    [MARGIN, length - MARGIN) exactly once, and every stretch sits at least MARGIN inside its tile.
    """
    if length < TILE:
        raise ValueError(f"padded length {length} is smaller than a tile")
    origins = list(range(0, length - TILE + 1, STRIDE))
    if origins[-1] + TILE < length:
        origins.append(length - TILE)
    segments, pos = [], MARGIN
    for origin in origins:
        end = min(origin + TILE - MARGIN, length - MARGIN)
        if end > pos:
            segments.append((origin, pos, end))
            pos = end
    return segments


def load_model():
    import coremltools as ct
    import pipersr
    unit = getattr(ct.ComputeUnit, "CPU_AND_NE", None) or getattr(ct.ComputeUnit, "CPU_AND_NEURAL_ENGINE")
    return ct.models.MLModel(str(pipersr._find_model()), compute_units=unit)


def predict_tile(model, tile):
    result = model.predict({"input_image": tile})
    out = result.get("output_image")
    if out is None:
        out = next(iter(result.values()))
    return out.convert("RGB")


def mix(plain, sharp, strength: float):
    """`strength` of the PiperSR picture over the plain resize (both RGB, same size)."""
    from PIL import Image
    return sharp if strength >= 1.0 else plain if strength <= 0.0 else Image.blend(plain, sharp, strength)


def _running(arr, op):
    """`op` (np.minimum or np.maximum) over a HALO_WINDOW square around every pixel, as two passes of shifts
    (a square window is separable); edges are extended. PIL's own 5x5 rank filters give the same numbers but take
    several seconds a 3072x2048 picture, and this runs once per video frame."""
    import numpy as np

    r = HALO_WINDOW // 2
    for axis in (0, 1):
        padded = np.pad(arr, [(r, r) if a == axis else (0, 0) for a in range(arr.ndim)], mode="edge")
        n = arr.shape[axis]
        out = np.take(padded, range(0, n), axis=axis)
        for k in range(1, HALO_WINDOW):
            out = op(out, np.take(padded, range(k, k + n), axis=axis))
        arr = out
    return arr


def limit_halo(plain, sharp):
    """`sharp` with every pixel held between the darkest and brightest `plain` value in its HALO_WINDOW square."""
    import numpy as np
    from PIL import Image

    plain = np.asarray(plain)
    return Image.fromarray(np.clip(np.asarray(sharp), _running(plain, np.minimum), _running(plain, np.maximum)))


def upscale_image(model, img, strength: float = DEFAULT_STRENGTH):
    """One RGB PIL image -> its 2x version (same pixels' neighbourhood used for every output pixel)."""
    import numpy as np
    from PIL import Image

    w, h = img.size
    padded = Image.fromarray(np.pad(np.asarray(img.convert("RGB")), ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)),
                                    mode="reflect"))
    pw, ph = padded.size
    out = Image.new("RGB", (w * 2, h * 2))
    for oy, ys, ye in tile_segments(ph):
        for ox, xs, xe in tile_segments(pw):
            sr = predict_tile(model, padded.crop((ox, oy, ox + TILE, oy + TILE)))
            piece = sr.crop(((xs - ox) * 2, (ys - oy) * 2, (xe - ox) * 2, (ye - oy) * 2))
            out.paste(piece, ((xs - MARGIN) * 2, (ys - MARGIN) * 2))
    plain = img.convert("RGB").resize((w * 2, h * 2), Image.LANCZOS)
    return mix(plain, limit_halo(plain, out), strength)


def ffprobe(clip: Path, entries: str, stream: str = "v:0") -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries,
                           "-of", "default=nw=1:nk=1", str(clip)], capture_output=True, text=True).stdout.strip()


PICTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def sharpen_picture(src: Path, out: Path, strength: float = DEFAULT_STRENGTH) -> int:
    from PIL import Image

    print(ATTRIBUTION, flush=True)
    t0 = time.perf_counter()
    try:
        img = Image.open(src)
        img.load()
    except Exception as e:  # noqa: BLE001 - a bad picture is the caller's to hear about, not a traceback
        print(f"[pipersr] error cannot read that picture: {e}", flush=True)
        return 2
    if min(img.size) < 32:
        print("[pipersr] error the picture is too small", flush=True)
        return 2
    upscale_image(load_model(), img.convert("RGB"), strength).save(out, format="PNG")
    print(f"[pipersr] done {time.perf_counter() - t0:.1f}s {out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", type=Path, help="a clip, or a picture (png, jpg, jpeg, webp)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="default: <name>_sr2x.mp4 (clip) or <name>_sr2x.png (picture) beside the input")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH,
                    help="0 (plain resize) to 1 (pure PiperSR); default %(default)s")
    args = ap.parse_args(argv)
    if not 0.0 <= args.strength <= 1.0:
        print(f"[pipersr] error strength must be 0 to 1 (got {args.strength:g})", flush=True)
        return 2
    if not args.clip.is_file():
        print(f"[pipersr] error no such file: {args.clip}", flush=True)
        return 2
    if args.clip.suffix.lower() in PICTURE_SUFFIXES:
        return sharpen_picture(args.clip, args.output or args.clip.with_name(args.clip.stem + "_sr2x.png"), args.strength)
    out = args.output or args.clip.with_name(args.clip.stem + "_sr2x.mp4")
    probe = ffprobe(args.clip, "stream=width,height,r_frame_rate").splitlines()
    if len(probe) < 3:
        print("[pipersr] error that file has no video", flush=True)
        return 2
    width, height, fps = int(probe[0]), int(probe[1]), probe[2]
    if width < 32 or height < 32:
        print("[pipersr] error the picture is too small", flush=True)
        return 2
    has_audio = bool(ffprobe(args.clip, "stream=index", "a"))

    print(ATTRIBUTION, flush=True)
    from PIL import Image
    t0 = time.perf_counter()
    model = load_model()
    with tempfile.TemporaryDirectory(prefix="pipersr_x2_") as work:
        frames, sr = Path(work) / "in", Path(work) / "sr"
        frames.mkdir()
        sr.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.clip), "-map", "0:v:0",
                        str(frames / "f_%06d.png")], check=True)
        files = sorted(frames.glob("f_*.png"))
        for i, f in enumerate(files, 1):
            upscale_image(model, Image.open(f).convert("RGB"), args.strength).save(sr / f.name)
            if i == 1 or i == len(files) or i % 20 == 0:
                print(f"[pipersr] frame {i}/{len(files)}", flush=True)
        cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-i", str(sr / "f_%06d.png")]
        cmd += ["-i", str(args.clip), "-map", "0:v:0"] + (["-map", "1:a:0", "-c:a", "copy"] if has_audio else ["-an"])
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    print(f"[pipersr] done {time.perf_counter() - t0:.1f}s {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
