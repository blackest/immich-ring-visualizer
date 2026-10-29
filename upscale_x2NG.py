"""Upscale a finished clip 2x with LTX 2.5's pixel-spatial-upscaler IC-LoRA, keeping its sound.

Run with the ltx-2-mlx venv's python (it imports ltx_pipelines_mlx):

    <phosphene>/ltx-2-mlx/env/bin/python3.11 upscale_x2NG.py CLIP.mp4 [-o OUT.mp4] [--steps 3]

The clip's own latent is upsampled 2x and only --steps distilled steps run at full size, so the
shot, face and motion stay the clip's. Same recipe as Phosphene's "Upscale & Face Fix" (the
Faithful setting), minus its panel, queue and warm helper: ICLoraPipeline does the work.
LTX's frame grid is 1+8k, so a 124-frame clip renders as 129 (last frame held) and is cut back.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Read by ltx_core_mlx / MLX at import time, so they go in before the pipeline import below.
os.environ.setdefault("LTX2_DIT_EVAL_EVERY", "1")
os.environ.setdefault("LTX2_GEMMA_EVAL_EVERY", "1")
# M1/M2 GPUs kill a command buffer that runs too long ("Impacting Interactivity"). One op per buffer
# gives identical output and was the only setting that got through (10 died right after the text
# encoder loaded, three times), so it is the default there; --short-gpu forces it on other chips.
SHORT_GPU = "--short-gpu" in sys.argv
if SHORT_GPU or re.search(r"Apple M[12]\b", subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout):
    os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "1")
    os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "1")

PHOSPHENE = Path(os.environ.get("LTX_STUDIO_ROOT", "/Volumes/AI/Pinkio/api/phosphene.git"))
MODELS = PHOSPHENE / "mlx_models"
PACK = MODELS / "ltx-2.5-mlx-q8"
TEXT_ENCODER = MODELS / "gemma4-12b-ltx25-q4"  # LTX 2.5's own encoder; the 2.3 one gives a mosaic
ADAPTER = MODELS / "loras" / "ic" / "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_PROMPT = "The same shot, sharper and more detailed."
VAE_STREAMING_MAX_FRAMES = 121  # the patched decoder streams longer clips on its own


def probe(path: Path) -> tuple[int, int, int, float]:
    """(width, height, frames, fps) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = dict(line.split("=", 1) for line in out.splitlines())
    num, _, den = info["r_frame_rate"].partition("/")
    return int(info["width"]), int(info["height"]), int(info["nb_read_frames"]), float(num) / float(den or 1)


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clip", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--steps", type=int, default=3, help="refine steps from the clip's own latent")
    ap.add_argument("--keep", type=float, default=1.0, help="how tightly the clip is held (0.3 to 1)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--short-gpu", action="store_true", help="one op per Metal buffer (already the default on M1/M2)")
    args = ap.parse_args()

    for path in (PACK, TEXT_ENCODER, ADAPTER):
        if not path.exists():
            sys.exit(f"missing: {path}")
    width, height, frames, fps = probe(args.clip)
    out_w, out_h = width * 2 // 64 * 64, height * 2 // 64 * 64
    grid = 1 + 8 * math.ceil((frames - 1) / 8)
    output = args.output or args.clip.with_name(args.clip.stem + "_x2.mp4")
    keep = max(0.3, min(1.0, args.keep))
    print(f"{args.clip.name}: {width}x{height} {frames}f @ {fps:g} -> {out_w}x{out_h}, renders {grid}f", flush=True)

    if frames <= VAE_STREAMING_MAX_FRAMES:
        os.environ["LTX_VAE_STREAMING"] = "0"
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline  # after the environment is set

    with tempfile.TemporaryDirectory(prefix="upscale_x2_") as work:
        source = args.clip
        if grid > frames:
            # lossless copy with the last frame held, so the reference loader sees a full 1+8k clip
            source = Path(work) / "held.mp4"
            ffmpeg("-i", args.clip, "-map", "0:v:0", "-an", "-vf", f"tpad=stop_mode=clone:stop={grid - frames}",
                   "-frames:v", grid, "-c:v", "libx264rgb", "-crf", 0, "-preset", "veryfast",
                   "-pix_fmt", "rgb24", "-f", "mp4", source)
        rendered = Path(work) / "x2.mp4"
        pipe = ICLoraPipeline(model_dir=PACK, lora_paths=[(str(ADAPTER), 1.0)],
                              gemma_model_id=str(TEXT_ENCODER), low_memory=True)
        pipe.generate_and_save(
            prompt=args.prompt, output_path=str(rendered), video_conditioning=[[str(source), keep]],
            height=out_h, width=out_w, num_frames=grid, frame_rate=fps, seed=args.seed,
            stage1_steps=4, stage2_steps=3, conditioning_attention_strength=keep,
            source_video=str(source), refine_steps=args.steps)

        # the clip's own sound, padded then cut to frames/fps (never -shortest, which eats a frame)
        has_audio = bool(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
             str(args.clip)], capture_output=True, text=True).stdout.strip())
        cmd = ["-i", rendered] + (["-i", args.clip] if has_audio else []) + ["-map", "0:v:0"]
        cmd += ["-map", "1:a:0"] if has_audio else []
        cmd += ["-frames:v", frames]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", 18, "-preset", "medium"] if grid > frames else ["-c:v", "copy"]
        cmd += ["-af", f"apad,atrim=end={frames / fps:.6f}", "-c:a", "aac", "-b:a", "192k"] if has_audio else ["-an"]
        ffmpeg(*cmd, "-movflags", "+faststart", "-f", "mp4", output)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
