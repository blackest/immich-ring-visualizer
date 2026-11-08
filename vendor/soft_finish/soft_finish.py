#!/usr/bin/env python3
"""Soft finish: a per-frame colour pass over a finished clip. mp4 in, mp4 out, the audio copied through.

    soft_finish.py --in clip.mp4 --out finished.mp4 [--clarity -0.2] [--radius 24] [--tone 0.25] [--saturation 0.9]

Two steps per frame, in this order (gamma-encoded RGB 0..1, no linearisation -- the same as ComfyUI feeds
the nodes, which is what the recipe was tuned on):

  1. Clarity (luma mode): local contrast in the midtones. A small NEGATIVE strength softens it (skin texture).
  2. Tone map (Cineon-ish) blended in at a low strength, then a saturation trim: the black and white points stay,
     low shadows lift gently. The Filmic curves are deliberately not offered -- on normal gamma video they darken
     the picture (measured: mean 64.6 -> 26.5).

It does not recover shadows that were clipped during generation.

The maths is a numpy/OpenCV port of two nodes from Orion4D_FXMax v1.0.0 (MIT, https://github.com/orion4d/
Orion4D_FXMax): `_clarity` / `gaussian_blur` / `midtone_mask` / `_apply_luma_detail` from Orion4D_FXMax.py and
utils.py, and `_apply_dctl_tone_mapper` (mode "Cineon-ish") from color_fx_common.py. Ported (torch -> numpy,
torch's reflect-padded conv -> cv2.GaussianBlur with BORDER_REFLECT_101, the same thing) so it runs in any Python with numpy and
OpenCV, with no ComfyUI and no GPU. See VENDORED.md.

Frames travel through ffmpeg pipes (rgb24 in, rgb24 out), so the clip's size, frame rate and audio come from
ffprobe/ffmpeg and nothing is decoded by OpenCV. The output is written to `<out>.partial` and renamed when
complete, so a killed run never leaves a half file under the real name. Progress goes to stderr as
`[finish] frame N/M`; exit 0 = the output exists.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

EPS = 1e-6
# The recipe: Clarity -0.20 at radius 24, Cineon-ish at strength 0.25, saturation 0.9.
DEFAULTS = {"clarity": -0.20, "radius": 24.0, "tone": 0.25, "saturation": 0.9}
MIDTONE_SOFTNESS = 0.35
CINEON_ROLLOFF = 0.65
CRF = "16"

_W709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma(img: np.ndarray) -> np.ndarray:
    return img @ _W709


def gaussian_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Orion's heuristic: sigma = max(0.15, radius*0.5 + 0.25), kernel to 3 sigma, reflect padding."""
    if radius <= 0:
        return img
    sigma = max(0.15, radius * 0.5 + 0.25)
    half = int(3.0 * sigma + 0.5)
    k = 2 * half + 1
    return cv2.GaussianBlur(img, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)


def midtone_mask(lum: np.ndarray, softness: float = MIDTONE_SOFTNESS) -> np.ndarray:
    """Near 1 around luma 0.5, 0 toward shadows and highlights."""
    s = max(1e-6, min(0.5, float(softness)))
    m = (np.clip(np.cos((lum - 0.5) * math.pi / (2 * s)), -1, 1) + 1) / 2.0
    m[(lum > 0.5 + s) | (lum < 0.5 - s)] = 0.0
    return m


def clarity(img: np.ndarray, strength: float, radius: float) -> np.ndarray:
    if abs(strength) < 1e-6:
        return img
    low = gaussian_blur(img, radius)
    lum = luma(img)
    detail = (lum - luma(low)) * midtone_mask(lum)
    lum_out = np.clip(lum + strength * detail, 0.0, 1.0)
    scale = (lum_out + EPS) / (lum + EPS)
    return np.clip(img * scale[..., None], 0.0, 1.0)


def tone_cineon(img: np.ndarray, strength: float, saturation: float) -> np.ndarray:
    if strength <= 0.0 and abs(saturation - 1.0) < 1e-6:
        return img
    k = 4.0 + CINEON_ROLLOFF * 4.0
    mapped = np.log1p(np.clip(img, 0.0, None) * k) / math.log1p(k)
    if abs(saturation - 1.0) > 1e-6:
        gray = luma(mapped)[..., None]
        mapped = gray + (mapped - gray) * saturation
    return np.clip(img + (mapped - img) * strength, 0.0, 1.0)


def finish_frame(rgb_u8: np.ndarray, p: dict) -> np.ndarray:
    img = rgb_u8.astype(np.float32) / 255.0
    img = clarity(img, p["clarity"], p["radius"])
    img = tone_cineon(img, p["tone"], p["saturation"])
    return (img * 255.0 + 0.5).astype(np.uint8)


# ---- the clip -------------------------------------------------------------------------------------------

def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate,nb_frames",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise SystemExit("the file has no video")
    num, _, den = str(video.get("r_frame_rate") or "24/1").partition("/")
    fps = float(num) / float(den or 1)
    return {"w": int(video["width"]), "h": int(video["height"]), "fps": fps if 1.0 <= fps <= 120.0 else 24.0,
            "frames": int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else 0,
            "audio": any(s.get("codec_type") == "audio" for s in streams)}


def run(src: str, dst: str, p: dict) -> None:
    info = probe(src)
    w, h = info["w"], info["h"]
    partial = dst + ".partial"
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", repr(info["fps"]), "-i", "-"]
    if info["audio"]:
        enc_cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    enc_cmd += ["-c:v", "libx264", "-crf", CRF, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-f", "mp4", partial]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    size = w * h * 3
    n = 0
    try:
        while True:
            buf = dec.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            enc.stdin.write(finish_frame(frame, p).tobytes())
            n += 1
            if n % 12 == 0 or n == info["frames"]:
                print(f"[finish] frame {n}/{info['frames'] or '?'}", file=sys.stderr, flush=True)
        enc.stdin.close()
        if enc.wait() != 0 or dec.wait() != 0 or n == 0:
            raise SystemExit("ffmpeg failed")
        os.replace(partial, dst)
    except BaseException:
        for proc in (enc, dec):
            if proc.poll() is None:
                proc.kill()
        if os.path.exists(partial):
            os.unlink(partial)
        raise
    print(f"[finish] done, {n} frames", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--clarity", type=float, default=DEFAULTS["clarity"], help="local contrast, -1..1 (negative softens)")
    ap.add_argument("--radius", type=float, default=DEFAULTS["radius"], help="clarity radius")
    ap.add_argument("--tone", type=float, default=DEFAULTS["tone"], help="Cineon-ish tone map strength, 0..1")
    ap.add_argument("--saturation", type=float, default=DEFAULTS["saturation"])
    a = ap.parse_args(argv)
    if not os.path.isfile(a.src):
        print(f"no such file: {a.src}", file=sys.stderr)
        return 2
    run(a.src, a.dst, {"clarity": a.clarity, "radius": a.radius, "tone": a.tone, "saturation": a.saturation})
    return 0


if __name__ == "__main__":
    sys.exit(main())
