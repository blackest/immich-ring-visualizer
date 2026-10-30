"""Decode a RefMod into a contact sheet and re-encode it, to check our encoder against a library file.

    python scripts/refmod_roundtrip.py FILE.safetensors --compact-root <models>/ddalcu-q8 --out sheet.png

Each stored latent frame is un-normalised and decoded to a picture (the sheet shows what the author
fed in, at the latent's own resolution). The pictures are then encoded again with
``refmod_encode.encode_images``; if our encoder follows the same conventions the new latents land
close to the stored ones. Prints the mean absolute error against the latents' own spread, for the
posterior mean and for a sample, so the sheet also says which one the library used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from minimax_h3_mlx.load import load_compact_video_vae  # noqa: E402
from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD  # noqa: E402
from minimax_h3_mlx.refmod_encode import LATENT_RATIO, encode_images  # noqa: E402
from minimax_h3_mlx.refmod_io import load_refmod  # noqa: E402


def decode_frames(vae, latent: np.ndarray) -> list[Image.Image]:
    """``(1, C, T, H, W)`` normalised latent -> one picture per latent frame (spatial decode only)."""
    cfg = vae.config
    mean = np.array(cfg.latents_mean, np.float32).reshape(1, -1, 1, 1, 1)
    std = np.array(cfg.latents_std, np.float32).reshape(1, -1, 1, 1, 1)
    z = latent * std + mean
    pix_mean = np.array(PIXEL_MEAN, np.float32)
    pix_std = np.array(PIXEL_STD, np.float32)
    out = []
    for t in range(z.shape[2]):
        frame = mx.array(z[:, :, t : t + 1].transpose(0, 2, 3, 4, 1))  # (1, 1, H, W, C)
        pixels = vae._run_decoder(frame)
        mx.eval(pixels)
        arr = np.array(pixels)[0, -1]  # (H', W', 3); the last frame of the decoder's output
        arr = np.clip(arr * pix_std + pix_mean, 0.0, 1.0)
        out.append(Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)))
    return out


def sheet(images: list[Image.Image], cell: int = 256, cols: int = 6) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell))
    for i, im in enumerate(images):
        canvas.paste(im.resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refmod", type=Path)
    ap.add_argument("--compact-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("refmod_sheet.png"))
    args = ap.parse_args()

    mod = load_refmod(args.refmod)[0]
    vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
    _, _, t, h, w = mod.latent.shape
    print(f"{args.refmod.name}: {t} frames, latent {h}x{w} -> pictures {w * LATENT_RATIO}x{h * LATENT_RATIO}")
    pics = decode_frames(vae, mod.latent)
    sheet(pics).save(args.out)
    print("contact sheet:", args.out)

    spread = float(np.abs(mod.latent - mod.latent.mean()).mean())
    for label, sample in (("mean", False), ("sample", True)):
        again = encode_images(vae, pics, w * LATENT_RATIO, h * LATENT_RATIO, sample=sample)
        err = float(np.abs(again - mod.latent).mean())
        print(f"re-encode ({label}): mean |new - stored| = {err:.3f}  (latent spread {spread:.3f}, ratio {err / spread:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
