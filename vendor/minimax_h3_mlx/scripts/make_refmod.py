"""Make a RefMod from a folder of pictures (about 20 is what the library uses).

    python scripts/make_refmod.py ~/photos/lake --name lake_v1 \\
        --compact-root <models>/ddalcu-q8 --out <models>/refmods

Writes ``<out>/<name>.safetensors``; the app's RefMod list picks it up on the next reload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from minimax_h3_mlx.load import load_compact_video_vae  # noqa: E402
from minimax_h3_mlx.refmod_encode import encode_images, list_images, refmod_meta, save_refmod  # noqa: E402

LANDSCAPE = (768, 448)  # 24x14 latent cells: a 16:9-ish scene keeps its shape, 336 rows a picture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--name", required=True)
    ap.add_argument("--compact-root", type=Path, required=True, help="folder holding video_vae.safetensors")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--landscape", action="store_true", help=f"use a {LANDSCAPE[0]}x{LANDSCAPE[1]} canvas instead of 512x512")
    ap.add_argument("--sample", action="store_true", help="sample the posterior (default: its mean)")
    ap.add_argument("--description", default="")
    ap.add_argument("--concept-type", default="identity")
    args = ap.parse_args()

    if args.landscape:
        args.width, args.height = LANDSCAPE
    paths = list_images(args.images)
    print(f"{len(paths)} images -> {args.width}x{args.height}")
    vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
    images = [Image.open(p) for p in paths]
    latent = encode_images(vae, images, args.width, args.height, sample=args.sample)
    rows = latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)
    meta = refmod_meta(args.name, latent, (args.width, args.height), args.description, args.concept_type)
    args.out.mkdir(parents=True, exist_ok=True)
    out = save_refmod(args.out / args.name, latent, meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {rows:,} reference rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
