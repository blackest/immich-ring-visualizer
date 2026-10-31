"""Build Pika H3 Flex's Ref2VA adapter for the MLX H3 runner, on this Mac.

    python3 scripts/build_pika_ref2va_adapter.py --dir ~/models/h3-ref2va/pika-flex-ref2va

Phosphene 4.19.15 ships only the FL2VA adapter (`scripts/fetch_h3_flex.py`); its docs say the Ref2VA one "targets the
separate Ref2VA trunk, which this pack does not carry". That trunk is the model our "H3 Ref2VA Q8" pack is built from
(scripts/build_ref2va_q8.py, scripts/prune_ref2va_adaln.py). Pika's Ref2VA adapter has the same shape as the FL2VA one
(rank 128, alpha 128, the same target modules, 2,767,274,120 bytes, the same re-noise sampler knots and shifts), so
Phosphene's own re-layout code applies unchanged. This script reuses it, with the Ref2VA constants:

  * fetch the ORIGINAL file from Pika's Hugging Face repo at the pinned commit, checking the SHA-256 Pika publishes in
    its SHA256SUMS (never re-hosted), with Pika's LICENSE, NOTICE and the release's own Ref2VA manifest beside it;
  * re-lay it out for the runner's `--lora` loader (fused per-head qkv, SwiGLU halves swapped, float32 -> bfloat16) and
    write `pika_h3_flex_ref2va_phosphene.safetensors` (about 1.96 GB) with a modification notice in its metadata.

The result is a MiniMax H3 Model Derivative (MiniMax H3 Community License; Pika's code is Apache-2.0). Build your own;
do not re-host it without the licence, the NOTICE and a "modified" notice.

Unproven: whether the runner applies this adapter on a Ref2VA DiT together with `--refmod` / `--reference` (the app
refuses Flex with references until that has been tried). The conversion itself ends with the same checks as Phosphene's.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SOURCE_FILE = "ref2va/adapter.safetensors"
SOURCE_SHA256 = "eef684ba1258f0cdbbbda6128188ad2688417f2eb17ff875375e74a32b6d61b5"  # Pika's SHA256SUMS
MANIFEST = ("ref2va/manifest.json", "758497d3b316ea1a956a5b460d6c02ef380ecb5bf95f16f29e932d438c3ccff8")
ASSET_NAME = "pika_h3_flex_ref2va_phosphene.safetensors"


def _load_phosphene_fetcher(phosphene: Path):
    spec = importlib.util.spec_from_file_location("fetch_h3_flex", phosphene / "scripts" / "fetch_h3_flex.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_ng(fetch) -> None:
    """Point Phosphene's FL2VA fetcher at the Ref2VA file. Its functions read these module globals when they run."""
    fetch.SOURCE_FILE = SOURCE_FILE
    fetch.SOURCE_SHA256 = SOURCE_SHA256  # the byte size is the same as FL2VA's (checked below by the download)
    fetch.ASSET_NAME = ASSET_NAME
    fetch.ASSET_SHA256 = ""  # no pinned digest for our result yet; the build still checks the layout as it writes
    fetch.ASSET_BYTES = 0
    fetch.SIDE_FILES = {MANIFEST[0]: MANIFEST[1], **{k: v for k, v in fetch.SIDE_FILES.items() if "/" not in k}}
    fetch.MODIFICATION_NOTICE = fetch.MODIFICATION_NOTICE.replace("FL2VA", "Ref2VA")
    fetch.METADATA.update({
        "converted_from": f"{fetch.SOURCE_REPO}@{fetch.SOURCE_REVISION}:{SOURCE_FILE}",
        "source_sha256": SOURCE_SHA256,
        "modification_notice": fetch.MODIFICATION_NOTICE,
        "trunk": "transformer_ref (the Ref2VA DiT)",
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, type=Path, help="where to put the adapter and Pika's licence files")
    ap.add_argument("--phosphene", type=Path, default=Path("/Volumes/AI/Pinkio/api/phosphene.git"),
                    help="Phosphene's checkout (for scripts/fetch_h3_flex.py)")
    ap.add_argument("--keep-source", action="store_true", help="keep the 2.8 GB upstream file after a good build")
    args = ap.parse_args()

    fetch = _load_phosphene_fetcher(args.phosphene.expanduser())
    configure_ng(fetch)
    target_dir = args.dir.expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    for name, sha in fetch.SIDE_FILES.items():  # licence, notice, manifest
        dest = target_dir / ("pika_" + name.replace("/", "_"))
        if dest.is_file() and fetch.file_sha256(dest) == sha:
            continue
        if fetch._download(fetch.source_url(name), dest, None, sha, name):
            return 1

    target = target_dir / ASSET_NAME
    if target.is_file():
        print(f"already built: {target}")
        return 0
    source = target_dir / "pika_ref2va_adapter.safetensors.download"
    if fetch._download(fetch.source_url(SOURCE_FILE), source, fetch.SOURCE_BYTES, SOURCE_SHA256, SOURCE_FILE):
        return 1
    if fetch.build(source, target):
        return 1
    if not args.keep_source:
        source.unlink(missing_ok=True)
    print(f"Pika H3 Flex Ref2VA adapter ready: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
