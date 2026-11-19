#!/usr/bin/env python3
"""Cut the small AdaLN overlay pack out of an official-layout Ref2VA DiT.

The hybrid ("Clean" look) runs FL2VA everywhere and borrows only Ref2VA's AdaLN for the last blocks
(see vendor/minimax_h3_mlx/minimax_h3_mlx/adaln_overlay.py). That reads 14 tensors out of a 47.5 GB model:
``time_embedder.*`` and ``blocks.45-49.adaln_proj.linear.{weight,bias}`` (about 2.7 GB). This writes
exactly those into a folder the overlay accepts in place of the full model:

    <dest>/config.json         copied from the source (the overlay reads the DiT config from it)
    <dest>/model.safetensors   only the chosen tensors, byte-for-byte (dtype untouched)
    <dest>/LICENSE*, NOTICE*   copied from the source if it has any

    python3 scripts/extract_adaln_overlay_pack.py /Volumes/AI/models/h3-ref2va/q8 /Volumes/AI/models/h3-ref2va/adaln-overlay-45-49

The full model keeps working as an overlay source: the overlay takes either folder. A pack built here
can hold fewer blocks than a run asks for, and then the overlay says which tensor is missing.

The pack is a derivative of Ref2VA, so MiniMax's licence applies to it as to the model it came from.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

TIME_EMBEDDER = [f"time_embedder.{part}.{kind}" for part in ("proj_in", "proj_out") for kind in ("weight", "bias")]


def parse_blocks(spec: str, num_blocks: int) -> list[int]:
    """``"45-49"``, ``"45-"``, ``"3"`` or ``"0,2,45-49"`` -> sorted block indices."""
    blocks: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), (int(hi_s) if hi_s else num_blocks - 1)
        else:
            lo = hi = int(part)
        if lo > hi:
            raise ValueError(f"block range {part!r} runs backwards")
        blocks.update(range(lo, hi + 1))
    bad = sorted(b for b in blocks if not 0 <= b < num_blocks)
    if not blocks or bad:
        raise ValueError(f"blocks {bad or spec!r} are outside 0-{num_blocks - 1}")
    return sorted(blocks)


def weight_map(source: Path) -> dict[str, str]:
    index = source / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    single = source / "model.safetensors"
    if single.is_file():
        return {k: single.name for k in mx.load(str(single))}
    raise FileNotFoundError(f"{source} has neither model.safetensors.index.json nor model.safetensors")


def extract(source: Path, dest: Path, blocks_spec: str = "45-49") -> dict:
    """Write the pack; returns {"blocks": [...], "tensors": n, "bytes": n}."""
    config = json.loads((source / "config.json").read_text())
    blocks = parse_blocks(blocks_spec, int(config["num_layers"]))
    names = list(TIME_EMBEDDER)
    for b in blocks:
        names += [f"blocks.{b}.adaln_proj.linear.weight", f"blocks.{b}.adaln_proj.linear.bias"]
    wmap = weight_map(source)
    missing = [n for n in names if n not in wmap]
    if missing:
        raise KeyError(f"{source.name} has no tensor {missing[0]!r}" + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""))
    if any(f"blocks.{b}.adaln_proj.linear.scales" in wmap for b in blocks):
        raise ValueError(f"{source.name} has quantized AdaLN projections; the overlay needs them unquantized")
    if config.get("adaln_curve_grid") is not None:
        raise ValueError(f"{source.name} is a pruned DiT; the overlay needs the official layout")

    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    tensors: dict[str, mx.array] = {}
    for shard, wanted in by_shard.items():
        loaded = mx.load(str(source / shard))
        for n in wanted:
            tensors[n] = loaded[n]
        mx.eval(*tensors.values())
        del loaded

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "config.json", dest / "config.json")
    out = dest / "model.safetensors"
    tmp = dest / "model.partial.safetensors"  # save_safetensors wants the .safetensors suffix itself
    mx.save_safetensors(str(tmp), tensors, {"format": "mlx", "adaln_overlay_blocks": blocks_spec,
                                           "source": source.name})
    tmp.replace(out)
    for extra in sorted(source.glob("LICENSE*")) + sorted(source.glob("NOTICE*")):
        shutil.copy2(extra, dest / extra.name)
    return {"blocks": blocks, "tensors": len(tensors), "bytes": out.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="official-layout Ref2VA folder (config.json + shards)")
    ap.add_argument("dest", type=Path, help="folder to write (created; must not already hold a model.safetensors)")
    ap.add_argument("--blocks", default="45-49", help="blocks to keep (default 45-49)")
    args = ap.parse_args(argv)
    if (args.dest / "model.safetensors").exists():
        print(f"{args.dest} already has a model.safetensors; remove it or pick another folder.", file=sys.stderr)
        return 2
    info = extract(args.source, args.dest, args.blocks)
    print(f"wrote {args.dest / 'model.safetensors'}: {info['tensors']} tensors, blocks {info['blocks'][0]}-{info['blocks'][-1]}, "
          f"{info['bytes'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
