"""Build the Ref2VA 8-bit DiT pack the H3 view uses for "H3 Ref2VA Q8" (references and RefMods).

    RUNNER/.venv/bin/python scripts/build_ref2va_q8.py --runner RUNNER \\
        --checkpoint /path/to/MiniMax-H3/Ref2VA --out ~/models/h3-ref2va/q8
    python3 scripts/build_ref2va_q8.py --verify ~/models/h3-ref2va/q8      # check a finished pack, no MLX needed

RUNNER is the minimax-h3-mlx checkout (Phosphene's, or the vendored copy in vendor/minimax_h3_mlx).

Why this is not just the runner's `scripts/build_quant.py`: that script copies every bf16 weight into a dict so it
can quantize several bit widths from one load. For the 66 GB Ref2VA transformer that copy peaks near 93 GB of a
103 GB Mac. One width needs no copy, so this loads the model once and quantizes it in place. Everything else is the
runner's own code (`load_dit`, `quantize_dit`, `save_sharded`), so the pack has the same layout `build_quant.py`
writes: 8-bit, group 64, the 13B `adaln_proj` left in bf16 (it is read once to build the modulation cache and then
dropped), 208 quantized layers, 10 shards of at most 5 GB, `quant_config.json`, `model.safetensors.index.json`.

The Ref2VA transformer is its own fine-tune, not the FL2VA one (different shard bytes), so it is its own model.
Get the weights from the official repo, about 66 GB, 1-2 h:

    hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/transformer/*" --local-dir DIR     # DIR/Ref2VA is --checkpoint

Licence: the result is a MiniMax H3 Model Derivative (MiniMax H3 Community License; the US, EU, UK and South Korea
are excluded territories). Build your own; do not re-host it without the licence text, a "modified" notice and
"Powered by MiniMax H3" (the runner's `scripts/upload.py` enforces these).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

BITS, GROUP_SIZE, EXPECTED_LAYERS, MAX_SHARD_BYTES = 8, 64, 208, 5 * 1024**3
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


def verify_ng(pack: Path) -> list:
    """Problems with a finished pack, as strings (empty = matches what this script writes). Pure JSON and file
    checks, so it runs anywhere."""
    problems = []
    try:
        quant = json.loads((pack / "quant_config.json").read_text())
        index = json.loads((pack / "model.safetensors.index.json").read_text())
    except (OSError, ValueError) as e:
        return [f"cannot read quant_config.json / model.safetensors.index.json: {e}"]
    if (quant.get("bits"), quant.get("group_size")) != (BITS, GROUP_SIZE):
        problems.append(f"expected {BITS}-bit group {GROUP_SIZE}, found {quant.get('bits')}-bit group {quant.get('group_size')}")
    if quant.get("quantize_adaln"):
        problems.append("AdaLN is quantized; the runner expects it in bf16")
    if quant.get("quantized_layers") != {str(BITS): EXPECTED_LAYERS}:
        problems.append(f"expected {{'{BITS}': {EXPECTED_LAYERS}}} quantized layers, found {quant.get('quantized_layers')}")
    names = sorted({v for v in index.get("weight_map", {}).values()})
    totals = {SHARD_RE.match(n).group(2) for n in names if SHARD_RE.match(n)}
    if not names or len(totals) != 1 or any(not SHARD_RE.match(n) for n in names):
        problems.append(f"shard names are not model-NNNNN-of-NNNNN.safetensors: {names[:3]}")
    elif int(next(iter(totals))) != len(names):
        problems.append(f"index names {len(names)} shards but they say {next(iter(totals))}")
    missing = [n for n in names if not (pack / n).is_file()]
    if missing:
        problems.append(f"missing shard files: {missing}")
    biggest = max(((pack / n).stat().st_size for n in names if (pack / n).is_file()), default=0)
    if biggest > MAX_SHARD_BYTES + 64 * 1024**2:
        problems.append(f"a shard is {biggest / 1024**3:.1f} GB; shards should be at most 5 GB")
    if not (pack / "config.json").is_file():
        problems.append("config.json missing")
    return problems


def _load_runner(runner: Path):
    """The runner's package (on sys.path) and its build_quant module, for save_sharded."""
    sys.path.insert(0, str(runner))
    spec = importlib.util.spec_from_file_location("runner_build_quant", runner / "scripts" / "build_quant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ng(runner: Path, checkpoint: Path, out: Path) -> None:
    build_quant = _load_runner(runner)
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.quantize import QuantConfig, quantize_dit, resident_footprint

    source = checkpoint / "transformer" if (checkpoint / "transformer").is_dir() else checkpoint
    print(f"loading {source}", flush=True)
    started = time.perf_counter()
    model = load_dit(source)
    print(f"  loaded in {time.perf_counter() - started:.1f}s", flush=True)

    config = QuantConfig(bits=BITS, group_size=GROUP_SIZE, quantize_adaln=False)
    print(f"quantizing in place at {BITS}-bit (group {GROUP_SIZE}), adaln left in bf16", flush=True)
    summary = quantize_dit(model, config, verbose=True)
    footprint = resident_footprint(model)
    print(f"  on disk {footprint['total_gb']:.1f} GB; resident after the adaln drop {footprint['resident_gb']:.1f} GB", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "config.json", out / "config.json")
    meta = {"bits": BITS, "group_size": GROUP_SIZE, "quantize_adaln": False, "adaln_bits": None,
            "quantized_layers": {str(k): v for k, v in summary["quantized_layers"].items()},
            "gb_on_disk": round(footprint["total_gb"], 2),
            "gb_resident_after_adaln_drop": round(footprint["resident_gb"], 2)}
    (out / "quant_config.json").write_text(json.dumps(meta, indent=2))
    started = time.perf_counter()
    names = build_quant.save_sharded(model, out, {"quantization": json.dumps(meta)})
    print(f"  wrote {len(names)} shards in {time.perf_counter() - started:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", type=Path, help="the minimax-h3-mlx checkout (has minimax_h3_mlx/ and scripts/)")
    ap.add_argument("--checkpoint", type=Path, help="the official Ref2VA folder (contains transformer/) or the transformer folder")
    ap.add_argument("--out", type=Path, help="where the q8 pack is written")
    ap.add_argument("--verify", type=Path, metavar="PACK", help="only check a finished pack and exit")
    args = ap.parse_args()
    if args.verify:
        problems = verify_ng(args.verify)
        print("\n".join(problems) if problems else f"{args.verify}: matches the Ref2VA Q8 layout")
        return 1 if problems else 0
    if not (args.runner and args.checkpoint and args.out):
        ap.error("--runner, --checkpoint and --out are required (or use --verify PACK)")
    build_ng(args.runner.expanduser(), args.checkpoint.expanduser(), args.out.expanduser())
    problems = verify_ng(args.out.expanduser())
    print("\n".join(problems) if problems else "verified: layout matches")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
