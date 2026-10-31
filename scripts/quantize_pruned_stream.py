"""Quantize a pruned H3 DiT (single file with `adaln_t_table`) to a pack, one tensor at a time, at chosen bit widths.

    RUNNER/.venv/bin/python scripts/quantize_pruned_stream.py --runner RUNNER \\
        --src Ref2VA-pruned_bf16.safetensors --out ~/models/h3-ref2va/q4-pruned --bits 4

The same job as the runner's `scripts/quantize_stream.py` (same CPU stream, same shard size, same file names and
`config.json` / `quant_config.json` layout, so `load_dit` reads the result), with one difference: the core linears and
the AdaLN projections take separate widths, and the AdaLN can be left unquantized (`--adaln-bits 0`, the default). The
upstream script quantizes it at the same width as the core. The runner's own notes say AdaLN error biases every block
identically at every step, and on Ref2VA an 8-bit AdaLN changed a seeded render where pruning alone (1.7e-6 relative)
should not. The pruned projections are only 64 wide, so keeping them in float32 costs about 1.3 GB.
`--bits 8 --adaln-bits 8` reproduces upstream's Q8 pack byte for byte.

Bounded memory: the source is memory-mapped and only one tensor is quantized at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import sys
import time
from pathlib import Path

GROUP = 64
MAX_SHARD = 5 * 1024**3


def read_header_ng(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def plan_ng(header: dict, qc) -> list:
    """[(key, kind, shape, bits)] with kind "q" for a tensor to quantize, else its dtype (copied unchanged). Only 2-D
    `.weight` tensors of modules the recipe names, whose input width divides the group, are quantized."""
    plan = []
    for key, meta in sorted(header.items()):
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        bits = qc.bits_for(base) if base else None
        if bits and len(meta["shape"]) == 2 and meta["shape"][1] % GROUP == 0:
            plan.append((key, "q", meta["shape"], bits))
        else:
            plan.append((key, meta["dtype"], meta["shape"], None))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", type=Path, required=True, help="the minimax-h3-mlx checkout")
    ap.add_argument("--src", type=Path, required=True, help="the pruned single-file checkpoint")
    ap.add_argument("--out", type=Path, required=True, help="the pack folder to write")
    ap.add_argument("--bits", type=int, default=8, choices=[2, 3, 4, 6, 8], help="core linears (default 8)")
    ap.add_argument("--adaln-bits", type=int, default=0, choices=[0, 2, 3, 4, 6, 8],
                    help="AdaLN projections; 0 leaves them unquantized in float32 (default 0)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.runner.expanduser()))
    import mlx.core as mx
    from safetensors import safe_open

    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.quantize import QuantConfig

    src_path, out = a.src.expanduser(), a.out.expanduser()
    t0 = time.time()
    header = read_header_ng(src_path)
    with safe_open(str(src_path), framework="np") as h:
        grid, rank = h.get_slice("adaln_t_table").get_shape()
    cfg = DiTConfig(time_embed_dim=rank, adaln_curve_grid=grid)
    qc = QuantConfig(bits=a.bits, group_size=GROUP, quantize_adaln=a.adaln_bits > 0, adaln_bits=a.adaln_bits or 8)
    plan = plan_ng(header, qc)
    print(f"{len(plan)} tensors, {sum(1 for p in plan if p[1] == 'q')} quantized "
          f"(core {a.bits}-bit, AdaLN {str(a.adaln_bits) + '-bit' if a.adaln_bits else 'unquantized'}, group {GROUP})", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    src = mx.load(str(src_path))  # lazy, mmap-backed
    shard_idx, shard_bytes, shard, names, weight_map = 1, 0, {}, [], {}

    def flush() -> None:
        nonlocal shard_idx, shard_bytes, shard
        if not shard:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        mx.save_safetensors(str(out / name), shard)
        weight_map.update({k: name for k in shard})
        names.append(name)
        print(f"  wrote {name} ({shard_bytes / 1e9:.2f} GB, {len(shard)} tensors)", flush=True)
        shard_idx, shard_bytes, shard = shard_idx + 1, 0, {}

    with mx.stream(mx.cpu):  # CPU quantization: deterministic and RAM-bounded
        for key, kind, _shape, bits in plan:
            if kind == "q":
                w, sc, bi = mx.quantize(src[key], group_size=GROUP, bits=bits)
                base = key[: -len(".weight")]
                add = {base + ".weight": w, base + ".scales": sc.astype(mx.bfloat16), base + ".biases": bi.astype(mx.bfloat16)}
            else:
                add = {key: src[key]}
            mx.eval(list(add.values()))
            nbytes = sum(t.nbytes for t in add.values())
            if shard_bytes + nbytes > MAX_SHARD and shard:
                flush()
            shard.update(add)
            shard_bytes += nbytes
    flush()

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_shards": len(names)}, "weight_map": weight_map}, indent=1))
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    (out / "quant_config.json").write_text(json.dumps(
        {"bits": a.bits, "group_size": GROUP, "quantize_adaln": a.adaln_bits > 0, "adaln_bits": a.adaln_bits or None,
         "writer": "quantize_stream", "source": src_path.name}, indent=1))

    # Check each width on a real tensor, then make load_dit build the whole tree from the pack.
    for want in sorted({p[3] for p in plan if p[1] == "q"}):
        key = next(p[0] for p in plan if p[1] == "q" and p[3] == want)
        base = key[: -len(".weight")]
        packed = mx.load(str(out / weight_map[base + ".weight"]))
        deq = mx.dequantize(packed[base + ".weight"], packed[base + ".scales"].astype(mx.float32),
                            packed[base + ".biases"].astype(mx.float32), group_size=GROUP, bits=want)
        err = float(mx.abs(deq - src[key].astype(mx.float32)).max())
        print(f"dequant probe {base} ({want}-bit): max abs err {err:.5f}", flush=True)
    load_dit(out, verbose=False)
    print(f"LOAD OK -- pack valid. Total {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
