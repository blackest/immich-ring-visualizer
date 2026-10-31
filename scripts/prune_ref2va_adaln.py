"""Prune the official Ref2VA transformer to the single-file "pruned AdaLN" format, with bounded memory.

    RUNNER/.venv/bin/python scripts/prune_ref2va_adaln.py --runner RUNNER \\
        --src /path/to/MiniMax-H3/Ref2VA/transformer --out /path/to/Ref2VA-pruned_bf16.safetensors
    RUNNER/.venv/bin/python RUNNER/scripts/quantize_stream.py --src <that file> --out ~/models/h3-ref2va/q8-pruned

What this is. Every block's AdaLN projection reads `silu(time_embedder(sinusoid(t)))`, a 2688-dim vector that depends
only on the scalar timestep t. That curve is almost one-dimensional: on Ref2VA's own embedder a 16-dim basis already
holds all but 7e-14 of its energy. The pruned format (the one DeepBeepMeep published for FL2VA, which the runner's
`load_dit` reads when `adaln_curve_grid` is set) stores a sampled `adaln_t_table` of shape (grid, rank) and re-projects
each of the 51 AdaLN weights from (out, 2688) onto the rank-dim basis, dropping the `time_embedder` weights. The
projected AdaLN outputs differ from the originals by about 1.7e-6 relative at rank 64, which is float32 noise.

What it does: SVD of the curve sampled at `grid` points in [0, 1] (the runner interpolates linearly between them) ->
basis V (2688 x rank) and table T = S V; every `*.adaln_proj.linear.weight` W becomes W V in float32; `time_embedder.*`
is dropped; every other tensor is copied byte for byte. It writes the safetensors file by hand, one tensor at a time,
so peak memory is a few GB (the largest tensor is a 1 GB float32 projection), not the 41 GB output.

The pruned file is then quantized by the runner's own `scripts/quantize_stream.py` (also bounded memory), which gives a
pack of about 22 GB that needs no AdaLN weights, no AdaLN cache and works for any schedule.

Licence: the result is a MiniMax H3 Model Derivative (MiniMax H3 Community License; the US, EU, UK and South Korea are
excluded territories). Build your own from the official download; do not re-host it without the licence text, a
"modified" notice and "Powered by MiniMax H3".
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

CHUNK = 256 * 1024**2
ADALN_SUFFIX = ".adaln_proj.linear.weight"
EMBEDDER_PREFIX = "time_embedder."
TABLE_KEY = "adaln_t_table"
EMBEDDER_KEYS = ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")


def read_header_ng(path: Path) -> tuple:
    """(header dict without __metadata__, byte offset where the data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def bf16_to_f32_ng(raw: bytes, shape) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _read_f32(path: Path, meta: dict, data_start: int) -> np.ndarray:
    a, b = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + a)
        raw = f.read(b - a)
    if meta["dtype"] == "BF16":
        return bf16_to_f32_ng(raw, meta["shape"])
    if meta["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"]).copy()
    raise ValueError(f"unexpected dtype {meta['dtype']} for a tensor read as float32")


def curve_basis_ng(embedder: dict, grid: int, rank: int):
    """(S, V, T): silu(temb) on the grid (grid x 2688, float32), the basis (2688 x rank) and the table S V."""
    import mlx.core as mx
    from minimax_h3_mlx.dit import timestep_embedding

    w = {k: mx.array(embedder[k]) for k in EMBEDDER_KEYS}
    t = mx.linspace(0, 1, grid)
    h = timestep_embedding(t, 256) @ w["proj_in.weight"].T + w["proj_in.bias"]
    h = h * mx.sigmoid(h)
    e = h @ w["proj_out.weight"].T + w["proj_out.bias"]
    s = e * mx.sigmoid(e)
    mx.eval(s)
    s = np.array(s)
    _u, _sv, vt = np.linalg.svd(s.astype(np.float64), full_matrices=False)
    v = vt[:rank].T.astype(np.float32)
    return s, v, (s @ v).astype(np.float32)


def write_safetensors_ng(out: Path, entries: list) -> None:
    """entries: [(key, dtype, shape, nbytes, producer)] where producer() returns the tensor's bytes. Writes the header
    first (offsets are known from the sizes), then each tensor in turn."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for key, dtype, shape, nbytes, _ in entries:
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for key, _dtype, _shape, nbytes, producer in entries:
            data = producer()
            if len(data) != nbytes:
                raise RuntimeError(f"{key}: wrote {len(data)} bytes, header says {nbytes}")
            f.write(data)


def prune_ng(src: Path, out: Path, grid: int, rank: int) -> dict:
    shards = sorted(src.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no model-*.safetensors in {src}")
    where = {}  # key -> (shard path, meta, data_start)
    for shard in shards:
        header, start = read_header_ng(shard)
        for key, meta in header.items():
            where[key] = (shard, meta, start)
    missing = [k for k in EMBEDDER_KEYS if EMBEDDER_PREFIX + k not in where]
    if missing:
        raise SystemExit(f"{src} has no time_embedder weights ({missing}); is it already pruned?")

    embedder = {k: _read_f32(*[where[EMBEDDER_PREFIX + k][i] for i in (0, 1, 2)]) for k in EMBEDDER_KEYS}
    s, v, table = curve_basis_ng(embedder, grid, rank)
    print(f"curve basis: grid {grid}, rank {rank}", flush=True)

    entries, n_proj = [], 0
    for key in sorted(where):
        if key.startswith(EMBEDDER_PREFIX):
            continue
        shard, meta, start = where[key]
        if key.endswith(ADALN_SUFFIX):
            out_f = meta["shape"][0]
            n_proj += 1

            def project(shard=shard, meta=meta, start=start):
                return (_read_f32(shard, meta, start) @ v).astype(np.float32).tobytes()

            entries.append((key, "F32", [out_f, rank], out_f * rank * 4, project))
        else:
            a, b = meta["data_offsets"]

            def copy(shard=shard, start=start, a=a, b=b):
                with open(shard, "rb") as f:
                    f.seek(start + a)
                    return f.read(b - a)

            entries.append((key, meta["dtype"], meta["shape"], b - a, copy))
    entries.append((TABLE_KEY, "F32", [grid, rank], grid * rank * 4, lambda: table.tobytes()))
    write_safetensors_ng(out, entries)
    return {"projected": n_proj, "tensors": len(entries), "bytes": out.stat().st_size, "S": s, "V": v, "T": table}


def self_check_ng(out: Path, src: Path, prune: dict, grid: int) -> list:
    """Numbers, not a verdict: relative error of the pruned path against the original AdaLN on block 0 and the final
    layer, at the grid nodes and halfway between nodes (where the runner interpolates)."""
    s, t_tab = prune["S"], prune["T"]
    mids = (np.arange(grid - 1) + 0.5) / (grid - 1)
    import mlx.core as mx
    from minimax_h3_mlx.dit import timestep_embedding

    shards = {k: (p, m, st) for p in sorted(src.glob("model-*.safetensors")) for st in [read_header_ng(p)[1]]
              for k, m in read_header_ng(p)[0].items()}
    emb = {k: mx.array(_read_f32(*shards[EMBEDDER_PREFIX + k])) for k in EMBEDDER_KEYS}
    h = timestep_embedding(mx.array(mids.astype(np.float32)), 256) @ emb["proj_in.weight"].T + emb["proj_in.bias"]
    h = h * mx.sigmoid(h)
    e = h @ emb["proj_out.weight"].T + emb["proj_out.bias"]
    s_mid = np.array(e * mx.sigmoid(e))
    t_mid = 0.5 * (t_tab[:-1] + t_tab[1:])  # linear interpolation halfway between nodes
    pruned_header, pruned_start = read_header_ng(out)
    lines = []
    for name in ("blocks.0", "final_layer"):
        key = name + ADALN_SUFFIX
        w = _read_f32(*shards[key])
        wp = _read_f32(out, pruned_header[key], pruned_start)
        for label, full_in, pruned_in in (("nodes", s, t_tab), ("midpoints", s_mid, t_mid)):
            full, approx = w @ full_in.T, wp @ pruned_in.T
            lines.append(f"{name:12s} {label:9s} rel-fro {np.linalg.norm(full - approx) / np.linalg.norm(full):.2e}"
                         f"  max-rel {np.abs(full - approx).max() / np.abs(full).max():.2e}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", type=Path, required=True, help="the minimax-h3-mlx checkout (for timestep_embedding)")
    ap.add_argument("--src", type=Path, required=True, help="the official Ref2VA/transformer folder")
    ap.add_argument("--out", type=Path, required=True, help="the pruned single .safetensors file to write")
    ap.add_argument("--grid", type=int, default=1001)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--no-check", action="store_true", help="skip the accuracy numbers at the end")
    a = ap.parse_args()
    sys.path.insert(0, str(a.runner.expanduser()))
    src, out = a.src.expanduser(), a.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = prune_ng(src, out, a.grid, a.rank)
    print(f"wrote {out} ({result['bytes'] / 1e9:.1f} GB, {result['tensors']} tensors, "
          f"{result['projected']} AdaLN projections)", flush=True)
    if not a.no_check:
        print("\n".join(self_check_ng(out, src, result, a.grid)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
