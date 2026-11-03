"""AdaLN overlay: take some blocks' AdaLN modulation from a different DiT.

MiniMax-H3's FL2VA and Ref2VA checkpoints are nearly identical except for the per-block
``adaln_proj`` projections. A *hybrid* runs the cleaner FL2VA weights everywhere and takes the
Ref2VA modulation only for a range of the last blocks (community testing found blocks 45-49 the
best trade).

The AdaLN modulation depends on the timestep embedding alone, and the runner precomputes it into a
``ModulationCache`` of per-block tables. So the hybrid is done at the table level: build the
overlay model's tables for the chosen blocks straight from its own checkpoint (time embedder + those
blocks' projections, nothing else is read) and splice them over the base cache. The two DiTs may
use different AdaLN layouts (the pruned FL2VA pack feeds a 64-value timestep curve, the official
Ref2VA pack a 2688-value embedding); the tables have the same shape either way.

Only the overlay's own time embedder and the chosen blocks' projections are loaded, about
``len(blocks) * 0.5 GB`` of bf16 weights, and they are freed again straight after.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

from .config import DiTConfig
from .dit import AdaLayerNormModulation, TimestepEmbedder, timestep_embedding


def parse_block_range(spec: str, num_blocks: int) -> list[int]:
    """``"45-49"``, ``"45-"`` (to the last block), ``"3"`` or ``"0,2,45-49"`` -> sorted block indices."""
    blocks: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s) if hi_s else num_blocks - 1
        else:
            lo = hi = int(part)
        if lo > hi:
            raise ValueError(f"block range {part!r} runs backwards")
        blocks.update(range(lo, hi + 1))
    if not blocks:
        raise ValueError("no blocks selected")
    bad = [b for b in blocks if not 0 <= b < num_blocks]
    if bad:
        raise ValueError(f"blocks {sorted(bad)} are outside 0-{num_blocks - 1}")
    return sorted(blocks)


def _weight_map(overlay_dir: Path) -> dict[str, str]:
    index = overlay_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    single = overlay_dir / "model.safetensors"
    if single.is_file():
        return {k: single.name for k in mx.load(str(single))}
    raise FileNotFoundError(f"{overlay_dir} has neither model.safetensors.index.json nor model.safetensors")


def _load_tensors(overlay_dir: Path, names: list[str], weight_map: dict[str, str]) -> dict[str, mx.array]:
    """Only the named tensors, reading each shard once."""
    by_shard: dict[str, list[str]] = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"{overlay_dir.name} has no tensor {name!r}")
        by_shard.setdefault(weight_map[name], []).append(name)
    out: dict[str, mx.array] = {}
    for shard, wanted in by_shard.items():
        loaded = mx.load(str(overlay_dir / shard))
        for name in wanted:
            out[name] = loaded[name]
        del loaded
    return out


def build_overlay_tables(
    overlay_dir: str | Path,
    blocks: list[int],
    timesteps: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
) -> dict[int, tuple[mx.array, ...]]:
    """``{block: six modulation tables}`` for ``blocks``, from the official-layout DiT in ``overlay_dir``.

    ``timesteps`` is the same ``(T,)`` array ``ModulationCache.build`` receives. The overlay must be
    an official-layout checkpoint (it carries ``time_embedder.*``, not a pruned ``adaln_t_table``) and
    unquantized in the AdaLN projections.
    """
    overlay_dir = Path(overlay_dir)
    config = DiTConfig.from_json(overlay_dir / "config.json")
    if config.adaln_curve_grid is not None:
        raise ValueError(f"{overlay_dir.name} is a pruned DiT; an AdaLN overlay needs the official layout")
    weight_map = _weight_map(overlay_dir)

    embedder_names = [f"time_embedder.{part}.{kind}" for part in ("proj_in", "proj_out") for kind in ("weight", "bias")]
    names = list(embedder_names)
    for b in blocks:
        names += [f"blocks.{b}.adaln_proj.linear.weight", f"blocks.{b}.adaln_proj.linear.bias"]
    tensors = _load_tensors(overlay_dir, names, weight_map)
    for b in blocks:
        if f"blocks.{b}.adaln_proj.linear.scales" in weight_map:
            raise ValueError(f"{overlay_dir.name} has quantized AdaLN projections; the overlay needs bf16")

    embedder = TimestepEmbedder(config)
    embedder.load_weights([(k.split(".", 1)[1], tensors[k]) for k in embedder_names])
    temb = embedder(timestep_embedding(timesteps, config.timestep_input_dim))
    mx.eval(temb)

    tables: dict[int, tuple[mx.array, ...]] = {}
    for b in blocks:
        proj = AdaLayerNormModulation(config)
        proj.load_weights([
            ("linear.weight", tensors[f"blocks.{b}.adaln_proj.linear.weight"]),
            ("linear.bias", tensors[f"blocks.{b}.adaln_proj.linear.bias"]),
        ])
        table = tuple(t.astype(dtype) for t in proj(temb))
        mx.eval(table)
        tables[b] = table
        del proj
    return tables


def splice_overlay(cache, tables: dict[int, tuple[mx.array, ...]]) -> None:
    """Replace ``cache``'s tables for the overlay's blocks, checking each has the base's shapes."""
    for b, table in tables.items():
        base = cache.tables[b]
        if len(base) != len(table) or any(x.shape != y.shape for x, y in zip(base, table)):
            raise ValueError(
                f"block {b}: overlay tables {[t.shape for t in table]} do not match "
                f"the base {[t.shape for t in base]}"
            )
        cache.tables[b] = table
