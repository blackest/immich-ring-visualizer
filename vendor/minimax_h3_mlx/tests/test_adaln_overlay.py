"""The AdaLN overlay (`--adaln-overlay`): some blocks' modulation tables come from another DiT.

* the overlay tables equal an independent computation (sinusoid -> time embedder -> silu -> block
  projection, split into the six modulation tensors) from the same tiny checkpoint,
* only the chosen blocks are replaced in the base cache; the others are untouched,
* a block range parses as written and rejects out-of-range or backwards input,
* a pruned or quantized overlay and a shape mismatch are refused,
* the cache key changes with the overlay and its blocks, and is unchanged when no overlay is set.

Nothing here loads a real model: the "checkpoint" is a tiny sharded safetensors folder.

Run with: python -m unittest tests.test_adaln_overlay -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.adaln_overlay import build_overlay_tables, parse_block_range, splice_overlay
from minimax_h3_mlx.draft_cache import DraftCache

HIDDEN = 4
EMBED = 6          # time_embed_dim
TS_IN = 8          # timestep_input_dim
TS_HID = 5         # time_embed_hidden_size
BLOCKS = 5
CONFIG = {
    "hidden_size": HIDDEN, "num_layers": BLOCKS, "time_embed_dim": EMBED,
    "timestep_input_dim": TS_IN, "time_embed_hidden_size": TS_HID,
    "adaln_out_features": 18 * HIDDEN, "final_adaln_out_features": 2 * HIDDEN,
}


def _rand(*shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def _make_overlay(root: Path, quantized: bool = False, pruned: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG)
    if pruned:
        cfg["adaln_curve_grid"] = 16
    (root / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "time_embedder.proj_in.weight": _rand(TS_HID, TS_IN, seed=1),
        "time_embedder.proj_in.bias": _rand(TS_HID, seed=2),
        "time_embedder.proj_out.weight": _rand(EMBED, TS_HID, seed=3),
        "time_embedder.proj_out.bias": _rand(EMBED, seed=4),
    }
    for b in range(BLOCKS):
        tensors[f"blocks.{b}.adaln_proj.linear.weight"] = _rand(18 * HIDDEN, EMBED, seed=10 + b).astype(mx.bfloat16)
        tensors[f"blocks.{b}.adaln_proj.linear.bias"] = _rand(18 * HIDDEN, seed=30 + b).astype(mx.bfloat16)
        if quantized:
            tensors[f"blocks.{b}.adaln_proj.linear.scales"] = _rand(18 * HIDDEN, 1, seed=50 + b)
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-1.safetensors": names[:half], "model-2.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(root / shard), {k: tensors[k] for k in keys})
        weight_map.update({k: shard for k in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tensors


def _reference_tables(tensors, block, timesteps):
    """The same maths in numpy, independent of the module under test."""
    t = np.asarray(timesteps, dtype=np.float32)
    half = TS_IN // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    emb = t[:, None] * freqs[None, :]
    sin = np.concatenate([np.cos(emb), np.sin(emb)], axis=-1)

    def f32(a):
        return np.asarray(a.astype(mx.float32))

    def lin(x, w, b):
        return x @ f32(w).T + f32(b)

    h = lin(sin, tensors["time_embedder.proj_in.weight"], tensors["time_embedder.proj_in.bias"])
    h = h / (1.0 + np.exp(-h))                       # silu
    temb = lin(h, tensors["time_embedder.proj_out.weight"], tensors["time_embedder.proj_out.bias"])
    act = temb / (1.0 + np.exp(-temb))               # silu on the way into the block projection
    act = np.asarray(mx.array(act).astype(mx.bfloat16).astype(mx.float32))
    out = lin(act, tensors[f"blocks.{block}.adaln_proj.linear.weight"],
              tensors[f"blocks.{block}.adaln_proj.linear.bias"])
    out = out.reshape(-1, 6 * HIDDEN)
    return [out[:, i * HIDDEN:(i + 1) * HIDDEN] for i in range(6)]


class ParseBlockRange(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_block_range("45-49", 50), [45, 46, 47, 48, 49])
        self.assertEqual(parse_block_range("45-", 50), [45, 46, 47, 48, 49])
        self.assertEqual(parse_block_range("3", 50), [3])
        self.assertEqual(parse_block_range("0, 2,48-49", 50), [0, 2, 48, 49])
        self.assertEqual(parse_block_range("none", 50), [])
        self.assertEqual(parse_block_range(" None ", 50), [])

    def test_rejects_bad_input(self):
        for spec in ("49-45", "50", "-3", "", "a-b"):
            with self.assertRaises(ValueError, msg=spec):
                parse_block_range(spec, 50)


class BuildAndSplice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ref2va"
        self.tensors = _make_overlay(self.root)
        self.timesteps = mx.array([0.0, 0.25, 0.5, 1.0])

    def tearDown(self):
        self.tmp.cleanup()

    def test_tables_match_independent_computation(self):
        tables = build_overlay_tables(self.root, [3, 4], self.timesteps, dtype=mx.float32)
        self.assertEqual(sorted(tables), [3, 4])
        for b in (3, 4):
            expected = _reference_tables(self.tensors, b, self.timesteps)
            for got, want in zip(tables[b], expected):
                self.assertEqual(got.shape, (12, HIDDEN))   # 3 modalities per timestep
                np.testing.assert_allclose(np.asarray(got), want, rtol=2e-2, atol=2e-2)

    def test_only_chosen_blocks_are_replaced(self):
        base = SimpleNamespace(tables=[tuple(mx.full((12, HIDDEN), float(b)) for _ in range(6)) for b in range(BLOCKS)])
        before = [tuple(np.asarray(t) for t in table) for table in base.tables]
        tables = build_overlay_tables(self.root, [3, 4], self.timesteps)
        splice_overlay(base, tables)
        for b in range(3):
            for got, want in zip(base.tables[b], before[b]):
                np.testing.assert_array_equal(np.asarray(got.astype(mx.float32)), want)
        for b in (3, 4):
            self.assertFalse(np.allclose(np.asarray(base.tables[b][0].astype(mx.float32)), before[b][0]))

    def test_shape_mismatch_refused(self):
        base = SimpleNamespace(tables=[tuple(mx.zeros((9, HIDDEN)) for _ in range(6)) for _ in range(BLOCKS)])
        tables = build_overlay_tables(self.root, [4], self.timesteps)
        with self.assertRaises(ValueError):
            splice_overlay(base, tables)

    def test_unknown_block_and_pruned_and_quantized_refused(self):
        with self.assertRaises(KeyError):
            build_overlay_tables(self.root, [BLOCKS + 3], self.timesteps)
        pruned = Path(self.tmp.name) / "pruned"
        _make_overlay(pruned, pruned=True)
        with self.assertRaises(ValueError):
            build_overlay_tables(pruned, [4], self.timesteps)
        quant = Path(self.tmp.name) / "quant"
        _make_overlay(quant, quantized=True)
        with self.assertRaises(ValueError):
            build_overlay_tables(quant, [4], self.timesteps)


class CacheKey(unittest.TestCase):
    def test_overlay_changes_the_key_and_no_overlay_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            dit = Path(tmp) / "dit"
            dit.mkdir()
            (dit / "model.safetensors").write_bytes(b"x")
            ov = Path(tmp) / "ov"
            _make_overlay(ov)
            table = np.arange(4, dtype=np.float32)
            plain = DraftCache.adaln_key(timestep_table=table, dit=dit, lora=None, lora_adaln=None)
            explicit_none = DraftCache.adaln_key(timestep_table=table, dit=dit, lora=None, lora_adaln=None, overlay=None)
            a = DraftCache.adaln_key(timestep_table=table, dit=dit, lora=None, lora_adaln=None, overlay=(ov, [3, 4]))
            b = DraftCache.adaln_key(timestep_table=table, dit=dit, lora=None, lora_adaln=None, overlay=(ov, [4]))
            self.assertEqual(plain, explicit_none)
            self.assertNotEqual(plain, a)
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
