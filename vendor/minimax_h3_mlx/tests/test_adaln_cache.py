"""The AdaLN table cache for ordinary renders (`--adaln-cache-dir`).

The tables depend only on the timestep schedule, the DiT files and the LoRA stack, so they are
stored once and reused. The points that matter:

* a stored table comes back bit-for-bit (bf16 -> float32 -> bf16 is exact),
* the key separates different schedules, DiT files and LoRA stacks, and a sharded DiT directory is
  identified by its files, because a directory's own mtime does not change when a shard inside it is
  replaced in place,
* a cache HIT never touches the per-block AdaLN projections -- that is what lets the ~24 GiB of
  weights stay unread,
* building with `drop_projections=True` gives the same tables as building without it and frees each
  block's projection.

Nothing here loads a model: the DiT is a fake whose blocks hold small matrices.

Run with: python -m unittest tests.test_adaln_cache -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights
from minimax_h3_mlx.draft_cache import DraftCache, path_fingerprint

DIM = 8
BLOCKS = 3


class _Linear:
    def __init__(self, weight):
        self.weight = weight


class _AdaLN:
    """Stands in for a block's adaln_proj: six modulation tensors from the timestep embedding."""

    def __init__(self, weight):
        self.linear = _Linear(weight)

    def __call__(self, temb):
        return tuple((temb @ self.linear.weight) * (i + 1) for i in range(6))


class _Block:
    def __init__(self, seed):
        mx.random.seed(seed)
        self.adaln_proj = _AdaLN(mx.random.normal((DIM, DIM)))


class _FakeDiT:
    def __init__(self):
        self.blocks = [_Block(i) for i in range(BLOCKS)]
        self.final_layer = SimpleNamespace(adaln_proj=SimpleNamespace())

    def embed_timesteps(self, timesteps):
        return mx.broadcast_to(timesteps[:, None], (timesteps.shape[0], DIM)).astype(mx.float32)


class _UntouchableBlock:
    """A block whose projection must never be read: a cache hit may not need it."""

    @property
    def adaln_proj(self):
        raise AssertionError("a cache hit read a block's AdaLN projection")


def _tables(steps=4):
    mx.random.seed(7)
    tables = [tuple(mx.random.normal((steps * 3, DIM)).astype(mx.bfloat16) for _ in range(6))
              for _ in range(BLOCKS)]
    return ModulationCache(tables, mx.arange(steps, dtype=mx.float32) / steps)


class KeyAndFingerprintTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.dit = self.tmp / "q8"
        self.dit.mkdir()
        (self.dit / "config.json").write_text("{}")
        (self.dit / "model-00001.safetensors").write_bytes(b"a" * 100)
        self.table = np.linspace(0.0, 1.0, 5, dtype=np.float32)

    def _key(self, **kw):
        args = dict(timestep_table=self.table, dit=self.dit, lora=None, lora_adaln=None)
        args.update(kw)
        return DraftCache.adaln_key(**args)

    def test_same_inputs_give_the_same_key(self):
        self.assertEqual(self._key(), self._key())

    def test_the_schedule_changes_the_key(self):
        self.assertNotEqual(self._key(), self._key(timestep_table=self.table[:4]))
        shifted = self.table.copy()
        shifted[1] += 1e-3
        self.assertNotEqual(self._key(), self._key(timestep_table=shifted))

    def test_the_lora_stack_changes_the_key(self):
        lora = self.tmp / "l.safetensors"
        lora.write_bytes(b"x" * 10)
        self.assertNotEqual(self._key(), self._key(lora=[(str(lora), 1.0)]))
        self.assertNotEqual(self._key(lora=[(str(lora), 1.0)]), self._key(lora=[(str(lora), 0.5)]))

    def test_a_replaced_shard_changes_the_directory_fingerprint(self):
        before, key_before = path_fingerprint(self.dit), self._key()
        (self.dit / "model-00001.safetensors").write_bytes(b"b" * 250)   # same name, new content
        after, key_after = path_fingerprint(self.dit), self._key()
        self.assertNotEqual(before, after)
        self.assertNotEqual(key_before, key_after)      # so a rebuilt DiT can never hit a stale table

    def test_unrelated_files_do_not_change_it(self):
        before = path_fingerprint(self.dit)
        (self.dit / "README.md").write_text("notes")
        (self.dit / ".DS_Store").write_bytes(b"junk")
        self.assertEqual(before, path_fingerprint(self.dit))

    def test_a_single_file_is_fingerprinted_as_before(self):
        f = self.tmp / "dit.safetensors"
        f.write_bytes(b"z" * 33)
        fp = path_fingerprint(f)
        self.assertEqual((fp["size"], sorted(fp)), (33, ["mtime_ns", "path", "size"]))


class StoreLoadTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = DraftCache(Path(tmp.name) / "adaln-cache", limit=3)

    def test_a_stored_table_comes_back_bit_for_bit(self):
        original = _tables()
        self.cache.store_adaln("k", original, _FakeDiT(), {"blocks": 3})
        loaded, report = self.cache.load_adaln("k", _FakeDiT())
        self.assertEqual(report, {"blocks": 3})
        self.assertTrue(mx.array_equal(loaded.timesteps, original.timesteps))
        for got, want in zip(loaded.tables, original.tables):
            for g, w in zip(got, want):
                self.assertEqual(g.dtype, mx.bfloat16)
                self.assertTrue(mx.array_equal(g, w))

    def test_a_miss_is_none(self):
        self.assertIsNone(self.cache.load_adaln("nothing-here", _FakeDiT()))

    def test_a_hit_never_touches_the_block_projections(self):
        self.cache.store_adaln("k", _tables(), _FakeDiT(), None)
        dit = _FakeDiT()
        dit.blocks = [_UntouchableBlock() for _ in range(BLOCKS)]
        loaded, _ = self.cache.load_adaln("k", dit)     # AssertionError here = the weights were read
        self.assertEqual(len(loaded.tables), BLOCKS)

    def test_a_damaged_entry_is_a_miss_and_is_removed(self):
        self.cache.store_adaln("k", _tables(), _FakeDiT(), None)
        path = self.cache.root / "adaln" / "k.npz"
        path.write_bytes(b"not an npz")
        self.assertIsNone(self.cache.load_adaln("k", _FakeDiT()))
        self.assertFalse(path.exists())

    def test_the_cache_is_bounded(self):
        for i in range(6):
            self.cache.store_adaln(f"k{i}", _tables(), _FakeDiT(), None)
        self.assertLessEqual(len(list((self.cache.root / "adaln").glob("*.npz"))), 3)


class StreamingBuildTests(unittest.TestCase):
    TABLE = mx.array([0.0, 0.25, 0.5, 1.0], dtype=mx.float32)

    def test_dropping_as_it_goes_gives_the_same_tables(self):
        plain = ModulationCache.build(_FakeDiT(), self.TABLE)
        streamed = ModulationCache.build(_FakeDiT(), self.TABLE, drop_projections=True)
        for a, b in zip(plain.tables, streamed.tables):
            for x, y in zip(a, b):
                self.assertTrue(mx.array_equal(x, y))

    def test_each_blocks_projection_is_freed_and_counted(self):
        dit = _FakeDiT()
        expected = sum(b.adaln_proj.linear.weight.nbytes for b in dit.blocks)
        cache = ModulationCache.build(dit, self.TABLE, drop_projections=True)
        self.assertEqual(cache.dropped_bytes, expected)
        self.assertTrue(all(not hasattr(b.adaln_proj.linear, "weight") for b in dit.blocks))
        self.assertEqual(drop_adaln_weights(dit), 0)       # nothing left to drop afterwards

    def test_without_the_flag_nothing_is_dropped(self):
        dit = _FakeDiT()
        cache = ModulationCache.build(dit, self.TABLE)
        self.assertEqual(cache.dropped_bytes, 0)
        self.assertTrue(all(hasattr(b.adaln_proj.linear, "weight") for b in dit.blocks))
        self.assertGreater(drop_adaln_weights(dit), 0)


if __name__ == "__main__":
    unittest.main()
