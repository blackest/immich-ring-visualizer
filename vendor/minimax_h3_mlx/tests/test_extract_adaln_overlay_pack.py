"""scripts/extract_adaln_overlay_pack.py (repo root): cut a small AdaLN overlay pack out of a Ref2VA DiT.

* the pack holds exactly the time embedder and the chosen blocks' projections, byte-for-byte,
* the overlay builds identical tables from the pack and from the full folder (both stay valid sources),
* a block the pack does not hold is named in the error, not silently skipped,
* a quantized, pruned or index-less source is refused, and an existing pack is never overwritten.

The "checkpoint" is the same tiny sharded safetensors folder test_adaln_overlay builds.

Run with: python -m unittest tests.test_extract_adaln_overlay_pack -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.adaln_overlay import build_overlay_tables
from tests.test_adaln_overlay import BLOCKS, _make_overlay

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import extract_adaln_overlay_pack as extractor  # noqa: E402


class ExtractPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "ref2va"
        self.dest = Path(self.tmp.name) / "pack"
        self.tensors = _make_overlay(self.src)
        self.timesteps = mx.array([0.0, 0.25, 0.5, 1.0])

    def tearDown(self):
        self.tmp.cleanup()

    def test_pack_holds_only_the_chosen_tensors_unchanged(self):
        info = extractor.extract(self.src, self.dest, "3-4")
        self.assertEqual(info["blocks"], [3, 4])
        packed = mx.load(str(self.dest / "model.safetensors"))
        want = {k for k in self.tensors if k.startswith("time_embedder.")
                or k.startswith(("blocks.3.adaln_proj.", "blocks.4.adaln_proj."))}
        self.assertEqual(set(packed), want)
        for name, tensor in packed.items():
            self.assertEqual(tensor.dtype, self.tensors[name].dtype, name)
            self.assertTrue(bool(mx.array_equal(tensor, self.tensors[name])), name)
        self.assertEqual(json.loads((self.dest / "config.json").read_text()),
                         json.loads((self.src / "config.json").read_text()))
        self.assertFalse((self.dest / "model.partial.safetensors").exists())

    def test_overlay_tables_are_identical_from_the_pack_and_the_full_folder(self):
        extractor.extract(self.src, self.dest, "3-4")
        from_full = build_overlay_tables(self.src, [3, 4], self.timesteps)
        from_pack = build_overlay_tables(self.dest, [3, 4], self.timesteps)
        for b in (3, 4):
            for x, y in zip(from_full[b], from_pack[b]):
                self.assertEqual(x.dtype, y.dtype)
                self.assertTrue(bool(mx.array_equal(x, y)))

    def test_a_block_the_pack_lacks_is_named(self):
        extractor.extract(self.src, self.dest, "3-4")
        with self.assertRaises(KeyError) as ctx:
            build_overlay_tables(self.dest, [2], self.timesteps)
        self.assertIn("blocks.2.adaln_proj.linear.weight", str(ctx.exception))

    def test_default_blocks_are_the_last_five_of_a_fifty_block_model(self):
        self.assertEqual(extractor.parse_blocks("45-49", 50), [45, 46, 47, 48, 49])
        self.assertEqual(extractor.parse_blocks("48-", 50), [48, 49])
        self.assertEqual(extractor.parse_blocks("0, 2,49", 50), [0, 2, 49])
        for bad in ("49-45", "50", "", "a-b"):
            with self.assertRaises(ValueError, msg=bad):
                extractor.parse_blocks(bad, 50)

    def test_blocks_outside_the_source_are_refused(self):
        with self.assertRaises(ValueError):
            extractor.extract(self.src, self.dest, f"0-{BLOCKS}")
        self.assertFalse((self.dest / "model.safetensors").exists())

    def test_quantized_and_pruned_sources_are_refused(self):
        quant = Path(self.tmp.name) / "quant"
        _make_overlay(quant, quantized=True)
        with self.assertRaises(ValueError) as ctx:
            extractor.extract(quant, self.dest, "3-4")
        self.assertIn("quantized", str(ctx.exception))
        pruned = Path(self.tmp.name) / "pruned"
        _make_overlay(pruned, pruned=True)
        with self.assertRaises(ValueError) as ctx:
            extractor.extract(pruned, self.dest, "3-4")
        self.assertIn("pruned", str(ctx.exception))
        self.assertFalse((self.dest / "model.safetensors").exists())

    def test_a_folder_without_weights_is_refused(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        (empty / "config.json").write_text(json.dumps({"num_layers": 5}))
        with self.assertRaises(FileNotFoundError):
            extractor.extract(empty, self.dest, "3-4")

    def test_licence_files_travel_with_the_pack(self):
        (self.src / "LICENSE").write_text("MiniMax licence text")
        (self.src / "NOTICE.txt").write_text("notice")
        extractor.extract(self.src, self.dest, "3-4")
        self.assertEqual((self.dest / "LICENSE").read_text(), "MiniMax licence text")
        self.assertEqual((self.dest / "NOTICE.txt").read_text(), "notice")

    def test_an_existing_pack_is_not_overwritten(self):
        extractor.extract(self.src, self.dest, "3-4")
        before = (self.dest / "model.safetensors").stat().st_mtime_ns
        self.assertEqual(extractor.main([str(self.src), str(self.dest), "--blocks", "3-4"]), 2)
        self.assertEqual((self.dest / "model.safetensors").stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()
