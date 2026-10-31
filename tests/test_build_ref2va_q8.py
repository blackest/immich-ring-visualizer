"""Unit tests for scripts/build_ref2va_q8.py's pack verifier (the build itself needs the 66 GB weights).

Run with: python3 -m unittest tests.test_build_ref2va_q8 -v
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_ref2va_q8", Path(__file__).resolve().parent.parent / "scripts" / "build_ref2va_q8.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def _pack(root, shards=2, **quant):
    names = [f"model-{i:05d}-of-{shards:05d}.safetensors" for i in range(1, shards + 1)]
    for n in names:
        (root / n).write_bytes(b"x")
    (root / "config.json").write_text("{}")
    meta = {"bits": 8, "group_size": 64, "quantize_adaln": False, "quantized_layers": {"8": 208}, **quant}
    (root / "quant_config.json").write_text(json.dumps(meta))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}}))


class Verify(unittest.TestCase):
    def check(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            _pack(Path(d), **kw)
            return build.verify_ng(Path(d))

    def test_a_matching_pack_has_no_problems(self):
        self.assertEqual(self.check(), [])

    def test_wrong_width_adaln_or_layer_count_is_reported(self):
        self.assertTrue(self.check(bits=4))
        self.assertTrue(self.check(quantize_adaln=True))
        self.assertTrue(self.check(quantized_layers={"8": 100}))

    def test_missing_shard_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            _pack(Path(d))
            (Path(d) / "model-00001-of-00002.safetensors").unlink()
            self.assertTrue(any("missing shard" in p for p in build.verify_ng(Path(d))))
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(build.verify_ng(Path(d)))


if __name__ == "__main__":
    unittest.main()
