"""Unit tests for the pure helpers in scripts/prune_ref2va_adaln.py (bf16 decode, the hand-written safetensors writer).
The prune itself needs the 66 GB weights and MLX.

Run with: python3 -m unittest tests.test_prune_ref2va_adalnNG -v
"""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "prune_ref2va_adaln", Path(__file__).resolve().parent.parent / "scripts" / "prune_ref2va_adaln.py")
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def _bf16_bytes(values):
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


class Helpers(unittest.TestCase):
    def test_bf16_decodes_exactly_representable_values(self):
        raw = _bf16_bytes([1.0, -2.5, 0.5, 3.0])
        np.testing.assert_array_equal(prune.bf16_to_f32_ng(raw, (2, 2)), [[1.0, -2.5], [0.5, 3.0]])

    def test_writer_round_trips_through_the_reader_and_aligns_the_header(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = _bf16_bytes([1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.safetensors"
            prune.write_safetensors_ng(path, [("a", "F32", [2, 3], a.nbytes, lambda: a.tobytes()),
                                               ("b", "BF16", [2], len(b), lambda: b)])
            header, start = prune.read_header_ng(path)
            self.assertEqual(start % 8, 0)
            self.assertEqual(header["a"]["data_offsets"], [0, 24])
            self.assertEqual(header["b"]["data_offsets"], [24, 28])
            np.testing.assert_array_equal(prune._read_f32(path, header["a"], start), a)
            np.testing.assert_array_equal(prune._read_f32(path, header["b"], start), [1.0, 2.0])

    def test_writer_refuses_a_tensor_whose_size_disagrees_with_the_header(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                prune.write_safetensors_ng(Path(d) / "t", [("a", "F32", [2], 8, lambda: b"xx")])


if __name__ == "__main__":
    unittest.main()
