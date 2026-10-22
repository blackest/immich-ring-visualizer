"""Unit tests for h3_loraNG.py -- the qkv row re-order for speed adapters.

Run with: python3 -m unittest tests.test_h3_loraNG -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_loraNG as lora

ROWS = 3 * lora.HEADS * lora.HEAD_DIM
COLS = 2  # bytes per row in the fake tensor


def _write_st(path: Path, tensors: dict, metadata=None) -> None:
    header, blobs, pos = {}, [], 0
    for name, (shape, blob) in tensors.items():
        header[name] = {"dtype": "U8", "shape": shape, "data_offsets": [pos, pos + len(blob)]}
        blobs.append(blob)
        pos += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"".join(blobs))


def _read_st(path: Path):
    data = path.read_bytes()
    hlen = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + hlen])
    body = data[8 + hlen:]
    out = {}
    for name, info in header.items():
        if name != "__metadata__":
            a, b = info["data_offsets"]
            out[name] = body[a:b]
    return header, out


def _blob():
    """Each row i holds bytes (i % 256, i // 256): distinguishable per row."""
    return b"".join(bytes((i % 256, i // 256)) for i in range(ROWS))


class RowReorderTests(unittest.TestCase):
    def test_contiguous_to_per_head(self):
        out = lora.rows_per_head_ng(_blob(), ROWS)
        row = lambda b, i: b[i * COLS:(i + 1) * COLS]
        src = _blob()
        for h in (0, 1, 55):
            for j in range(3):  # j: 0=q 1=k 2=v
                # source (j, h, :) block lands at per-head (h, j, :)
                s = (j * lora.HEADS + h) * lora.HEAD_DIM
                d = (h * 3 + j) * lora.HEAD_DIM
                self.assertEqual(out[d * COLS:(d + lora.HEAD_DIM) * COLS],
                                 src[s * COLS:(s + lora.HEAD_DIM) * COLS])
        self.assertEqual(len(out), len(src))
        self.assertNotEqual(out, src)

    def test_wrong_row_count_rejected(self):
        with self.assertRaises(ValueError):
            lora.rows_per_head_ng(b"\0" * 10, 5)


class BakeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.src = self.dir / "adapter.safetensors"
        self.other = b"\x07" * 16
        _write_st(self.src, {
            "blocks.0.attn.qkv_proj.lora_B.weight": ([ROWS, COLS], _blob()),
            "blocks.0.attn.qkv_proj.lora_A.weight": ([2, 8], self.other),
        }, metadata={"keep": "me"})

    def test_bake_moves_only_qkv_b_rows_and_stamps_metadata(self):
        logs = []
        dst = lora.bake_runner_layout_ng(self.src, logs.append)
        self.assertEqual(dst, self.dir / "runner-layout" / "adapter.safetensors")
        header, tensors = _read_st(dst)
        self.assertEqual(tensors["blocks.0.attn.qkv_proj.lora_B.weight"],
                         lora.rows_per_head_ng(_blob(), ROWS))
        self.assertEqual(tensors["blocks.0.attn.qkv_proj.lora_A.weight"], self.other)
        self.assertEqual(header["__metadata__"]["keep"], "me")
        self.assertEqual(header["__metadata__"][lora.LAYOUT_KEY], lora.LAYOUT_VALUE)
        self.assertEqual(len(logs), 1)

    def test_second_call_reuses_current_copy(self):
        dst = lora.bake_runner_layout_ng(self.src)
        mtime = dst.stat().st_mtime_ns
        logs = []
        self.assertEqual(lora.bake_runner_layout_ng(self.src, logs.append), dst)
        self.assertEqual(dst.stat().st_mtime_ns, mtime)
        self.assertEqual(logs, [])

    def test_changed_source_is_rebaked_and_source_untouched(self):
        dst = lora.bake_runner_layout_ng(self.src)
        before = self.src.read_bytes()
        os.utime(self.src, ns=(1, 1))
        self.assertFalse(lora._current(self.src, dst))
        lora.bake_runner_layout_ng(self.src)
        self.assertTrue(lora._current(self.src, dst))
        self.assertEqual(self.src.read_bytes(), before)

    def test_no_qkv_tensors_leaves_no_partial(self):
        bad = self.dir / "bad.safetensors"
        _write_st(bad, {"other": ([1, 1], b"x")})
        with self.assertRaisesRegex(ValueError, "no attn.qkv_proj"):
            lora.bake_runner_layout_ng(bad)
        self.assertFalse(list((self.dir / "runner-layout").glob("bad*")))

    def test_not_a_safetensors_file(self):
        junk = self.dir / "junk.safetensors"
        junk.write_bytes(b"\xff" * 64)
        with self.assertRaises(ValueError):
            lora.bake_runner_layout_ng(junk)


class FlagProbeTests(unittest.TestCase):
    def test_probe(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Path(d) / "run.py"
            runner.write_text("p.add_argument('--lora-qkv-permute')")
            self.assertTrue(lora.runner_has_flag_ng(runner, "--lora-qkv-permute"))
            self.assertFalse(lora.runner_has_flag_ng(runner, "--video-shift"))
            self.assertFalse(lora.runner_has_flag_ng(Path(d) / "missing.py", "--x"))


if __name__ == "__main__":
    unittest.main()
