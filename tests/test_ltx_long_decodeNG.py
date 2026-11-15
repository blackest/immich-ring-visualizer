"""LTX's opt-in chunked decode for long clips (LTX_LONG_DECODE_TILES), run in the LTX env's own python. Nothing is rendered.

Run with: python3 -m unittest tests.test_ltx_long_decodeNG -v
"""

import json
import os
import subprocess
import unittest
from pathlib import Path

ENV_PY = Path(__file__).resolve().parent.parent / "vendor" / "ltx_2_mlx" / "env" / "bin" / "python"
PROBE = ("import json,os;from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling as c;"
         "t=c((1,128,%d,18,32));"
         "print(json.dumps(None if t is None else [t.temporal_config.tile_size_in_frames,t.temporal_config.tile_overlap_in_frames]))")


@unittest.skipUnless(ENV_PY.is_file(), "the LTX env is not built")
class LongDecodeTests(unittest.TestCase):
    def probe(self, f_lat, value=None):
        env = {k: v for k, v in os.environ.items() if k != "LTX_LONG_DECODE_TILES"}
        if value is not None:
            env["LTX_LONG_DECODE_TILES"] = value
        out = subprocess.run([str(ENV_PY), "-c", PROBE % f_lat], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_off_by_default(self):
        self.assertIsNone(self.probe(31))

    def test_on_chunks_a_long_clip_only(self):
        self.assertEqual(self.probe(31, "on"), [128, 32])      # 241 frames
        self.assertIsNone(self.probe(16, "on"))                 # 121 frames: never chunked

    def test_custom_and_bad_values(self):
        self.assertEqual(self.probe(31, "96/16"), [96, 16])
        self.assertEqual(self.probe(31, "garbage"), [128, 32])  # unreadable: the measured default
        self.assertIsNone(self.probe(31, "off"))
        self.assertIsNone(self.probe(31, "100/32"))             # chunk not a multiple of 8: refused


if __name__ == "__main__":
    unittest.main()
