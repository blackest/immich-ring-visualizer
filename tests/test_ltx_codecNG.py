"""Unit tests for ltx_codecNG.py's idempotent library patch + preset env.

Run with: python3 -m unittest tests.test_ltx_codecNG -v
"""
import tempfile
import unittest
from pathlib import Path

import ltx_codecNG as C

UPSTREAM = f"def f(cmd, output_path):\n        {C.PATCH_OLD}\n"


class CodecPatchTests(unittest.TestCase):
    def _file(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "video_vae.py"
        p.write_text(text)
        return p

    def test_applies_then_is_idempotent(self):
        p = self._file(UPSTREAM)
        self.assertEqual(C.apply_codec_patch_ng(p), C.APPLIED)
        patched = p.read_text()
        compile(patched, "video_vae.py", "exec")
        self.assertIn("LTX_OUTPUT_CRF", patched)
        self.assertIn("+faststart", patched)
        self.assertEqual(C.apply_codec_patch_ng(p), C.ALREADY)
        self.assertEqual(p.read_text(), patched)

    def test_drift_when_upstream_line_changed(self):
        p = self._file("cmd.extend(['something else'])\n")
        self.assertEqual(C.apply_codec_patch_ng(p), C.DRIFT)

    def test_missing_file(self):
        self.assertEqual(C.apply_codec_patch_ng(Path("/nonexistent/x.py")), C.MISSING)

    def test_preset_env(self):
        self.assertEqual(C.codec_env_ng("web"),
                         {"LTX_OUTPUT_PIX_FMT": "yuv420p", "LTX_OUTPUT_CRF": "23"})
        self.assertEqual(C.codec_env_ng("archival")["LTX_OUTPUT_CRF"], "0")
        self.assertEqual(C.codec_env_ng("bogus"), C.codec_env_ng("web"))


if __name__ == "__main__":
    unittest.main()
