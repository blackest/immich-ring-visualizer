"""Unit tests for the H3 hybrid "clean look": FL2VA Q8 base + Ref2VA AdaLN on a block range.

h3_ref2vaNG (validation + argv), the h3_engineNG command line (the base `--dit` is the FL2VA Q8, the
overlay is the Ref2VA Q8), h3_jobsNG enqueue and the /generate `adaln_blocks` field. Nothing renders
and no model files are read.

Run with: python3 -m unittest tests.test_h3_hybridNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as eng
import h3_ref2vaNG as ref2va

OVERLAY_RUNNER = "--adaln-overlay REF2VA_DIT\n--adaln-blocks RANGE\n--sigma-subset\n--refmod PATH\n"


class _Captured(Exception):
    pass


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.runner = self.tmp / "generate_staged.py"
        self.runner.write_text(OVERLAY_RUNNER)


class ValidateTests(Base):
    def _v(self, spec, **kw):
        args = dict(model="h3ref", flex=False, turbo=False, v13=False, runner=self.runner)
        args.update(kw)
        return ref2va.validate_adaln_blocks_ng(spec, **args)

    def test_empty_means_no_hybrid_on_any_model(self):
        for model in ("h3", "h3q8", "h3ref"):
            self.assertEqual(self._v("", model=model), "")
            self.assertEqual(self._v(None, model=model, turbo=True), "")

    def test_accepts_ranges_and_strips_spaces(self):
        for spec, want in (("45-49", "45-49"), ("45-", "45-"), (" 3 ", "3"), ("0, 2, 45-49", "0,2,45-49"), ("none", "none"), (" None ", "none")):
            self.assertEqual(self._v(spec), want)

    def test_refusals(self):
        for kw, pat in (({"model": "h3q8"}, "Ref2VA render"), ({"model": "h3"}, "Ref2VA render"),
                        ({"flex": True}, "Flex"), ({"turbo": True}, "Turbo"), ({"v13": True}, "Turbo")):
            with self.assertRaisesRegex(ValueError, pat):
                self._v("45-49", **kw)
        for bad in ("abc", "45-49-50", "-3", "45;49", "4 5-49x"):
            with self.assertRaisesRegex(ValueError, "block range"):
                self._v(bad)
        self.runner.write_text("--refmod PATH\n")
        with self.assertRaisesRegex(ValueError, "no --adaln-overlay"):
            self._v("45-49")

    def test_argv(self):
        self.assertEqual(ref2va.adaln_overlay_argv_ng("/r/q8", ""), [])
        self.assertEqual(ref2va.adaln_overlay_argv_ng("/r/q8", "45-49"),
                         ["--adaln-overlay", "/r/q8", "--adaln-blocks", "45-49"])


class EngineTests(Base):
    def setUp(self):
        super().setUp()
        patches = (
            ("_resolve_h3_python_ng", lambda c: "/py"),
            ("_resolve_h3_runner_ng", lambda c: str(self.runner)),
            ("_resolve_h3_dit_ng", lambda c: "/fl2va-q8" if c.model == "h3q8" else "/ref2va-q8"),
            ("_resolve_h3_compact_root_ng", lambda c: "/compact"),
            ("_resolve_h3_text_config_ng", lambda c: "/text"),
        )
        for name, val in patches:
            p = mock.patch.object(eng, name, val)
            p.start()
            self.addCleanup(p.stop)

    def _cmd(self, model="h3ref", **kw):
        args = dict(prompt="p", image_path=None, duration_s=5.0, output_dir=self.tmp / "o",
                    seed=1, config=eng.H3Config(model=model))
        args.update(kw)
        with mock.patch.object(eng.subprocess, "Popen",
                               side_effect=lambda cmd, **k: (_ for _ in ()).throw(_Captured(cmd))), \
             mock.patch("model_offloadNG.offload_resident_models_ng"):
            with self.assertRaises(_Captured) as cm:
                eng.generate_h3_video_ng(**args)
        return cm.exception.args[0]

    def test_no_hybrid_keeps_the_ref2va_dit_and_adds_no_flag(self):
        cmd = self._cmd()
        self.assertEqual(cmd[cmd.index("--dit") + 1], "/ref2va-q8")
        self.assertNotIn("--adaln-overlay", cmd)

    def test_hybrid_swaps_the_base_and_overlays_ref2va(self):
        cmd = self._cmd(adaln_blocks="45-49")
        self.assertEqual(cmd[cmd.index("--dit") + 1], "/fl2va-q8")
        self.assertEqual(cmd[cmd.index("--adaln-overlay") + 1], "/ref2va-q8")
        self.assertEqual(cmd[cmd.index("--adaln-blocks") + 1], "45-49")

    def test_hybrid_is_refused_off_ref2va(self):
        with self.assertRaisesRegex(ValueError, "Ref2VA render"):
            self._cmd(model="h3q8", adaln_blocks="45-49")

    def test_missing_fl2va_base_is_a_plain_error(self):
        with mock.patch.object(eng, "_resolve_h3_dit_ng", lambda c: "/ref2va-q8" if c.model == "h3ref" else None):
            with self.assertRaisesRegex(FileNotFoundError, "FL2VA Q8 base"):
                self._cmd(adaln_blocks="45-49")


if __name__ == "__main__":
    unittest.main()
