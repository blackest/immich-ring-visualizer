"""Unit tests for the H3 hybrid "clean look": FL2VA Q8 base + Ref2VA AdaLN on a block range.

h3_ref2vaNG (validation + argv), the h3_engineNG command line (the base `--dit` is the FL2VA Q8, the
overlay is the Ref2VA Q8), h3_jobsNG enqueue and the /generate `adaln_blocks` field. Nothing renders
and no model files are read.

Run with: python3 -m unittest tests.test_h3_hybridNG -v
"""

import json
import os
import struct
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


def _make_pack(root: Path, blocks=range(45, 50), time_embedder=True, layers=50) -> Path:
    """A small AdaLN overlay pack: config.json + a model.safetensors holding only a header (and 2 bytes)
    for each tensor, which is all the up-front check reads."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"num_layers": layers}))
    names = [f"time_embedder.{p}.{k}" for p in ("proj_in", "proj_out") for k in ("weight", "bias")] if time_embedder else []
    for b in blocks:
        names += [f"blocks.{b}.adaln_proj.linear.weight", f"blocks.{b}.adaln_proj.linear.bias"]
    head = {n: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, n in enumerate(names)}
    head["__metadata__"] = {"format": "mlx"}
    raw = json.dumps(head).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0" * len(names))
    return root


class OverlaySourceSwitchTests(Base):
    def setUp(self):
        super().setUp()
        self.pack = str(_make_pack(self.tmp / "pack"))

    def test_mode_defaults_to_auto_and_reads_the_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ref2va.ADALN_SOURCE_ENV, None)
            self.assertEqual(ref2va.adaln_source_mode_ng(), "auto")
        for raw, want in (("", "auto"), (" Full ", "full"), ("PACK", "pack"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: raw}):
                self.assertEqual(ref2va.adaln_source_mode_ng(), want)
        with mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: "both"}):
            with self.assertRaisesRegex(ValueError, "must be one of auto, full, pack"):
                ref2va.adaln_source_mode_ng()

    def test_auto_prefers_the_full_model_and_falls_back_to_the_pack(self):
        pick = ref2va.pick_overlay_source_ng
        self.assertEqual(pick("auto", "/full", self.pack, "45-49"), "/full")
        self.assertEqual(pick("auto", None, self.pack, "45-49"), self.pack)

    def test_full_and_pack_are_each_exactly_what_they_say(self):
        pick = ref2va.pick_overlay_source_ng
        self.assertEqual(pick("full", "/full", self.pack, "45-49"), "/full")
        self.assertEqual(pick("pack", "/full", self.pack, "45-49"), self.pack)
        with self.assertRaisesRegex(FileNotFoundError, "full Ref2VA model.*not on this machine"):
            pick("full", None, self.pack, "45-49")
        with self.assertRaisesRegex(FileNotFoundError, "neither was found"):
            pick("pack", "/full", None, "45-49")
        with self.assertRaisesRegex(FileNotFoundError, "neither was found"):
            pick("auto", None, None, "45-49")

    def test_a_pack_that_lacks_a_block_is_refused_before_a_render_starts(self):
        pick = ref2va.pick_overlay_source_ng
        with self.assertRaisesRegex(ValueError, "does not hold block 44"):
            pick("pack", "/full", self.pack, "44-49")
        with self.assertRaisesRegex(ValueError, "does not hold block 40, 41"):
            pick("auto", None, str(_make_pack(self.tmp / "p2", blocks=range(42, 50))), "40-45")
        self.assertEqual(pick("pack", None, self.pack, "47"), self.pack)
        self.assertEqual(pick("pack", None, self.pack, "46,49"), self.pack)
        self.assertEqual(pick("pack", None, self.pack, "48-"), self.pack)   # open end = to the last block
        self.assertEqual(pick("pack", None, self.pack, "none"), self.pack)  # a control run needs no block

    def test_a_pack_without_the_time_embedder_is_refused(self):
        bad = str(_make_pack(self.tmp / "p3", time_embedder=False))
        with self.assertRaisesRegex(ValueError, "time embedder"):
            ref2va.pick_overlay_source_ng("pack", None, bad, "45-49")

    def test_pack_discovery_order(self):
        root = self.tmp / "models"
        env_pack = _make_pack(self.tmp / "envpack")
        with mock.patch.dict(os.environ, {ref2va.ADALN_PACK_ENV: str(env_pack)}):
            _make_pack(root / ref2va.ADALN_PACK_DIRNAME)
            self.assertEqual(ref2va.resolve_adaln_pack_ng(root), str(env_pack))
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(ref2va.ADALN_PACK_ENV, None)
            self.assertEqual(ref2va.resolve_adaln_pack_ng(root), str(root / ref2va.ADALN_PACK_DIRNAME))
            (root / ref2va.ADALN_PACK_DIRNAME / "model.safetensors").unlink()
            with mock.patch.object(ref2va.Path, "home", return_value=self.tmp / "nohome"):
                self.assertIsNone(ref2va.resolve_adaln_pack_ng(root))


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

    def _pack_only(self, mode="auto", pack="/pack"):
        """Full Ref2VA absent; the small pack (or not) present; the switch set to `mode`."""
        for target, val in (
            (eng, ("_resolve_h3_dit_ng", lambda c: None if c.model == "h3ref" else "/fl2va-q8")),
            (ref2va, ("resolve_adaln_pack_ng", lambda root: pack)),
            (ref2va, ("pack_missing_blocks_ng", lambda p, spec: [])),
        ):
            p = mock.patch.object(target, val[0], val[1])
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: mode})
        env.start()
        self.addCleanup(env.stop)

    def test_hybrid_runs_on_the_pack_when_the_full_model_is_absent(self):
        self._pack_only()
        cmd = self._cmd(adaln_blocks="45-49")
        self.assertEqual(cmd[cmd.index("--dit") + 1], "/fl2va-q8")
        self.assertEqual(cmd[cmd.index("--adaln-overlay") + 1], "/pack")

    def test_the_pack_switch_wins_even_when_the_full_model_is_present(self):
        with mock.patch.object(ref2va, "resolve_adaln_pack_ng", lambda root: "/pack"), \
             mock.patch.object(ref2va, "pack_missing_blocks_ng", lambda p, spec: []), \
             mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: "pack"}):
            cmd = self._cmd(adaln_blocks="45-49")
        self.assertEqual(cmd[cmd.index("--adaln-overlay") + 1], "/pack")

    def test_full_switch_with_no_full_model_is_a_plain_error(self):
        self._pack_only(mode="full")
        with self.assertRaisesRegex(FileNotFoundError, "full Ref2VA model"):
            self._cmd(adaln_blocks="45-49")

    def test_no_source_at_all_is_a_plain_error(self):
        self._pack_only(pack=None)
        with self.assertRaisesRegex(FileNotFoundError, "neither was found"):
            self._cmd(adaln_blocks="45-49")

    def test_a_plain_ref2va_render_still_needs_the_full_model_even_with_a_pack(self):
        self._pack_only()
        with self.assertRaisesRegex(FileNotFoundError, "DiT weights not found"):
            self._cmd()

    def test_health_reports_the_switch_and_the_pack(self):
        with mock.patch.object(ref2va, "resolve_adaln_pack_ng", lambda root: "/pack"), \
             mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: "pack"}):
            h = eng.h3_health_ng("h3ref")
        self.assertEqual(h["adaln_source"], "pack")
        self.assertTrue(h["adaln_pack_ok"])
        self.assertEqual(h["adaln_pack_path"], "/pack")
        with mock.patch.dict(os.environ, {ref2va.ADALN_SOURCE_ENV: "both"}):
            self.assertTrue(eng.h3_health_ng("h3ref")["adaln_source"].startswith("invalid:"))


if __name__ == "__main__":
    unittest.main()
