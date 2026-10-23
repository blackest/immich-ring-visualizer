"""Unit tests for hidream_engineNG.py -- NG twin of tests/test_hidream_engine.py,
plus the pure helpers, the argv built for the subprocess, the watchdog and
the color-match step. No real HiDream, venv or model is touched: Popen is
faked by a stand-in that "writes" the PNGs the real script would have.

Run with: python3 -m unittest tests.test_hidream_engineNG -v
"""

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

import hidream_engineNG as hd


def _png_bytes(w, h, color=(120, 60, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _output_paths(cmd):
    i = cmd.index("--output") + 1
    paths = []
    while i < len(cmd) and not str(cmd[i]).startswith("--"):
        paths.append(cmd[i])
        i += 1
    return paths


class _Proc:
    def __init__(self, rc=0, lines=(), delay=0.0):
        self.rc, self.pid, self.killed = rc, 99999, False
        self._lines, self._delay = list(lines), delay
        self.stdout = self._gen()

    def _gen(self):
        if self._delay:
            time.sleep(self._delay)
        for line in self._lines:
            yield line

    def wait(self):
        return self.rc

    def kill(self):
        self.killed = True


class HelperTests(unittest.TestCase):
    def test_snap_picks_closest_trained_aspect(self):
        self.assertEqual(hd._snap_to_trained_resolution_ng(1000, 1000), (2048, 2048))
        self.assertEqual(hd._snap_to_trained_resolution_ng(1920, 1080), (2560, 1440))
        self.assertEqual(hd._snap_to_trained_resolution_ng(1080, 1920), (1440, 2560))
        self.assertEqual(hd._snap_to_trained_resolution_ng(3000, 1200), (3104, 1312))

    def test_patch_align_floors_to_multiple_with_minimum(self):
        self.assertEqual(hd._patch_align_ng(100), 96)
        self.assertEqual(hd._patch_align_ng(64), 64)
        self.assertEqual(hd._patch_align_ng(10), 32)

    def test_clean_env_strips_malloc_vars_only(self):
        with mock.patch.dict(os.environ, {"MallocStackLogging": "1", "MallocNanoZone": "0", "KEEP_ME": "x"}):
            env = hd._clean_subprocess_env_ng()
        self.assertNotIn("MallocStackLogging", env)
        self.assertNotIn("MallocNanoZone", env)
        self.assertEqual(env["KEEP_ME"], "x")

    def test_lab_dir_env_var_wins_then_first_existing_candidate_then_default(self):
        with mock.patch.dict(os.environ, {"HIDREAM_LAB_DIR": "/somewhere/lab"}):
            self.assertEqual(hd._resolve_hidream_lab_dir_ng(), Path("/somewhere/lab"))
        with tempfile.TemporaryDirectory() as home, \
             mock.patch.dict(os.environ, {}, clear=False), \
             mock.patch.object(Path, "home", return_value=Path(home)):
            os.environ.pop("HIDREAM_LAB_DIR", None)
            default = Path(home) / "HIDREAM-O1-MLX-LAB-active"
            second = Path(home) / "AI" / "HIDREAM-O1-MLX-LAB-active"
            # Nothing exists (assuming the machine has no /Volumes/AI lab): falls back to the first.
            with mock.patch.object(Path, "exists", return_value=False):
                self.assertEqual(hd._resolve_hidream_lab_dir_ng(), default)
            second.mkdir(parents=True)
            with mock.patch.object(Path, "exists", lambda p: p == second):
                self.assertEqual(hd._resolve_hidream_lab_dir_ng(), second)

    def test_resolvers_need_real_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            py = d / "python"
            py.write_text("#!/bin/sh\n")
            self.assertIsNone(hd._resolve_hidream_python_ng(hd.HiDreamConfig(python_path=str(py))))
            py.chmod(0o755)
            self.assertEqual(hd._resolve_hidream_python_ng(hd.HiDreamConfig(python_path=str(py))), str(py))

            model = d / "model"
            (model / "extras").mkdir(parents=True)
            cfg = hd.HiDreamConfig(model_path=str(model))
            self.assertIsNone(hd._resolve_hidream_model_ng(cfg))
            (model / "model.safetensors").write_bytes(b"")
            self.assertIsNone(hd._resolve_hidream_model_ng(cfg))
            (model / "extras" / "custom_heads.safetensors").write_bytes(b"")
            self.assertEqual(hd._resolve_hidream_model_ng(cfg), str(model))

    def test_health_ready_only_when_python_model_and_script_all_ok(self):
        with mock.patch.object(hd, "_resolve_hidream_python_ng", return_value="/p"), \
             mock.patch.object(hd, "_resolve_hidream_model_ng", return_value="/m"), \
             mock.patch.object(hd, "HIDREAM_GENERATE_SCRIPT", Path(__file__)):
            self.assertTrue(hd.hidream_health_ng()["ready"])
        with mock.patch.object(hd, "_resolve_hidream_python_ng", return_value="/p"), \
             mock.patch.object(hd, "_resolve_hidream_model_ng", return_value=None), \
             mock.patch.object(hd, "HIDREAM_GENERATE_SCRIPT", Path(__file__)):
            h = hd.hidream_health_ng()
            self.assertFalse(h["ready"])
            self.assertFalse(h["model_ok"])
            self.assertTrue(h["python_ok"])


class ColorMatchTests(unittest.TestCase):
    def test_result_tonality_moves_toward_reference_and_mode_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            ref, res = Path(d) / "ref.png", Path(d) / "res.png"
            rng = np.random.default_rng(0)
            Image.fromarray(rng.integers(150, 250, (16, 16, 3), dtype=np.uint8)).save(ref)  # bright
            Image.fromarray(rng.integers(0, 100, (16, 16, 3), dtype=np.uint8)).save(res)     # dark
            before = np.asarray(Image.open(res).convert("L")).mean()
            hd._color_match_to_reference_ng(res, ref)
            after_img = Image.open(res)
            self.assertEqual(after_img.mode, "RGB")
            self.assertGreater(np.asarray(after_img.convert("L")).mean(), before + 50)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        self.script = self.out / "gen.py"
        self.script.write_text("# never run\n")
        for p in (mock.patch.object(hd, "_resolve_hidream_python_ng", return_value="/usr/bin/python3"),
                  mock.patch.object(hd, "_resolve_hidream_model_ng", return_value="/fake/model"),
                  mock.patch.object(hd, "HIDREAM_GENERATE_SCRIPT", self.script)):
            p.start()
            self.addCleanup(p.stop)
        self.cfg = hd.HiDreamConfig()
        self.logs = []
        self.cmds = []

    def _run(self, writer=None, proc=None, n=1, w=64, h=64, offspec=True, refs=None,
             seed=42, **kw):
        def popen(cmd, **kwargs):
            self.cmds.append((cmd, kwargs))
            if writer:
                writer(_output_paths(cmd))
            return proc or _Proc()
        with mock.patch("subprocess.Popen", popen):
            return hd.generate_hidream_ng(
                "a prompt", n, w, h, self.out, seed, self.cfg, refs=refs,
                allow_offspec_res=offspec, on_log=self.logs.append, **kw)

    @staticmethod
    def _valid(w=64, h=64):
        def write(paths):
            for p in paths:
                Path(p).write_bytes(_png_bytes(w, h))
        return write

    # --- preflight ---------------------------------------------------------
    def test_missing_python_model_or_script_raise_file_not_found(self):
        with mock.patch.object(hd, "_resolve_hidream_python_ng", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "venv python"):
                self._run(self._valid())
        with mock.patch.object(hd, "_resolve_hidream_model_ng", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "model not found"):
                self._run(self._valid())
        with mock.patch.object(hd, "HIDREAM_GENERATE_SCRIPT", self.out / "missing.py"):
            with self.assertRaisesRegex(FileNotFoundError, "script missing"):
                self._run(self._valid())
        self.assertEqual(self.cmds, [])

    # --- argv ---------------------------------------------------------------
    def test_argv_for_plain_offspec_render(self):
        res = self._run(self._valid(96, 64), n=2, w=100, h=70, seed=10)
        cmd, kwargs = self.cmds[0]
        self.assertEqual(cmd[:2], ["/usr/bin/python3", str(self.script)])
        flag = lambda name: cmd[cmd.index(name) + 1]
        self.assertEqual((flag("--width"), flag("--height")), ("96", "64"))   # patch-aligned, not snapped
        self.assertEqual(flag("--model-path"), "/fake/model")
        self.assertEqual(flag("--prompt"), "a prompt")
        self.assertEqual(flag("--num-inference-steps"), "28")
        self.assertIn("--no-snap-resolution", cmd)
        self.assertNotIn("--ref-images", cmd)
        self.assertEqual(cmd[cmd.index("--seed") + 1:cmd.index("--seed") + 3], ["10", "11"])
        self.assertEqual(len(_output_paths(cmd)), 2)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual([r["seed"] for r in res], [10, 11])
        self.assertTrue(any("patch-aligning 100x70 -> 96x64" in l for l in self.logs))

    def test_trained_resolution_snap_when_not_offspec(self):
        self._run(self._valid(2048, 2048), w=1024, h=1024, offspec=False)
        cmd = self.cmds[0][0]
        self.assertEqual(cmd[cmd.index("--width") + 1], "2048")
        self.assertNotIn("--no-snap-resolution", cmd)
        self.assertTrue(any("snapping 1024x1024 -> 2048x2048" in l for l in self.logs))

    def test_refs_switch_on_edit_mode_flags(self):
        ref = self.out / "ref.png"
        ref.write_bytes(_png_bytes(64, 64))
        self._run(self._valid(), refs=[ref])
        cmd = self.cmds[0][0]
        self.assertEqual(cmd[cmd.index("--ref-images") + 1], str(ref))
        self.assertEqual(cmd[cmd.index("--editing-scheduler") + 1], "flow_match")

    def test_random_seed_when_none_given(self):
        res = self._run(self._valid(), seed=None)
        self.assertTrue(0 <= res[0]["seed"] < 2**31)

    # --- process handling -------------------------------------------------
    def test_subprocess_output_is_logged_and_proc_handed_back(self):
        started = []
        proc = _Proc(lines=["loading\n", "\n", "step 1/28\n"])
        self._run(self._valid(), proc=proc, on_proc_start=started.append)
        self.assertEqual(started, [proc])
        self.assertIn("[hidream] loading", self.logs)
        self.assertIn("[hidream] step 1/28", self.logs)
        self.assertNotIn("[hidream] ", self.logs)  # blank lines skipped

    def test_nonzero_rc_and_killed_rc(self):
        with self.assertRaisesRegex(RuntimeError, "failed with rc=1"):
            self._run(self._valid(), proc=_Proc(rc=1))
        for rc in (-9, -15, 137, 143):
            with self.assertRaises(hd.ImageJobCancelled):
                self._run(self._valid(), proc=_Proc(rc=rc))

    def test_watchdog_timeout_kills_and_raises_cancelled(self):
        proc = _Proc(delay=0.3)
        with mock.patch.dict(os.environ, {"RINGVIZ_HIDREAM_TIMEOUT_S": "0.05"}), \
             mock.patch.object(hd.os, "killpg", side_effect=ProcessLookupError):
            with self.assertRaisesRegex(hd.ImageJobCancelled, "deadline"):
                self._run(self._valid(), proc=proc)
        self.assertTrue(proc.killed)

    # --- output validation ---------------------------------------------------
    def test_missing_png_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no PNG"):
            self._run(lambda paths: None)

    def test_truncated_png_raises(self):
        def write(paths):
            full = _png_bytes(64, 64)
            for p in paths:
                Path(p).write_bytes(full[: len(full) // 2])
        with self.assertRaisesRegex(RuntimeError, "incomplete/corrupt PNG"):
            self._run(write)

    def test_wrong_size_png_raises(self):
        with self.assertRaisesRegex(RuntimeError, "expected 64x64"):
            self._run(self._valid(32, 32))

    def test_valid_png_returns_result_dict(self):
        [r] = self._run(self._valid())
        self.assertEqual((r["width"], r["height"], r["engine"]), (64, 64, "hidream-o1-dev-bf16"))
        self.assertTrue(Path(r["png_path"]).is_file())

    # --- color match in edit mode ------------------------------------------
    def test_edit_mode_color_matches_and_survives_failure(self):
        ref = self.out / "ref.png"
        ref.write_bytes(_png_bytes(64, 64))
        with mock.patch.object(hd, "_color_match_to_reference_ng") as cm:
            self._run(self._valid(), refs=[ref])
        cm.assert_called_once()
        self.assertTrue(any("color-matched to ref.png" in l for l in self.logs))

        self.logs.clear()
        with mock.patch.object(hd, "_color_match_to_reference_ng", side_effect=ValueError("bad ref")):
            [r] = self._run(self._valid(), refs=[ref])
        self.assertTrue(Path(r["png_path"]).is_file())  # render kept
        self.assertTrue(any("color-match skipped (bad ref)" in l for l in self.logs))


if __name__ == "__main__":
    unittest.main()
