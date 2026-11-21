"""Unit tests for Sharp 2x (h3_sharpNG.py, the /sharp route, and the tile layout in vendor/pipersr_x2/pipersr_x2_run.py).
No model runs: the subprocess is faked, and the tile layout is plain arithmetic.

Run with: venv/bin/python3 -m unittest tests.test_h3_sharpNG -v
"""

import importlib.util
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_sharpNG as sharp
import routes.h3_toolsNG as tools

_real_pipersr_python = sharp.pipersr_python_ng  # setUp replaces it with a fake

_spec = importlib.util.spec_from_file_location("pipersr_x2_run", sharp.SCRIPT)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class FakeProc:
    def __init__(self, lines, code, make=None):
        self.stdout = io.StringIO("".join(l + "\n" for l in lines))
        self.code, self.make = code, make
        self.pid = 1

    def wait(self):
        if self.make:
            Path(self.make).write_bytes(b"mp4")
        return self.code


class SharpBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clip = self.tmp / "film.mp4"
        self.clip.write_bytes(b"x")
        self.queued = {}
        self.job_log = []

        def fake_start(task, label, run, meta=None):
            self.queued = dict(task=task, label=label, run=run, meta=meta)
            return SimpleNamespace(job_id="s1", meta=meta)

        for target, name, val in (
                (sharp.h3_reroll, "resolve_source_ng", lambda i: SimpleNamespace(mp4_path=str(self.clip))),
                (sharp, "pipersr_python_ng", lambda: "/venv/bin/python"),
                (sharp, "frames_of_ng", lambda p: 124),
                (sharp.h3_jobs, "start_task_job_ng", fake_start)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)

    def job(self):
        return SimpleNamespace(job_dir=self.tmp, cancelled=False, _proc=None,
                               append_log=lambda line: self.job_log.append(line),
                               log_tail=lambda n: self.job_log[-n:])


class Start(SharpBase):
    def test_queues_a_sharp_task_with_a_seconds_scale_estimate(self):
        sharp.start_sharp_ng("j1")
        self.assertEqual((self.queued["task"], self.queued["meta"]["frames"]), ("sharp", 124))
        self.assertLess(self.queued["meta"]["estimate_min"], 1.0)

    def test_the_default_is_the_halo_limited_picture_and_is_passed_to_the_runner(self):
        self.assertEqual(sharp.DEFAULT_STRENGTH, 1.0)
        self.assertEqual(runner.DEFAULT_STRENGTH, 1.0)
        sharp.start_sharp_ng("j1", strength=0.5)
        self.assertEqual(self.queued["meta"]["strength"], 0.5)
        with mock.patch.object(sharp.subprocess, "Popen", return_value=FakeProc(["ok"], 0, make=self.tmp / "film_sr2x.mp4")) as popen:
            self.queued["run"](self.job())
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--strength") + 1], "0.5")

    def test_a_strength_out_of_range_is_refused(self):
        for bad in (-1, 1.01, "x", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "strength"):
                sharp.start_sharp_ng("j1", strength=bad)

    def test_a_render_without_a_video_is_refused(self):
        self.clip.unlink()
        with self.assertRaisesRegex(ValueError, "no video"):
            sharp.start_sharp_ng("j1")

    def test_a_missing_env_says_how_to_build_it(self):
        with mock.patch.object(sharp, "PYTHON", self.tmp / "nope" / "python"), \
                mock.patch.dict(os.environ, {"RINGVIZ_PIPERSR_PYTHON": ""}):
            with self.assertRaisesRegex(FileNotFoundError, "build_venv.sh"):
                _real_pipersr_python()


class Run(SharpBase):
    def run_job(self, proc):
        sharp.start_sharp_ng("j1")
        with mock.patch.object(sharp.subprocess, "Popen", return_value=proc) as popen:
            try:
                return self.queued["run"](self.job()), popen
            except RuntimeError as e:
                return e, popen

    def test_a_clean_run_returns_the_output_and_logs_the_attribution(self):
        out = self.tmp / "film_sr2x.mp4"
        result, popen = self.run_job(FakeProc([sharp.ATTRIBUTION, "[pipersr] done 18s"], 0, make=out))
        self.assertEqual(result, str(out))
        self.assertEqual(popen.call_args.args[0][:2], ["/venv/bin/python", str(sharp.SCRIPT)])
        self.assertIn(sharp.ATTRIBUTION, self.job_log)

    def test_a_failed_run_says_why(self):
        result, _ = self.run_job(FakeProc(["[pipersr] error no such clip"], 2))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("exit 2", str(result))
        self.assertIn("no such clip", str(result))

    def test_exit_zero_without_a_file_is_still_a_failure(self):
        result, _ = self.run_job(FakeProc(["fine"], 0))
        self.assertIsInstance(result, RuntimeError)


class Route(SharpBase):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(tools.h3ToolsNG_bp)
        self.client = app.test_client()

    def test_accepts_and_reports_the_estimate(self):
        r = self.client.post("/api/ng/h3/jobs/j1/sharp")
        self.assertEqual((r.status_code, r.get_json()["job_id"]), (202, "s1"))
        self.assertIn("estimate_min", r.get_json())

    def test_a_strength_goes_through_and_a_bad_one_is_a_400(self):
        r = self.client.post("/api/ng/h3/jobs/j1/sharp", json={"strength": 0.25})
        self.assertEqual((r.status_code, self.queued["meta"]["strength"]), (202, 0.25))
        self.assertEqual(self.client.post("/api/ng/h3/jobs/j1/sharp", json={"strength": 7}).status_code, 400)

    def test_unknown_job_and_bad_source(self):
        with mock.patch.object(sharp.h3_reroll, "resolve_source_ng", side_effect=LookupError("no such job")):
            self.assertEqual(self.client.post("/api/ng/h3/jobs/zz/sharp").status_code, 404)
        self.clip.unlink()
        self.assertEqual(self.client.post("/api/ng/h3/jobs/j1/sharp").status_code, 400)


class RunnerRefusals(unittest.TestCase):
    """The runner says what is wrong in one line and exits 2, before the model is ever loaded."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch.object(runner, "load_model", side_effect=AssertionError("the model must not load"))
        p.start()
        self.addCleanup(p.stop)

    def test_a_missing_file(self):
        self.assertEqual(runner.main([str(self.tmp / "nope.png")]), 2)

    def test_a_picture_that_is_not_a_picture(self):
        bad = self.tmp / "bad.png"
        bad.write_bytes(b"not a png")
        self.assertEqual(runner.main([str(bad)]), 2)
        self.assertFalse((self.tmp / "bad_sr2x.png").exists())

    def test_a_strength_out_of_range(self):
        pic = self.tmp / "a.png"
        pic.write_bytes(b"x")
        self.assertEqual(runner.main([str(pic), "--strength", "1.5"]), 2)

    def test_a_clip_with_no_video(self):
        bad = self.tmp / "bad.mp4"
        bad.write_bytes(b"not a clip")
        self.assertEqual(runner.main([str(bad)]), 2)

    def test_pictures_and_clips_are_told_apart_by_their_suffix(self):
        for name in ("a.png", "a.JPG", "a.jpeg", "a.webp"):
            self.assertIn(Path(name).suffix.lower(), runner.PICTURE_SUFFIXES)
        self.assertNotIn(".mp4", runner.PICTURE_SUFFIXES)


class HaloLimit(unittest.TestCase):
    """The halo limit holds every pixel inside the plain resize's local range; mix is a straight blend."""

    def setUp(self):
        import numpy as np
        from PIL import Image
        self.np, self.Image = np, Image

    def img(self, arr):
        return self.Image.fromarray(self.np.array(arr, dtype=self.np.uint8))

    def test_running_min_and_max_match_a_brute_force_window(self):
        np = self.np
        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
        r = runner.HALO_WINDOW // 2
        padded = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
        for op, ref in ((np.minimum, np.min), (np.maximum, np.max)):
            want = np.zeros_like(a)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    want[y, x] = ref(padded[y:y + 2 * r + 1, x:x + 2 * r + 1], axis=(0, 1))
            self.assertTrue((runner._running(a, op) == want).all())

    def test_an_overshoot_beside_an_edge_is_pulled_back_and_a_flat_area_is_untouched(self):
        np = self.np
        plain = np.full((12, 12, 3), 100, dtype=np.uint8)
        plain[:, 6:] = 200                                  # a step edge
        sharp = plain.copy()
        sharp[:, 5] = 60                                    # dark halo on the dark side
        sharp[:, 6] = 255                                   # bright halo on the bright side
        sharp[0, 0] = 100
        out = np.asarray(runner.limit_halo(self.img(plain), self.img(sharp)))
        self.assertEqual((out[:, 5] == 100).all(), True)    # the dark halo is back inside the 100..200 range
        self.assertEqual((out[:, 6] == 200).all(), True)    # and the bright one is cut to 200
        self.assertEqual(int(out[0, 0, 0]), 100)

    def test_nothing_is_changed_when_sharp_stays_inside_the_range(self):
        np = self.np
        plain = np.tile(np.arange(0, 240, 20, dtype=np.uint8), (12, 1))[..., None].repeat(3, axis=2)
        out = np.asarray(runner.limit_halo(self.img(plain), self.img(plain)))
        self.assertTrue((out == plain).all())

    def test_mix_endpoints_and_midpoint(self):
        plain, sharp = self.img([[[0, 0, 0]]]), self.img([[[200, 100, 50]]])
        self.assertEqual(runner.mix(plain, sharp, 1.0).getpixel((0, 0)), (200, 100, 50))
        self.assertEqual(runner.mix(plain, sharp, 0.0).getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(runner.mix(plain, sharp, 0.5).getpixel((0, 0)), (100, 50, 25))


class TileLayout(unittest.TestCase):
    """Every padded pixel of the kept area is supplied by exactly one tile, always well inside that tile."""

    def check(self, length):
        segs = runner.tile_segments(length)
        covered = []
        for origin, start, end in segs:
            self.assertGreaterEqual(origin, 0)
            self.assertLessEqual(origin + runner.TILE, length)               # the tile fits in the image
            self.assertGreaterEqual(start, origin + runner.MARGIN)            # at least MARGIN from the tile's left edge
            self.assertLessEqual(end, origin + runner.TILE - runner.MARGIN)   # and from its right edge
            covered += list(range(start, end))
        self.assertEqual(covered, list(range(runner.MARGIN, length - runner.MARGIN)))

    def test_the_clip_sizes_we_make(self):
        for w, h in ((640, 384), (768, 448), (1024, 576), (1344, 768), (1280, 704)):
            with self.subTest(w=w, h=h):
                self.check(w + 2 * runner.MARGIN)
                self.check(h + 2 * runner.MARGIN)

    def test_odd_and_tile_sized_lengths(self):
        for n in (96, 97, 128, 129, 200, 351, 352, 353):
            with self.subTest(n=n):
                self.check(n + 2 * runner.MARGIN)

    def test_a_length_smaller_than_a_tile_is_refused(self):
        with self.assertRaises(ValueError):
            runner.tile_segments(runner.TILE - 1)


if __name__ == "__main__":
    unittest.main()
