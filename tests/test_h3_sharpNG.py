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

    def test_unknown_job_and_bad_source(self):
        with mock.patch.object(sharp.h3_reroll, "resolve_source_ng", side_effect=LookupError("no such job")):
            self.assertEqual(self.client.post("/api/ng/h3/jobs/zz/sharp").status_code, 404)
        self.clip.unlink()
        self.assertEqual(self.client.post("/api/ng/h3/jobs/j1/sharp").status_code, 400)


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
