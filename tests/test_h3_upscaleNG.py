"""Unit tests for the 2x upscale job (h3_upscaleNG.py, routes/h3_toolsNG.py): what it refuses, the
command it runs, how a finished or failed run is reported, and the route. No model runs; the
subprocess is faked.

Run with: python3 -m unittest tests.test_h3_upscaleNG -v
"""

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

import h3_upscaleNG as up
import routes.h3_toolsNG as tools


class FakeProc:
    def __init__(self, lines, code, make=None):
        self.stdout = io.StringIO("".join(l + "\n" for l in lines))
        self.code, self.make = code, make
        self.pid = 1

    def wait(self):
        if self.make:
            Path(self.make).write_bytes(b"mp4")
        return self.code


class UpscaleBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clip = self.tmp / "film.mp4"
        self.clip.write_bytes(b"x")
        self.queued = {}

        def fake_start(task, label, run, meta=None):
            self.queued = dict(task=task, label=label, run=run, meta=meta)
            return SimpleNamespace(job_id="u1", meta=meta)

        for target, name, val in (
                (up.h3_reroll, "resolve_source_ng", lambda i: SimpleNamespace(mp4_path=str(self.clip))),
                (up, "ltx_python_ng", lambda: "/venv/bin/python3.11"),
                (up, "frames_of_ng", lambda p: 248),
                (up.h3_jobs, "start_task_job_ng", fake_start),
                (up.ltx, "_unload_gemma_server_ng", lambda log=None: None)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)

    def job(self):
        return SimpleNamespace(job_dir=self.tmp, cancelled=False, _proc=None, log=[],
                               append_log=lambda line: self.job_log.append(line),
                               log_tail=lambda n: self.job_log[-n:])

    job_log = []


class Start(UpscaleBase):
    def test_queues_a_task_job_with_an_estimate_from_the_frames(self):
        up.start_upscale_ng("j1")
        self.assertEqual((self.queued["task"], self.queued["meta"]["frames"]), ("upscale", 248))
        self.assertEqual(self.queued["meta"]["estimate_min"], 17.0)  # two windows' worth

    def test_refusals(self):
        for kw, text in ((dict(steps=0), "steps must be"), (dict(steps="x"), "steps must be a number"),
                         (dict(keep=2), "keep must be"), (dict(keep=0.1), "keep must be")):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, text):
                up.start_upscale_ng("j1", **kw)
        self.clip.unlink()
        with self.assertRaisesRegex(ValueError, "no video"):
            up.start_upscale_ng("j1")


class Run(UpscaleBase):
    def run_job(self, proc):
        up.start_upscale_ng("j1", steps=2, keep=0.8, seed=7)
        self.job_log = []
        job = self.job()
        with mock.patch.object(up.subprocess, "Popen", return_value=proc) as popen:
            try:
                return self.queued["run"](job), popen
            except RuntimeError as e:
                return e, popen

    def test_a_clean_run_returns_the_output_path_and_passes_the_settings(self):
        out = self.tmp / "film_x2.mp4"
        result, popen = self.run_job(FakeProc(["film.mp4: 640x384", "wrote x"], 0, make=out))
        self.assertEqual(result, str(out))
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:2], ["/venv/bin/python3.11", str(up.SCRIPT)])
        self.assertEqual([cmd[cmd.index(f) + 1] for f in ("--steps", "--keep", "--seed")], ["2", "0.8", "7"])
        self.assertIn("wrote x", self.job_log)

    def test_a_failed_run_says_why(self):
        result, _ = self.run_job(FakeProc(["boom"], 1))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("exit 1", str(result))
        self.assertIn("boom", str(result))

    def test_the_metal_watchdog_is_retried_and_then_succeeds(self):
        out = self.tmp / "film_x2.mp4"
        procs = [FakeProc(["Denoising", "Impacting Interactivity"], -6), FakeProc(["wrote x"], 0, make=out)]
        up.start_upscale_ng("j1")
        self.job_log = []
        with mock.patch.object(up.subprocess, "Popen", side_effect=procs) as popen:
            result = self.queued["run"](self.job())
        self.assertEqual((result, popen.call_count), (str(out), 2))
        self.assertTrue(any("trying again" in l for l in self.job_log))

    def test_the_watchdog_gives_up_after_the_attempts_and_other_errors_do_not_retry(self):
        up.start_upscale_ng("j1")
        self.job_log = []
        dead = [FakeProc(["Impacting Interactivity"], -6) for _ in range(up.ATTEMPTS)]
        with mock.patch.object(up.subprocess, "Popen", side_effect=dead) as popen:
            with self.assertRaisesRegex(RuntimeError, "exit -6"):
                self.queued["run"](self.job())
        self.assertEqual(popen.call_count, up.ATTEMPTS)
        with mock.patch.object(up.subprocess, "Popen", return_value=FakeProc(["boom"], 1)) as popen:
            with self.assertRaises(RuntimeError):
                self.queued["run"](self.job())
        self.assertEqual(popen.call_count, 1)

    def test_exit_zero_without_a_file_is_still_a_failure(self):
        result, _ = self.run_job(FakeProc([], 0))
        self.assertIsInstance(result, RuntimeError)


class Route(UpscaleBase):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(tools.h3ToolsNG_bp)
        self.client = app.test_client()

    def test_accepts_and_reports_the_estimate(self):
        r = self.client.post("/api/ng/h3/jobs/j1/upscale", json={"steps": 2})
        self.assertEqual((r.status_code, r.get_json()["job_id"], r.get_json()["estimate_min"]), (202, "u1", 17.0))

    def test_bad_values_and_unknown_jobs(self):
        self.assertEqual(self.client.post("/api/ng/h3/jobs/j1/upscale", json={"steps": 99}).status_code, 400)
        with mock.patch.object(up.h3_reroll, "resolve_source_ng", side_effect=LookupError("no such job")):
            self.assertEqual(self.client.post("/api/ng/h3/jobs/zz/upscale").status_code, 404)


if __name__ == "__main__":
    unittest.main()
