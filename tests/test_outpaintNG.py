"""Make vertical (AI fill): the clip checks, the job queue, the routes and the runner's retry, with the model itself
replaced so nothing renders.

Run with: python3 -m unittest tests.test_outpaintNG -v
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import outpaint_engineNG as engine
import outpaint_jobsNG as jobs
import outpaint_runnerNG as runner

HAVE_FFMPEG = bool(shutil.which("ffmpeg") or os.path.exists("/opt/homebrew/bin/ffmpeg"))


def make_clip(path, w, h, frames=9):
    exe = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=24",
                    "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)], check=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is needed to make a test clip")
class PlanChecks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_widescreen_clip_is_planned_tall(self):
        clip = Path(self.dir) / "wide.mp4"
        make_clip(clip, 640, 360)
        plan = engine.plan_ng(str(clip), "9:16")
        self.assertEqual((plan["out_w"], plan["out_h"]), (720, 1280))
        self.assertEqual((plan["width"], plan["height"], plan["frames"]), (640, 360, 9))

    def test_square_shape_is_taller_than_the_clip(self):
        clip = Path(self.dir) / "wide.mp4"
        make_clip(clip, 640, 360)
        plan = engine.plan_ng(str(clip), "1:1")
        self.assertEqual(plan["out_w"], plan["out_h"])

    def test_a_clip_that_is_already_tall_is_refused(self):
        clip = Path(self.dir) / "tall.mp4"
        make_clip(clip, 360, 640)
        with self.assertRaises(ValueError):
            engine.plan_ng(str(clip), "9:16")

    def test_bad_shape_and_missing_file_are_refused(self):
        with self.assertRaises(ValueError):
            engine.plan_ng("/nowhere.mp4", "9:16")
        clip = Path(self.dir) / "wide.mp4"
        make_clip(clip, 640, 360)
        with self.assertRaises(ValueError):
            engine.plan_ng(str(clip), "21:9")


class Health(unittest.TestCase):
    def test_missing_lora_is_reported_in_plain_words(self):
        with mock.patch.object(engine, "LORA_PATH", Path("/nowhere/lora.safetensors")):
            h = engine.outpaint_health_ng()
        self.assertFalse(h["ok"])
        self.assertTrue(any("add-on" in p for p in h["problems"]))
        self.assertEqual(h["ratios"], ["9:16", "4:5", "1:1"])


class FakeVideoJob:
    def __init__(self, mp4, prompt="a woman by a door"):
        self.mp4_path, self.prompt = mp4, prompt


class Queue(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.src = Path(self.dir) / "src.mp4"
        self.src.write_bytes(b"x")
        patches = [
            mock.patch.object(jobs, "OUTPAINT_DIR", self.dir),
            mock.patch.object(jobs, "_other_gpu_work", lambda: False),
            mock.patch.object(jobs.video_jobsNG, "get_job_ng", lambda jid: FakeVideoJob(str(self.src)) if jid == "clip1" else None),
            mock.patch.object(jobs.engine, "outpaint_health_ng", lambda: {"ok": True, "problems": []}),
            mock.patch.object(jobs.engine, "plan_ng", lambda src, ratio: {"out_w": 720, "out_h": 1280}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def wait(self, job_id, timeout=5):
        end = time.time() + timeout
        while time.time() < end:
            j = jobs.get_ng(job_id)
            if j["finished_at"]:
                return j
            time.sleep(0.05)
        self.fail("the job did not finish")

    def test_a_job_runs_and_exposes_its_video(self):
        def fake(source, output, ratio, prompt, on_log=None, should_stop=None):
            self.assertEqual((source, ratio, prompt), (str(self.src), "9:16", "a woman by a door"))
            Path(output).write_bytes(b"mp4")
            on_log("Outpaint: stage 1")
            return {"output": output, "seconds": 12.5, "windows": 1}
        with mock.patch.object(jobs.engine, "outpaint_vertical_ng", fake):
            started = jobs.start_job_ng("clip1", "9:16")
            done = self.wait(started["id"])
        pub = jobs.public_ng(done)
        self.assertEqual((pub["status"], pub["seconds"]), ("done", 12.5))
        self.assertTrue(pub["video_url"].endswith("/video"))
        self.assertIn("Outpaint: stage 1", pub["log_tail"])

    def test_a_failure_is_reported_not_raised(self):
        with mock.patch.object(jobs.engine, "outpaint_vertical_ng", side_effect=RuntimeError("the AI fill failed: boom")):
            done = self.wait(jobs.start_job_ng("clip1", "4:5")["id"])
        self.assertEqual((done["status"], done["error"]), ("failed", "the AI fill failed: boom"))
        self.assertIsNone(jobs.public_ng(done)["video_url"])

    def test_unknown_or_unfinished_clips_are_refused(self):
        with self.assertRaises(LookupError):
            jobs.start_job_ng("nope", "9:16")
        with mock.patch.object(jobs.video_jobsNG, "get_job_ng", lambda jid: FakeVideoJob(None)):
            with self.assertRaises(LookupError):
                jobs.start_job_ng("clip1", "9:16")

    def test_missing_pieces_stop_the_job_before_it_queues(self):
        with mock.patch.object(jobs.engine, "outpaint_health_ng", lambda: {"ok": False, "problems": ["the AI fill add-on is not downloaded (1.3 GB)"]}):
            with self.assertRaises(ValueError) as cm:
                jobs.start_job_ng("clip1", "9:16")
        self.assertIn("add-on", str(cm.exception))

    def test_cancel_before_start_and_after_finish(self):
        with mock.patch.object(jobs, "_other_gpu_work", lambda: True):    # held back, so it is still queued
            started = jobs.start_job_ng("clip1", "9:16")
            self.assertTrue(jobs.cancel_ng(started["id"]))
            done = self.wait(started["id"], 8)
        self.assertEqual(done["status"], "cancelled")
        self.assertFalse(jobs.cancel_ng(started["id"]))
        self.assertFalse(jobs.cancel_ng("no-such-job"))


class Routes(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from routes.outpaintNG import outpaintNG_bp
        app = Flask(__name__)
        app.register_blueprint(outpaintNG_bp)
        self.c = app.test_client()

    def test_status_lists_the_shapes(self):
        r = self.c.get("/api/ng/outpaint/status")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["ratios"], ["9:16", "4:5", "1:1"])

    def test_start_maps_errors_to_codes(self):
        with mock.patch.object(jobs, "start_job_ng", side_effect=LookupError("gone")):
            self.assertEqual(self.c.post("/api/ng/outpaint/start", json={"job_id": "x"}).status_code, 404)
        with mock.patch.object(jobs, "start_job_ng", side_effect=ValueError("already vertical")):
            r = self.c.post("/api/ng/outpaint/start", json={"job_id": "x", "ratio": "9:16"})
        self.assertEqual((r.status_code, r.get_json()["error"]), (400, "already vertical"))

    def test_unknown_job_and_unfinished_video(self):
        self.assertEqual(self.c.get("/api/ng/outpaint/jobs/nope").status_code, 404)
        self.assertEqual(self.c.get("/api/ng/outpaint/jobs/nope/video").status_code, 404)
        self.assertEqual(self.c.get("/api/ng/outpaint/jobs/nope/download").status_code, 404)


class FakeProc:
    def __init__(self, lines, code):
        self.stdin, self.stdout, self._code = io.StringIO(), iter(lines), code

    def wait(self):
        return self._code


WATCHDOG = "[METAL] Command buffer execution failed: Impacting Interactivity (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"


class RunnerRetry(unittest.TestCase):
    def run_child(self, procs):
        with mock.patch("subprocess.Popen", side_effect=procs) as popen, mock.patch.object(runner, "say"):
            try:
                runner.child({"model": "m"}, stage=1)
                return popen.call_count, None
            except RuntimeError as e:
                return popen.call_count, str(e)

    def test_a_watchdog_kill_is_retried_in_a_fresh_process(self):
        calls, err = self.run_child([FakeProc([WATCHDOG], 1), FakeProc(["done"], 0)])
        self.assertEqual((calls, err), (2, None))

    def test_three_kills_in_a_row_give_up(self):
        calls, err = self.run_child([FakeProc([WATCHDOG], 1) for _ in range(3)])
        self.assertEqual(calls, 3)
        self.assertIn("stage 1 failed", err)

    def test_any_other_failure_is_not_retried(self):
        calls, err = self.run_child([FakeProc(["ValueError: bad lora"], 1), FakeProc(["done"], 0)])
        self.assertEqual(calls, 1)
        self.assertIn("bad lora", err)


if __name__ == "__main__":
    unittest.main()
