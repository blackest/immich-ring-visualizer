"""Unit tests for the soft-finish pass: the settings checks, the ComfyUI workflow it builds, a run with ComfyUI's HTTP
replaced, the job queue, the routes and how Talk uses it. Nothing talks to ComfyUI and nothing is rendered.

Run with: python3 -m unittest tests.test_finishNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import finish_engineNG as eng
import finish_jobsNG as jobs
import routes.finishNG as finish_routes
import talk_jobsNG as talk


class ParamTests(unittest.TestCase):
    def test_defaults_fill_in_and_values_are_kept(self):
        self.assertEqual(eng.normalize_params_ng(None), eng.DEFAULTS)
        self.assertEqual(eng.normalize_params_ng({"saturation": 1.0})["saturation"], 1.0)
        self.assertEqual(eng.normalize_params_ng({"tone_strength": ""})["tone_strength"], 0.25)

    def test_out_of_range_unknown_and_non_numbers_are_refused(self):
        for bad in ({"clarity_strength": 0.3}, {"clarity_strength": -0.9}, {"saturation": 3}, {"tone_strength": -1},
                    {"mode": "Filmic Soft"}, {"saturation": True}, {"clarity_radius": "24"}):
            with self.subTest(bad=bad), self.assertRaises(eng.FinishRequestError):
                eng.normalize_params_ng(bad)


def _soft_finish_module():
    """The vendored script as a module (its filename is not importable as a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("soft_finish_script", eng.FINISH_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.sf, cls.np = _soft_finish_module(), np

    def test_a_flat_picture_is_untouched_by_clarity(self):
        flat = self.np.full((32, 32, 3), 0.5, dtype=self.np.float32)
        self.assertTrue(self.np.allclose(self.sf.clarity(flat, -0.5, 8.0), flat, atol=1e-5))

    def test_negative_clarity_softens_midtone_detail_and_positive_sharpens_it(self):
        np = self.np
        rng = np.random.default_rng(1)
        img = np.clip(0.5 + rng.normal(0, 0.05, (64, 64, 1)), 0, 1).repeat(3, axis=2).astype(np.float32)
        soft = self.sf.clarity(img, -0.5, 6.0)
        hard = self.sf.clarity(img, 0.5, 6.0)
        self.assertLess(self.sf.luma(soft).std(), self.sf.luma(img).std())
        self.assertGreater(self.sf.luma(hard).std(), self.sf.luma(img).std())

    def test_the_tone_map_lifts_shadows_and_keeps_black_and_white_where_they_are(self):
        np = self.np
        ramp = np.linspace(0, 1, 11, dtype=np.float32)[None, :, None].repeat(3, axis=2)
        out = self.sf.tone_cineon(ramp, 0.25, 1.0)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[0, -1, 0]), 1.0, places=5)
        self.assertTrue(np.all(out[0, 1:-1, 0] > ramp[0, 1:-1, 0]))          # every shadow/mid is lifted
        lift = out[0, :, 0] - ramp[0, :, 0]
        self.assertGreater(lift[3], lift[9])        # a low-mid is lifted more than a near-white; the lift tapers to nothing at white

    def test_strength_zero_and_saturation_one_is_the_identity_and_saturation_trims_chroma(self):
        np = self.np
        img = np.random.default_rng(2).random((8, 8, 3)).astype(np.float32)
        self.assertTrue(np.array_equal(self.sf.tone_cineon(img, 0.0, 1.0), img))
        grey = lambda a: a - self.sf.luma(a)[..., None]
        self.assertTrue(np.array_equal(self.sf.tone_cineon(img, 0.0, 0.5), img))      # strength scales the whole pass
        self.assertLess(np.abs(grey(self.sf.tone_cineon(img, 1.0, 0.5))).mean(), np.abs(grey(img)).mean())


def _make_clip(path, audio=True, seconds=1):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=24:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    import subprocess
    subprocess.run(cmd, check=True)


def _probe(path):
    import json
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,width,height,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


class HealthTests(unittest.TestCase):
    def test_ready_here(self):
        self.assertTrue(eng.finish_health_ng()["ok"], eng.finish_health_ng())

    def test_each_problem_is_named(self):
        with mock.patch.object(eng.shutil, "which", lambda t: None):
            self.assertEqual(len(eng.finish_health_ng()["problems"]), 2)
        with mock.patch.object(eng, "FINISH_SCRIPT", Path("/nope/soft_finish.py")):
            self.assertIn("vendored finish script", eng.finish_health_ng()["problems"][0])
        with mock.patch.object(eng.importlib.util, "find_spec", lambda m: None):
            self.assertIn("opencv-python", eng.finish_health_ng()["problems"][0])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.src = self.t / "clip.mp4"
        _make_clip(self.src)

    def test_a_clip_goes_in_and_a_clip_of_the_same_size_length_and_audio_comes_out(self):
        logs = []
        result = eng.finish_video_ng(str(self.src), str(self.t / "out" / "f.mp4"), on_log=logs.append)
        info = _probe(result["path"])
        self.assertEqual((int(info["video"]["width"]), int(info["video"]["height"])), (96, 64))
        self.assertEqual(info["video"]["nb_read_frames"], _probe(self.src)["video"]["nb_read_frames"])
        self.assertIn("audio", info)
        self.assertTrue(any("done" in line for line in logs))
        self.assertFalse((self.t / "out" / "f.mp4.partial").exists())

    def test_a_silent_clip_stays_silent(self):
        silent = self.t / "silent.mp4"
        _make_clip(silent, audio=False)
        out = eng.finish_video_ng(str(silent), str(self.t / "s.mp4"))["path"]
        self.assertNotIn("audio", _probe(out))

    def test_the_picture_actually_changes_and_the_original_does_not(self):
        before = self.src.read_bytes()
        out = Path(eng.finish_video_ng(str(self.src), str(self.t / "f.mp4"))["path"])
        self.assertEqual(self.src.read_bytes(), before)
        self.assertNotEqual(out.read_bytes(), before)

    def test_stop_kills_it_and_leaves_no_file(self):
        with self.assertRaises(eng.FinishCancelled):
            eng.finish_video_ng(str(self.src), str(self.t / "f.mp4"), should_cancel=lambda: True)
        self.assertEqual([p.name for p in self.t.iterdir() if p.name.startswith("f.mp4")], [])

    def test_a_file_that_is_not_a_video_fails_with_a_reason(self):
        junk = self.t / "junk.mp4"
        junk.write_bytes(b"not a video")
        with self.assertRaisesRegex(RuntimeError, "finish failed"):
            eng.finish_video_ng(str(junk), str(self.t / "j.mp4"))

    def test_request_errors(self):
        with self.assertRaises(eng.FinishRequestError):
            eng.finish_video_ng(str(self.t / "nope.mp4"), str(self.t / "o.mp4"))
        with self.assertRaises(eng.FinishRequestError):
            eng.finish_video_ng(str(self.src), str(self.t / "o.mp4"), {"saturation": 9})


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.clip = self.t / "h3_1.mp4"
        self.clip.write_bytes(b"clip")
        self.h3 = SimpleNamespace(job_id="h1", mp4_path=str(self.clip), finished_at=1.0, error=None)
        for target, name, val in (
                (jobs, "_QUEUE", mock.MagicMock()),
                (jobs, "_ensure_worker_started", lambda: None),
                (jobs.h3_reroll, "resolve_source_ng", lambda jid: self.h3 if jid == "h1" else (_ for _ in ()).throw(
                    LookupError("no such job"))),
                (talk, "_QUEUE", mock.MagicMock()),
                (talk, "_ensure_worker_started", lambda: None),
                (talk.h3_lora_lib, "resolve_h3_lora_path_ng", lambda n: "/l/face.safetensors")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        jobs._JOBS.clear()
        talk._JOBS.clear()
        self.addCleanup(jobs._JOBS.clear)
        self.addCleanup(talk._JOBS.clear)


class JobTests(_Base):
    def test_a_job_is_built_server_side_and_writes_beside_the_source(self):
        job = jobs.start_finish_job_ng("h1", {"saturation": 1.0})
        self.assertEqual(Path(job.out_path).parent, self.clip.parent)
        self.assertTrue(Path(job.out_path).name.startswith("h3_1_finished_"))
        self.assertEqual(job.params["saturation"], 1.0)
        self.assertEqual(jobs.job_status_ng(job.job_id)["status"], "queued")

    def test_unknown_render_unfinished_clip_and_bad_settings_are_refused_and_queue_nothing(self):
        with self.assertRaises(LookupError):
            jobs.start_finish_job_ng("nope")
        self.clip.unlink()
        with self.assertRaises(eng.FinishRequestError):
            jobs.start_finish_job_ng("h1")
        self.clip.write_bytes(b"c")
        with self.assertRaises(eng.FinishRequestError):
            jobs.start_finish_job_ng("h1", {"saturation": 9})
        self.assertEqual(jobs.list_jobs_ng(), [])

    def test_run_records_the_result_or_the_error(self):
        job = jobs.make_job_ng("h1")
        with mock.patch.object(eng, "finish_video_ng", return_value={"path": job.out_path}):
            jobs.run_finish_job_ng(job)
        self.assertEqual(job.result_path, job.out_path)
        bad = jobs.make_job_ng("h1")
        with mock.patch.object(eng, "finish_video_ng", side_effect=RuntimeError("ComfyUI failed: x")):
            jobs.run_finish_job_ng(bad)
        self.assertEqual((bad.error, bad.error_type), ("ComfyUI failed: x", "RuntimeError"))

    def test_cancelling_a_queued_job_skips_it(self):
        job = jobs.start_finish_job_ng("h1")
        self.assertTrue(jobs.cancel_finish_job_ng(job.job_id))
        with mock.patch.object(eng, "finish_video_ng") as run:
            jobs.run_finish_job_ng(job)
        run.assert_not_called()
        self.assertEqual(jobs.job_status_ng(job.job_id)["status"], "failed")


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(finish_routes.finishNG_bp)
        self.client = app.test_client()

    def test_start_poll_cancel_and_the_error_codes(self):
        r = self.client.post("/api/ng/finish/jobs", data={"h3_job_id": "h1", "saturation": "1.0"})
        self.assertEqual(r.status_code, 202)
        jid = r.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/ng/finish/jobs/{jid}").get_json()["status"], "queued")
        self.assertEqual(len(self.client.get("/api/ng/finish/jobs").get_json()["jobs"]), 1)
        self.assertEqual(self.client.get(f"/api/ng/finish/jobs/{jid}/video").status_code, 404)   # not ready
        self.assertTrue(self.client.delete(f"/api/ng/finish/jobs/{jid}").get_json()["ok"])
        self.assertEqual(self.client.post("/api/ng/finish/jobs", data={"h3_job_id": "zzz"}).status_code, 404)
        for data in ({"h3_job_id": "h1", "saturation": "9"}, {"h3_job_id": "h1", "saturation": "abc"}):
            self.assertEqual(self.client.post("/api/ng/finish/jobs", data=data).status_code, 400, data)

    def test_a_finished_clip_is_served(self):
        job = jobs.make_job_ng("h1")
        job.result_path = str(self.clip)
        job.finished_at = 1.0
        jobs._JOBS[job.job_id] = job
        self.assertEqual(self.client.get(f"/api/ng/finish/jobs/{job.job_id}/video").data, b"clip")
        r = self.client.get(f"/api/ng/finish/jobs/{job.job_id}/download")
        self.assertIn("attachment", r.headers["Content-Disposition"])


class TalkFinishTests(_Base):
    def done_job(self, **kw):
        job = talk.start_talk_job_ng("Hello.", "preset", voice="aiden", lora_name="face", **kw)
        job.stage, job.h3_job_id, job.finished_at = "done", "h1", 2.0
        return job

    def test_the_request_validates_its_settings(self):
        job = talk.start_talk_job_ng("Hi.", "preset", voice="aiden", lora_name="face", finish_pass=True,
                                     finish_params={"saturation": 1.0})
        self.assertEqual((job.finish, job.finish_params["saturation"]), (True, 1.0))
        with self.assertRaises(talk.TalkRequestError):
            talk.start_talk_job_ng("Hi.", "preset", voice="aiden", lora_name="face", finish_pass=True,
                                   finish_params={"saturation": 9})
        self.assertFalse(talk.start_talk_job_ng("Hi.", "preset", voice="aiden", lora_name="face").finish)

    def test_the_status_prefers_the_finished_clip_and_keeps_the_original(self):
        job = self.done_job()
        st = talk.job_status_ng(job.job_id)
        self.assertEqual((st["video_url"], st["finished"], st["finish_status"]), ("/api/ng/h3/jobs/h1/video", False, None))
        fj = jobs.make_job_ng("h1")
        fj.result_path, fj.finished_at = str(self.clip), 3.0
        jobs._JOBS[fj.job_id] = fj
        job.finish_job_id = fj.job_id
        st = talk.job_status_ng(job.job_id)
        self.assertTrue(st["finished"])
        self.assertEqual(st["video_url"], f"/api/ng/finish/jobs/{fj.job_id}/video")
        self.assertEqual(st["original_video_url"], "/api/ng/h3/jobs/h1/video")
        self.assertTrue(st["download_url"].endswith("/download") and fj.job_id in st["download_url"])

    def test_a_failed_finish_keeps_the_render_and_reports_why(self):
        job = talk.start_talk_job_ng("Hi.", "preset", voice="aiden", lora_name="face", finish_pass=True)
        sj = SimpleNamespace(job_id="s1", finished_at=1.0, error=None, audio_seconds=3.0,
                             audio_path=str(self.t / "s.wav"))
        Path(sj.audio_path).write_bytes(b"wav")
        hj = SimpleNamespace(job_id="h1", finished_at=2.0, error=None, mp4_path=str(self.clip))
        with mock.patch.object(talk.speech_jobs, "start_speech_job_ng", return_value=sj), \
                mock.patch.object(talk.h3_jobs, "start_h3_job_ng", return_value=hj), \
                mock.patch.object(talk.finish_jobs, "start_finish_job_ng", side_effect=eng.FinishRequestError("ComfyUI is down")), \
                mock.patch.object(talk, "_POLL_S", 0):
            talk._run_job(job)           # does not raise
        self.assertEqual(job.mp4_path, str(self.clip))
        self.assertEqual(job.finish_error, "ComfyUI is down")

    def test_finishing_a_done_clip_by_hand(self):
        job = self.done_job()
        fj = talk.start_finish_ng(job.job_id, {"saturation": 1.0})
        self.assertEqual(job.finish_job_id, fj.job_id)
        with self.assertRaisesRegex(talk.TalkRequestError, "already running"):
            talk.start_finish_ng(job.job_id)
        job.stage = "failed"
        with self.assertRaises(talk.TalkRequestError):
            talk.start_finish_ng(job.job_id)
        with self.assertRaises(LookupError):
            talk.start_finish_ng("nosuch")


if __name__ == "__main__":
    unittest.main()
