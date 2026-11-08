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


class WorkflowTests(unittest.TestCase):
    def test_the_graph_is_load_clarity_tone_write_with_the_recipe_values(self):
        wf = eng.build_workflow_ng("/c/clip.mp4", {}, 24.0, True, "pre_")
        self.assertEqual([wf[k]["class_type"] for k in "1234"],
                         [eng.NODE_LOAD, eng.NODE_CLARITY, eng.NODE_TONE, eng.NODE_COMBINE])
        self.assertEqual(wf["1"]["inputs"]["video"], "/c/clip.mp4")
        self.assertEqual((wf["2"]["inputs"]["image"], wf["3"]["inputs"]["image"], wf["4"]["inputs"]["images"]),
                         (["1", 0], ["2", 0], ["3", 0]))
        import json
        clar = json.loads(wf["2"]["inputs"]["raw_settings"])
        tone = json.loads(wf["3"]["inputs"]["raw_settings"])
        self.assertEqual((clar["clarity_strength"], clar["clarity_radius"]), (-0.2, 24.0))
        self.assertEqual((tone["mode"], tone["strength"], tone["saturation"]), ("Cineon-ish", 0.25, 0.9))
        # this pass softens: every sharpening effect in the Clarity pipeline is off
        for key in ("sharpen_amount", "smart_amount", "highpass_amount", "texture_strength"):
            self.assertEqual(clar[key], 0.0)
        self.assertEqual(wf["4"]["inputs"]["filename_prefix"], "pre_")
        self.assertEqual(wf["4"]["inputs"]["audio"], ["1", 2])

    def test_a_silent_clip_has_no_audio_link_and_the_frame_rate_follows_the_clip(self):
        wf = eng.build_workflow_ng("/c/clip.mp4", {"saturation": 1.1}, 25.0, False)
        self.assertNotIn("audio", wf["4"]["inputs"])
        self.assertEqual(wf["4"]["inputs"]["frame_rate"], 25.0)


class HealthTests(unittest.TestCase):
    def check(self, nodes_present, local=True, reachable=True):
        def fake_get(path, timeout=15.0):
            if not reachable:
                raise OSError("down")
            node = path.rsplit("/", 1)[1]
            if node not in nodes_present:
                raise eng.urllib.error.HTTPError(path, 404, "nf", {}, None)
            return {node: {}}
        with mock.patch.object(eng, "_get", fake_get), mock.patch.object(eng, "comfy_is_local_ng", lambda b=None: local), \
                mock.patch.object(eng, "_base", lambda: "http://c:8188"):
            return eng.finish_health_ng()

    def test_ready_when_local_and_all_four_nodes_exist(self):
        self.assertTrue(self.check(eng.REQUIRED_NODES)["ok"])

    def test_each_problem_is_named(self):
        self.assertIn("Orion4D_FXMax", self.check([eng.NODE_LOAD, eng.NODE_COMBINE])["problems"][0])
        self.assertIn("VideoHelperSuite", self.check([eng.NODE_CLARITY, eng.NODE_TONE])["problems"][0])
        self.assertIn("not on this Mac", self.check(eng.REQUIRED_NODES, local=False)["problems"][0])
        self.assertIn("not reachable", self.check([], reachable=False)["problems"][-1])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.src = self.t / "clip.mp4"
        self.src.write_bytes(b"src")
        self.comfy_out = self.t / "comfy_out"
        self.comfy_out.mkdir()
        self.history = {}
        self.posts = []

    def run_finish(self, history, **kw):
        def fake_post(path, body, timeout=30.0):
            self.posts.append((path, body))
            if path == "/prompt":
                prefix = body["prompt"]["4"]["inputs"]["filename_prefix"]
                for suffix in ("-audio.mp4", ".mp4", ".png"):
                    (self.comfy_out / f"{prefix}_00001{suffix}").write_bytes(b"x" + suffix.encode())
                self.prefix = prefix
                return {"prompt_id": "p1"}
            return {}
        with mock.patch.object(eng, "finish_health_ng", lambda: {"ok": True, "problems": []}), \
                mock.patch.object(eng, "probe_clip_ng", lambda p: {"fps": 24.0, "has_audio": True}), \
                mock.patch.object(eng, "_post", fake_post), \
                mock.patch.object(eng, "_get", lambda path, timeout=15.0: history(self)), \
                mock.patch.object(eng, "_POLL_S", 0):
            return eng.finish_video_ng(str(self.src), str(self.t / "out" / "f.mp4"), **kw)

    @staticmethod
    def ok_history(t):
        return {"p1": {"status": {"completed": True, "status_str": "success"},
                       "outputs": {"4": {"gifs": [{"fullpath": str(t.comfy_out / f"{t.prefix}_00001-audio.mp4")}]}}}}

    def test_success_copies_the_clip_and_clears_every_file_ComfyUI_left(self):
        result = self.run_finish(self.ok_history)
        self.assertEqual(Path(result["path"]).read_bytes(), b"x-audio.mp4")
        self.assertEqual(list(self.comfy_out.iterdir()), [])        # the .mp4 and .png beside it went too
        self.assertEqual(self.src.read_bytes(), b"src")               # the original is untouched

    def test_a_comfy_error_is_reported_with_its_message(self):
        hist = lambda t: {"p1": {"status": {"completed": False, "status_str": "error", "messages": [
            ["execution_error", {"exception_message": "CUDA out of memory"}]]}, "outputs": {}}}
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.run_finish(hist)

    def test_cancel_deletes_the_queued_run_and_interrupts_it(self):
        with self.assertRaises(eng.FinishCancelled):
            self.run_finish(lambda t: {}, should_cancel=lambda: True)
        paths = [p for p, _b in self.posts]
        self.assertIn("/queue", paths)
        self.assertEqual(self.posts[-1], ("/interrupt", {"prompt_id": "p1"}))

    def test_a_missing_clip_and_a_failed_health_check_are_request_errors(self):
        self.src.unlink()
        with self.assertRaises(eng.FinishRequestError):
            self.run_finish(self.ok_history)
        with mock.patch.object(eng, "finish_health_ng", lambda: {"ok": False, "problems": ["ComfyUI is down"]}):
            self.src.write_bytes(b"s")
            with self.assertRaisesRegex(eng.FinishRequestError, "ComfyUI is down"):
                eng.finish_video_ng(str(self.src), str(self.t / "o.mp4"))


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
