"""Unit tests for the Talk view: the request checks, the Speech -> H3 chain (both stages replaced), cancel, the
status shape and the HTTP routes. Nothing renders and no model is read.

Run with: python3 -m unittest tests.test_talkNG -v
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.talkNG as talk_routes
import speech_engineNG as speech
import talk_jobsNG as talk


class PromptTests(unittest.TestCase):
    def test_subject_scene_and_the_default(self):
        self.assertIn("a grey-haired man looks straight into the camera", talk.build_prompt_ng("a grey-haired man"))
        self.assertIn("a person looks straight", talk.build_prompt_ng(""))
        self.assertTrue(talk.build_prompt_ng("X", "Rain on the window.").endswith("Rain on the window."))
        self.assertNotIn(" his ", talk.build_prompt_ng("X"))   # neutral wording, fits any subject


class _Base(unittest.TestCase):
    def setUp(self):
        self.queue = mock.MagicMock()
        for target, name, val in (
                (talk, "_QUEUE", self.queue),
                (talk, "_ensure_worker_started", lambda: None),
                (talk.h3, "_resolve_h3_tristep_ng", lambda: ("/adapters/tristep.safetensors", None)),
                (talk.h3_lora_lib, "resolve_h3_lora_path_ng", lambda n: f"/loras/{n}.safetensors" if n == "face" else None)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        talk._JOBS.clear()
        self.addCleanup(talk._JOBS.clear)

    def start(self, **kw):
        args = dict(text="Hello there.", kind="preset", voice="aiden", lora_name="face")
        args.update(kw)
        return talk.start_talk_job_ng(**args)


class StartTests(_Base):
    def test_a_good_request_is_queued_with_its_resolved_pieces(self):
        job = self.start(subject="a man", quality="standard")
        self.assertEqual((job.stage, job.lora_path, job.quality), ("queued", "/loras/face.safetensors", "standard"))
        self.assertEqual((job.width, job.height), (768, 448))
        self.queue.put.assert_called_once_with(job.job_id)
        self.assertIn("a man", talk.job_status_ng(job.job_id)["prompt"])

    def test_refusals_queue_nothing(self):
        for kw, exc in (({"lora_name": "nobody"}, talk.TalkRequestError),
                        ({"lora_name": ""}, talk.TalkRequestError),            # no face at all
                        ({"quality": "ultra"}, talk.TalkRequestError),
                        ({"lora_strength": 9}, talk.TalkRequestError),
                        ({"text": "  "}, speech.SpeechRequestError),
                        ({"kind": "clip", "clip_bytes": b"RIFF"}, speech.SpeechRequestError)):   # no consent
            with self.subTest(kw=kw), self.assertRaises(exc):
                self.start(**kw)
        self.queue.put.assert_not_called()

    def test_a_face_picture_alone_is_enough(self):
        job = self.start(lora_name="", first_frame=b"\x89PNG", first_frame_ext=".png")
        self.assertEqual(job.lora_path, "")


def _fake_speech(seconds=3.0, error=None, tmp=None):
    wav = Path(tmp) / "speech.wav"
    wav.write_bytes(b"RIFFwavdata")
    return SimpleNamespace(job_id="s1", finished_at=1.0, error=error, audio_seconds=seconds, audio_path=str(wav))


class ChainTests(_Base):
    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def run_chain(self, speech_job, h3_job=None, **kw):
        job = self.start(**kw)
        with mock.patch.object(talk.speech_jobs, "start_speech_job_ng", return_value=speech_job) as sp, \
             mock.patch.object(talk.h3_jobs, "start_h3_job_ng", return_value=h3_job) as h3, \
             mock.patch.object(talk, "_POLL_S", 0):
            try:
                talk._run_job(job)
            finally:
                self.sp, self.h3 = sp, h3
        return job

    def test_speech_then_an_audio_drive_render(self):
        h3_job = SimpleNamespace(job_id="h1", finished_at=2.0, error=None, mp4_path="/x/h3.mp4")
        job = self.run_chain(_fake_speech(tmp=self.tmpdir.name), h3_job, first_frame=b"IMG", first_frame_ext=".png")
        args, kw = self.h3.call_args
        self.assertEqual(args[:4], (b"IMG", ".png", job.prompt, talk.TALK_CLIP_S))
        self.assertEqual(kw["audio_bytes"], b"RIFFwavdata")
        self.assertEqual(kw["loras"], [{"name": "face", "path": "/loras/face.safetensors", "strength": 1.0}])
        self.assertEqual((kw["model"], kw["steps"]), ("h3q8", talk.TALK_STEPS))
        self.assertEqual((job.speech_job_id, job.h3_job_id, job.mp4_path), ("s1", "h1", "/x/h3.mp4"))

    def test_speech_too_long_for_the_window_is_refused_before_any_render(self):
        with self.assertRaisesRegex(talk.TalkRequestError, "shorten the script"):
            self.run_chain(_fake_speech(seconds=7.2, tmp=self.tmpdir.name), None)
        self.h3.assert_not_called()

    def test_a_failed_speech_stage_stops_the_chain(self):
        with self.assertRaisesRegex(RuntimeError, "speech: boom"):
            self.run_chain(_fake_speech(error="boom", tmp=self.tmpdir.name), None)
        self.h3.assert_not_called()

    def test_a_failed_render_reports_its_error(self):
        h3_job = SimpleNamespace(job_id="h1", finished_at=2.0, error="OOM", mp4_path=None)
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            self.run_chain(_fake_speech(tmp=self.tmpdir.name), h3_job)

    def test_cancel_during_a_stage_cancels_it_and_ends_the_job(self):
        stage = SimpleNamespace(job_id="s1", finished_at=None, error=None, audio_seconds=3.0, audio_path="")
        job = self.start()
        job.cancelled = True

        def finish(_id):
            stage.finished_at = 1.0
            return True
        with mock.patch.object(talk.speech_jobs, "start_speech_job_ng", return_value=stage), \
             mock.patch.object(talk.speech_jobs, "cancel_speech_job_ng", side_effect=finish) as cancel, \
             mock.patch.object(talk, "_POLL_S", 0):
            with self.assertRaises(talk.TalkJobCancelled):
                talk._run_job(job)
        cancel.assert_called_once_with("s1")


class FastTests(ChainTests):
    def test_off_by_default_and_the_render_is_a_full_one(self):
        h3_job = SimpleNamespace(job_id="h1", finished_at=2.0, error=None, mp4_path="/x/h3.mp4")
        job = self.run_chain(_fake_speech(tmp=self.tmpdir.name), h3_job)
        self.assertFalse(job.fast)
        self.assertFalse(self.h3.call_args.kwargs["tristep"])
        self.assertFalse(talk.job_status_ng(job.job_id)["fast"])

    def test_fast_asks_the_h3_job_for_the_three_step_adapter(self):
        h3_job = SimpleNamespace(job_id="h1", finished_at=2.0, error=None, mp4_path="/x/h3.mp4")
        job = self.run_chain(_fake_speech(tmp=self.tmpdir.name), h3_job, fast=True)
        self.assertTrue(job.fast)
        self.assertTrue(self.h3.call_args.kwargs["tristep"])
        self.assertTrue(talk.job_status_ng(job.job_id)["fast"])

    def test_fast_without_the_adapter_is_refused_before_anything_is_queued(self):
        with mock.patch.object(talk.h3, "_resolve_h3_tristep_ng", lambda: None):
            with self.assertRaisesRegex(talk.TalkRequestError, "3-step adapter"):
                self.start(fast=True)
            self.start()       # a full-quality job does not need it
        self.assertEqual(self.queue.put.call_count, 1)


class StatusTests(_Base):
    def test_stages_map_to_the_queue_statuses(self):
        job = self.start()
        for stage, status in (("queued", "queued"), ("speaking", "rendering"), ("rendering", "rendering"),
                              ("done", "completed"), ("failed", "failed"), ("cancelled", "failed")):
            job.stage = stage
            self.assertEqual(talk.job_status_ng(job.job_id)["status"], status, stage)

    def test_urls_only_when_there_is_something_to_play(self):
        job = self.start()
        job.speech_job_id, job.h3_job_id = "s1", "h1"
        job.stage = "speaking"
        st = talk.job_status_ng(job.job_id)
        self.assertEqual((st["audio_url"], st["video_url"]), (None, None))
        job.stage = "rendering"
        self.assertTrue(talk.job_status_ng(job.job_id)["audio_url"].endswith("/s1/audio"))
        job.stage = "done"
        st = talk.job_status_ng(job.job_id)
        self.assertTrue(st["video_url"].endswith("/h1/video") and st["download_url"].endswith("/h1/download"))

    def test_progress_reads_the_h3_log(self):
        job = self.start()
        job.stage, job.h3_job_id = "rendering", "h1"
        with mock.patch.object(talk.h3_jobs, "job_status_ng",
                               return_value={"status": "rendering", "log_tail": ["== joint_denoise ==", "step 4/15: 72.0s"]}):
            self.assertEqual(talk.job_status_ng(job.job_id)["progress"],
                             {"phase": "joint_denoise", "step": 4, "steps": 15})

    def test_cancel_flags_a_live_job_only(self):
        job = self.start()
        self.assertTrue(talk.cancel_talk_job_ng(job.job_id))
        job.finished_at = 1.0
        self.assertFalse(talk.cancel_talk_job_ng(job.job_id))
        self.assertFalse(talk.cancel_talk_job_ng("nosuch"))
        self.assertEqual([j["job_id"] for j in talk.list_jobs_ng()], [job.job_id])


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(talk_routes.talkNG_bp)
        self.client = app.test_client()

    def test_generate_poll_list_cancel(self):
        r = self.client.post("/api/ng/talk/generate", data={"text": "Hello.", "voice": "aiden", "lora": "face"})
        self.assertEqual(r.status_code, 202)
        jid = r.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/ng/talk/jobs/{jid}").get_json()["status"], "queued")
        self.assertEqual(len(self.client.get("/api/ng/talk/jobs").get_json()["jobs"]), 1)
        self.assertTrue(self.client.delete(f"/api/ng/talk/jobs/{jid}").get_json()["ok"])

    def test_the_fast_field_is_read(self):
        for data, expected in (({"fast": "true"}, True), ({}, False)):
            r = self.client.post("/api/ng/talk/generate", data={"text": "Hi.", "voice": "aiden", "lora": "face", **data})
            self.assertEqual(talk.get_job_ng(r.get_json()["job_id"]).fast, expected)

    def test_the_face_picture_and_voice_clip_are_read(self):
        r = self.client.post("/api/ng/talk/generate", data={
            "text": "Hi.", "kind": "clip", "consent": "true",
            "file": (io.BytesIO(b"RIFF"), "v.wav"), "face": (io.BytesIO(b"\x89PNG"), "f.PNG")},
            content_type="multipart/form-data")
        self.assertEqual(r.status_code, 202)
        job = talk.get_job_ng(r.get_json()["job_id"])
        self.assertEqual((job.first_frame, job.first_frame_ext, job.clip_ext), (b"\x89PNG", ".png", ".wav"))

    def test_bad_input_is_a_400_not_a_500(self):
        for data in ({"text": "Hi.", "voice": "aiden"},                           # no face
                     {"text": "Hi.", "voice": "aiden", "lora": "nobody"},
                     {"text": "", "voice": "aiden", "lora": "face"},
                     {"text": "Hi.", "voice": "aiden", "lora": "face", "speed": "fast"},
                     {"text": "Hi.", "kind": "clip", "lora": "face"}):
            self.assertEqual(self.client.post("/api/ng/talk/generate", data=data).status_code, 400, data)

    def test_unknown_job_is_404_and_status_reports_readiness(self):
        self.assertEqual(self.client.get("/api/ng/talk/jobs/nope").status_code, 404)
        d = self.client.get("/api/ng/talk/status").get_json()
        self.assertEqual((d["clip_s"], d["qualities"]), (5.0, ["draft", "standard"]))
        self.assertIn("ok", d)


if __name__ == "__main__":
    unittest.main()
