"""Unit tests for the Speech view: the engine's request checks and command line, the character-voice lookup,
the job queue (enqueue, cancel, list, status) and the HTTP routes. Nothing renders and no model is read: the
runner call is replaced.

Run with: python3 -m unittest tests.test_speechNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.speechNG as speech_routes
import speech_engineNG as eng
import speech_jobsNG as jobs


class ValidateRequestTests(unittest.TestCase):
    def check(self, **kw):
        base = dict(kind="preset", text="Hello.", voice="aiden", language="english", speed=1.0,
                    has_ref_audio=False, voice_consent=False)
        base.update(kw)
        eng.validate_request_ng(**base)

    def test_a_good_preset_passes(self):
        self.check()

    def test_empty_script_refused(self):
        with self.assertRaises(eng.SpeechRequestError):
            self.check(text="   ")

    def test_too_long_refused(self):
        with self.assertRaises(eng.SpeechRequestError):
            self.check(text="x" * (eng.SPEECH_MAX_CHARS + 1))

    def test_unknown_voice_type_language_speed_and_preset_refused(self):
        for kw in (dict(kind="robot"), dict(language="klingon"), dict(speed=3.0), dict(speed=0.1),
                   dict(voice="nobody")):
            with self.subTest(kw=kw), self.assertRaises(eng.SpeechRequestError):
                self.check(**kw)

    def test_clone_needs_a_clip_and_confirmation(self):
        with self.assertRaises(eng.SpeechRequestError):
            self.check(kind="clip", has_ref_audio=False, voice_consent=True)
        with self.assertRaises(eng.SpeechRequestError):
            self.check(kind="clip", has_ref_audio=True, voice_consent=False)
        # only the boolean True counts, not a truthy string
        with self.assertRaises(eng.SpeechRequestError):
            self.check(kind="character", has_ref_audio=True, voice_consent="true")
        self.check(kind="clip", has_ref_audio=True, voice_consent=True)


class DescribedVoiceTests(unittest.TestCase):
    """The third voice kind: a voice made from a written description (Qwen3-TTS VoiceDesign)."""
    DESIGN = "a warm, slow, older male voice with a slight rasp"

    def check(self, **kw):
        base = dict(kind="design", text="Hello there.", voice="", language="english", speed=1.0, has_ref_audio=False,
                    voice_consent=False, design_text=self.DESIGN)
        base.update(kw)
        eng.validate_request_ng(**base)

    def test_a_description_is_all_it_needs_no_clip_no_consent_no_preset(self):
        self.check()

    def test_the_description_has_a_length_window_and_whitespace_does_not_count(self):
        for bad in ("", "   ", "short", "x" * (eng.DESIGN_MAX_CHARS + 1), "a    b    c "):
            with self.subTest(bad=bad[:12]), self.assertRaisesRegex(eng.SpeechRequestError, "Describe the voice"):
                self.check(design_text=bad)
        self.check(design_text="x" * eng.DESIGN_MAX_CHARS)
        self.check(design_text="  " + "a " * 6 + "  ")

    def test_the_other_checks_still_apply(self):
        for kw in (dict(text="  "), dict(language="klingon"), dict(speed=3.0)):
            with self.subTest(kw=kw), self.assertRaises(eng.SpeechRequestError):
                self.check(**kw)

    def test_health_reports_the_model_without_making_the_engine_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d)
            runner = Path(d) / "design_run.py"
            runner.write_text("#")
            with mock.patch.object(eng, "SPEECH_PACK", pack), mock.patch.object(eng, "SPEECH_DESIGN_RUNNER", runner):
                self.assertFalse(eng.speech_health_ng()["design_ok"])
                (pack / "voicedesign").mkdir()
                (pack / "voicedesign" / "config.json").write_text("{}")
                self.assertTrue(eng.speech_health_ng()["design_ok"])
            with mock.patch.object(eng, "SPEECH_PACK", pack), mock.patch.object(eng, "SPEECH_DESIGN_RUNNER", Path(d) / "gone.py"):
                self.assertFalse(eng.speech_health_ng()["design_ok"])             # no runner, no option

    def test_the_command_uses_the_design_runner_and_the_description_not_the_preset_or_clone_flags(self):
        class Captured(Exception):
            pass

        def fake(cmd, **kw):
            raise Captured(cmd)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(eng, "speech_health_ng", lambda: {"ok": True, "design_ok": True, "missing": []}), \
                mock.patch.object(eng.subprocess, "Popen", side_effect=fake):
            with self.assertRaises(Captured) as cm:
                eng.generate_speech_ng("Hello.", Path(d), "design", design_text="  " + self.DESIGN + "  ", seed=5)
        cmd = cm.exception.args[0]
        self.assertEqual(cmd[1], str(eng.SPEECH_DESIGN_RUNNER))
        self.assertEqual(cmd[cmd.index("--design") + 1], self.DESIGN)
        self.assertEqual(cmd[cmd.index("--seed") + 1], "5")
        for flag in ("--kind", "--voice", "--ref-audio", "--ref-text", "--style"):
            self.assertNotIn(flag, cmd)
        with mock.patch.object(eng, "speech_health_ng", lambda: {"ok": True, "design_ok": False, "missing": []}):
            with self.assertRaisesRegex(RuntimeError, "voice-design model is not installed"):
                eng.generate_speech_ng("Hello.", Path(tempfile.gettempdir()) / "x", "design", design_text=self.DESIGN)


class StyleTests(unittest.TestCase):
    def test_free_text_wins_then_the_chosen_style_then_natural(self):
        self.assertEqual(eng.instruction_for("deadpan", "whispering, like a secret"), "whispering, like a secret")
        self.assertIn("deadpan", eng.instruction_for("deadpan", "").lower())
        self.assertEqual(eng.instruction_for("natural", ""), "")
        self.assertEqual(eng.instruction_for("nonsense", ""), "")
        self.assertEqual(len(eng.instruction_for("x", "a" * 500)), 300)

    def test_the_nine_presets_are_the_models_own(self):
        self.assertEqual(len(eng.PRESETS), 9)
        self.assertIn("aiden", eng.PRESET_IDS)


class VoiceSampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "alice.voice.mp3").write_bytes(b"x")
        (d / "bob.voice.WAV").write_bytes(b"x")
        (d / "alice_v2.safetensors").write_bytes(b"x")       # a face LoRA, not a voice
        (d / "evil name.voice.mp3").write_bytes(b"x")         # fails the name rule
        (d / "notes.voice.txt").write_bytes(b"x")             # not an audio type
        self.patch = mock.patch.object(eng, "SPEECH_VOICES_DIR", d)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_only_real_voice_samples_are_listed(self):
        self.assertEqual([v["id"] for v in eng.list_voice_samples_ng()], ["alice", "bob"])

    def test_lookup_is_by_name_never_by_path(self):
        self.assertIsNotNone(eng.resolve_voice_sample_ng("alice"))
        for bad in ("../alice", "/etc/passwd", "alice.voice.mp3", "", "nobody"):
            self.assertIsNone(eng.resolve_voice_sample_ng(bad), bad)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(jobs, "SPEECHGEN_DIR", self.tmp.name),
            mock.patch.object(jobs, "_ensure_worker_started", lambda: None),   # no worker: jobs stay queued
        ]
        for p in self.patches:
            p.start()
        jobs._JOBS.clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        jobs._JOBS.clear()
        self.tmp.cleanup()

    def test_enqueue_status_and_list(self):
        job = jobs.start_speech_job_ng("Hello there.", "preset", voice="ryan", style_id="dramatic", seed=3)
        st = jobs.job_status_ng(job.job_id)
        self.assertEqual((st["status"], st["kind"], st["voice"], st["cancelled"]), ("queued", "preset", "Ryan", False))
        self.assertIn("theatrical", job.style)
        listed = jobs.list_jobs_ng()
        self.assertEqual([j["job_id"] for j in listed], [job.job_id])
        self.assertNotIn("log_tail", listed[0])

    def test_a_clone_without_confirmation_is_refused_and_nothing_is_queued(self):
        with self.assertRaises(eng.SpeechRequestError):
            jobs.start_speech_job_ng("Hi.", "clip", clip_bytes=b"RIFF", clip_ext=".wav", voice_consent=False)
        self.assertEqual(jobs.list_jobs_ng(), [])

    def test_a_clip_is_written_into_the_job_folder(self):
        job = jobs.start_speech_job_ng("Hi.", "clip", clip_bytes=b"RIFFdata", clip_ext=".wav", voice_consent=True)
        self.assertTrue(Path(job.ref_audio_path).is_file())
        self.assertEqual(Path(job.ref_audio_path).parent, job.job_dir)

    def test_a_described_voice_job_keeps_its_description_and_refuses_a_missing_one(self):
        job = jobs.start_speech_job_ng("Hello there.", "design", design_text="  a warm   slow older male voice  ")
        st = jobs.job_status_ng(job.job_id)
        self.assertEqual((st["kind"], st["voice"], st["design"]), ("design", "described voice", "a warm slow older male voice"))
        with self.assertRaisesRegex(eng.SpeechRequestError, "Describe the voice"):
            jobs.start_speech_job_ng("Hello there.", "design", design_text="")

    def test_a_character_voice_is_looked_up_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "alice.voice.mp3").write_bytes(b"x")
            with mock.patch.object(eng, "SPEECH_VOICES_DIR", Path(d)):
                job = jobs.start_speech_job_ng("Hi.", "character", character="alice", voice_consent=True)
                self.assertTrue(job.ref_audio_path.endswith("alice.voice.mp3"))
                with self.assertRaises(eng.SpeechRequestError):
                    jobs.start_speech_job_ng("Hi.", "character", character="../alice", voice_consent=True)

    def test_cancelling_a_queued_job_flags_it(self):
        job = jobs.start_speech_job_ng("Hello.", "preset", voice="aiden")
        self.assertTrue(jobs.cancel_speech_job_ng(job.job_id))
        self.assertTrue(jobs.job_status_ng(job.job_id)["cancelled"])
        self.assertFalse(jobs.cancel_speech_job_ng("nosuchjob"))

    def test_trimming_the_list_never_deletes_a_finished_jobs_audio(self):
        made = []
        for i in range(jobs._MAX_JOBS_KEPT + 3):
            j = jobs.start_speech_job_ng(f"Line {i}.", "preset", voice="aiden")
            (j.job_dir / "speech.wav").write_bytes(b"wav")
            j.finished_at = 1000.0 + i
            made.append(j)
            with jobs._JOBS_LOCK:
                jobs._prune_old_jobs_locked()
        self.assertLessEqual(len(jobs._JOBS), jobs._MAX_JOBS_KEPT)
        self.assertTrue(all((j.job_dir / "speech.wav").is_file() for j in made))


class RouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(speech_routes.speechNG_bp)
        self.client = app.test_client()
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(jobs, "SPEECHGEN_DIR", self.tmp.name),
            mock.patch.object(jobs, "_ensure_worker_started", lambda: None),
        ]
        for p in self.patches:
            p.start()
        jobs._JOBS.clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        jobs._JOBS.clear()
        self.tmp.cleanup()

    def test_voices_lists_presets_styles_and_languages(self):
        d = self.client.get("/api/ng/speech/voices").get_json()
        self.assertEqual((len(d["presets"]), len(d["styles"])), (9, 8))
        self.assertIn("english", d["languages"])

    def test_generate_preset_then_poll_cancel_and_list(self):
        r = self.client.post("/api/ng/speech/generate", data={"kind": "preset", "voice": "aiden",
                                                              "text": "Hello from the route."})
        self.assertEqual(r.status_code, 202)
        jid = r.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/ng/speech/jobs/{jid}").get_json()["status"], "queued")
        self.assertEqual(len(self.client.get("/api/ng/speech/jobs").get_json()["jobs"]), 1)
        self.assertTrue(self.client.delete(f"/api/ng/speech/jobs/{jid}").get_json()["ok"])

    def test_generate_clone_without_consent_is_a_400(self):
        r = self.client.post("/api/ng/speech/generate", data={"kind": "clip", "text": "Hi.",
                                                              "file": (__import__("io").BytesIO(b"RIFF"), "c.wav")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("confirmation", r.get_json()["error"])

    def test_bad_input_is_a_400_not_a_500(self):
        for data in ({"kind": "preset", "voice": "aiden", "text": ""},
                     {"kind": "preset", "voice": "aiden", "text": "Hi.", "speed": "fast"},
                     {"kind": "robot", "text": "Hi."}):
            self.assertEqual(self.client.post("/api/ng/speech/generate", data=data).status_code, 400, data)

    def test_unknown_job_and_unfinished_audio_are_404(self):
        self.assertEqual(self.client.get("/api/ng/speech/jobs/nope").status_code, 404)
        r = self.client.post("/api/ng/speech/generate", data={"kind": "preset", "voice": "aiden", "text": "Hi."})
        jid = r.get_json()["job_id"]
        self.assertEqual(self.client.get(f"/api/ng/speech/jobs/{jid}/audio").status_code, 404)

    def test_the_generate_route_takes_a_description(self):
        r = self.client.post("/api/ng/speech/generate", data={"kind": "design", "text": "Hi there.",
                                                              "design_text": "a bright, quick young female voice"})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(self.client.get("/api/ng/speech/jobs").get_json()["jobs"][0]["design"],
                         "a bright, quick young female voice")
        bad = self.client.post("/api/ng/speech/generate", data={"kind": "design", "text": "Hi.", "design_text": "x"})
        self.assertEqual(bad.status_code, 400)

    def test_a_finished_take_can_be_kept_as_a_saved_voice_by_name_never_overwriting(self):
        job = jobs.start_speech_job_ng("Hi there.", "design", design_text="a bright, quick young female voice")
        voices = Path(self.tmp.name) / "voices"
        wav = Path(self.tmp.name) / "take.wav"
        wav.write_bytes(b"RIFFwave")
        with mock.patch.object(eng, "SPEECH_VOICES_DIR", voices):
            url = f"/api/ng/speech/jobs/{job.job_id}/save-voice"
            self.assertEqual(self.client.post(url, data={"name": "mara"}).status_code, 409)       # not finished yet
            job.finished_at, job.audio_path = 1.0, str(wav)
            for bad in ("", "bad name", "../x", "a" * 65):
                self.assertEqual(self.client.post(url, data={"name": bad}).status_code, 400, bad)
            r = self.client.post(url, data={"name": "mara"})
            self.assertEqual((r.status_code, r.get_json()["name"]), (200, "mara"))
            self.assertEqual((voices / "mara.voice.wav").read_bytes(), b"RIFFwave")
            self.assertEqual([v["id"] for v in eng.list_voice_samples_ng()], ["mara"])           # it is a usable saved voice now
            self.assertEqual(self.client.post(url, data={"name": "MARA"}).status_code, 409)       # no overwrite, any case
            self.assertEqual(self.client.post("/api/ng/speech/jobs/nope/save-voice", data={"name": "z"}).status_code, 404)

    def test_a_voice_sample_is_served_by_name_only(self):
        self.assertEqual(self.client.get("/api/ng/speech/voices/..%2Fetc/sample").status_code, 404)


if __name__ == "__main__":
    unittest.main()
