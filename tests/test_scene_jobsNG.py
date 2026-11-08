"""Unit tests for the Scene job chain: request checks, one speech pass per character cut into per-line audio, one H3
clip per line (face, setting, audio), the exposure-held join, cancel, and failures. Speech and H3 are replaced by fakes;
the finish and join steps run for real on tiny generated clips.

Run with: python3 -m unittest tests.test_scene_jobsNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finish_engineNG as finish
import scene_jobsNG as scene
import scene_voiceNG as sv
from tests.test_sceneNG import speech as synth, SR

SCRIPT = ("KNIGHT [on the battlements]: The wall stands. It has for years.\n"
          "WOMAN [in the great hall]: Come in out of the wind.\n"
          "KNIGHT: I count the stones.")


def tiny_clip(path, seconds=5):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=24:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)


def probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def character(name, **kw):
    base = {"name": name, "subject": f"the {name.lower()}", "lora": "face", "voice": {"kind": "preset", "voice": "aiden"}}
    base.update(kw)
    return base


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.clip = self.t / "h3.mp4"
        tiny_clip(self.clip)
        self.speech_calls, self.h3_calls = [], []
        # one take per speaker, with a real pause between sentences (0.3 s inside a line, 0.5 s between lines)
        self.takes = takes = {"The wall stands. It has for years. I count the stones.":
                 [("s", .1, 0), ("v", .8, .2), ("s", .3, 0), ("v", .9, .2), ("s", .5, 0), ("v", 1.0, .2), ("s", .1, 0)],
                 "Come in out of the wind.": [("s", .1, 0), ("v", 1.5, .12), ("s", .1, 0)]}

        def fake_speech(text, kind, **kw):
            self.speech_calls.append((text, kind, kw))
            path = self.t / f"take{len(self.speech_calls)}.wav"
            sv.write_wav_ng(path, synth(self.takes[text]), SR)
            return SimpleNamespace(job_id=f"s{len(self.speech_calls)}", finished_at=1.0, error=None, audio_path=str(path))

        def fake_h3(first, ext, prompt, duration, **kw):
            self.h3_calls.append(dict(first=first, ext=ext, prompt=prompt, duration=duration, **kw))
            return SimpleNamespace(job_id=f"h{len(self.h3_calls)}", finished_at=2.0, error=None, mp4_path=str(self.clip))

        for target, name, val in (
                (scene, "SCENEGEN_DIR", str(self.t / "scenes")),
                (scene, "_QUEUE", mock.MagicMock()),
                (scene, "_ensure_worker_started", lambda: None),
                (scene.speech_jobs, "start_speech_job_ng", fake_speech),
                (scene.h3_jobs, "start_h3_job_ng", fake_h3),
                (scene.h3, "_resolve_h3_tristep_ng", lambda: ("/adapters/tristep.safetensors", None)),
                (scene.h3_lora_lib, "resolve_h3_lora_path_ng", lambda n: f"/loras/{n}.safetensors" if n == "face" else None),
                (scene.talk, "_POLL_S", 0)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        scene._JOBS.clear()
        self.addCleanup(scene._JOBS.clear)

    def start(self, script=SCRIPT, characters=None, **kw):
        chars = characters or [
            character("Knight", looks=[{"name": "battlements", "bytes": b"PNGK", "ext": ".png"},
                                       {"name": "hall", "bytes": b"PNGKH", "ext": ".png"}], default_look="battlements"),
            character("Woman", lora="")]
        chars[1].setdefault("looks", [{"name": "hall", "bytes": b"PNGW", "ext": ".png"}])
        return scene.start_scene_job_ng(script, chars, **kw)


class RequestTests(_Base):
    def test_a_good_request_is_queued_and_its_script_saved(self):
        job = self.start(seed=5)
        self.assertEqual((job.stage, len(job.clips), job.fast, job.match_exposure), ("queued", 3, True, True))
        self.assertTrue((job.job_dir / "script.txt").is_file())
        self.assertEqual([c.setting for c in job.clips], ["on the battlements", "in the great hall", "on the battlements"])

    def test_refusals_in_plain_words_and_nothing_queued(self):
        bad = [
            ({"script": "GHOST: Boo."}, "no face and voice"),
            ({"quality": "ultra"}, "quality"),
            ({"characters": [character("Knight", lora="", looks=[]), character("Woman")]}, "no face"),
            ({"characters": [character("Knight"), character("Knight"), character("Woman")]}, "listed twice"),
            ({"characters": [character("Knight", voice={"kind": "clip"}), character("Woman")]}, "voice clip"),
            ({"characters": [character("Knight", voice={"kind": "clip", "clip_bytes": b"RIFF"}), character("Woman")]},
             "confirmation"),
            ({"characters": [character("Knight", lora="nope"), character("Woman")]}, "unknown or unusable LoRA"),
        ]
        for kw, why in bad:
            with self.subTest(why=why), self.assertRaisesRegex(scene.SceneRequestError, why):
                self.start(**kw)
        with self.assertRaisesRegex(scene.SceneRequestError, "at least one character"):
            scene.start_scene_job_ng(SCRIPT, [])
        with mock.patch.object(scene.h3, "_resolve_h3_tristep_ng", lambda: None):
            with self.assertRaisesRegex(scene.SceneRequestError, "3-step adapter"):
                self.start()
            self.start(fast=False)
        with mock.patch.object(scene.finish, "finish_health_ng", lambda: {"ok": False, "problems": ["no ffmpeg"]}):
            with self.assertRaisesRegex(scene.SceneRequestError, "no ffmpeg"):
                self.start()
            self.start(match_exposure=False)
        self.assertEqual(len(scene._JOBS), 2)


class RunTests(_Base):
    def run_scene(self, **kw):
        job = self.start(seed=11, **kw)
        scene._run_scene(job)
        return job

    def test_one_speech_pass_per_character_cut_into_lines_then_one_clip_per_line_and_a_joined_scene(self):
        job = self.run_scene()
        self.assertEqual(len(self.speech_calls), 2)                       # one pass per character, not per line
        self.assertEqual(self.speech_calls[0][0], "The wall stands. It has for years. I count the stones.")
        self.assertEqual(len(self.h3_calls), 3)
        # script order, each clip with its own speaker's audio, face and setting
        self.assertIn("Setting: on the battlements.", self.h3_calls[0]["prompt"])
        self.assertIn("Setting: in the great hall.", self.h3_calls[1]["prompt"])
        self.assertIn("the knight", self.h3_calls[0]["prompt"])
        self.assertEqual([c["first"] for c in self.h3_calls], [b"PNGK", b"PNGW", b"PNGK"])   # looks picked by setting
        self.assertEqual([c["seed"] for c in self.h3_calls], [11, 12, 13])
        self.assertTrue(all(c["tristep"] and c["audio_bytes"][:4] == b"RIFF" for c in self.h3_calls))
        self.assertEqual(self.h3_calls[0]["loras"][0]["name"], "face")
        self.assertEqual(self.h3_calls[1]["loras"], [])                    # the woman has no LoRA, only her picture
        # the knight's first line holds two sentences: its audio keeps the pause inside it
        self.assertGreater(job.clips[0].audio_seconds, job.clips[2].audio_seconds + 0.8)
        out = Path(job.output_path)
        self.assertTrue(out.is_file())
        expect = sum(min(5.0, c.audio_seconds + scene.TAIL_S) for c in job.clips)
        self.assertAlmostEqual(probe_duration(out), expect, delta=0.25)
        self.assertTrue((job.job_dir / "scene.json").is_file())
        self.assertEqual([c.status for c in job.clips], ["done"] * 3)

    def test_the_scene_is_held_to_one_exposure_per_clip_and_the_full_look_is_optional(self):
        calls = []
        real = finish.finish_video_ng

        def spy(src, out, params=None, **kw):
            calls.append(dict(params or {}))
            return real(src, out, params, **kw)
        with mock.patch.object(scene.finish, "finish_video_ng", spy):
            self.run_scene()
            self.run_scene(finish_pass=True, finish_params={"saturation": 1.0})
        self.assertEqual(calls[0], {"clarity_strength": 0.0, "tone_strength": 0.0, "saturation": 1.0, "hold": 1.0})
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[3]["saturation"], 1.0)
        self.assertEqual(calls[3]["clarity_strength"], finish.DEFAULTS["clarity_strength"])

    def test_without_matching_or_finishing_the_clips_are_joined_as_they_came(self):
        with mock.patch.object(scene.finish, "finish_video_ng") as fin:
            job = self.run_scene(match_exposure=False)
        fin.assert_not_called()
        self.assertTrue(Path(job.output_path).is_file())

    def test_a_failed_finish_keeps_the_render_and_says_why(self):
        with mock.patch.object(scene.finish, "finish_video_ng", side_effect=RuntimeError("boom")):
            job = self.run_scene()
        self.assertTrue(Path(job.output_path).is_file())
        self.assertTrue(all(c.finish_error == "boom" for c in job.clips))

    def test_an_h3_failure_names_the_line_and_keeps_the_clips_already_made(self):
        n = {"i": 0}
        real = scene.h3_jobs.start_h3_job_ng

        def flaky(*a, **kw):
            n["i"] += 1
            job = real(*a, **kw)
            if n["i"] == 2:
                job.error = "out of memory"
            return job
        with mock.patch.object(scene.h3_jobs, "start_h3_job_ng", flaky):
            job = self.start()
            with self.assertRaisesRegex(RuntimeError, r"shot 2 \(WOMAN\): out of memory"):
                scene._run_scene(job)
        self.assertEqual([c.status for c in job.clips], ["done", "failed", "waiting"])
        self.assertIsNone(job.output_path)

    def test_a_voice_that_cannot_be_cut_stops_before_any_render(self):
        run_together = [("s", .1, 0), ("v", 3.5, .2), ("s", .1, 0)]
        with mock.patch.object(scene.speech_jobs, "start_speech_job_ng", side_effect=lambda text, kind, **kw: (
                sv.write_wav_ng(self.t / "x.wav", synth(run_together), SR) or
                SimpleNamespace(job_id="s", finished_at=1.0, error=None, audio_path=str(self.t / "x.wav")))):
            job = self.start()
            with self.assertRaisesRegex(RuntimeError, "full stop"):
                scene._run_scene(job)
        self.assertEqual(len(self.h3_calls), 0)

    def test_cancel_during_a_clip_cancels_it_and_ends_the_scene(self):
        job = self.start()
        job.cancelled = True
        with mock.patch.object(scene.h3_jobs, "cancel_h3_job_ng", return_value=True) as cancel, \
                mock.patch.object(scene.h3_jobs, "start_h3_job_ng", return_value=SimpleNamespace(
                    job_id="h1", finished_at=None, error=None, mp4_path=None)) as start:
            with mock.patch.object(scene, "_voice_stage", return_value={
                    c.index: (np.zeros(SR, dtype=np.float32) + .1, SR) for c in job.clips}):
                def end(_):                       # the cancelled job winds down once told
                    start.return_value.finished_at = 1.0
                    return True
                cancel.side_effect = end
                with self.assertRaises(scene.SceneCancelled):
                    scene._run_scene(job)
        cancel.assert_called_once_with("h1")
        self.assertEqual(job.clips[0].error, "cancelled")


class StatusTests(_Base):
    def test_status_shape_urls_and_cancel(self):
        job = self.start()
        st = scene.job_status_ng(job.job_id)
        self.assertEqual((st["status"], st["stage"], st["lines"], st["video_url"]), ("queued", "queued", 3, None))
        self.assertEqual([c["speaker"] for c in st["clips"]], ["KNIGHT", "WOMAN", "KNIGHT"])
        scene._run_scene(job)
        job.stage = "done"
        st = scene.job_status_ng(job.job_id)
        self.assertTrue(st["video_url"].endswith("/video") and st["download_url"].endswith("/download"))
        self.assertTrue(all(c["video_url"] for c in st["clips"]))
        live = self.start()
        self.assertTrue(scene.cancel_scene_job_ng(live.job_id))
        live.finished_at = 1.0
        self.assertFalse(scene.cancel_scene_job_ng(live.job_id))
        self.assertEqual(len(scene.list_jobs_ng()), 2)
        with self.assertRaises(LookupError):
            scene.job_status_ng("nope")


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        import io
        from flask import Flask
        import routes.sceneNG as scene_routes
        self.io = io
        app = Flask(__name__)
        app.register_blueprint(scene_routes.sceneNG_bp)
        self.client = app.test_client()

    def post(self, **over):
        import json
        chars = [
            {"name": "Knight", "subject": "the knight", "lora": "face",
             "looks": [{"name": "battlements", "file": "look_0_0"}], "default_look": "battlements",
             "voice": {"kind": "preset", "voice": "aiden"}},
            {"name": "Woman", "subject": "the woman",
             "looks": [{"name": "hall", "file": "look_1_0"}],
             "voice": {"kind": "clip", "clip_file": "voice_1", "consent": True}}]
        data = {"script": SCRIPT, "characters": json.dumps(chars), "quality": "draft", "seed": "11",
                "look_0_0": (self.io.BytesIO(b"PNGK"), "k.PNG"), "look_1_0": (self.io.BytesIO(b"PNGW"), "w.jpg"),
                "voice_1": (self.io.BytesIO(b"RIFFvoice"), "v.wav")}
        data.update(over)
        return self.client.post("/api/ng/scene/generate", data=data, content_type="multipart/form-data")

    def test_generate_reads_the_uploads_and_the_defaults_then_poll_list_cancel(self):
        r = self.post()
        self.assertEqual(r.status_code, 202, r.get_json())
        job = scene.get_job_ng(r.get_json()["job_id"])
        k, w = job.characters["knight"], job.characters["woman"]
        self.assertEqual((k.looks["battlements"][:2], w.looks["hall"][:2]), ((b"PNGK", ".png"), (b"PNGW", ".jpg")))
        self.assertEqual((w.voice["clip_bytes"], w.voice["clip_ext"]), (b"RIFFvoice", ".wav"))
        self.assertEqual((job.fast, job.match_exposure, job.finish, job.seed), (True, True, False, 11))   # fast and match default on
        self.assertFalse(job.ambience)                                                                   # ambience is opt-in
        jid = job.job_id
        self.assertEqual(self.client.get(f"/api/ng/scene/jobs/{jid}").get_json()["status"], "queued")
        self.assertEqual(len(self.client.get("/api/ng/scene/jobs").get_json()["jobs"]), 1)
        self.assertEqual(self.client.get(f"/api/ng/scene/jobs/{jid}/video").status_code, 404)
        self.assertTrue(self.client.delete(f"/api/ng/scene/jobs/{jid}").get_json()["ok"])

    def test_options_off_and_errors_are_400s_not_500s(self):
        job = scene.get_job_ng(self.post(fast="false", match_exposure="false", finish="true", saturation="1.0").get_json()["job_id"])
        self.assertEqual((job.fast, job.match_exposure, job.finish, job.finish_params["saturation"]), (False, False, True, 1.0))
        for over, why in (({"script": "GHOST: hi."}, "no face and voice"), ({"characters": "not json"}, "JSON"),
                          ({"characters": "{}"}, "list"), ({"saturation": "9", "finish": "true"}, "between"),
                          ({"saturation": "abc"}, "")):
            r = self.post(**over)
            self.assertEqual(r.status_code, 400, (over, r.get_json()))
            self.assertIn(why, r.get_json()["error"])
        missing = self.post(look_0_0=(self.io.BytesIO(b""), "k.png"))
        self.assertEqual(missing.status_code, 400)

    def test_an_empty_option_means_its_default(self):
        job = scene.get_job_ng(self.post(fast="", match_exposure="").get_json()["job_id"])
        self.assertEqual((job.fast, job.match_exposure), (True, True))

    def test_a_missing_upload_and_unknown_job_and_status(self):
        import json
        r = self.client.post("/api/ng/scene/generate", data={"script": SCRIPT, "characters": json.dumps(
            [{"name": "Knight", "looks": [{"name": "x", "file": "nope"}]}])}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("was not uploaded", r.get_json()["error"])
        self.assertEqual(self.client.get("/api/ng/scene/jobs/nope").status_code, 404)
        d = self.client.get("/api/ng/scene/status").get_json()
        self.assertEqual((d["clip_s"], d["qualities"], d["max_lines"]), (5.0, ["draft", "standard"], 12))
        self.assertIn("ok", d)

    def test_a_finished_scene_is_served(self):
        job = self.start()
        scene._run_scene(job)
        job.stage = "done"
        r = self.client.get(f"/api/ng/scene/jobs/{job.job_id}/video")
        self.assertEqual((r.status_code, r.mimetype), (200, "video/mp4"))
        self.assertIn("attachment", self.client.get(f"/api/ng/scene/jobs/{job.job_id}/download").headers["Content-Disposition"])


def solid_png(color, size=(200, 120)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class AmbienceTests(_Base):
    def test_one_bed_per_setting_is_made_first_then_mixed_under_the_clips_of_that_setting(self):
        job = self.start(seed=11, ambience_pass=True)
        scene._run_scene(job)
        beds = [c for c in self.h3_calls if c["first"] is None]
        self.assertEqual(len(beds), 2)                                     # two settings: battlements and great hall
        self.assertIn("on the battlements", beds[0]["prompt"])
        self.assertIn("in the great hall", beds[1]["prompt"])
        self.assertTrue(all("no speech" in c["prompt"] and c["loras"] == [] for c in beds))
        self.assertEqual(set(job.beds), {"on the battlements", "in the great hall"})
        self.assertEqual(job.bed_errors, {})
        self.assertEqual(self.h3_calls[0]["first"], None)                  # beds come before the dialogue clips
        self.assertEqual(len(self.h3_calls), 5)
        self.assertTrue(Path(job.output_path).is_file())
        self.assertTrue(all((job.job_dir / f"shot{i:02d}_ambience.mp4").is_file() for i in (1, 2, 3)))
        self.assertTrue(scene.job_status_ng(job.job_id)["ambience"])

    def test_off_by_default_and_a_failed_bed_leaves_only_that_setting_dry(self):
        job = self.start()
        scene._run_scene(job)
        self.assertEqual((len(self.h3_calls), job.beds), (3, {}))
        self.h3_calls.clear()
        real = scene.h3_jobs.start_h3_job_ng

        def flaky(first, *a, **kw):
            j = real(first, *a, **kw)
            if first is None and "great hall" in a[1]:
                j.error = "no sound"
            return j
        with mock.patch.object(scene.h3_jobs, "start_h3_job_ng", flaky):
            job2 = self.start(ambience_pass=True)
            scene._run_scene(job2)
        self.assertEqual(set(job2.beds), {"on the battlements"})
        self.assertIn("no sound", job2.bed_errors["in the great hall"])
        self.assertTrue(Path(job2.output_path).is_file())                  # the scene still came out
        self.assertTrue((job2.job_dir / "shot01_ambience.mp4").is_file())
        self.assertFalse((job2.job_dir / "shot02_ambience.mp4").exists())


class SharedFrameTests(_Base):
    SHARED = ("KNIGHT [in the tower]: Where are you now?\n"
              "& WOMAN [in the kitchen]: Right here, the pot boils.\n"
              "KNIGHT: Good.")

    def setUp(self):
        super().setUp()
        self.takes["Where are you now? Good."] = [("s", .1, 0), ("v", 1.0, .2), ("s", .5, 0), ("v", .6, .2), ("s", .1, 0)]
        self.takes["Right here, the pot boils."] = [("s", .1, 0), ("v", 1.4, .2), ("s", .1, 0)]

    def chars(self, woman_pictures=True, loras=False):
        knight = character("Knight", looks=[{"name": "tower", "bytes": solid_png((200, 0, 0)), "ext": ".png"}])
        woman = character("Woman", lora="", looks=[{"name": "kitchen", "bytes": solid_png((0, 0, 200)), "ext": ".png"}]
                          if woman_pictures else [])
        if loras or not woman_pictures:        # without pictures she must have a LoRA to count as having a face at all
            woman["lora"] = "face"
        return [knight, woman]

    def test_a_shot_with_two_speakers_is_one_clip_with_a_split_screen_first_frame_and_both_voices(self):
        import io
        import wave
        from PIL import Image
        job = scene.start_scene_job_ng(self.SHARED, self.chars(loras=True), seed=4)
        self.assertEqual([c.shared for c in job.clips], [True, False])
        self.assertEqual(job.clips[0].speaker, "KNIGHT + WOMAN")
        scene._run_scene(job)
        self.assertEqual(len(self.h3_calls), 2)                              # three lines, two shots
        call = self.h3_calls[0]
        im = Image.open(io.BytesIO(call["first"])).convert("RGB")
        self.assertEqual(im.size, (job.width, job.height))
        left, right = im.getpixel((10, job.height // 2)), im.getpixel((job.width - 10, job.height // 2))
        self.assertGreater(left[0], 150)                                     # the knight's picture on the left
        self.assertGreater(right[2], 150)                                    # the woman's on the right
        p = call["prompt"]
        for part in ("vertical split-screen", "on the left half", "(S1), in the tower", "<d>[English] Where are you now?</d>",
                     "On the right half", "(S2), in the kitchen", "<d>[English] Right here, the pot boils.</d>"):
            self.assertIn(part, p)
        self.assertEqual({l["name"] for l in call["loras"]}, {"face"})       # both have the same LoRA: listed once
        with wave.open(io.BytesIO(call["audio_bytes"])) as w:
            secs = w.getnframes() / w.getframerate()
        a, b = job.clips[0].audio_seconds, 0
        self.assertAlmostEqual(secs, a, delta=0.02)
        self.assertGreater(a, 1.0 + scene.SHARED_GAP_S + 1.4)               # both voices plus the breath between
        self.assertEqual(self.h3_calls[1]["first"], solid_png((200, 0, 0)))  # the next shot is the knight alone again
        self.assertTrue(Path(job.output_path).is_file())
        self.assertEqual(job.clips[0].line_indexes, [0, 1])

    def test_two_voices_that_do_not_fit_one_window_stop_before_any_render(self):
        self.takes["Right here, the pot boils."] = [("s", .1, 0), ("v", 4.6, .12), ("s", .1, 0)]
        job = scene.start_scene_job_ng(self.SHARED, self.chars())
        with self.assertRaisesRegex(RuntimeError, r"shared frame KNIGHT \+ WOMAN runs .* shorten"):
            scene._run_scene(job)
        self.assertEqual(self.h3_calls, [])

    def test_a_shared_frame_needs_a_picture_for_both_and_the_status_reports_shots(self):
        with self.assertRaisesRegex(scene.SceneRequestError, "Woman needs one|WOMAN needs one"):
            scene.start_scene_job_ng(self.SHARED, self.chars(woman_pictures=False))
        job = scene.start_scene_job_ng(self.SHARED, self.chars())
        st = scene.job_status_ng(job.job_id)
        self.assertEqual((st["lines"], st["shots"]), (3, 2))
        self.assertEqual([c["shared"] for c in st["clips"]], [True, False])
        self.assertEqual(st["clips"][0]["text"], "Where are you now? / Right here, the pot boils.")


class JoinTests(unittest.TestCase):
    def test_parts_are_trimmed_and_joined_in_order_with_audio(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.mp4", Path(d) / "b.mp4"
            tiny_clip(a)
            tiny_clip(b)
            out = Path(d) / "o.mp4"
            scene.join_clips_ng([(a, 2.0), (b, 3.0)], str(out))
            self.assertAlmostEqual(probe_duration(out), 5.0, delta=0.15)
            streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
                                     capture_output=True, text=True).stdout.split()
            self.assertEqual(sorted(streams), ["audio", "video"])
            with self.assertRaises(ValueError):
                scene.join_clips_ng([], str(out))


if __name__ == "__main__":
    unittest.main()
