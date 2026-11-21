"""Pictures kept in EVERY window of a chained clip ("faces" reference mode: the first-frame model with the runner's
`--reference`, h3_referencesNG.py): what is refused, the runner arguments, the sentence that names the pictures in each
window's prompt, the engine's command for a chain, the job queue and the /generate field. Nothing renders.

Run with: venv/bin/python3 -m unittest tests.test_h3_face_referencesNG -v
"""

import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_jobsNG as jobs
import h3_referencesNG as refs
import routes.h3NG as h3_routes
from tests.test_h3_tristepNG import _Base, RUNNER

FACES_RUNNER = RUNNER + "--reference IMAGE\n--reference-size PIXELS\n"
DESC = ("integrated_multimodal_description: [Shot 1] The fairy looks up.\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A")


class _FacesBase(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(FACES_RUNNER)
        self.pic = self.tmp / "fairy.png"
        self.pic.write_bytes(b"png")
        self.pic2 = self.tmp / "mage.png"
        self.pic2.write_bytes(b"png")
        self.clip = self.tmp / "scene.mp4"
        self.clip.write_bytes(b"mp4")

    def _validate(self, paths=None, size="match", **kw):
        args = dict(model="h3q8", image_path=None, runner=str(self.runner), mode="faces")
        args.update(kw)
        return refs.validate_references_ng([str(self.pic)] if paths is None else paths, size, **args)


class Validation(_FacesBase):
    def test_pictures_pass_in_order_on_either_first_frame_model_and_in_a_chain(self):
        for model in ("h3q8", "h3"):
            self.assertEqual(self._validate([str(self.pic), str(self.pic2)], model=model, chain_windows=3),
                             [str(self.pic), str(self.pic2)])

    def test_a_first_frame_picture_may_come_with_them(self):
        self.assertEqual(self._validate(image_path="keyframe"), [str(self.pic)])

    def test_refusals(self):
        for kw, text in (
                (dict(model="h3ref"), "first-frame model"),
                (dict(last_frame_path="last"), "last or timed"),
                (dict(keyframes=[("a", "1")]), "last or timed"),
                (dict(extend_from="x"), "extending"),
                (dict(refmods=[{"path": "r"}]), "RefMod"),
                (dict(turbo=True), "Turbo"),
                (dict(v13=True), "Turbo"),
                (dict(paths=[str(self.clip)]), "only pictures"),
                (dict(paths=[str(self.pic)] * 4), "at most 3"),
                (dict(size="huge"), "size must be"),
                (dict(mode="faced"), "mode must be")):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, text):
                self._validate(**kw)

    def test_a_runner_without_the_flag_says_so(self):
        self.runner.write_text(RUNNER)
        with self.assertRaisesRegex(ValueError, "no --reference"):
            self._validate()

    def test_a_missing_file_is_named_unless_the_check_is_off(self):
        with self.assertRaises(FileNotFoundError):
            self._validate([str(self.tmp / "gone.png")])
        self.assertEqual(self._validate([str(self.tmp / "gone.png")], check_files=False), [str(self.tmp / "gone.png")])

    def test_the_default_mode_still_refuses_a_chain_and_the_wrong_model(self):
        self.runner.write_text(FACES_RUNNER + "--ref2va-reference PICTURE|CLIP\n")
        with self.assertRaisesRegex(ValueError, "single 5 s window"):
            refs.validate_references_ng([str(self.pic)], "match", model="h3ref", image_path=None, chain_windows=2,
                                        runner=str(self.runner))
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
            refs.validate_references_ng([str(self.pic)], "match", model="h3q8", image_path=None,
                                        runner=str(self.runner))


class Argv(unittest.TestCase):
    def test_one_flag_per_picture_in_order_then_the_short_edge(self):
        self.assertEqual(refs.reference_argv_ng(["a.png", "b.png"], "match", "faces"),
                         ["--reference", "a.png", "--reference", "b.png", "--reference-size", "1024"])
        self.assertEqual(refs.reference_argv_ng(["a.png"], "max", "faces"),
                         ["--reference", "a.png", "--reference-size", "2048"])
        self.assertEqual(refs.reference_argv_ng([], "max", "faces"), [])

    def test_the_default_mode_is_unchanged(self):
        self.assertEqual(refs.reference_argv_ng(["a.png"], "max"),
                         ["--ref2va-reference", "a.png", "--ref2va-reference-size", "max"])


class Naming(unittest.TestCase):
    def test_the_sentence_goes_at_the_end_of_the_description_before_the_sound_sections(self):
        out = refs.name_faces_in_prompt_ng(DESC, 1, 1)
        self.assertIn("The fairy looks up. The person from <Picture 1> is in this shot,", out)
        self.assertLess(out.index("<Picture 1>"), out.index("overall_soundscape:"))
        self.assertTrue(out.endswith("non_diegetic_music: N/A"))

    def test_pictures_come_after_a_first_frame_and_each_gets_a_sentence(self):
        out = refs.name_faces_in_prompt_ng(DESC, 2, 2)
        self.assertIn("<Picture 2>", out)
        self.assertIn("<Picture 3>", out)
        self.assertNotIn("<Picture 1>", out)

    def test_a_picture_the_prompt_already_names_is_left_alone(self):
        text = DESC.replace("looks up.", "looks up at <Picture 2>.")
        out = refs.name_faces_in_prompt_ng(text, 2, 2)
        self.assertEqual(out.count("<Picture 2>"), 1)      # only the user's own mention; nothing added for it
        self.assertEqual(out.count("The person from <Picture 2>"), 0)
        self.assertIn("The person from <Picture 3>", out)

    def test_a_first_frame_mention_does_not_hide_the_kept_picture(self):
        text = "For the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n" + DESC
        self.assertIn("The person from <Picture 2>", refs.name_faces_in_prompt_ng(text, 1, 2))

    def test_a_prompt_with_no_description_section_gets_it_at_the_end(self):
        self.assertEqual(refs.name_faces_in_prompt_ng("a fairy", 1, 1),
                         "a fairy " + refs.FACES_SENTENCE.format(n=1))

    def test_nothing_to_name_changes_nothing(self):
        self.assertEqual(refs.name_faces_in_prompt_ng(DESC, 0, 1), DESC)


class EngineCommand(_FacesBase):
    def test_a_chain_gets_the_flags_and_every_window_names_the_pictures(self):
        windows = [DESC, DESC.replace("looks up", "backs away")]
        cmd = self._cmd(model="h3q8", duration_s=10.0, chain_prompts=windows, tristep=True,
                        references=[str(self.pic), str(self.pic2)], reference_mode="faces")
        self.assertEqual([cmd[i + 1] for i, a in enumerate(cmd) if a == "--reference"], [str(self.pic), str(self.pic2)])
        self.assertEqual(self._value(cmd, "--reference-size"), "1024")
        self.assertNotIn("--ref2va-reference", cmd)
        self.assertEqual(self._value(cmd, "--chain-windows"), "2")
        saved = json.loads(open(self._value(cmd, "--chain-prompts")).read())
        self.assertEqual(len(saved), 2)
        for text in saved:
            self.assertIn("The person from <Picture 1>", text)
            self.assertIn("The person from <Picture 2>", text)

    def test_a_single_window_names_them_in_the_prompt_and_a_first_frame_moves_the_numbers(self):
        cmd = self._cmd(model="h3q8", prompt=DESC, image_path=str(self.pic2), references=[str(self.pic)],
                        reference_mode="faces")
        self.assertIn("--first-frame", cmd)
        self.assertIn("The person from <Picture 2>", cmd[2])          # [python, runner, prompt, ...]

    def test_flex_is_allowed_with_kept_pictures_but_not_with_ref2va_ones(self):
        self.runner.write_text(FACES_RUNNER + "--pika-flex\n")
        with mock.patch.object(__import__("h3_flexNG"), "resolve_flex_for_model_ng", return_value="/adapter"):
            cmd = self._cmd(model="h3q8", prompt=DESC, flex=True, references=[str(self.pic)], reference_mode="faces")
        self.assertIn("--reference", cmd)

    def test_the_wrong_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "first-frame model"):
            self._cmd(model="h3ref", references=[str(self.pic)], reference_mode="faces")

    def test_the_default_mode_command_is_unchanged(self):
        self.runner.write_text(FACES_RUNNER + "--ref2va-reference PICTURE|CLIP\n")
        cmd = self._cmd(model="h3ref", references=[str(self.pic)], tristep=True)
        self.assertIn("--ref2va-reference", cmd)
        self.assertNotIn("--reference", cmd)


class JobAndRoute(_FacesBase):
    def setUp(self):
        super().setUp()
        for target, name, val in (
                (jobs, "H3GEN_DIR", str(self.tmp / "gen")),
                (jobs, "_QUEUE", mock.MagicMock()),
                (jobs, "_ensure_worker_started", lambda: None),
                (jobs.h3, "_resolve_h3_runner_ng", lambda c: str(self.runner))):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: jobs._JOBS.clear())
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def _start(self, **kw):
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=10.0, model="h3q8", tristep=True,
                    chain_prompts=["a", "b"], references=[(b"PNG", ".png")], reference_mode="faces")
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_a_chained_job_with_kept_pictures_is_queued_and_reports_its_mode(self):
        job = self._start()
        self.assertEqual((job.references, job.reference_mode), (["reference_0.png"], "faces"))
        status = jobs.job_status_ng(job.job_id)
        self.assertEqual((status["references"], status["reference_mode"]), (1, "faces"))

    def test_a_job_without_references_reports_no_mode(self):
        self.assertIsNone(jobs.job_status_ng(self._start(references=None).job_id)["reference_mode"])

    def test_the_default_mode_still_refuses_a_chain(self):
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
            self._start(reference_mode="ref2va")

    def test_a_refused_combination_queues_nothing(self):
        for kw, text in ((dict(model="h3ref"), "first-frame model"), (dict(references=[(b"x", ".mp4")]), "only pictures"),
                         (dict(references=[(b"x", ".png")] * 4), "at most 3")):
            with self.subTest(kw=kw), self.assertRaisesRegex((ValueError, FileNotFoundError), text):
                self._start(**kw)
        self.assertEqual(jobs._JOBS, {})

    def test_the_route_forwards_the_mode(self):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j1")
            r = self.client.post("/api/ng/h3/generate", content_type="multipart/form-data", data={
                "prompt": "p", "reference_mode": "FACES", "reference_files": [(io.BytesIO(b"a"), "face.png")]})
        self.assertEqual((r.status_code, start.call_args.kwargs["reference_mode"]), (202, "faces"))
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j2")
            self.client.post("/api/ng/h3/generate", data={"prompt": "p"})
        self.assertEqual(start.call_args.kwargs["reference_mode"], "ref2va")


if __name__ == "__main__":
    unittest.main()
