"""Unit tests for H3 `--ref2va-reference` (h3_referencesNG.py): what is refused, the runner arguments, the
job queue storing the uploaded pictures and clips, the /generate fields, and the health flag.
Nothing renders and no real model files are read.

Run with: python3 -m unittest tests.test_h3_referencesNG -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_engineNG as eng
import h3_jobsNG as jobs
import h3_referencesNG as refs
import routes.h3NG as h3_routes
from tests.test_h3_tristepNG import _Base, RUNNER

REF_RUNNER = RUNNER + "--ref2va-reference PICTURE|CLIP\n--ref2va-reference-size match|max\n"


class _RefBase(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(REF_RUNNER)
        self.pic = self.tmp / "portrait.png"
        self.pic.write_bytes(b"png")
        self.clip = self.tmp / "scene.mp4"
        self.clip.write_bytes(b"mp4")

    def _validate(self, paths=None, size="match", **kw):
        args = dict(model="h3ref", image_path=None, runner=str(self.runner))
        args.update(kw)
        return refs.validate_references_ng([str(self.pic), str(self.clip)] if paths is None else paths,
                                           size, **args)


class ValidationTests(_RefBase):
    def test_none_is_always_fine(self):
        self.assertEqual(self._validate([], model="h3q8"), [])

    def test_a_picture_and_a_clip_pass_in_order(self):
        self.assertEqual(self._validate(), [str(self.pic), str(self.clip)])

    def test_refusals(self):
        for kw, text in (
                (dict(model="h3q8"), "Ref2VA"),
                (dict(image_path="keyframe"), "first, last or keyframe"),
                (dict(last_frame_path="last"), "first, last or keyframe"),
                (dict(keyframes=[("a", "1")]), "first, last or keyframe"),
                (dict(extend_from="x"), "extending"),
                (dict(refmods=[{"path": "r"}]), "RefMod"),
                (dict(turbo=True), "Turbo"),
                (dict(v13=True), "Turbo"),
                (dict(chain_windows=2), "single 5 s window"),
                (dict(size="huge"), "size must be")):
            with self.subTest(kw=kw):
                with self.assertRaisesRegex(ValueError, text):
                    self._validate(**kw)

    def test_a_runner_without_the_flag_is_refused(self):
        self.runner.write_text(RUNNER)
        with self.assertRaisesRegex(ValueError, "no --ref2va-reference support"):
            self._validate()

    def test_limits_and_unknown_files(self):
        many = [str(self.pic)] * (refs.MAX_PICTURES + 1)
        with self.assertRaisesRegex(ValueError, "at most 9 picture"):
            self._validate(many)
        with self.assertRaisesRegex(ValueError, "at most 3 clip"):
            self._validate([str(self.clip)] * 4)
        with self.assertRaisesRegex(ValueError, "picture or a clip"):
            self._validate([str(self.tmp / "notes.txt")])
        with self.assertRaises(FileNotFoundError):
            self._validate([str(self.tmp / "gone.png")])
        self.assertEqual(self._validate([str(self.tmp / "gone.png")], check_files=False),
                         [str(self.tmp / "gone.png")])

    def test_kind_and_extension(self):
        self.assertEqual((refs.kind_of_ng("a.PNG"), refs.kind_of_ng("b.mov"), refs.kind_of_ng("c.wav")),
                         ("image", "video", None))
        self.assertEqual(refs.clean_ext_ng("Face.JPG"), ".jpg")
        with self.assertRaises(ValueError):
            refs.clean_ext_ng("face")


class ArgvTests(_RefBase):
    def test_one_flag_per_file_in_order_and_size_only_when_not_match(self):
        self.assertEqual(refs.reference_argv_ng(["a.png", "b.mp4"]),
                         ["--ref2va-reference", "a.png", "--ref2va-reference", "b.mp4"])
        self.assertEqual(refs.reference_argv_ng(["a.png"], "max"),
                         ["--ref2va-reference", "a.png", "--ref2va-reference-size", "max"])
        self.assertEqual(refs.reference_argv_ng([], "max"), [])

    def test_engine_command_carries_them(self):
        cmd = self._cmd(references=[str(self.pic), str(self.clip)], tristep=True)
        flags = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--ref2va-reference"]
        self.assertEqual(flags, [str(self.pic), str(self.clip)])
        self.assertNotIn("--ref2va-reference-size", cmd)
        self.assertEqual(self._value(self._cmd(references=[str(self.pic)], reference_size="max"),
                                     "--ref2va-reference-size"), "max")

    def test_engine_without_references_adds_nothing(self):
        self.assertNotIn("--ref2va-reference", self._cmd())

    def test_engine_refuses_the_wrong_model(self):
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
            self._cmd(model="h3q8", references=[str(self.pic)])

    def test_health_reports_support(self):
        self.assertTrue(eng.h3_health_ng("h3ref")["reference_supported"])
        self.runner.write_text(RUNNER)
        self.assertFalse(eng.h3_health_ng("h3ref")["reference_supported"])


class JobTests(_RefBase):
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

    def _start(self, **kw):
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3ref",
                    tristep=True, references=[(b"PNGDATA", ".png"), (b"MP4DATA", ".MP4")])
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_files_are_stored_in_order_and_reported(self):
        job = self._start(reference_size="max")
        self.assertEqual(job.references, ["reference_0.png", "reference_1.mp4"])
        self.assertEqual((job.job_dir / "reference_0.png").read_bytes(), b"PNGDATA")
        self.assertEqual((job.job_dir / "reference_1.mp4").read_bytes(), b"MP4DATA")
        self.assertEqual((job.reference_size, jobs.job_status_ng(job.job_id)["references"]), ("max", 2))

    def test_a_job_without_references_says_zero(self):
        job = self._start(references=None)
        self.assertEqual(jobs.job_status_ng(job.job_id)["references"], 0)

    def test_bad_combinations_fail_the_post_synchronously(self):
        for kw, text in (
                (dict(model="h3q8"), "Ref2VA"),
                (dict(image_bytes=b"x"), "first, last or keyframe"),
                (dict(last_bytes=b"x"), "first, last or keyframe"),
                (dict(extend_bytes=b"x"), "extending"),
                (dict(turbo=True, tristep=False, model="h3ref"), "turbo"),
                (dict(duration_s=20.0), "single 5 s window"),
                (dict(references=[(b"x", ".txt")]), "picture"),
                (dict(reference_size="huge"), "size must be")):
            with self.subTest(kw=kw), self.assertRaisesRegex((ValueError, FileNotFoundError), text):
                self._start(**kw)
        self.assertEqual(jobs._JOBS, {})  # nothing queued, nothing written for a refused job


class RouteTests(_RefBase):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def _post(self, **data):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j1")
            r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", **data},
                                 content_type="multipart/form-data")
        return r, start

    def test_files_and_size_are_forwarded_in_order(self):
        import io
        r, start = self._post(model="h3ref", reference_size="MAX", reference_files=[
            (io.BytesIO(b"a"), "face.png"), (io.BytesIO(b"b"), "scene.mp4")])
        self.assertEqual(r.status_code, 202)
        kw = start.call_args.kwargs
        self.assertEqual(kw["references"], [(b"a", ".png"), (b"b", ".mp4")])
        self.assertEqual(kw["reference_size"], "max")

    def test_absent_means_none_and_match(self):
        _r, start = self._post()
        self.assertEqual(start.call_args.kwargs["references"], [])
        self.assertEqual(start.call_args.kwargs["reference_size"], "match")

    def test_a_refusal_comes_back_as_400(self):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng", side_effect=ValueError("nope")):
            r = self.client.post("/api/ng/h3/generate", data={"prompt": "p"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
