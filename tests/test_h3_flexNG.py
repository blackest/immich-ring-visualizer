"""Unit tests for H3 "Flex" (Pika H3 Flex, `flex`): the runner command line and its refusals, health, the
job enqueue and status, the reroll's memory of it, and the /generate `flex` field. Nothing renders and no
real model files are read.

Run with: python3 -m unittest tests.test_h3_flexNG -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_engineNG as eng
import h3_flexNG as flexmod
import h3_jobsNG as jobs
import routes.h3NG as h3_routes
from tests.test_h3_tristepNG import _Base, RUNNER

FLEX_RUNNER = RUNNER + "--pika-flex\n"


class _FlexBase(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(FLEX_RUNNER)
        self.flex_file = self.tmp / "pika.safetensors"
        self.flex_file.write_bytes(b"x")
        self.flex_ref2va_file = self.tmp / "pika_ref2va.safetensors"
        self.flex_ref2va_file.write_bytes(b"x")
        for target, name, val in ((flexmod, "resolve_flex_ng", lambda root: str(self.flex_file)),
                                  (flexmod, "resolve_flex_ref2va_ng", lambda root: str(self.flex_ref2va_file))):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)

    def _flex(self, **kw):
        return self._cmd(model="h3q8", flex=True, **kw)


class CommandTests(_FlexBase):
    def test_the_sampler_preset_and_the_adapter_at_full_strength(self):
        cmd = self._flex()
        self.assertIn("--pika-flex", cmd)
        self.assertEqual(self._value(cmd, "--lora"), f"{self.flex_file}:1.0")

    def test_steps_are_the_seven_knots_whatever_was_asked(self):
        self.assertEqual(self._value(self._flex(steps=16), "--steps"), "7")

    def test_it_runs_on_the_quantized_model(self):
        self.assertIn("--pika-flex", self._cmd(model="h3q8", flex=True))

    def test_a_user_lora_stacks_after_it(self):
        user = self.tmp / "face.safetensors"
        user.write_bytes(b"x")
        with mock.patch.object(eng.h3_lora, "lora_for_runner_ng", return_value=(user, False)):
            cmd = self._flex(loras=[(str(user), 0.7)])
        loras = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--lora"]
        self.assertEqual(loras, [f"{self.flex_file}:1.0", f"{user}:0.7"])

    def test_off_means_no_flag(self):
        self.assertNotIn("--pika-flex", self._cmd(model="h3q8"))

    def test_refused_with_other_speed_options_and_with_refmods(self):
        for kw in ({"tristep": True}, {"turbo": True}, {"v13": True}, {"refmods": [self._item()]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self._cmd(model="h3q8", flex=True, **kw)

    def test_the_ref2va_model_takes_the_ref2va_adapter_never_the_fl2va_one(self):
        cmd = self._cmd(model="h3ref", flex=True)
        self.assertIn("--pika-flex", cmd)
        self.assertEqual(self._value(cmd, "--lora"), f"{self.flex_ref2va_file}:1.0")
        self.assertNotIn(str(self.flex_file), cmd)

    def test_the_ref2va_model_without_its_adapter_is_refused_even_if_the_fl2va_one_exists(self):
        with mock.patch.object(flexmod, "resolve_flex_ref2va_ng", return_value=None), \
                self.assertRaises(FileNotFoundError):
            self._cmd(model="h3ref", flex=True)

    def test_refused_without_the_adapter_or_a_runner_that_has_the_flag(self):
        with mock.patch.object(flexmod, "resolve_flex_ng", return_value=None), self.assertRaises(FileNotFoundError):
            self._flex()
        self.runner.write_text(RUNNER)
        with self.assertRaisesRegex(ValueError, "--pika-flex"):
            self._flex()


class HealthTests(_FlexBase):
    def test_fields(self):
        h = eng.h3_health_ng("h3q8")
        self.assertTrue(h["flex_ok"] and h["flex_supported"])
        self.runner.write_text(RUNNER)
        self.assertFalse(eng.h3_health_ng("h3q8")["flex_supported"])
        with mock.patch.object(flexmod, "resolve_flex_ng", return_value=None):
            self.assertFalse(eng.h3_health_ng("h3q8")["flex_ok"])


class AdapterFileTests(unittest.TestCase):
    def test_only_the_built_file_at_its_exact_size_counts(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / flexmod.FLEX_DIRNAME
            folder.mkdir()
            self.assertIsNone(flexmod.resolve_flex_ng(tmp))
            (folder / flexmod.FLEX_FILE).write_bytes(b"x" * 10)
            self.assertIsNone(flexmod.resolve_flex_ng(tmp))  # truncated, or the 2.8 GB upstream file


class JobTests(_FlexBase):
    def setUp(self):
        super().setUp()
        self.queue = mock.MagicMock()
        for target, name, val in (
                (jobs, "H3GEN_DIR", str(self.tmp / "gen")),
                (jobs, "_QUEUE", self.queue),
                (jobs, "_ensure_worker_started", lambda: None),
                (jobs.h3, "_resolve_h3_runner_ng", lambda c: str(self.runner))):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: jobs._JOBS.clear())

    def _start(self, **kw):
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3q8", flex=True)
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_a_flex_job_forces_the_steps_and_reports_itself(self):
        job = self._start(steps=16)
        self.assertEqual((job.steps, job.flex), (7, True))
        status = jobs.job_status_ng(job.job_id)
        self.assertTrue(status["flex"])
        self.assertEqual(status["steps"], 7)
        saved = (job.job_dir / jobs.SETTINGS_NAME).read_text()
        self.assertIn('"flex": true', saved)

    def test_bad_combinations_fail_the_post_synchronously(self):
        for kw in ({"tristep": True}, {"refmods": [{"name": "a", **self._item()}]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self._start(**kw)
        with mock.patch.object(flexmod, "resolve_flex_ng", return_value=None), self.assertRaises(FileNotFoundError):
            self._start()
        self.queue.put.assert_not_called()


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def test_the_field_is_forwarded_and_absent_means_off(self):
        for data, expected in (({"flex": "true"}, True), ({}, False)):
            with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
                start.return_value = mock.Mock(job_id="j1")
                r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", **data})
            self.assertEqual(r.status_code, 202)
            self.assertEqual(start.call_args.kwargs["flex"], expected)


class Ref2vaAdapterResolve(unittest.TestCase):
    def test_only_the_exact_size_counts_and_the_env_override_wins(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d, mock.patch.object(flexmod.Path, "home", return_value=Path(d) / "home"):
            root = Path(d) / "models"
            (root / flexmod.FLEX_DIRNAME).mkdir(parents=True)
            wrong = root / flexmod.FLEX_DIRNAME / flexmod.FLEX_REF2VA_FILE
            wrong.write_bytes(b"x")
            self.assertIsNone(flexmod.resolve_flex_ref2va_ng(root))
            with wrong.open("r+b") as fh:
                fh.truncate(flexmod.FLEX_REF2VA_BYTES)
            self.assertEqual(flexmod.resolve_flex_ref2va_ng(root), str(wrong))
            self.assertEqual(flexmod.resolve_flex_for_model_ng(root, "h3ref"), str(wrong))
            self.assertIsNone(flexmod.resolve_flex_for_model_ng(root, "h3q8"))  # the FL2VA adapter is not there


if __name__ == "__main__":
    unittest.main()
