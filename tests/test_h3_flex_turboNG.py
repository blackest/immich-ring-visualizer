"""Unit tests for H3 "Flex Turbo" (Pika's four-forward Flex LoRA, `flex_turbo`): the runner command line and its
refusals, health, the job enqueue and status, the re-roll's memory of it, and the /generate `flex_turbo` field.
Nothing renders and no real model files are read.

Run with: python3 -m unittest tests.test_h3_flex_turboNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_engineNG as eng
import h3_flexNG as flexmod
import h3_jobsNG as jobs
import routes.h3NG as h3_routes
from tests.test_h3_tristepNG import _Base, RUNNER

TURBO_RUNNER = RUNNER + "--sigma-knots\n--video-shift\n--audio-shift\n--lora-fc1-swap\n"


class _TurboBase(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(TURBO_RUNNER)
        self.turbo_file = self.tmp / "turbo.safetensors"
        self.turbo_file.write_bytes(b"x")
        p = mock.patch.object(flexmod, "resolve_flex_turbo_ng", lambda root: str(self.turbo_file))
        p.start()
        self.addCleanup(p.stop)

    def _turbo(self, **kw):
        return self._cmd(model="h3q8", flex_turbo=True, **kw)


class CommandTests(_TurboBase):
    def test_the_recipe_is_euler_over_five_knots_with_pikas_shifts_and_layout_flags(self):
        cmd = self._turbo()
        self.assertEqual(self._value(cmd, "--sampler"), "euler")
        self.assertEqual(self._value(cmd, "--sigma-knots"), "1000,750,500,250,0")
        self.assertEqual(self._value(cmd, "--video-shift"), "6")
        self.assertEqual(self._value(cmd, "--audio-shift"), "3")
        self.assertEqual(self._value(cmd, "--lora"), f"{self.turbo_file}:1.0")
        self.assertIn("--lora-qkv-permute", cmd)
        self.assertIn("--lora-fc1-swap", cmd)
        self.assertNotIn("--pika-flex", cmd)

    def test_steps_are_the_five_knots_whatever_was_asked(self):
        self.assertEqual(self._value(self._turbo(steps=16), "--steps"), "5")

    def test_it_runs_on_the_bf16_and_q8_models(self):
        for model in ("h3", "h3q8"):
            with self.subTest(model=model):
                self.assertIn("--sigma-knots", self._cmd(model=model, flex_turbo=True))

    def test_off_means_none_of_it(self):
        cmd = self._cmd(model="h3q8")
        for flag in ("--sigma-knots", "--lora-fc1-swap"):
            self.assertNotIn(flag, cmd)

    def test_a_user_lora_stacks_after_it(self):
        user = self.tmp / "face.safetensors"
        user.write_bytes(b"x")
        with mock.patch.object(eng.h3_lora, "lora_for_runner_ng", return_value=(user, False)):
            cmd = self._turbo(loras=[(str(user), 0.7)])
        loras = [cmd[i + 1] for i, v in enumerate(cmd) if v == "--lora"]
        self.assertEqual(loras, [f"{self.turbo_file}:1.0", f"{user}:0.7"])

    def test_refused_with_other_speed_options_flex_refmods_and_the_ref2va_model(self):
        for kw in ({"tristep": True}, {"turbo": True}, {"v13": True}, {"flex": True},
                   {"refmods": [self._item()]}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self._cmd(model="h3q8", flex_turbo=True, **kw)
        with self.assertRaisesRegex(ValueError, "FL2VA"):
            self._cmd(model="h3ref", flex_turbo=True)

    def test_refused_without_the_file_or_a_runner_with_the_flags(self):
        with mock.patch.object(flexmod, "resolve_flex_turbo_ng", return_value=None), \
                self.assertRaises(FileNotFoundError):
            self._turbo()
        self.runner.write_text(RUNNER)  # no --sigma-knots, --video-shift, ...
        with self.assertRaisesRegex(ValueError, "--sigma-knots"):
            self._turbo()


class HealthTests(_TurboBase):
    def test_fields(self):
        h = eng.h3_health_ng("h3q8")
        self.assertTrue(h["flex_turbo_ok"] and h["flex_turbo_supported"])
        self.runner.write_text(RUNNER)
        self.assertFalse(eng.h3_health_ng("h3q8")["flex_turbo_supported"])
        with mock.patch.object(flexmod, "resolve_flex_turbo_ng", return_value=None):
            self.assertFalse(eng.h3_health_ng("h3q8")["flex_turbo_ok"])


class FileTests(unittest.TestCase):
    def test_only_the_published_file_at_its_exact_size_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / flexmod.FLEX_DIRNAME
            folder.mkdir()
            self.assertIsNone(flexmod.resolve_flex_turbo_ng(tmp))
            target = folder / flexmod.FLEX_TURBO_FILE
            target.write_bytes(b"x" * 10)
            self.assertIsNone(flexmod.resolve_flex_turbo_ng(tmp))  # truncated download
            with target.open("r+b") as fh:
                fh.truncate(flexmod.FLEX_TURBO_BYTES)
            self.assertEqual(flexmod.resolve_flex_turbo_ng(tmp), str(target))

    def test_the_knots_match_the_step_count(self):
        self.assertEqual(len(flexmod.FLEX_TURBO_KNOTS.split(",")), flexmod.FLEX_TURBO_STEPS)


class JobTests(_TurboBase):
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
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3q8", flex_turbo=True)
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_a_turbo_job_forces_the_steps_and_reports_itself(self):
        job = self._start(steps=16)
        self.assertEqual((job.steps, job.flex_turbo, job.flex), (5, True, False))
        status = jobs.job_status_ng(job.job_id)
        self.assertTrue(status["flex_turbo"])
        self.assertEqual(status["steps"], 5)
        self.assertIn('"flex_turbo": true', (job.job_dir / jobs.SETTINGS_NAME).read_text())

    def test_bad_combinations_fail_the_post_synchronously(self):
        for kw in ({"tristep": True}, {"flex": True}, {"model": "h3ref"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self._start(**kw)
        with mock.patch.object(flexmod, "resolve_flex_turbo_ng", return_value=None), \
                self.assertRaises(FileNotFoundError):
            self._start()
        self.queue.put.assert_not_called()


class RouteTests(_Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def test_the_field_is_forwarded_and_absent_means_off(self):
        for data, expected in (({"flex_turbo": "true"}, True), ({}, False)):
            with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
                start.return_value = mock.Mock(job_id="j1")
                r = self.client.post("/api/ng/h3/generate", data={"prompt": "p", **data})
            self.assertEqual(r.status_code, 202)
            self.assertEqual(start.call_args.kwargs["flex_turbo"], expected)


if __name__ == "__main__":
    unittest.main()
