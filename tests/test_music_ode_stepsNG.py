"""The Music "Speed" option: generation_config.ode_steps (32 standard, 8 fast).

Engine request builder, validation, the job layer and the /api/ng/music/generate
route. No engine is run and no job is queued for real.

Run with: python3 -m unittest tests.test_music_ode_stepsNG -v
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import music_engineNG as music
import music_jobsNG as jobs
import routes.musicNG as music_routes


def _request(ode_steps=None):
    return music._build_request_ng("calm piano", "", 8.0, 7, "full", None, None, ode_steps)


class RequestBuilderTests(unittest.TestCase):
    def test_default_and_standard_leave_the_request_exactly_as_before(self):
        before = _request()
        self.assertNotIn("generation_config", before)
        self.assertEqual(_request(32), before)

    def test_fast_adds_only_generation_config(self):
        fast = _request(8)
        self.assertEqual(fast["generation_config"], {"ode_steps": 8})
        without = dict(fast)
        del without["generation_config"]
        self.assertEqual(without, _request())


class ValidationTests(unittest.TestCase):
    def test_only_the_two_listened_to_values(self):
        music._validate_music_ode_steps_ng(8)
        music._validate_music_ode_steps_ng(32)
        for bad in (0, 1, 16, 64, -8, "8", 8.0, None, True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    music._validate_music_ode_steps_ng(bad)

    def test_default_is_one_of_the_choices(self):
        self.assertIn(music.MUSIC_DEFAULT_ODE_STEPS, music.MUSIC_ODE_STEPS_CHOICES)
        self.assertEqual(music.MUSIC_DEFAULT_ODE_STEPS, 32)


class JobLayerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(jobs, "_ensure_worker_started"),
            mock.patch.object(jobs, "_QUEUE"),
            mock.patch.object(jobs, "MUSICGEN_DIR", self.tmp.name),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def _start(self, **kw):
        job = jobs.start_music_job_ng("calm piano", "", 8.0, **kw)
        self.addCleanup(jobs._JOBS.pop, job.job_id, None)
        return job

    def test_ode_steps_is_stored_and_reported(self):
        job = self._start(ode_steps=8)
        self.assertEqual(job.ode_steps, 8)
        self.assertEqual(jobs.job_status_ng(job.job_id)["ode_steps"], 8)

    def test_omitted_stays_none(self):
        self.assertIsNone(self._start().ode_steps)

    def test_invalid_is_rejected_before_queueing(self):
        with self.assertRaises(ValueError):
            jobs.start_music_job_ng("calm piano", "", 8.0, ode_steps=16)

    def test_cover_jobs_refuse_it(self):
        with self.assertRaises(ValueError) as cm:
            jobs.start_music_job_ng("", "", None, audio_bytes=b"x", audio_ext=".mp3", ode_steps=8)
        self.assertIn("cover", str(cm.exception))


class RouteTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(music_routes.musicNG_bp)
        self.client = app.test_client()

    def _post(self, body):
        fake = mock.Mock(job_id="abc123def456")
        with mock.patch.object(music_routes.music_jobs, "start_music_job_ng", return_value=fake) as start:
            resp = self.client.post("/api/ng/music/generate", json=body)
        return resp, start

    def test_ode_steps_is_passed_through(self):
        resp, start = self._post({"style": "x", "duration_s": 8, "ode_steps": 8})
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(start.call_args.kwargs["ode_steps"], 8)

    def test_omitted_is_none(self):
        resp, start = self._post({"style": "x", "duration_s": 8})
        self.assertEqual(resp.status_code, 202)
        self.assertIsNone(start.call_args.kwargs["ode_steps"])

    def test_wrong_type_is_a_400(self):
        for bad in ("8", 8.5, True, [8]):
            with self.subTest(bad=bad):
                resp, start = self._post({"style": "x", "duration_s": 8, "ode_steps": bad})
                self.assertEqual(resp.status_code, 400)
                start.assert_not_called()

    def test_wrong_value_is_a_400_from_the_job_layer(self):
        with mock.patch.object(music_routes.music_jobs, "start_music_job_ng",
                               side_effect=ValueError("ode_steps must be one of (8, 32) (got 16)")):
            resp = self.client.post("/api/ng/music/generate",
                                    json={"style": "x", "duration_s": 8, "ode_steps": 16})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("ode_steps", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
