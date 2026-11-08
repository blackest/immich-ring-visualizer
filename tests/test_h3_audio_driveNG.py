"""Unit tests for H3 audio drive (lip-sync, `--audio-drive`): the request checks, the runner command line, the job
enqueue and status, and the /generate `audio_drive` upload. Nothing renders and no real model files are read.

Run with: python3 -m unittest tests.test_h3_audio_driveNG -v
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_audio_driveNG as drive
import h3_jobsNG as jobs
import routes.h3NG as h3_routes
from tests.test_h3_tristepNG import _Base, RUNNER

DRIVE_RUNNER = RUNNER + "--audio-drive AUDIO\n--last-frame\n--keyframe\n--extend-from\n"


class _DriveBase(_Base):
    def setUp(self):
        super().setUp()
        self.runner.write_text(DRIVE_RUNNER)
        self.wav = self.tmp / "voice.wav"
        self.wav.write_bytes(b"RIFFxxxx")


class ModuleTests(_DriveBase):
    def test_no_audio_is_a_no_op(self):
        drive.check_audio_drive_ng(None, refmods=[{"x": 1}], extend_from="e.mp4")
        self.assertEqual(drive.audio_drive_argv_ng(None), [])

    def test_a_good_request_passes_and_builds_the_flag(self):
        drive.check_audio_drive_ng(str(self.wav), runner=self.runner)
        self.assertEqual(drive.audio_drive_argv_ng(str(self.wav)), ["--audio-drive", str(self.wav)])

    def test_refused_combinations(self):
        with self.assertRaises(ValueError):
            drive.check_audio_drive_ng(str(self.wav), extend_from="e.mp4")

    def test_a_refmod_or_reference_is_allowed(self):
        # lifted 2026-11-08: the runner takes --refmod with --audio-drive, and the mouth followed the words
        drive.check_audio_drive_ng(str(self.wav), refmods=[{"path": "r"}], runner=self.runner)
        drive.check_audio_drive_ng(str(self.wav), references=["pic.png"], runner=self.runner)

    def test_missing_file_and_old_runner(self):
        with self.assertRaises(FileNotFoundError):
            drive.check_audio_drive_ng(str(self.tmp / "nope.wav"))
        self.runner.write_text(RUNNER)
        with self.assertRaisesRegex(ValueError, "no --audio-drive"):
            drive.check_audio_drive_ng(str(self.wav), runner=self.runner)

    def test_extension_cleaning(self):
        self.assertEqual(drive.clean_ext_ng(".MP3"), ".mp3")
        for bad in ("", ".exe", ".wav/../x", None):
            self.assertEqual(drive.clean_ext_ng(bad), ".wav")


class CommandTests(_DriveBase):
    def test_the_flag_is_in_the_command_with_a_lora_and_a_first_frame(self):
        pic = self.tmp / "face.png"
        pic.write_bytes(b"x")
        cmd = self._cmd(model="h3q8", image_path=str(pic), audio_drive_path=str(self.wav))
        self.assertEqual(self._value(cmd, "--audio-drive"), str(self.wav))
        self.assertEqual(self._value(cmd, "--first-frame"), str(pic))

    def test_off_means_no_flag(self):
        self.assertNotIn("--audio-drive", self._cmd(model="h3q8"))

    def test_extend_is_refused_before_launch_and_a_refmod_is_not(self):
        cmd = self._cmd(model="h3ref", refmods=[self._item()], audio_drive_path=str(self.wav))
        self.assertEqual(self._value(cmd, "--audio-drive"), str(self.wav))
        self.assertIn("--refmod", cmd)
        src = self.tmp / "src.mp4"
        src.write_bytes(b"x")
        with self.assertRaises(ValueError):
            self._cmd(model="h3q8", extend_from=str(src), audio_drive_path=str(self.wav))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            self._cmd(model="h3q8", audio_drive_path=str(self.tmp / "gone.wav"))


class JobTests(_DriveBase):
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
        args = dict(image_bytes=None, image_ext=".jpg", prompt="p", duration_s=5.0, model="h3q8")
        args.update(kw)
        return jobs.start_h3_job_ng(**args)

    def test_the_audio_is_written_into_the_job_folder_and_reported(self):
        job = self._start(audio_bytes=b"RIFFdata", audio_ext=".MP3")
        self.assertTrue(job.has_audio_drive)
        self.assertEqual((job.job_dir / "drive.mp3").read_bytes(), b"RIFFdata")
        self.assertTrue(jobs.job_status_ng(job.job_id)["has_audio_drive"])
        self.assertIn('"has_audio_drive": true', (job.job_dir / jobs.SETTINGS_NAME).read_text())

    def test_no_audio_means_no_file(self):
        job = self._start()
        self.assertFalse(job.has_audio_drive)
        self.assertEqual(list(job.job_dir.glob("drive.*")), [])

    def test_bad_requests_fail_the_post_synchronously(self):
        for kw in ({"audio_bytes": b""}, {"audio_bytes": b"x", "extend_bytes": b"y"}):
            with self.subTest(kw=list(kw)), self.assertRaises(ValueError):
                self._start(**kw)
        self.queue.put.assert_not_called()


class RouteTests(_DriveBase):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        self.client = app.test_client()

    def test_the_upload_is_forwarded_and_absent_means_none(self):
        with mock.patch.object(h3_routes.h3_jobs, "start_h3_job_ng") as start:
            start.return_value = mock.Mock(job_id="j1")
            r = self.client.post("/api/ng/h3/generate", data={
                "prompt": "p", "audio_drive": (io.BytesIO(b"RIFFdata"), "voice.wav")},
                content_type="multipart/form-data")
            self.assertEqual(r.status_code, 202)
            self.assertEqual(start.call_args.kwargs["audio_bytes"], b"RIFFdata")
            self.assertEqual(start.call_args.kwargs["audio_ext"], ".wav")
            start.reset_mock()
            self.client.post("/api/ng/h3/generate", data={"prompt": "p"})
            self.assertIsNone(start.call_args.kwargs["audio_bytes"])


if __name__ == "__main__":
    unittest.main()
