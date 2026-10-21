"""Unit tests for video_jobsNG.py -- the Animate render queue. The LTX
engine, the durable job log and the worker thread are all mocked/driven
by hand; no real render runs.

Run with: python3 -m unittest tests.test_video_jobsNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import video_jobsNG as vj


class _OneShotQueue:
    """Stands in for vj._QUEUE: hands out the given ids, then ends the
    otherwise-infinite worker loop."""
    def __init__(self, *ids):
        self.ids = list(ids)

    def get(self):
        if not self.ids:
            raise SystemExit
        return self.ids.pop(0)

    def put(self, _):
        pass


class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for p in (mock.patch.object(vj, "VIDEOGEN_DIR", str(self.tmp)),
                  mock.patch.object(vj, "_JOBS", {}),
                  mock.patch.object(vj, "_QUEUE", _OneShotQueue()),
                  mock.patch.object(vj, "_ensure_worker_started")):
            p.start()
            self.addCleanup(p.stop)

    def _start(self, **kw):
        args = dict(image_bytes=b"img", image_ext=".png", prompt="a cat", duration_s=2.0)
        args.update(kw)
        return vj.start_video_job_ng(**args)

    def _run_worker(self, job, engine=None):
        vj._QUEUE.ids = [job.job_id]
        with mock.patch.object(vj.ltx, "generate_ltx_video_ng", engine or mock.Mock(
                return_value={"mp4_path": "/x.mp4", "seed": 5})) as gen, \
             mock.patch.object(vj.job_logsNG, "record_job_ng") as record:
            with self.assertRaises(SystemExit):
                vj._worker_loop()
        return gen, record

    # --- start_video_job_ng -------------------------------------------
    def test_start_writes_reference_into_job_dir(self):
        job = self._start()
        self.assertTrue((job.job_dir / "ref.png").read_bytes() == b"img")
        self.assertIs(vj.get_job_ng(job.job_id), job)
        self.assertEqual(vj.job_status_ng(job.job_id)["status"], "queued")

    def test_bad_extension_falls_back_to_jpg_and_text_to_video_has_no_ref(self):
        self.assertTrue((self._start(image_ext=".exe").job_dir / "ref.jpg").exists())
        t2v = self._start(image_bytes=None)
        self.assertFalse(t2v.has_image)
        self.assertEqual(list(t2v.job_dir.glob("ref.*")), [])

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "prompt is required"):
            self._start(prompt="   ")
        for bad in (0.1, 31):
            with self.assertRaisesRegex(ValueError, "duration_s"):
                self._start(duration_s=bad)
        with self.assertRaises(FileNotFoundError):
            self._start(lora_path=str(self.tmp / "missing.safetensors"))
        self.assertEqual(vj._JOBS, {})

    def test_defaults_come_from_engine(self):
        job = self._start()
        self.assertEqual((job.width, job.height, job.frame_rate),
                         (vj.ltx.LTX_WIDTH, vj.ltx.LTX_HEIGHT, vj.ltx.LTX_FRAME_RATE))

    def test_prune_only_evicts_finished_jobs_oldest_first(self):
        with mock.patch.object(vj, "_MAX_JOBS_KEPT", 2):
            a, b = self._start(), self._start()
            a.finished_at, b.finished_at = 1.0, 2.0
            c = self._start()  # 3 > 2: oldest finished (a) goes, dir deleted
            self.assertIsNone(vj.get_job_ng(a.job_id))
            self.assertFalse(a.job_dir.exists())
            self.assertIsNotNone(vj.get_job_ng(b.job_id))
            d = self._start()  # only c, d unfinished + b finished
            self.assertIsNone(vj.get_job_ng(b.job_id))
            self.assertIsNotNone(vj.get_job_ng(c.job_id))
            self.assertIsNotNone(vj.get_job_ng(d.job_id))

    # --- worker --------------------------------------------------------
    def test_worker_success_records_log_and_completes(self):
        job = self._start(seed=3)
        gen, record = self._run_worker(job)
        kw = gen.call_args.kwargs
        self.assertEqual((kw["prompt"], kw["seed"], kw["duration_s"]), ("a cat", 3, 2.0))
        self.assertTrue(kw["image_path"].endswith("ref.png"))
        self.assertEqual(job.mp4_path, "/x.mp4")
        record.assert_called_once_with(job, {"mp4_path": "/x.mp4", "seed": 5})
        status = vj.job_status_ng(job.job_id)
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["video_url"].endswith(f"{job.job_id}/video"))

    def test_worker_engine_failure_is_reported_not_raised(self):
        job = self._start()
        _, record = self._run_worker(job, mock.Mock(side_effect=RuntimeError("oom")))
        status = vj.job_status_ng(job.job_id)
        self.assertEqual((status["status"], status["error"], status["error_type"]),
                         ("failed", "oom", "RuntimeError"))
        self.assertIsNone(status["video_url"])
        record.assert_called_once()

    def test_worker_survives_log_failure(self):
        job = self._start()
        vj._QUEUE.ids = [job.job_id]
        with mock.patch.object(vj.ltx, "generate_ltx_video_ng", return_value={"mp4_path": "/x"}), \
             mock.patch.object(vj.job_logsNG, "record_job_ng", side_effect=OSError("disk")):
            with self.assertRaises(SystemExit):
                vj._worker_loop()
        self.assertEqual(vj.job_status_ng(job.job_id)["status"], "completed")

    def test_worker_missing_reference_fails_job(self):
        job = self._start()
        (job.job_dir / "ref.png").unlink()
        gen, _ = self._run_worker(job)
        gen.assert_not_called()
        self.assertEqual(vj.job_status_ng(job.job_id)["error_type"], "FileNotFoundError")

    def test_worker_skips_cancelled_queued_job(self):
        job = self._start()
        self.assertTrue(vj.cancel_video_job_ng(job.job_id))
        gen, record = self._run_worker(job)
        gen.assert_not_called()
        self.assertEqual(job.error_type, "VideoJobCancelled")
        self.assertIsNone(job.dispatched_at)

    def test_worker_ignores_unknown_job_id(self):
        vj._QUEUE.ids = ["ghost"]
        with self.assertRaises(SystemExit):
            vj._worker_loop()

    # --- cancel / status -----------------------------------------------
    def test_cancel_unknown_or_finished_returns_false(self):
        self.assertFalse(vj.cancel_video_job_ng("nope"))
        job = self._start()
        job.finished_at = 1.0
        self.assertFalse(vj.cancel_video_job_ng(job.job_id))

    def test_cancel_running_job_kills_process_group(self):
        job = self._start()
        job.dispatched_at = 1.0
        job._proc = mock.Mock(pid=4242)
        with mock.patch.object(vj.os, "getpgid", return_value=77), \
             mock.patch.object(vj.os, "killpg") as killpg:
            self.assertTrue(vj.cancel_video_job_ng(job.job_id))
        killpg.assert_called_once_with(77, vj.signal.SIGKILL)
        self.assertEqual(vj.job_status_ng(job.job_id)["status"], "rendering")

    def test_cancel_falls_back_to_proc_kill(self):
        job = self._start()
        job.dispatched_at = 1.0
        job._proc = mock.Mock(pid=4242)
        with mock.patch.object(vj.os, "getpgid", side_effect=ProcessLookupError):
            vj.cancel_video_job_ng(job.job_id)
        job._proc.kill.assert_called_once()

    def test_status_unknown_job_raises_lookup(self):
        with self.assertRaises(LookupError):
            vj.job_status_ng("nope")

    def test_log_ring_buffer_keeps_last_200(self):
        job = self._start()
        for i in range(250):
            job.append_log(f"l{i}")
        self.assertEqual(len(job.log_lines), 200)
        self.assertEqual(job.log_tail(2), ["l248", "l249"])


if __name__ == "__main__":
    unittest.main()
