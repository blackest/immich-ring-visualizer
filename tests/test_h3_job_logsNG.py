"""Unit tests for h3_job_logsNG.py's record/list/sidecar/delete logic.

Run with: python3 -m unittest tests.test_h3_job_logsNG -v
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_job_logsNG as logs


def _job(tmp: Path, **overrides):
    job_dir = tmp / "job"
    job_dir.mkdir(exist_ok=True)
    (job_dir / "ref.png").write_bytes(b"png")
    (job_dir / "h3_1.mp4").write_bytes(b"mp4")
    (job_dir / "h3_1.metrics.json").write_text("{}")
    now = time.time()
    fields = dict(
        job_id="abc123def456", job_dir=job_dir, mp4_path=str(job_dir / "h3_1.mp4"),
        cancelled=False, dispatched_at=now - 300, finished_at=now, error=None,
        error_type=None, prompt="a prompt", duration_s=5.0, seed=7, model="h3",
        width=1344, height=768, steps=4, turbo=True, queued_at=now - 301,
        _log_lock=threading.Lock(), log_lines=["line one", "line two"])
    fields.update(overrides)
    return SimpleNamespace(**fields)


class H3JobLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        patcher = mock.patch.object(logs, "H3_JOB_LOG_DIR", str(self.tmp / "logs"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_record_copies_sidecars_and_lists(self):
        job = _job(self.tmp)
        logs.record_job_ng(job)
        entries = logs.list_recent_logs_ng()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual((e["status"], e["has_video"], e["has_ref"], e["has_metrics"]),
                         ("completed", True, True, True))
        self.assertEqual(logs.log_video_path_ng(e["date"], job.job_id).name, "abc123def456.mp4")
        self.assertEqual(logs.log_ref_path_ng(e["date"], job.job_id).name, "abc123def456.ref.png")
        self.assertEqual(logs.log_text_path_ng(e["date"], job.job_id).read_text(), "line one\nline two")

    def test_failed_job_without_video(self):
        job = _job(self.tmp, mp4_path=None, error="boom", error_type="RuntimeError")
        logs.record_job_ng(job)
        e = logs.list_recent_logs_ng()[0]
        self.assertEqual((e["status"], e["has_video"], e["error"]), ("failed", False, "boom"))

    def test_cancelled_before_start_is_not_logged(self):
        logs.record_job_ng(_job(self.tmp, cancelled=True, dispatched_at=None))
        self.assertEqual(logs.list_recent_logs_ng(), [])

    def test_notes_and_delete(self):
        job = _job(self.tmp)
        logs.record_job_ng(job)
        date = logs.list_recent_logs_ng()[0]["date"]
        self.assertEqual(logs.update_log_notes_ng(date, job.job_id, "hi")["notes"], "hi")
        self.assertTrue(logs.delete_log_ng(date, job.job_id))
        self.assertEqual(logs.list_recent_logs_ng(), [])
        self.assertIsNone(logs.log_video_path_ng(date, job.job_id))

    def test_rejects_bad_ids_and_dates(self):
        self.assertIsNone(logs.get_log_entry_ng("2026-09-29", "../etc/passwd"))
        self.assertIsNone(logs.get_log_entry_ng("../..", "abc123def456"))
        self.assertFalse(logs.delete_log_ng("2026-09-29", "nothex!"))

    def test_past_month_days_are_archived(self):
        old = self.tmp / "logs" / "2020-01-05"
        old.mkdir(parents=True)
        (old / "abc123def456.json").write_text('{"job_id": "abc123def456", "finished_at": 1}')
        self.assertEqual(logs.list_recent_logs_ng(), [])
        self.assertEqual(logs.list_archive_years_ng(), ["2020"])
        self.assertEqual(logs.list_archive_months_ng("2020"), ["01"])
        self.assertEqual(logs.list_archive_day_entries_ng("2020", "01")[0]["date"], "2020-01-05")


if __name__ == "__main__":
    unittest.main()
