"""Unit tests for job_logsNG.py's record/list/archive/notes/delete logic.

Run with: python3 -m unittest tests.test_job_logsNG -v
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import job_logsNG as logs

JOB_ID = "abc123def456"


def _job(tmp: Path, **overrides):
    job_dir = tmp / "job"
    job_dir.mkdir(exist_ok=True)
    (job_dir / "ref.PNG").write_bytes(b"png")
    (job_dir / "out.mp4").write_bytes(b"mp4")
    now = time.time()
    fields = dict(
        job_id=JOB_ID, job_dir=job_dir, mp4_path=str(job_dir / "out.mp4"),
        has_image=True, cancelled=False, dispatched_at=now - 60, finished_at=now,
        error=None, error_type=None, prompt="a prompt", duration_s=5.0, seed=7,
        width=768, height=512, frame_rate=24, lora_path=None, lora_strength=1.0,
        queued_at=now - 61)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class VideoJobLogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "logs"
        patcher = mock.patch.object(logs, "JOB_LOG_DIR", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_record_copies_ref_and_video(self):
        logs.record_job_ng(_job(self.tmp), {"seed": 99})
        e = logs.list_recent_logs_ng()[0]
        self.assertEqual((e["status"], e["has_ref"], e["has_video"], e["resolved_seed"]),
                         ("completed", True, True, 99))
        self.assertEqual(logs.log_ref_path_ng(e["date"], JOB_ID).name, f"{JOB_ID}_ref.png")
        self.assertEqual(logs.log_video_path_ng(e["date"], JOB_ID).name, f"{JOB_ID}.mp4")

    def test_failed_and_cancelled_status(self):
        logs.record_job_ng(_job(self.tmp, mp4_path=None, error="boom"), None)
        self.assertEqual(logs.list_recent_logs_ng()[0]["status"], "failed")
        logs.record_job_ng(_job(self.tmp, job_id="bbb123def456", cancelled=True), None)
        by_id = {e["job_id"]: e for e in logs.list_recent_logs_ng()}
        self.assertEqual(by_id["bbb123def456"]["status"], "cancelled")

    def test_cancelled_before_dispatch_not_logged(self):
        logs.record_job_ng(_job(self.tmp, cancelled=True, dispatched_at=None), None)
        self.assertEqual(logs.list_recent_logs_ng(), [])

    def test_lora_strength_only_kept_with_lora(self):
        logs.record_job_ng(_job(self.tmp), None)
        self.assertIsNone(logs.list_recent_logs_ng()[0]["lora_strength"])

    def test_record_never_raises(self):
        logs.record_job_ng(SimpleNamespace(), None)  # missing every attribute
        self.assertEqual(logs.list_recent_logs_ng(), [])

    def test_notes_roundtrip_and_bad_ids(self):
        logs.record_job_ng(_job(self.tmp), None)
        date = logs.list_recent_logs_ng()[0]["date"]
        self.assertEqual(logs.update_log_notes_ng(date, JOB_ID, "keeper")["notes"], "keeper")
        self.assertEqual(logs.get_log_entry_ng(date, JOB_ID)["notes"], "keeper")
        for bad in ("../etc", "", None, "XYZ"):
            self.assertIsNone(logs.get_log_entry_ng(date, bad))
            self.assertIsNone(logs.update_log_notes_ng(date, bad, "x"))
        self.assertIsNone(logs.get_log_entry_ng("not-a-date", JOB_ID))

    def test_delete_removes_all_sidecars(self):
        logs.record_job_ng(_job(self.tmp), None)
        date = logs.list_recent_logs_ng()[0]["date"]
        self.assertTrue(logs.delete_log_ng(date, JOB_ID))
        self.assertEqual(list((self.root / date).iterdir()), [])
        self.assertFalse(logs.delete_log_ng(date, JOB_ID))
        self.assertFalse(logs.delete_log_ng(date, "../x"))

    def test_past_months_roll_into_archive(self):
        old = self.root / "2020-03-15"
        old.mkdir(parents=True)
        (old / f"{JOB_ID}.json").write_text('{"job_id": "%s", "finished_at": 5}' % JOB_ID)
        (self.root / "notes.txt").write_text("ignored")
        self.assertEqual(logs.list_recent_logs_ng(), [])
        self.assertFalse(old.exists())
        self.assertEqual(logs.list_archive_years_ng(), ["2020"])
        self.assertEqual(logs.list_archive_months_ng("2020"), ["03"])
        entries = logs.list_archive_day_entries_ng("2020", "03")
        self.assertEqual([e["date"] for e in entries], ["2020-03-15"])
        # archived entries stay reachable through the date-keyed lookups
        self.assertEqual(logs.get_log_entry_ng("2020-03-15", JOB_ID)["job_id"], JOB_ID)

    def test_archive_never_overwrites_existing_dest(self):
        (self.root / "2020-03-15").mkdir(parents=True)
        (self.root / "2020" / "03" / "15").mkdir(parents=True)
        logs._archive_past_months()
        self.assertTrue((self.root / "2020-03-15").exists())

    def test_listing_guards_and_corrupt_json(self):
        self.assertEqual(logs.list_recent_logs_ng(), [])  # root missing
        self.assertEqual(logs.list_archive_years_ng(), [])
        self.assertEqual(logs.list_archive_months_ng("../"), [])
        self.assertEqual(logs.list_archive_day_entries_ng("2020", "zz"), [])
        day = self.root / time.strftime("%Y-%m-%d")
        day.mkdir(parents=True)
        (day / "bad.json").write_text("{not json")
        self.assertEqual(logs.list_recent_logs_ng(), [])

    def test_recent_sorted_newest_first(self):
        now = time.time()
        logs.record_job_ng(_job(self.tmp, job_id="aaa123def456", finished_at=now - 100), None)
        logs.record_job_ng(_job(self.tmp, job_id="bbb123def456", finished_at=now), None)
        ids = [e["job_id"] for e in logs.list_recent_logs_ng()]
        self.assertEqual(ids, ["bbb123def456", "aaa123def456"])


if __name__ == "__main__":
    unittest.main()
