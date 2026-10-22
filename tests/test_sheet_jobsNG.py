"""Unit tests for sheet_jobsNG.py -- the character-sheet render queue.
The HiDream engine / character_sheetNG are mocked at the boundary and the
worker loop is driven by hand; per-shot status inference runs against real
files in a temp dir.

Run with: python3 -m unittest tests.test_sheet_jobsNG -v
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

import sheet_jobsNG as sj


class _OneShotQueue:
    def __init__(self, *ids):
        self.ids = list(ids)

    def get(self):
        if not self.ids:
            raise SystemExit
        return self.ids.pop(0)

    def put(self, _):
        pass


def _spec(key):
    return SimpleNamespace(key=key)


class SheetJobTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        cs = sj.character_sheet
        patches = [
            mock.patch.object(sj, "_JOBS", {}),
            mock.patch.object(sj, "_QUEUE", _OneShotQueue()),
            mock.patch.object(sj, "_ensure_worker_started"),
            mock.patch.object(cs, "character_exists_ng", return_value=True),
            mock.patch.object(cs, "character_avatar_ng", return_value=self.tmp / "a.png"),
            mock.patch.object(cs, "resolve_shots_ng", return_value=[_spec("front"), _spec("side")]),
            mock.patch.object(cs, "_shot_prompt_ng", side_effect=lambda spec, *a, **k: f"prompt-{spec.key}"),
            mock.patch.object(cs, "_character_dir_ng", return_value=self.tmp),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _start(self, **kw):
        return sj.start_job_ng("tom", **kw)

    def _run_worker(self, job):
        sj._QUEUE.ids = [job.job_id]
        with self.assertRaises(SystemExit):
            sj._worker_loop()

    def _make_png(self, shot, mtime):
        d = self.tmp / "sheet_views" / shot
        d.mkdir(parents=True, exist_ok=True)
        p = d / "cand_1.png"
        p.write_bytes(b"x")
        os.utime(p, (mtime, mtime))
        return p

    # --- start_job_ng ---------------------------------------------------
    def test_start_queues_job_with_prompts_and_params(self):
        job = self._start(wardrobe="red coat")
        self.assertEqual(job.shot_keys, ["front", "side"])
        self.assertEqual(job.shot_prompts, {"front": "prompt-front", "side": "prompt-side"})
        self.assertEqual(job.params["preset"], "default")
        self.assertEqual(job.params["wardrobe"], "red coat")
        self.assertEqual(sj.job_status_ng(job.job_id)["status"], "queued")

    def test_custom_shots_or_views_report_preset_custom(self):
        self.assertEqual(self._start(shots=[{"x": 1}]).params["preset"], "custom")
        self.assertEqual(self._start(views=["back"]).params["preset"], "custom")

    def test_validation_errors_raise_before_queueing(self):
        cs = sj.character_sheet
        with mock.patch.object(cs, "character_exists_ng", return_value=False):
            with self.assertRaises(LookupError):
                self._start()
        with mock.patch.object(cs, "character_avatar_ng", return_value=None):
            with self.assertRaises(FileNotFoundError):
                self._start()
        with self.assertRaises(Exception):
            self._start(character_id="../evil")
        self.assertEqual(sj._JOBS, {})

    def test_prune_keeps_unfinished_and_drops_oldest_finished(self):
        with mock.patch.object(sj, "_MAX_JOBS_KEPT", 2):
            a, b = self._start(), self._start()
            a.finished_at, b.finished_at = 1.0, 2.0
            c = self._start()
            self.assertIsNone(sj.get_job_ng(a.job_id))
            self.assertIsNotNone(sj.get_job_ng(b.job_id))
            self.assertIsNotNone(sj.get_job_ng(c.job_id))

    # --- worker ---------------------------------------------------------
    def test_worker_runs_target_with_log_and_proc_hooks(self):
        job = self._start(seed=9)
        with mock.patch.object(sj.character_sheet, "generate_character_sheet_ng") as gen:
            def fake(cid, **kw):
                kw["on_log"]("rendering front")
                kw["on_proc_start"]("PROC")
            gen.side_effect = fake
            self._run_worker(job)
        self.assertEqual(gen.call_args.args, ("tom",))
        self.assertEqual(gen.call_args.kwargs["seed"], 9)
        self.assertEqual(job.log_tail(), ["rendering front"])
        self.assertEqual(job._proc, "PROC")
        self.assertIsNotNone(job.finished_at)
        self.assertIsNone(job.error)

    def test_worker_failure_is_recorded_not_raised(self):
        job = self._start()
        with mock.patch.object(sj.character_sheet, "generate_character_sheet_ng",
                               side_effect=RuntimeError("gpu fell over")):
            self._run_worker(job)
        self.assertEqual((job.error, job.error_type), ("gpu fell over", "RuntimeError"))
        self.assertIsNotNone(job.finished_at)

    def test_worker_skips_cancelled_job_and_unknown_ids(self):
        job = self._start()
        self.assertTrue(sj.cancel_job_ng(job.job_id))
        with mock.patch.object(sj.character_sheet, "generate_character_sheet_ng") as gen:
            self._run_worker(job)
            sj._QUEUE.ids = ["ghost"]
            with self.assertRaises(SystemExit):
                sj._worker_loop()
        gen.assert_not_called()
        self.assertEqual(job.error_type, "SheetJobCancelled")

    # --- cancel ----------------------------------------------------------
    def test_cancel_unknown_or_finished(self):
        self.assertFalse(sj.cancel_job_ng("nope"))
        job = self._start()
        job.finished_at = 1.0
        self.assertFalse(sj.cancel_job_ng(job.job_id))

    def test_cancel_running_kills_group_then_falls_back_to_proc_kill(self):
        job = self._start()
        job.dispatched_at = 1.0
        job._proc = mock.Mock(pid=5)
        with mock.patch.object(sj.os, "getpgid", return_value=50), \
             mock.patch.object(sj.os, "killpg") as killpg:
            self.assertTrue(sj.cancel_job_ng(job.job_id))
        killpg.assert_called_once_with(50, sj.signal.SIGKILL)
        with mock.patch.object(sj.os, "getpgid", side_effect=ProcessLookupError):
            sj.cancel_job_ng(job.job_id)
        job._proc.kill.assert_called_once()

    # --- reroll -----------------------------------------------------------
    def test_reroll_validates_shot_exists_and_carries_prompt(self):
        cs = sj.character_sheet
        meta = {"views": [{"key": "front", "prompt": "old prompt"}]}
        with mock.patch.object(cs, "character_sheet_meta_ng", return_value=meta):
            job = sj.start_reroll_ng("tom", "front", seed=3)
            self.assertEqual(job.shot_prompts, {"front": "old prompt"})
            self.assertTrue(job.params["reroll"])
            with self.assertRaisesRegex(LookupError, "no shot 'back'"):
                sj.start_reroll_ng("tom", "back")
            custom = sj.start_reroll_ng("tom", "front", prompt="new one")
            self.assertEqual(custom.shot_prompts["front"], "new one")
        with mock.patch.object(cs, "character_exists_ng", return_value=False):
            with self.assertRaises(LookupError):
                sj.start_reroll_ng("tom", "front")

    def test_reroll_worker_calls_regenerate(self):
        cs = sj.character_sheet
        with mock.patch.object(cs, "character_sheet_meta_ng",
                               return_value={"views": [{"key": "front"}]}):
            job = sj.start_reroll_ng("tom", "front", seed=3, steps=10)
        with mock.patch.object(cs, "regenerate_shot_ng") as regen:
            self._run_worker(job)
        self.assertEqual(regen.call_args.args, ("tom", "front"))
        self.assertEqual((regen.call_args.kwargs["seed"], regen.call_args.kwargs["steps"]), (3, 10))

    # --- status inference -------------------------------------------------
    def test_status_unknown_job(self):
        with self.assertRaises(LookupError):
            sj.job_status_ng("nope")

    def test_running_status_infers_per_shot_from_disk_after_dispatch(self):
        job = self._start()
        job.dispatched_at = time.time() - 100
        stale = self._make_png("front", job.dispatched_at - 50)  # from before this job
        status = sj.job_status_ng(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["rendering", "queued"])
        self.assertEqual(status["status"], "running")
        stale.unlink()
        fresh = self._make_png("front", job.dispatched_at + 10)
        status = sj.job_status_ng(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["done", "rendering"])
        self.assertEqual(status["shots"][0]["thumbnail"], str(fresh))

    def test_failed_job_marks_first_unfinished_shot_failed(self):
        job = self._start()
        job.dispatched_at = time.time() - 100
        self._make_png("front", job.dispatched_at + 10)
        job.finished_at, job.error = time.time(), "boom"
        status = sj.job_status_ng(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["done", "failed"])
        self.assertEqual((status["status"], status["error"]), ("failed", "boom"))
        self.assertIsNone(status["sheet_url"])

    def test_completed_job_reports_sheet_url_and_all_done(self):
        job = self._start()
        job.dispatched_at, job.finished_at = time.time() - 5, time.time()
        status = sj.job_status_ng(job.job_id)
        self.assertEqual(status["status"], "completed")
        self.assertTrue(all(s["status"] == "done" for s in status["shots"]))
        self.assertEqual(status["sheet_url"], "/api/ng/generate/characters/tom/sheet")

    def test_log_ring_buffer_keeps_last_200(self):
        job = self._start()
        for i in range(250):
            job.append_log(f"l{i}")
        self.assertEqual(len(job.log_lines), 200)
        self.assertEqual(job.log_tail(2), ["l248", "l249"])


if __name__ == "__main__":
    unittest.main()
