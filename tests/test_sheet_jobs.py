"""Unit tests for sheet_jobs.py -- the original thread-per-job character-sheet
runner (the NG twin, sheet_jobsNG, has its own file). character_sheet is mocked
at the boundary; threads run synchronously via a stand-in so tests are
deterministic; per-shot status inference runs against real files in a temp dir.

Run with: python3 -m unittest tests.test_sheet_jobs -v
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

import sheet_jobs as sj


class _SyncThread:
    """Runs the target inside start() so a job is finished when start_* returns."""
    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


class _NoRunThread(_SyncThread):
    """Never runs the target, so a job stays 'running'."""
    def start(self):
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
            mock.patch.object(sj.threading, "Thread", _SyncThread),
            mock.patch.object(cs, "_safe_id", side_effect=lambda c: c),
            mock.patch.object(cs, "character_exists", return_value=True),
            mock.patch.object(cs, "character_avatar", return_value=self.tmp / "a.png"),
            mock.patch.object(cs, "resolve_shots", return_value=[_spec("front"), _spec("side")]),
            mock.patch.object(cs, "_shot_prompt", side_effect=lambda spec, *a, **k: f"prompt-{spec.key}"),
            mock.patch.object(cs, "_character_dir", return_value=self.tmp),
            mock.patch.object(cs, "generate_character_sheet"),
            mock.patch.object(cs, "regenerate_shot"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _make_png(self, shot, mtime):
        d = self.tmp / "sheet_views" / shot
        d.mkdir(parents=True, exist_ok=True)
        p = d / "cand_1.png"
        p.write_bytes(b"x")
        os.utime(p, (mtime, mtime))
        return p

    # --- start_job ---------------------------------------------------------
    def test_start_job_runs_generation_with_all_params_and_collects_prompts(self):
        gen = sj.character_sheet.generate_character_sheet

        def fake(cid, **kw):
            kw["on_log"]("rendering front")
        gen.side_effect = fake
        job = sj.start_job("tom", wardrobe="red coat", seed=9, style="noir")
        self.assertEqual(gen.call_args.args, ("tom",))
        self.assertEqual(gen.call_args.kwargs["wardrobe"], "red coat")
        self.assertEqual(gen.call_args.kwargs["seed"], 9)
        self.assertEqual(gen.call_args.kwargs["style"], "noir")
        self.assertEqual(job.shot_keys, ["front", "side"])
        self.assertEqual(job.shot_prompts, {"front": "prompt-front", "side": "prompt-side"})
        self.assertEqual(job.log_tail(), ["rendering front"])
        self.assertIsNotNone(job.finished_at)
        self.assertIsNone(job.error)

    def test_custom_shots_or_views_report_preset_custom(self):
        self.assertEqual(sj.start_job("tom").params["preset"], "default")
        self.assertEqual(sj.start_job("tom", shots=[{"x": 1}]).params["preset"], "custom")
        self.assertEqual(sj.start_job("tom", views=["back"]).params["preset"], "custom")

    def test_validation_errors_raise_before_any_job_is_created(self):
        cs = sj.character_sheet
        with mock.patch.object(cs, "character_exists", return_value=False):
            with self.assertRaises(LookupError):
                sj.start_job("tom")
        with mock.patch.object(cs, "character_avatar", return_value=None):
            with self.assertRaises(FileNotFoundError):
                sj.start_job("tom")
        with mock.patch.object(cs, "_safe_id", side_effect=ValueError("bad id")):
            with self.assertRaises(ValueError):
                sj.start_job("../evil")
        self.assertEqual(sj._JOBS, {})
        cs.generate_character_sheet.assert_not_called()

    def test_engine_failure_is_recorded_not_raised(self):
        sj.character_sheet.generate_character_sheet.side_effect = RuntimeError("gpu fell over")
        job = sj.start_job("tom")
        self.assertEqual((job.error, job.error_type), ("gpu fell over", "RuntimeError"))
        self.assertIsNotNone(job.finished_at)

    # --- reroll ----------------------------------------------------------------
    def test_reroll_validates_shot_exists_and_carries_prompt(self):
        cs = sj.character_sheet
        meta = {"views": [{"key": "front", "prompt": "old prompt"}]}
        with mock.patch.object(cs, "character_sheet_meta", return_value=meta):
            job = sj.start_reroll("tom", "front", seed=3)
            self.assertEqual(job.shot_prompts, {"front": "old prompt"})
            self.assertTrue(job.params["reroll"])
            cs.regenerate_shot.assert_called_once()
            self.assertEqual(cs.regenerate_shot.call_args.args, ("tom", "front"))
            self.assertEqual(cs.regenerate_shot.call_args.kwargs["seed"], 3)
            with self.assertRaisesRegex(LookupError, "no shot 'back'"):
                sj.start_reroll("tom", "back")
            custom = sj.start_reroll("tom", "front", prompt="new one")
            self.assertEqual(custom.shot_prompts["front"], "new one")
        with mock.patch.object(cs, "character_exists", return_value=False):
            with self.assertRaises(LookupError):
                sj.start_reroll("tom", "front")

    # --- job registry ------------------------------------------------------
    def test_get_job_and_unknown_status(self):
        job = sj.start_job("tom")
        self.assertIs(sj.get_job(job.job_id), job)
        self.assertIsNone(sj.get_job("nope"))
        with self.assertRaises(LookupError):
            sj.job_status("nope")

    def test_prune_keeps_unfinished_and_drops_oldest_finished(self):
        with mock.patch.object(sj, "_MAX_JOBS_KEPT", 2):
            a, b = sj.start_job("tom"), sj.start_job("tom")
            a.finished_at, b.finished_at = 1.0, 2.0
            c = sj.start_job("tom")
            self.assertIsNone(sj.get_job(a.job_id))
            self.assertIsNotNone(sj.get_job(b.job_id))
            self.assertIsNotNone(sj.get_job(c.job_id))

    def test_log_ring_buffer_keeps_last_200(self):
        job = sj.start_job("tom")
        for i in range(250):
            job.append_log(f"l{i}")
        self.assertEqual(len(job.log_lines), 200)
        self.assertEqual(job.log_tail(2), ["l248", "l249"])

    # --- status inference --------------------------------------------------
    def test_running_status_infers_per_shot_from_disk_and_ignores_stale_files(self):
        with mock.patch.object(sj.threading, "Thread", _NoRunThread):
            job = sj.start_job("tom")
        job.started_at = time.time() - 100
        stale = self._make_png("front", job.started_at - 50)  # left over from an older run
        status = sj.job_status(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["rendering", "queued"])
        self.assertEqual(status["status"], "running")
        self.assertIsNone(status["sheet_url"])
        stale.unlink()
        fresh = self._make_png("front", job.started_at + 10)
        status = sj.job_status(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["done", "rendering"])
        self.assertEqual(status["shots"][0]["thumbnail"], str(fresh))
        self.assertEqual(status["shots"][0]["prompt"], "prompt-front")

    def test_failed_job_marks_first_unfinished_shot_failed_rest_not_started(self):
        with mock.patch.object(sj.threading, "Thread", _NoRunThread):
            job = sj.start_job("tom")
        job.started_at = time.time() - 100
        job.finished_at, job.error, job.error_type = time.time(), "boom", "RuntimeError"
        status = sj.job_status(job.job_id)
        self.assertEqual([s["status"] for s in status["shots"]], ["failed", "not_started"])
        self.assertEqual((status["status"], status["error"], status["error_type"]),
                         ("failed", "boom", "RuntimeError"))
        self.assertIsNone(status["sheet_url"])

    def test_completed_job_reports_sheet_url_and_all_done(self):
        job = sj.start_job("tom")
        status = sj.job_status(job.job_id)
        self.assertEqual(status["status"], "completed")
        self.assertTrue(all(s["status"] == "done" for s in status["shots"]))
        self.assertEqual(status["sheet_url"], "/api/charsheet/characters/tom/sheet")
        self.assertEqual(status["params"]["preset"], "default")


if __name__ == "__main__":
    unittest.main()
