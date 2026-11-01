"""A video render waits while a Klein picture is on the GPU, and a cancel ends the wait.

Run with: python3 -m unittest tests.test_klein_h3_waitNG -v
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_jobsNG as h3
import klein_jobsNG as klein


class Wait(unittest.TestCase):
    def test_klein_registers_its_guard(self):
        self.assertIn(klein.running_ng, h3._GPU_GUARDS)

    def test_running_reflects_job_status(self):
        with mock.patch.dict(klein._JOBS, {"x": {"status": "queued"}}, clear=True):
            self.assertFalse(klein.running_ng())
            klein._JOBS["x"]["status"] = "running"
            self.assertTrue(klein.running_ng())

    def test_waits_until_the_guard_clears(self):
        answers = iter([True, True, False])
        job = SimpleNamespace(cancelled=False)
        with mock.patch.object(h3, "_GPU_GUARDS", [lambda: next(answers)]), mock.patch.object(h3.time, "sleep") as sl:
            h3._wait_for_other_gpu_work_ng(job)
        self.assertEqual(sl.call_count, 2)

    def test_cancel_ends_the_wait(self):
        job = SimpleNamespace(cancelled=True)
        with mock.patch.object(h3, "_GPU_GUARDS", [lambda: True]), mock.patch.object(h3.time, "sleep") as sl:
            h3._wait_for_other_gpu_work_ng(job)
        sl.assert_not_called()

    def test_no_guards_means_no_wait(self):
        with mock.patch.object(h3, "_GPU_GUARDS", []), mock.patch.object(h3.time, "sleep") as sl:
            h3._wait_for_other_gpu_work_ng(SimpleNamespace(cancelled=False))
        sl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
