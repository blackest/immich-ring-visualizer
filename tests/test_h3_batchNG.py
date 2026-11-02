"""Unit tests for a scene of windows with their own references (h3_batchNG.py): what each window is
given, that the next window is queued only when the one before it finishes, and the join at the end.
The job queue, ffmpeg and the job log are replaced with fakes, so no GPU is needed.

Run with: python3 -m unittest tests.test_h3_batchNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_batchNG as b


class FakeJob:
    def __init__(self, job_id, on_done, mp4):
        self.job_id, self.on_done, self.mp4_path = job_id, on_done, mp4


class Rig:
    """Records every job the batch queues; `finish(n)` plays the worker finishing job n."""

    def __init__(self, tmp):
        self.tmp, self.calls, self.jobs, self.stitched, self.filed = Path(tmp), [], [], [], []

    def submit(self, image, ext, prompt, seconds, seed=None, **kw):
        mp4 = self.tmp / f"job{len(self.jobs)}.mp4"
        mp4.write_bytes(b"clip%d" % len(self.jobs))
        job = FakeJob(f"job{len(self.jobs)}", kw["on_done"], str(mp4))
        self.calls.append({"prompt": prompt, "seconds": seconds, "image": image,
                           **{k: v for k, v in kw.items() if k != "on_done"}})
        self.jobs.append(job)
        return job

    def last_frame(self, clip):
        return b"last:" + Path(clip).name.encode()

    def stitcher(self, clips, joins, out):
        self.stitched.append((list(clips), joins))
        Path(out).write_bytes(b"film")
        return {"seconds": 15.0}

    def recorder(self, **kw):
        self.filed.append(kw)
        return {"job_id": "abc123abc123", "date": "2026-11-02"}

    def start(self, windows, **kw):
        return b.start_batch_ng(windows, submit=self.submit, last_frame=self.last_frame,
                                stitcher=self.stitcher, recorder=self.recorder, **kw)

    def finish(self, n):
        self.jobs[n].on_done(self.jobs[n])


def win(text, *refs):
    return {"prompt": text, "refs": [(r, ".png") for r in refs]}


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.old = b.H3GEN_DIR
        b.H3GEN_DIR = os.path.join(self.dir.name, "gen")
        self.addCleanup(setattr, b, "H3GEN_DIR", self.old)
        self.rig = Rig(self.dir.name)

    def test_only_the_first_window_is_queued_at_the_start(self):
        self.rig.start([win("a", b"A"), win("b", b"B"), win("c", b"C")])
        self.assertEqual(len(self.rig.calls), 1)
        call = self.rig.calls[0]
        self.assertEqual((call["prompt"], call["seconds"], call["model"]), ("a", 5.0, "h3ref"))
        self.assertEqual([p[0] for p in call["references"]], [b"A"])

    def test_each_window_gets_shared_then_own_then_the_carried_frame(self):
        self.rig.start([win("a", b"A"), win("b", b"B")], shared=[(b"S", ".png")], carry=True)
        self.rig.finish(0)
        self.assertEqual([p[0] for p in self.rig.calls[0]["references"]], [b"S", b"A"])
        second = self.rig.calls[1]["references"]
        self.assertEqual([p[0] for p in second], [b"S", b"B", b"last:w1.mp4"])
        self.assertEqual(second[-1][1], ".png")

    def test_no_carry_means_no_extra_picture(self):
        self.rig.start([win("a", b"A"), win("b", b"B")], carry=False)
        self.rig.finish(0)
        self.assertEqual([p[0] for p in self.rig.calls[1]["references"]], [b"B"])

    def test_the_last_window_joins_and_files_the_film(self):
        batch = self.rig.start([win("a"), win("b"), win("c")], joins=[{"seam": 2, "type": "dissolve", "seconds": 0.5}],
                               settings={"width": 640, "height": 384, "tristep": True})
        for n in range(3):
            self.rig.finish(n)
        self.assertEqual(len(self.rig.calls), 3)
        clips, joins = self.rig.stitched[0]
        self.assertEqual([Path(c).name for c in clips], ["w1.mp4", "w2.mp4", "w3.mp4"])
        self.assertEqual([j["type"] for j in joins], ["cut", "dissolve"])
        self.assertTrue(all(c["tristep"] for c in self.rig.calls))
        filed = self.rig.filed[0]
        self.assertEqual((filed["width"], filed["height"], filed["duration_s"]), (640, 384, 15.0))
        self.assertIn("job0, job1, job2", filed["notes"])
        status = b.batch_status_ng(batch["id"], job_status=lambda j: {"status": "completed"})
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["film"]["video_url"], "/api/ng/h3/logs/2026-11-02/abc123abc123/video")

    def test_status_while_running_and_when_a_window_fails(self):
        batch = self.rig.start([win("a"), win("b")])
        states = {"job0": "rendering"}
        status = b.batch_status_ng(batch["id"], job_status=lambda j: {"status": states[j]})
        self.assertEqual(status["status"], "running")
        self.assertEqual([w["status"] for w in status["windows"]], ["rendering", "waiting"])
        states["job0"] = "failed"
        self.assertEqual(b.batch_status_ng(batch["id"], job_status=lambda j: {"status": states[j]})["status"], "failed")

    def test_cancel_stops_the_next_window_being_queued(self):
        batch = self.rig.start([win("a"), win("b")])
        cancelled = []
        old = b.h3_jobsNG.cancel_h3_job_ng
        b.h3_jobsNG.cancel_h3_job_ng = lambda j: cancelled.append(j) or True
        self.addCleanup(setattr, b.h3_jobsNG, "cancel_h3_job_ng", old)
        self.assertTrue(b.cancel_batch_ng(batch["id"]))
        self.rig.finish(0)
        self.assertEqual(len(self.rig.calls), 1)
        self.assertEqual(cancelled, ["job0"])

    def test_a_failure_after_a_render_is_the_batch_error_not_a_raise(self):
        self.rig.last_frame = lambda clip: (_ for _ in ()).throw(RuntimeError("no frame"))
        batch = self.rig.start([win("a"), win("b")])
        self.rig.finish(0)  # must not raise
        self.assertIn("no frame", batch["error"])

    def test_refused_before_anything_is_queued(self):
        for kwargs, text in [
            (dict(windows=[win("a")]), "needs 2 to 12"),
            (dict(windows=[win("a"), win("")]), "window 2 has no prompt"),
            (dict(windows=[win("a"), win("b")], settings={"model": "h3q8"}), "Ref2VA"),
            (dict(windows=[win("a", *[b"x"] * 9), win("b")], carry=True), None),  # 9 is fine for window 1
        ]:
            if text is None:
                self.rig.start(**kwargs)
                continue
            with self.assertRaisesRegex(ValueError, text):
                self.rig.start(**kwargs)
        n = len(self.rig.calls)
        with self.assertRaisesRegex(ValueError, "window 2 has 10 pictures"):
            self.rig.start([win("a"), win("b", *[b"x"] * 9)], carry=True)
        self.assertEqual(len(self.rig.calls), n)


if __name__ == "__main__":
    unittest.main()
