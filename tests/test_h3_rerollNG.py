"""Unit tests for re-rolling one window of a chain (h3_rerollNG.py and routes/h3_rerollNG.py): the
window manifest built from a job's own files, the one-window job a re-roll starts, the new film
version it makes, and the routes. Clips are tiny real ffmpeg files; no model runs.

Run with: python3 -m unittest tests.test_h3_rerollNG -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_jobsNG as jobs
import h3_rerollNG as rr
import routes.h3_rerollNG as routes
from tests.test_h3_stitchNG import HAVE_FFMPEG, make_clip, stream_info


def make_job(tmp, windows=3, frames=48):
    """A finished chained job's folder: window clips, the runner's metrics, the stitched film."""
    job_dir = Path(tmp) / "job1"
    job_dir.mkdir()
    for n in range(1, windows + 1):
        make_clip(job_dir / f"h3_100_w{n}.mp4", frames, 300 + 50 * n)
    shutil.copyfile(job_dir / "h3_100_w1.mp4", job_dir / "h3_100.mp4")
    (job_dir / "h3_100.metrics.json").write_text(json.dumps({
        "delivered_frames": frames + (frames - 1) * (windows - 1) - 2,
        "windows": [{"window": n, "seed": 10 + n, "prompt": f"prompt {n}"} for n in range(1, windows + 1)]}))
    return job_dir


def live_job(job_dir, **kw):
    fields = dict(job_id="job1", job_dir=job_dir, mp4_path=str(job_dir / "h3_100.mp4"), finished_at=1.0,
                  error=None, model="h3q8", width=640, height=384, steps=4, turbo=False, v13=False,
                  tristep=True, loras=[{"name": "face", "path": "/l/face.safetensors", "strength": 1.0}],
                  has_image=False)
    fields.update(kw)
    return SimpleNamespace(**fields)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class RerollBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.job_dir = make_job(self.tmp)
        self.src = live_job(self.job_dir)
        patch = mock.patch.object(rr.h3_jobs, "get_job_ng", side_effect=lambda i: self.src if i == "job1" else None)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(rr.h3_jobs, "H3GEN_DIR", self.tmp)
        patch.start()
        self.addCleanup(patch.stop)


class Manifest(RerollBase):
    def test_built_from_the_jobs_own_files_and_kept(self):
        man = rr.load_manifest_ng(self.src)
        self.assertEqual([w["index"] for w in man["windows"]], [1, 2, 3])
        self.assertEqual(man["windows"][1]["seed"], 12)
        self.assertEqual(man["windows"][2]["prompt"], "prompt 3")
        self.assertEqual(man["trim_tail"], 2)
        self.assertTrue((self.job_dir / rr.MANIFEST_NAME).is_file())

    def test_a_single_window_render_has_nothing_to_reroll(self):
        for p in self.job_dir.glob("h3_100_w[23].mp4"):
            p.unlink()
        with self.assertRaisesRegex(ValueError, "single window"):
            rr.load_manifest_ng(self.src)

    def test_describe_lists_windows_joins_and_the_original(self):
        d = rr.describe_ng("job1")
        self.assertEqual(len(d["windows"]), 3)
        self.assertEqual([j["type"] for j in d["joins"]], ["cut", "cut"])
        self.assertEqual(d["versions"], [{"n": 0, "label": "original"}])

    def test_an_unknown_or_unfinished_job_is_refused(self):
        with self.assertRaises(LookupError):
            rr.describe_ng("nope")
        self.src.finished_at = None
        with self.assertRaisesRegex(ValueError, "not finished"):
            rr.describe_ng("job1")

    def test_a_job_gone_from_memory_is_found_by_its_saved_settings(self):
        (self.job_dir / jobs.SETTINGS_NAME).write_text(json.dumps({
            "model": "h3q8", "width": 640, "height": 384, "steps": 4, "turbo": False, "v13": False,
            "tristep": True, "has_image": False, "timeout_s": None, "loras": []}))
        shutil.move(str(self.job_dir), str(Path(self.tmp) / "abc123"))
        with mock.patch.object(rr.h3_jobs, "get_job_ng", return_value=None):
            src = rr.resolve_source_ng("abc123")
        self.assertEqual((src.model, src.tristep, src.width), ("h3q8", True, 640))


class StartReroll(RerollBase):
    def start(self, window, **kw):
        with mock.patch.object(rr.h3_jobs, "start_h3_job_ng", return_value=SimpleNamespace(job_id="r1")) as start, \
                mock.patch.object(rr, "last_frame_png_ng", return_value=b"PNG") as last:
            rr.start_reroll_ng("job1", window, **kw)
        return start, last

    def test_one_window_job_with_the_source_settings(self):
        start, _ = self.start(2, seed=77)
        args, kw = start.call_args
        self.assertEqual(args[0], b"PNG")  # starts on the previous window's last frame
        self.assertEqual(args[2:5], ("prompt 2", rr.window_seconds_ng(), 77))
        self.assertEqual((kw["model"], kw["width"], kw["height"], kw["tristep"]), ("h3q8", 640, 384, True))
        self.assertEqual(kw["loras"], [{"name": "face", "path": "/l/face.safetensors", "strength": 1.0}])
        self.assertEqual(kw["reroll"]["window"], 2)
        self.assertIs(kw["on_done"], rr.finish_reroll_ng)

    def test_the_prompt_can_be_changed_and_the_start_frame_declined(self):
        start, last = self.start(3, prompt="  in his pale shirt  ", from_previous=False)
        self.assertEqual(start.call_args.args[2], "in his pale shirt")
        self.assertIsNone(start.call_args.args[0])
        last.assert_not_called()

    def test_window_one_never_starts_from_a_previous_frame(self):
        start, last = self.start(1)
        self.assertIsNone(start.call_args.args[0])
        last.assert_not_called()

    def test_a_random_seed_when_none_is_given(self):
        start, _ = self.start(2)
        self.assertIsInstance(start.call_args.args[4], int)

    def test_refusals(self):
        for kw, text in ((dict(window=0), "window must be"), (dict(window=4), "window must be"),
                         (dict(window=2, prompt="  "), "no prompt"),
                         (dict(window=2, joins=[{"seam": 1, "type": "wipe"}]), "join type")):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, text):
                self.start(**kw)


class Finish(RerollBase):
    def test_the_new_clip_replaces_the_window_and_a_new_film_version_is_made(self):
        new = Path(self.tmp) / "new.mp4"
        make_clip(new, 48, 900)
        job = SimpleNamespace(reroll={"source": "job1", "window": 2, "joins": [{"seam": 1, "type": "dissolve", "seconds": 0.5}],
                                     "version": None}, mp4_path=str(new), seed=555, prompt="new prompt",
                              append_log=lambda line: None)
        rr.finish_reroll_ng(job)
        man = json.loads((self.job_dir / rr.MANIFEST_NAME).read_text())
        self.assertEqual(man["windows"][1]["clip"], "reroll_w2_v1.mp4")
        self.assertEqual((man["windows"][1]["seed"], man["windows"][1]["rerolled"]), (555, True))
        self.assertEqual(man["windows"][0]["rerolled"], False)
        self.assertEqual(job.reroll["version"], 1)
        self.assertEqual(self.src.mp4_path, str(self.job_dir / "film_v1.mp4"))
        # 48 + (47 - 12) + 47 - 2 trimmed
        self.assertEqual(stream_info(self.job_dir / "film_v1.mp4")["video"]["frames"], 48 + 35 + 47 - 2)
        self.assertTrue((self.job_dir / "h3_100_w2.mp4").is_file())  # the original window is kept

    def test_restitch_changes_only_the_joins(self):
        rr.restitch_ng("job1", [{"seam": 2, "type": "fade", "seconds": 1.0}])
        d = rr.describe_ng("job1")
        self.assertEqual([j["type"] for j in d["joins"]], ["cut", "fade"])
        self.assertEqual([v["n"] for v in d["versions"]], [0, 1])
        with self.assertRaisesRegex(ValueError, "join type"):
            rr.restitch_ng("job1", [{"seam": 1, "type": "wipe"}])
        self.assertEqual(len(rr.describe_ng("job1")["versions"]), 2)  # the refused one made nothing


class JobSettings(unittest.TestCase):
    """start_h3_job_ng saves what a re-roll needs, and carries the re-roll record and hook."""

    def test_settings_are_saved_beside_the_clips(self):
        from tests.test_h3_tristepNG import _Base  # noqa: F401 -- same runner stub as the other job tests
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        runner = tmp / "runner.py"
        runner.write_text("--lora PATH\nRepeat the flag to stack adapters\n--sigma-subset X\n")
        hook = lambda job: None  # noqa: E731
        for target, name, val in ((jobs, "H3GEN_DIR", str(tmp / "gen")), (jobs, "_QUEUE", mock.MagicMock()),
                                  (jobs, "_ensure_worker_started", lambda: None),
                                  (jobs.h3, "_resolve_h3_runner_ng", lambda c: str(runner)),
                                  (jobs.h3, "_resolve_h3_tristep_ng", lambda: "/x/tristep.safetensors")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(jobs._JOBS.clear)
        job = jobs.start_h3_job_ng(None, ".png", "p", 5.0, 1, tristep=True, reroll={"source": "a", "window": 2},
                                   on_done=hook)
        saved = json.loads((job.job_dir / jobs.SETTINGS_NAME).read_text())
        self.assertEqual((saved["model"], saved["tristep"], saved["steps"], saved["has_image"]),
                         ("h3q8", True, jobs.h3.H3_TRISTEP_STEPS, False))
        self.assertIs(job.on_done, hook)
        self.assertEqual(jobs.job_status_ng(job.job_id)["reroll"], {"source": "a", "window": 2})


class Routes(RerollBase):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(routes.h3RerollNG_bp)
        self.client = app.test_client()

    def test_windows_listing_and_unknown_job(self):
        r = self.client.get("/api/ng/h3/jobs/job1/windows")
        self.assertEqual((r.status_code, len(r.get_json()["windows"])), (200, 3))
        self.assertEqual(self.client.get("/api/ng/h3/jobs/zzz/windows").status_code, 404)

    def test_reroll_starts_a_job_or_says_why_not(self):
        with mock.patch.object(routes.reroll, "start_reroll_ng", return_value=SimpleNamespace(job_id="r9")) as start:
            r = self.client.post("/api/ng/h3/jobs/job1/reroll", json={"window": 2, "seed": 5, "from_previous": False})
        self.assertEqual((r.status_code, r.get_json()["job_id"]), (202, "r9"))
        self.assertEqual(start.call_args.kwargs["from_previous"], False)
        self.assertEqual(self.client.post("/api/ng/h3/jobs/job1/reroll", json={"window": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/api/ng/h3/jobs/job1/reroll", json={"window": 9}).status_code, 400)

    def test_a_job_gone_from_memory_still_plays_its_latest_film(self):
        import routes.h3NG as h3_routes
        (self.job_dir / jobs.SETTINGS_NAME).write_text(json.dumps({
            "model": "h3q8", "width": 640, "height": 384, "steps": 4, "turbo": False, "v13": False,
            "tristep": True, "has_image": False, "timeout_s": None, "loras": []}))
        shutil.move(str(self.job_dir), str(Path(self.tmp) / "abc123"))
        app = Flask(__name__)
        app.register_blueprint(h3_routes.h3NG_bp)
        client = app.test_client()
        with mock.patch.object(rr.h3_jobs, "get_job_ng", return_value=None), \
                mock.patch.object(h3_routes.h3_jobs, "get_job_ng", return_value=None):
            self.assertEqual(client.get("/api/ng/h3/jobs/abc123/video").status_code, 200)
            self.assertEqual(client.get("/api/ng/h3/jobs/nope/video").status_code, 404)

    def test_restitch_and_serving_the_versions(self):
        r = self.client.post("/api/ng/h3/jobs/job1/restitch", json={"joins": [{"seam": 1, "type": "fade", "seconds": 0.5}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/ng/h3/jobs/job1/versions/0/video").status_code, 200)
        self.assertEqual(self.client.get("/api/ng/h3/jobs/job1/versions/1/video").status_code, 200)
        self.assertEqual(self.client.get("/api/ng/h3/jobs/job1/versions/5/video").status_code, 404)
        self.assertEqual(self.client.get("/api/ng/h3/jobs/job1/windows/2/video").status_code, 200)
        self.assertEqual(self.client.get("/api/ng/h3/jobs/job1/windows/7/video").status_code, 404)


if __name__ == "__main__":
    unittest.main()
