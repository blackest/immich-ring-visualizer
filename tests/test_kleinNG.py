"""Klein pictures: the worker's command line, the engine's checks, the cache lookup, and the routes
(with the model itself replaced, so nothing renders).

Run with: python3 -m unittest tests.test_kleinNG -v
"""

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import klein_engineNG as engine
import klein_jobsNG as jobs
import klein_workerNG as worker

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82")


class WorkerArgs(unittest.TestCase):
    base = {"model": "/m", "variant": "distilled", "prompt": "p", "width": 1024, "height": 576, "refs": []}

    def test_distilled_text_to_image(self):
        a = worker.cli_args(self.base, 5, Path("o.png"))
        self.assertEqual(a[a.index("--steps") + 1], "16")
        self.assertEqual(a[a.index("--base-model") + 1], "flux3-image-klein")
        self.assertNotIn("--guidance", a)
        self.assertNotIn("--image-paths", a)

    def test_base_uses_fifty_steps_and_guidance(self):
        a = worker.cli_args({**self.base, "variant": "base"}, 5, Path("o.png"))
        self.assertEqual(a[a.index("--steps") + 1], "50")
        self.assertEqual(a[a.index("--base-model") + 1], "flux3-image-klein-base")
        self.assertEqual(a[a.index("--guidance") + 1], "3.5")

    def test_references_are_passed_in_order(self):
        a = worker.cli_args({**self.base, "refs": ["a.png", "b.png", "c.png"]}, 5, Path("o.png"))
        self.assertEqual(a[a.index("--image-paths") + 1:a.index("--image-paths") + 4], ["a.png", "b.png", "c.png"])


class EngineChecks(unittest.TestCase):
    def test_size_rules(self):
        engine.validate_ng(1024, 576, [], 1)
        for w, h in ((1000, 576), (2048, 1024), (128, 128)):
            with self.subTest(size=(w, h)), self.assertRaises(ValueError):
                engine.validate_ng(w, h, [], 1)

    def test_reference_and_count_limits(self):
        with self.assertRaises(ValueError):
            engine.validate_ng(512, 512, ["x"] * 11, 1)
        with self.assertRaises(ValueError):
            engine.validate_ng(512, 512, [], 0)

    def test_snapshot_lookup_needs_a_complete_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "models--mflux-community--flux-3-image-klein-mflux-q8" / "snapshots" / "abc"
            (snap / "transformer").mkdir(parents=True)
            with mock.patch.object(engine, "cache_roots", return_value=[Path(tmp)]):
                self.assertIsNone(engine.snapshot_ng("distilled-q8"))
                (snap / "manifest.json").write_text("{}")
                (snap / "transformer" / "0.safetensors").write_bytes(b"x")
                self.assertEqual(engine.snapshot_ng("distilled-q8"), snap)
                self.assertIsNone(engine.snapshot_ng("base-q4"))
                with self.assertRaises(ValueError):
                    engine.snapshot_ng("nonsense")


class Routes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        import routes.kleinNG as routes
        self.routes = routes
        for mod in (routes, jobs):
            patcher = mock.patch.object(mod, "KLEIN_DIR", self.tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        ready = {"ready": True, "worker_found": True, "models": ["distilled-q8"], "default": "distilled-q8",
                 "worker_python": "x"}
        for target, value in (("klein_health_ng", ready),):
            patcher = mock.patch.object(engine, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls = []

        def fake(prompt, n, width, height, output_dir, seed, refs=(), model="", on_log=None, should_stop=None,
                 on_result=None):
            self.calls.append({"prompt": prompt, "n": n, "refs": [Path(r).name for r in refs], "seed": seed})
            os.makedirs(output_dir, exist_ok=True)
            for i in range(n):
                path = Path(output_dir) / f"klein_{seed + i}.png"
                path.write_bytes(PNG)
                on_result({"png_path": str(path), "seed": seed + i, "width": width, "height": height,
                           "elapsed_seconds": 1.0})
        patcher = mock.patch.object(engine, "generate_klein_ng", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("h3_jobsNG.busy_ng", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(routes.kleinNG_bp)
        self.client = app.test_client()

    def wait(self, job_id):
        for _ in range(100):
            body = self.client.get(f"/api/ng/klein/status/{job_id}").get_json()
            if body["status"] in ("done", "error", "cancelled"):
                return body
            time.sleep(0.05)
        self.fail("job did not finish")

    def post(self, **data):
        return self.client.post("/api/ng/klein/generate", data=data, content_type="multipart/form-data")

    def test_text_only_job_runs_and_serves_its_pictures(self):
        r = self.post(prompt="a kitchen", n="2", width="768", height="432", seed="40")
        self.assertEqual(r.status_code, 200)
        body = self.wait(r.get_json()["jobId"])
        self.assertEqual(body["status"], "done")
        self.assertEqual([p["seed"] for p in body["results"]], [40, 41])
        self.assertEqual(self.calls[-1]["refs"], [])
        img = self.client.get(f"/api/ng/klein/image/{body['id']}/{body['results'][0]['file']}")
        self.assertEqual(img.status_code, 200)

    def test_slots_keep_their_order_and_skip_empty_ones(self):
        r = self.post(prompt="x", n="1", ref0=(io.BytesIO(PNG), "a.png"), ref2=(io.BytesIO(PNG), "c.jpg"))
        body = self.wait(r.get_json()["jobId"])
        self.assertEqual(self.calls[-1]["refs"], ["subject.png", "extra.jpg"])
        self.assertEqual(body["refs"], ["subject", "extra"])

    def test_an_earlier_picture_can_fill_a_slot(self):
        first = self.wait(self.post(prompt="x", n="1", seed="1").get_json()["jobId"])
        ref = f"{first['id']}/{first['results'][0]['file']}"
        second = self.post(prompt="y", n="1", ref_from1=ref)
        self.assertEqual(self.wait(second.get_json()["jobId"])["refs"], ["location"])

    def test_bad_requests_are_refused_and_leave_nothing_behind(self):
        for data in ({"prompt": ""}, {"prompt": "x", "width": "1000"}, {"prompt": "x", "ref_from0": "../etc/passwd"},
                     {"prompt": "x", "ref_from0": "nojob/nofile.png"}):
            with self.subTest(data=data):
                self.assertEqual(self.post(**data).status_code, 400)
        self.assertEqual([p for p in Path(self.tmp.name).glob("*/job.json")], [])

    def test_pictures_are_only_served_from_a_job_folder(self):
        self.assertEqual(self.client.get("/api/ng/klein/image/nope/x.png").status_code, 404)
        for ref in ("../x/y.png", "a/../b.png", "a\\b.png", "a/b/c.png", "a/"):
            with self.subTest(ref=ref):
                self.assertIsNone(self.routes._picture_path(ref))

    def test_the_job_list_survives_a_restart(self):
        job = self.wait(self.post(prompt="remember me", n="1").get_json()["jobId"])
        with jobs._LOCK:
            jobs._JOBS.clear()
        listed = self.client.get("/api/ng/klein/jobs").get_json()["jobs"]
        self.assertIn(job["id"], [j["id"] for j in listed])


if __name__ == "__main__":
    unittest.main()
