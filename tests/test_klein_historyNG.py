"""Klein history: ten reference slots, reference files kept, search, paging, delete.

Run with: python3 -m unittest tests.test_klein_historyNG -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import klein_jobsNG as jobs
from tests.test_kleinNG import PNG


def make_job(root, jid, prompt, created, status="done", refs=()):
    d = Path(root) / jid
    (d / "out").mkdir(parents=True); (d / "refs").mkdir()
    (d / "out" / "a.png").write_bytes(PNG)
    rs = []
    for slot in refs:
        (d / "refs" / f"{slot}.png").write_bytes(PNG)
        rs.append({"slot": slot, "path": str(d / "refs" / f"{slot}.png")})
    job = {"id": jid, "prompt": prompt, "n": 1, "width": 64, "height": 64, "seed": 1, "model": "distilled-q8",
           "refs": rs, "options": {"steps": 8}, "outpaint": None, "status": status, "error": None,
           "results": [{"file": "a.png", "seed": 1, "width": 64, "height": 64, "seconds": 1}], "log": [],
           "created_at": created, "cancel": False}
    (d / "job.json").write_text(json.dumps(job))
    return job


class History(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for patch in (mock.patch.object(jobs, "KLEIN_DIR", self.root), mock.patch("routes.kleinNG.KLEIN_DIR", self.root),
                      mock.patch.dict(jobs._JOBS, {}, clear=True)):
            patch.start(); self.addCleanup(patch.stop)
        make_job(self.root, "j1", "a red vase", 1)
        make_job(self.root, "j2", "a green apple", 2, refs=("subject", "ref4"))
        make_job(self.root, "j3", "mountains", 3, status="running")
        from flask import Flask
        from routes.kleinNG import kleinNG_bp
        app = Flask(__name__); app.register_blueprint(kleinNG_bp); self.c = app.test_client()
        self.addCleanup(self.tmp.cleanup)

    def test_ten_slots_in_order(self):
        self.assertEqual(len(jobs.SLOT_NAMES), 10)
        self.assertEqual(jobs.SLOT_NAMES[:4], ("subject", "location", "extra", "ref4"))

    def test_search_and_paging(self):
        b = self.c.get("/api/ng/klein/jobs?q=APPLE").get_json()
        self.assertEqual([j["id"] for j in b["jobs"]], ["j2"]); self.assertEqual(b["total"], 1)
        b = self.c.get("/api/ng/klein/jobs?limit=1").get_json()
        self.assertEqual([j["id"] for j in b["jobs"]], ["j3"]); self.assertEqual(b["total"], 3)
        self.assertEqual(self.c.get("/api/ng/klein/jobs?limit=x").status_code, 400)

    def test_reference_files_listed_and_served(self):
        j = [x for x in self.c.get("/api/ng/klein/jobs").get_json()["jobs"] if x["id"] == "j2"][0]
        self.assertEqual(j["ref_files"], ["subject.png", "ref4.png"])
        self.assertEqual(self.c.get("/api/ng/klein/ref/j2/ref4.png").status_code, 200)
        self.assertEqual(self.c.get("/api/ng/klein/ref/j2/..%2Fjob.json").status_code, 404)
        self.assertEqual(self.c.get("/api/ng/klein/ref/j1/nothing.png").status_code, 404)

    def test_delete(self):
        self.assertEqual(self.c.post("/api/ng/klein/delete/j1").status_code, 200)
        self.assertFalse((Path(self.root) / "j1").exists())
        self.assertEqual(self.c.post("/api/ng/klein/delete/j1").status_code, 404)
        jobs._JOBS["j3"] = json.loads((Path(self.root) / "j3" / "job.json").read_text())
        self.assertEqual(self.c.post("/api/ng/klein/delete/j3").status_code, 409)
        self.assertTrue((Path(self.root) / "j3").exists())
        self.assertEqual(self.c.post("/api/ng/klein/delete/..").status_code, 404)


if __name__ == "__main__":
    unittest.main()
