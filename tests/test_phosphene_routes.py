"""Flask-test-client tests for routes/phosphene.py's "Add sheet to ring"
endpoint (POST /api/phosphene/characters/<id>/sheet/add-to-ring).

This is the automation of the existing folder-import flow: it hands the
character's avatar (reference) plus every rendered shot's *current*
image (character_sheet.sheet_shot_image_paths()) to the same
run_folder_analysis() function /api/analyze-folder drives, minus the
browser upload step, since these files already live on this server's
disk.

run_folder_analysis itself (InsightFace/onnxruntime) is stubbed here --
same tradeoff test_character_sheet.py makes for hidream_engine -- these
tests are about the route's request/response contract and the exact
image list it hands off, not the face-detection pipeline.

Run with:

    python3 -m unittest tests.test_phosphene_routes -v

from the repo root.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from PIL import Image

import character_sheet
import routes.phosphene as phosphene_routes
from routes.phosphene import phosphene_bp
from state import _analysis_jobs


def _write_tiny_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path, format="PNG")


def _make_app():
    app = Flask(__name__)
    app.register_blueprint(phosphene_bp)
    app.testing = True
    return app


class AddSheetToRingRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        character_sheet.EXPORT_DIR = self._tmp.name

        self._src = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        _write_tiny_png(Path(self._src.name))
        self.addCleanup(lambda: os.path.exists(self._src.name) and os.remove(self._src.name))

        self.app = _make_app()
        self.client = self.app.test_client()

        self._job_ids_created = []
        self.addCleanup(self._cleanup_jobs)

        # Stub the actual face-detection pipeline -- these tests check
        # what gets handed to it and the route's response, not
        # InsightFace/onnxruntime itself.
        self._run_calls = []
        patcher = mock.patch.object(
            phosphene_routes, "run_folder_analysis",
            side_effect=self._fake_run_folder_analysis)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _cleanup_jobs(self):
        for jid in self._job_ids_created:
            _analysis_jobs.pop(jid, None)

    def _fake_run_folder_analysis(self, job_id, image_paths, sim_threshold,
                                   blur_threshold, ref_index, cache_format):
        # Runs synchronously in the request thread in these tests (the
        # route still launches it via threading.Thread, but a fast fake
        # finishes long before the test asserts anything) -- record what
        # it was called with, then mark the job done like the real
        # function would on success.
        self._run_calls.append({
            "job_id": job_id, "image_paths": list(image_paths),
            "sim_threshold": sim_threshold, "blur_threshold": blur_threshold,
            "ref_index": ref_index, "cache_format": cache_format,
        })
        _analysis_jobs[job_id]["status"] = "done"

    def _wait_for_call(self, timeout=2.0):
        deadline = time.time() + timeout
        while not self._run_calls and time.time() < deadline:
            time.sleep(0.01)

    def test_unknown_character_404(self):
        resp = self.client.post("/api/phosphene/characters/nobody/sheet/add-to-ring")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("reference image", resp.get_json()["error"])

    def test_character_with_no_shots_yet_404(self):
        character_sheet.create_draft_character("ada", self._src.name)
        resp = self.client.post("/api/phosphene/characters/ada/sheet/add-to-ring")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("no rendered shots", resp.get_json()["error"])

    def test_bad_character_id_400(self):
        resp = self.client.post("/api/phosphene/characters/..%2F..%2Fetc/sheet/add-to-ring")
        self.assertIn(resp.status_code, (400, 404))

    def test_avatar_and_shots_sent_with_avatar_first(self):
        character_sheet.create_draft_character("bea", self._src.name)
        char_dir = character_sheet._character_dir("bea")
        front_png = char_dir / "sheet_views" / "front" / "cand_00_hidream_1000.png"
        profile_png = char_dir / "sheet_views" / "profile_left" / "cand_00_hidream_1000.png"
        _write_tiny_png(front_png)
        _write_tiny_png(profile_png)

        resp = self.client.post("/api/phosphene/characters/bea/sheet/add-to-ring")
        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        self.assertIn("jobId", body)
        self._job_ids_created.append(body["jobId"])
        self.assertEqual(body["imageCount"], 3)  # avatar + 2 shots

        self._wait_for_call()
        self.assertEqual(len(self._run_calls), 1)
        call = self._run_calls[0]
        # Avatar must be first (ref_index=1 -> run_folder_analysis's
        # first entry) so shots are scored against the real photo, not
        # against another generated image.
        avatar_path = character_sheet.character_avatar("bea")
        self.assertEqual(Path(call["image_paths"][0]), avatar_path)
        self.assertEqual(set(call["image_paths"][1:]),
                          {str(front_png), str(profile_png)})
        self.assertEqual(call["ref_index"], 1)

        # Job record shape matches what /api/analyze-folder produces, so
        # the existing frontend polling/rendering code needs no changes.
        job = _analysis_jobs[body["jobId"]]
        self.assertEqual(job["sourceType"], "folder")

    def test_reroll_reflected_without_sheet_json(self):
        # sheet.json is only written once a whole generate/reroll job
        # finishes -- this route must still work off of what's actually
        # on disk (sheet_shot_image_paths()), not a stale/missing
        # sheet.json, so a shot that only exists as a file on disk (as
        # if mid-job, or freshly rerolled) is still picked up.
        character_sheet.create_draft_character("cleo", self._src.name)
        char_dir = character_sheet._character_dir("cleo")
        self.assertFalse((char_dir / "sheet.json").exists())
        _write_tiny_png(char_dir / "sheet_views" / "front" / "cand_00_hidream_999.png")

        resp = self.client.post("/api/phosphene/characters/cleo/sheet/add-to-ring")
        self.assertEqual(resp.status_code, 202)
        self._job_ids_created.append(resp.get_json()["jobId"])
        self.assertEqual(resp.get_json()["imageCount"], 2)  # avatar + 1 shot


if __name__ == "__main__":
    unittest.main()
