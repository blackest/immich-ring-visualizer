"""Regression coverage for /api/analyze-folder and /api/analyze-immich
holding source images in memory instead of writing them to disk.

Both routes used to write every uploaded/downloaded original to a temp
directory under config.FRAME_STORE and never cleaned it up on a
successful job -- an unbounded disk leak for content that's cheap to
just re-upload/re-fetch (unlike character-sheet generation, which
deliberately does persist to disk, since redoing a 30-minute render is
the expensive case worth protecting against). These tests pin down both
halves of the fix: nothing lands under FRAME_STORE, and the bytes that
would have been written are instead the exact bytes handed to
run_folder_analysis and stored in job["srcImages"] for /api/export-job
to read back later.

run_folder_analysis itself (InsightFace) is stubbed -- same tradeoff
test_phosphene_routes.py makes -- these tests are about what data reaches
it and where it's held, not the face-detection pipeline.

Run with:

    python3 -m unittest tests.test_ingest_no_disk -v

from the repo root.
"""

import io
import os
import sys
import time
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import config
from state import _analysis_jobs


def _snapshot_frame_store():
    try:
        return set(os.listdir(config.FRAME_STORE))
    except OSError:
        return set()


class AnalyzeFolderNoDiskTestCase(unittest.TestCase):
    def setUp(self):
        import routes.folder as folder_routes
        self.folder_routes = folder_routes

        app = Flask(__name__)
        app.register_blueprint(folder_routes.folder_bp)
        app.testing = True
        self.client = app.test_client()

        self._job_ids = []
        self.addCleanup(lambda: [_analysis_jobs.pop(j, None) for j in self._job_ids])

        self._calls = []
        patcher = mock.patch.object(
            folder_routes, "run_folder_analysis",
            side_effect=self._fake_run_folder_analysis)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_run_folder_analysis(self, job_id, images, sim_threshold,
                                   blur_threshold, ref_index, cache_format):
        self._calls.append({"job_id": job_id, "images": list(images)})
        _analysis_jobs[job_id]["status"] = "done"

    def test_multi_file_upload_stays_in_memory(self):
        before = _snapshot_frame_store()
        data = {
            "images": [
                (io.BytesIO(b"fake-jpeg-bytes-1"), "b_photo.jpg"),
                (io.BytesIO(b"fake-jpeg-bytes-2"), "a_photo.png"),
            ],
        }
        resp = self.client.post("/api/analyze-folder", data=data,
                                 content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self._job_ids.append(body["jobId"])
        self.assertEqual(body["imageCount"], 2)

        # Nothing new on disk under FRAME_STORE.
        self.assertEqual(_snapshot_frame_store(), before,
                          "analyze-folder must not write uploaded images to disk")

        self.assertEqual(len(self._calls), 1)
        images = self._calls[0]["images"]
        # Sorted by lowercase filename, exactly like the old saved_paths list.
        self.assertEqual([name for name, _ in images], ["a_photo.png", "b_photo.jpg"])
        by_name = dict(images)
        self.assertEqual(by_name["a_photo.png"], b"fake-jpeg-bytes-2")
        self.assertEqual(by_name["b_photo.jpg"], b"fake-jpeg-bytes-1")
        # Each entry is real in-memory bytes, never a filesystem path.
        for _name, blob in images:
            self.assertIsInstance(blob, (bytes, bytearray))

        job = _analysis_jobs[body["jobId"]]
        self.assertEqual(job["srcImages"], {
            "a_photo.png": b"fake-jpeg-bytes-2",
            "b_photo.jpg": b"fake-jpeg-bytes-1",
        })
        self.assertNotIn("srcDir", job)

    def test_zip_upload_stays_in_memory(self):
        before = _snapshot_frame_store()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("shot1.jpg", b"zip-bytes-1")
            zf.writestr("nested/shot2.png", b"zip-bytes-2")
            zf.writestr("notes.txt", b"not an image, must be skipped")
        buf.seek(0)

        resp = self.client.post(
            "/api/analyze-folder",
            data={"zip": (buf, "batch.zip")},
            content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self._job_ids.append(body["jobId"])
        self.assertEqual(body["imageCount"], 2)

        self.assertEqual(_snapshot_frame_store(), before,
                          "zip contents must be extracted in memory, not to disk")

        images = self._calls[0]["images"]
        by_name = dict(images)
        self.assertEqual(set(by_name), {"shot1.jpg", "shot2.png"})
        self.assertEqual(by_name["shot1.jpg"], b"zip-bytes-1")
        self.assertEqual(by_name["shot2.png"], b"zip-bytes-2")


class AnalyzeImmichNoDiskTestCase(unittest.TestCase):
    def setUp(self):
        import routes.immich as immich_routes
        self.immich_routes = immich_routes

        app = Flask(__name__)
        app.register_blueprint(immich_routes.immich_bp)
        app.testing = True
        self.client = app.test_client()

        self._job_ids = []
        self.addCleanup(lambda: [_analysis_jobs.pop(j, None) for j in self._job_ids])

        self._calls = []
        patcher = mock.patch.object(
            immich_routes, "run_folder_analysis",
            side_effect=self._fake_run_folder_analysis)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_run_folder_analysis(self, job_id, images, sim_threshold,
                                   blur_threshold, ref_index, cache_format):
        self._calls.append({"job_id": job_id, "images": list(images)})
        _analysis_jobs[job_id]["status"] = "done"

    def _fake_requests_get(self, assets):
        """assets: {asset_id: (orig_filename, content_bytes)}."""
        def _get(url, headers=None, timeout=None, **kwargs):
            resp = mock.Mock()
            for asset_id, (orig_name, content) in assets.items():
                if url.endswith(f"/assets/{asset_id}"):
                    resp.status_code = 200
                    resp.json.return_value = {"originalFileName": orig_name}
                    return resp
                if url.endswith(f"/assets/{asset_id}/original"):
                    resp.status_code = 200
                    resp.content = content
                    return resp
            resp.status_code = 404
            resp.content = b""
            resp.json.return_value = {}
            return resp
        return _get

    def test_immich_download_stays_in_memory(self):
        before = _snapshot_frame_store()
        assets = {
            "aid-1": ("holiday.jpg", b"immich-bytes-1"),
            "aid-2": ("party.png", b"immich-bytes-2"),
        }
        with mock.patch.object(self.immich_routes.requests, "get",
                                side_effect=self._fake_requests_get(assets)):
            resp = self.client.post("/api/analyze-immich",
                                     json={"assetIds": ["aid-1", "aid-2"]})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self._job_ids.append(body["jobId"])
        self.assertEqual(body["imageCount"], 2)
        self.assertEqual(body["fetchErrors"], [])

        self.assertEqual(_snapshot_frame_store(), before,
                          "downloaded Immich originals must not be written to disk")

        images = self._calls[0]["images"]
        by_name = dict(images)
        self.assertEqual(by_name["aid-1_holiday.jpg"], b"immich-bytes-1")
        self.assertEqual(by_name["aid-2_party.png"], b"immich-bytes-2")

        job = _analysis_jobs[body["jobId"]]
        self.assertEqual(set(job["srcImages"]), {"aid-1_holiday.jpg", "aid-2_party.png"})
        self.assertNotIn("srcDir", job)

    def test_partial_fetch_failure_still_stays_in_memory(self):
        before = _snapshot_frame_store()
        assets = {"aid-1": ("ok.jpg", b"immich-bytes-ok")}
        with mock.patch.object(self.immich_routes.requests, "get",
                                side_effect=self._fake_requests_get(assets)):
            resp = self.client.post("/api/analyze-immich",
                                     json={"assetIds": ["aid-1", "aid-missing"]})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self._job_ids.append(body["jobId"])
        self.assertEqual(body["imageCount"], 1)
        self.assertEqual(len(body["fetchErrors"]), 1)

        self.assertEqual(_snapshot_frame_store(), before)


if __name__ == "__main__":
    unittest.main()
