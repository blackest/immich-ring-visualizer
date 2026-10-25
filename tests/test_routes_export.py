"""Flask-test-client tests for routes/export.py.

Uses real (tiny) cv2 images through the real image_ops crop/resize, with
the frame cache, Immich download/export and face detector mocked. EXPORT_DIR
is redirected to a temp dir, so nothing is written to the real export folder.

Run with: python3 -m unittest tests.test_routes_export -v
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask

import routes.export as export
from state import _analysis_jobs


def _client():
    app = Flask(__name__)
    app.register_blueprint(export.export_bp)
    app.testing = True
    return app.test_client()


def _jpg(w=64, h=48):
    ok, enc = cv2.imencode(".jpg", np.full((h, w, 3), 128, dtype=np.uint8))
    assert ok
    return enc.tobytes()


class _TmpExportDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        p = mock.patch.object(export, "EXPORT_DIR", self._tmp.name)
        p.start()
        self.addCleanup(p.stop)
        self.job_ids = []
        self.addCleanup(lambda: [_analysis_jobs.pop(j, None) for j in self.job_ids])

    def _job(self, job_id, results, **extra):
        _analysis_jobs[job_id] = {"sourceName": "src", "results": results, **extra}
        self.job_ids.append(job_id)


class ExportJobTests(_TmpExportDir):
    def test_unknown_job(self):
        self.assertEqual(_client().post("/api/export-job/nope").status_code, 404)

    def test_exports_passed_frames_from_cache(self):
        self._job("j1", [
            {"frame": 3, "sim": 0.9, "passed": True, "frameId": "f3"},
            {"frame": 4, "sim": 0.5, "passed": False, "frameId": "f4"},
            {"frame": 5, "sim": 0.8, "passed": True, "frameId": None},
        ])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            body = _client().post("/api/export-job/j1", json={"width": 32, "height": 32}).get_json()
        self.assertEqual(body["exported"], 1)
        self.assertEqual(body["files"], ["fr00003_sim0.90.png"])
        self.assertEqual(body["reencodedFromCache"], 1)
        out = cv2.imread(os.path.join(body["path"], "fr00003_sim0.90.png"))
        self.assertEqual(out.shape[:2], (32, 32))

    def test_frames_filter_selects_subset(self):
        self._job("j2", [
            {"frame": 1, "sim": 0.9, "passed": True, "frameId": "a"},
            {"frame": 2, "sim": 0.9, "passed": True, "frameId": "b"},
        ])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            body = _client().post("/api/export-job/j2", json={"frames": [2]}).get_json()
        self.assertEqual(body["files"], ["fr00002_sim0.90.png"])

    def test_empty_frames_list_exports_nothing(self):
        self._job("j3", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            body = _client().post("/api/export-job/j3", json={"frames": []}).get_json()
        self.assertEqual(body["exported"], 0)

    def test_missing_cache_frame_is_skipped_silently(self):
        self._job("j4", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(None, None)):
            body = _client().post("/api/export-job/j4", json={}).get_json()
        self.assertEqual(body["exported"], 0)
        self.assertEqual(body["errors"], [])

    def test_prefers_original_from_src_images(self):
        self._job("j5", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a", "origName": "o.jpg"}],
                  srcImages={"o.jpg": _jpg()})
        with mock.patch.object(export, "find_cache_frame") as cache:
            body = _client().post("/api/export-job/j5", json={}).get_json()
        cache.assert_not_called()
        self.assertEqual(body["exported"], 1)
        self.assertEqual(body["reencodedFromCache"], 0)

    def test_small_face_is_skipped_with_reason(self):
        self._job("j6", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a", "bbox": [0, 0, 4, 4]}])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            body = _client().post("/api/export-job/j6", json={"minFacePx": 500}).get_json()
        self.assertEqual(body["exported"], 0)
        self.assertEqual(body["skipped"], 1)
        self.assertIn("skipped", body["errors"][0])

    def test_immich_asset_ids_are_merged(self):
        self._job("j7", [])
        fake = {"saved": ["a.png"], "errors": [], "widened": 2, "padded": 1}
        with mock.patch.object(export, "export_immich_asset_ids", return_value=fake) as exp:
            body = _client().post("/api/export-job/j7", json={"assetIds": ["x"]}).get_json()
        self.assertEqual(exp.call_args.args[0], ["x"])
        self.assertEqual((body["exported"], body["frameExported"], body["immichExported"]), (1, 0, 1))
        self.assertEqual((body["widened"], body["padded"]), (2, 1))


    def test_video_job_exports_original_frame_via_memory_video(self):
        self._job("j8", [{"frame": 7, "sim": 0.9, "passed": True, "frameId": "a"}], videoBytes=b"vid")
        mv = mock.Mock()
        mv.seek_frame.return_value = np.full((48, 64, 3), 99, dtype=np.uint8)
        with mock.patch.object(export, "MemoryVideo", return_value=mv) as mv_cls, \
             mock.patch.object(export, "find_cache_frame") as cache:
            body = _client().post("/api/export-job/j8", json={"width": 16, "height": 16}).get_json()
        mv_cls.assert_called_once_with(b"vid")
        mv.seek_frame.assert_called_once_with(7)
        cache.assert_not_called()
        self.assertEqual((body["exported"], body["reencodedFromCache"]), (1, 0))

    def test_video_frame_that_cannot_be_read_falls_back_to_cache(self):
        self._job("j9", [{"frame": 7, "sim": 0.9, "passed": True, "frameId": "a"}], videoBytes=b"vid")
        mv = mock.Mock()
        mv.seek_frame.return_value = None
        with mock.patch.object(export, "MemoryVideo", return_value=mv), \
             mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            body = _client().post("/api/export-job/j9", json={}).get_json()
        self.assertEqual((body["exported"], body["reencodedFromCache"]), (1, 1))

    def test_legacy_job_without_bbox_detects_face_on_demand(self):
        self._job("j10", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a"}])
        face = mock.Mock(bbox=np.array([4.0, 4.0, 40.0, 40.0]))
        app = mock.Mock()
        app.get.return_value = [face]
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")), \
             mock.patch.object(export, "get_face_app", return_value=app) as get_app, \
             mock.patch.object(export, "pick_largest_face", return_value=face) as pick:
            body = _client().post("/api/export-job/j10", json={"cropMode": "face", "width": 16, "height": 16}).get_json()
        get_app.assert_called_once()
        pick.assert_called_once()
        self.assertEqual(body["exported"], 1)

    def test_legacy_detection_failure_falls_back_to_no_bbox(self):
        self._job("j11", [{"frame": 1, "sim": 0.9, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")), \
             mock.patch.object(export, "get_face_app", side_effect=RuntimeError("no model")):
            body = _client().post("/api/export-job/j11", json={"cropMode": "face", "width": 16, "height": 16}).get_json()
        self.assertEqual(body["exported"], 1)


class ExportImmichAssetsTests(_TmpExportDir):
    def test_requires_asset_ids(self):
        self.assertEqual(_client().post("/api/export-immich-assets", json={}).status_code, 400)

    def test_delegates_to_exporter(self):
        fake = {"saved": ["a.png"], "errors": ["e"], "widened": 0, "padded": 3}
        with mock.patch.object(export, "export_immich_asset_ids", return_value=fake):
            body = _client().post("/api/export-immich-assets", json={"assetIds": ["x"]}).get_json()
        self.assertEqual(body["files"], ["a.png"])
        self.assertEqual(body["padded"], 3)
        self.assertTrue(body["path"].endswith("immich_selected"))
        self.assertTrue(os.path.isdir(body["path"]))


class PreviewFrameTests(_TmpExportDir):
    def test_unknown_job_and_frame(self):
        c = _client()
        self.assertEqual(c.get("/api/export-preview/nope/1").status_code, 404)
        self._job("p1", [{"frame": 1, "passed": False, "frameId": "a"}])
        self.assertEqual(c.get("/api/export-preview/p1/1").status_code, 404)
        self.assertEqual(c.get("/api/export-preview/p1/9").status_code, 404)

    def test_uncached_frame(self):
        self._job("p2", [{"frame": 1, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(None, None)):
            self.assertEqual(_client().get("/api/export-preview/p2/1").status_code, 404)

    def test_undecodable_frame(self):
        self._job("p3", [{"frame": 1, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(b"junk", "image/jpeg")):
            self.assertEqual(_client().get("/api/export-preview/p3/1").status_code, 404)

    def test_renders_jpeg_at_requested_size(self):
        self._job("p4", [{"frame": 1, "passed": True, "frameId": "a"}])
        with mock.patch.object(export, "find_cache_frame", return_value=(_jpg(), "image/jpeg")):
            r = _client().get("/api/export-preview/p4/1?width=40&height=30")
        self.assertEqual(r.mimetype, "image/jpeg")
        img = cv2.imdecode(np.frombuffer(r.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (30, 40))


class PreviewImmichTests(_TmpExportDir):
    def test_upstream_failure(self):
        with mock.patch.object(export.requests, "get", return_value=mock.Mock(status_code=404)):
            r = _client().get("/api/export-preview-immich/abc")
        self.assertEqual(r.status_code, 502)

    def test_undecodable_image(self):
        resp = mock.Mock(status_code=200, content=b"junk")
        with mock.patch.object(export.requests, "get", return_value=resp):
            self.assertEqual(_client().get("/api/export-preview-immich/abc").status_code, 500)

    def test_renders_without_detection_in_contain_mode(self):
        resp = mock.Mock(status_code=200, content=_jpg())
        with mock.patch.object(export.requests, "get", return_value=resp), \
             mock.patch.object(export, "get_face_app") as face_app:
            r = _client().get("/api/export-preview-immich/abc?width=20&height=20")
        face_app.assert_not_called()
        self.assertEqual(r.mimetype, "image/jpeg")

    def test_face_mode_detects_and_survives_detector_failure(self):
        resp = mock.Mock(status_code=200, content=_jpg())
        with mock.patch.object(export.requests, "get", return_value=resp), \
             mock.patch.object(export, "get_face_app", side_effect=RuntimeError("no model")):
            r = _client().get("/api/export-preview-immich/abc?cropMode=face&width=20&height=20")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
