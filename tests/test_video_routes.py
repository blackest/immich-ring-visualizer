"""Flask-test-client tests for routes/video.py.

Covers the request/response contract of the video-preview and job-lookup
endpoints without needing a GPU or the face-detection pipeline (the heavy
part of /api/analyze-video's background thread is intentionally not
exercised here, same tradeoff test_character_sheet.py makes for
hidream_engine -- these tests stub nothing and instead only cover request
handling that doesn't require insightface/onnxruntime to actually run).

Uses the same tiny in-memory synthetic video helper as
test_video_analysis.py, generated via `av` rather than reading a fixture
file from disk.

Run with:

    python3 -m unittest tests.test_video_routes -v

from the repo root.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import state
from routes.video import video_bp
from tests.test_video_analysis import _make_test_video_bytes


def _make_app():
    app = Flask(__name__)
    app.register_blueprint(video_bp)
    app.testing = True
    return app


class PreviewVideoRouteTestCase(unittest.TestCase):
    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()
        self._preview_ids_created = []
        self.addCleanup(self._cleanup_preview_jobs)

    def _cleanup_preview_jobs(self):
        for pid in self._preview_ids_created:
            state._preview_jobs.pop(pid, None)

    def _upload_valid_video(self):
        video_bytes = _make_test_video_bytes(n_frames=10, width=64, height=48, fps=10)
        resp = self.client.post(
            "/api/preview-video",
            data={"video": (io.BytesIO(video_bytes), "clip.mp4")},
            content_type="multipart/form-data",
        )
        if resp.status_code == 200:
            self._preview_ids_created.append(resp.get_json()["previewId"])
        return resp

    def test_missing_video_field_400(self):
        resp = self.client.post("/api/preview-video", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "video file required")

    def test_garbage_video_bytes_400(self):
        resp = self.client.post(
            "/api/preview-video",
            data={"video": (io.BytesIO(b"not a real video"), "clip.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Could not open video", resp.get_json()["error"])

    def test_valid_video_returns_preview_metadata(self):
        resp = self._upload_valid_video()
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("previewId", body)
        self.assertGreater(body["fps"], 0)
        self.assertGreater(body["totalFrames"], 0)
        self.assertGreater(body["duration"], 0)
        self.assertIn(body["previewId"], state._preview_jobs)


class PreviewFrameRouteTestCase(unittest.TestCase):
    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()
        self._preview_ids_created = []
        self.addCleanup(self._cleanup_preview_jobs)

    def _cleanup_preview_jobs(self):
        for pid in self._preview_ids_created:
            state._preview_jobs.pop(pid, None)

    def _make_preview(self):
        video_bytes = _make_test_video_bytes(n_frames=10, width=64, height=48, fps=10)
        resp = self.client.post(
            "/api/preview-video",
            data={"video": (io.BytesIO(video_bytes), "clip.mp4")},
            content_type="multipart/form-data",
        )
        preview_id = resp.get_json()["previewId"]
        self._preview_ids_created.append(preview_id)
        return preview_id

    def test_unknown_preview_id_404(self):
        resp = self.client.get("/api/preview-frame/does-not-exist/1")
        self.assertEqual(resp.status_code, 404)

    def test_known_preview_returns_jpeg_frame(self):
        preview_id = self._make_preview()
        resp = self.client.get(f"/api/preview-frame/{preview_id}/1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "image/jpeg")
        self.assertGreater(len(resp.data), 0)

    def test_frame_number_beyond_range_is_clamped_not_error(self):
        preview_id = self._make_preview()
        resp = self.client.get(f"/api/preview-frame/{preview_id}/9999")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "image/jpeg")


class JobLookupRouteTestCase(unittest.TestCase):
    """The 404/400 guard-clause paths for job-lookup routes -- these don't
    require an actual finished analysis job, just the "job not found" /
    "job not ready" contract that other code (and the frontend) relies on."""

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    def test_analysis_status_unknown_job_404(self):
        resp = self.client.get("/api/analysis-status/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "unknown job")

    def test_frame_file_unknown_frame_404(self):
        resp = self.client.get("/api/framefile/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_build_playback_unknown_job_404(self):
        resp = self.client.post("/api/build-playback/does-not-exist")
        self.assertEqual(resp.status_code, 404)

    def test_build_playback_job_not_done_400(self):
        job_id = "test-job-not-done"
        state._analysis_jobs[job_id] = {"status": "running", "results": [], "error": None}
        self.addCleanup(lambda: state._analysis_jobs.pop(job_id, None))
        resp = self.client.post(f"/api/build-playback/{job_id}")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "analysis not finished yet")

    def test_playback_file_unknown_job_404(self):
        resp = self.client.get("/api/playback-file/does-not-exist")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
