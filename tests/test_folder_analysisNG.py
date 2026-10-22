"""Unit tests for folder_analysisNG.run_folder_analysis_ng -- the NG twin
of tests/test_folder_analysis.py, plus the error and input-shape paths
that file doesn't cover.

InsightFace is stubbed; sim is controlled with fixed fake embeddings and
blur with real image content (flat = fails, checkerboard = passes).

Run with: python3 -m unittest tests.test_folder_analysisNG -v
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import folder_analysisNG as fa
import stateNG
import video_analysisNG as va


def _png(flat: bool, size: int = 40) -> bytes:
    if flat:
        img = np.full((size, size, 3), 128, dtype=np.uint8)
    else:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(0, size, 4):
            for x in range(0, size, 4):
                if ((x // 4) + (y // 4)) % 2 == 0:
                    img[y:y + 4, x:x + 4] = 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


E_REF = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
E_MISMATCH = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
BBOX = [0, 0, 40, 40]


class _Face:
    def __init__(self, embedding, bbox=BBOX, pose=(1.0, 2.0, 3.0)):
        self.normed_embedding = embedding
        self.bbox = bbox
        self.pose = pose


class _FaceApp:
    """One canned face list per call: the reference detection first, then
    one per image in list order."""
    def __init__(self, *faces_per_call):
        self._queue = list(faces_per_call)

    def get(self, frame):
        return self._queue.pop(0)


class FolderAnalysisNGTests(unittest.TestCase):
    def setUp(self):
        self._ids = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for jid in self._ids:
            stateNG._analysis_jobs_ng.pop(jid, None)
            va.clear_frame_cache_ng(jid)

    def _run(self, job_id, images, face_app, sim=0.5, blur=10.0, **kw):
        stateNG._analysis_jobs_ng[job_id] = {"status": "running", "results": [], "error": None}
        self._ids.append(job_id)
        with mock.patch.object(fa, "get_face_app_ng", return_value=face_app):
            fa.run_folder_analysis_ng(job_id, images, sim, blur, **kw)
        return stateNG._analysis_jobs_ng[job_id]

    # --- cache behaviour -----------------------------------------------
    def test_passing_frame_is_cached_and_embedded(self):
        job = self._run("ng_pass", [("a.png", _png(False)), ("b.png", _png(False))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], [_Face(E_REF)]))
        self.assertEqual(job["status"], "done", job.get("error"))
        r = job["results"][-1]
        self.assertTrue(r["passed"])
        self.assertIsNone(r["failReason"])
        self.assertEqual(r["origName"], "b.png")
        self.assertIsNotNone(va.find_cache_frame_ng(r["frameId"])[0])
        self.assertEqual(job["frame_embeddings"][2], E_REF.tolist())

    def test_failing_frame_not_cached_by_default(self):
        job = self._run("ng_fail", [("a.png", _png(False)), ("b.png", _png(False))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], [_Face(E_MISMATCH)]))
        r = job["results"][-1]
        self.assertEqual((r["passed"], r["failReason"], r["frameId"]), (False, "sim", None))
        self.assertIsNone(va.find_cache_frame_ng("ng_fail_00002")[0])
        self.assertNotIn(2, job.get("frame_embeddings", {}))

    def test_always_cache_keeps_failed_frames_viewable(self):
        job = self._run("ng_always", [("a.png", _png(False)), ("b.png", _png(False))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], [_Face(E_MISMATCH)]),
                        always_cache=True)
        r = job["results"][-1]
        self.assertEqual(r["failReason"], "sim")
        self.assertIsNotNone(va.find_cache_frame_ng(r["frameId"])[0])
        self.assertIn(2, job["frame_embeddings"])

    def test_blur_failure_reported_as_blur(self):
        job = self._run("ng_blur", [("a.png", _png(False)), ("flat.png", _png(True))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], [_Face(E_REF)]))
        r = job["results"][-1]
        self.assertEqual((r["passed"], r["failReason"]), (False, "blur"))

    # --- error and edge paths -----------------------------------------
    def test_unreadable_reference_errors_the_job(self):
        job = self._run("ng_badref", [("bad.png", b"not an image")], _FaceApp())
        self.assertEqual(job["status"], "error")
        self.assertIn("Could not read reference image bad.png", job["error"])

    def test_no_face_in_reference_errors_the_job(self):
        job = self._run("ng_noface_ref", [("a.png", _png(False))], _FaceApp([]))
        self.assertEqual(job["status"], "error")
        self.assertIn("No face detected in reference image a.png", job["error"])

    def test_unreadable_and_faceless_images_get_placeholder_rows(self):
        job = self._run("ng_placeholder",
                        [("a.png", _png(False)), ("junk.png", b"junk"), ("nofaces.png", _png(False))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], []))
        self.assertEqual(job["status"], "done", job.get("error"))
        _, junk, nofaces = job["results"]
        for row in (junk, nofaces):
            self.assertFalse(row["passed"])
            self.assertFalse(row["hasFace"])
            self.assertEqual(row["sim"], 0.0)
        self.assertNotIn("width", junk)              # never decoded
        self.assertEqual((nofaces["width"], nofaces["height"]), (40, 40))

    def test_ref_index_is_clamped_and_one_based(self):
        job = self._run("ng_refidx", [("a.png", _png(False)), ("b.png", _png(False))],
                        _FaceApp([_Face(E_REF)], [_Face(E_REF)], [_Face(E_REF)]),
                        ref_index=99)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertIsNotNone(job["anchorFrameId"])

    def test_engine_exception_becomes_job_error(self):
        boom = mock.Mock(side_effect=RuntimeError("onnx died"))
        stateNG._analysis_jobs_ng["ng_boom"] = {"status": "running", "results": [], "error": None}
        self._ids.append("ng_boom")
        with mock.patch.object(fa, "get_face_app_ng", boom):
            fa.run_folder_analysis_ng("ng_boom", [("a.png", _png(False))], 0.5, 10.0)
        job = stateNG._analysis_jobs_ng["ng_boom"]
        self.assertEqual((job["status"], job["error"]), ("error", "onnx died"))

    # --- _read_image_ng ------------------------------------------------
    def test_read_image_accepts_path_or_name_bytes_tuple(self):
        import tempfile
        frame, name = fa._read_image_ng(("mem.png", _png(False)))
        self.assertEqual((frame.shape[:2], name), ((40, 40), "mem.png"))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "disk.png")
            with open(p, "wb") as f:
                f.write(_png(False))
            frame, name = fa._read_image_ng(p)
            self.assertEqual((frame.shape[:2], name), ((40, 40), "disk.png"))
        self.assertIsNone(fa._read_image_ng(("x.png", b"junk"))[0])


if __name__ == "__main__":
    unittest.main()
