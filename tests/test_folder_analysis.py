"""Unit tests for folder_analysis.run_folder_analysis's always_cache flag.

InsightFace itself is stubbed (same tradeoff test_character_sheet.py makes
for hidream_engine) -- these tests are about whether a frame that fails
sim/blur still gets write_cache_frame()'d and a real frameId when
always_cache=True, not about face-detection accuracy itself.

Sim is controlled via fixed fake embeddings (matching vs. orthogonal
vectors -> sim 1.0 vs 0.0); blur is controlled via real image content fed
through the real get_blur_score/cv2.Laplacian path (a flat solid-color
image has ~0 variance = fails any positive blur_threshold; a checkerboard
has strong edges = comfortably passes).

Run with:

    python3 -m unittest tests.test_folder_analysis -v

from the repo root.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import folder_analysis
import state
import video_analysis


def _write_test_image(path: Path, flat: bool, size: int = 40):
    import cv2
    if flat:
        # Zero edges -> Laplacian variance ~0 -> fails any positive
        # blur_threshold.
        img = np.full((size, size, 3), 128, dtype=np.uint8)
    else:
        # A checkerboard has strong hard edges everywhere -> high
        # Laplacian variance -> comfortably passes a small blur_threshold.
        img = np.zeros((size, size, 3), dtype=np.uint8)
        step = 4
        for y in range(0, size, step):
            for x in range(0, size, step):
                if ((x // step) + (y // step)) % 2 == 0:
                    img[y:y + step, x:x + step] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


E_REF = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
E_MATCH = E_REF.copy()  # sim == 1.0 against the reference
E_MISMATCH = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # sim == 0.0


class _FakeFace:
    def __init__(self, embedding, bbox, pose=(0.0, 0.0, 0.0)):
        self.normed_embedding = embedding
        self.bbox = bbox
        self.pose = pose


class _FakeFaceApp:
    """Returns one canned face list per call, in order. First call is
    always the reference-frame detection; the rest are one per
    image_paths entry, in list order (matching run_folder_analysis's own
    call sequence)."""

    def __init__(self, faces_per_call):
        self._queue = list(faces_per_call)

    def get(self, frame):
        return self._queue.pop(0)


class RunFolderAnalysisAlwaysCacheTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._job_ids = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for jid in self._job_ids:
            state._analysis_jobs.pop(jid, None)
            state._frame_cache.pop(jid, None)

    def _run(self, job_id, shot_flat, shot_embedding, sim_threshold=0.5,
              blur_threshold=10.0, always_cache=False):
        tmp = Path(self._tmp.name)
        ref_path = tmp / f"{job_id}_ref.png"
        shot_path = tmp / f"{job_id}_shot.png"
        _write_test_image(ref_path, flat=False)  # ref: always sharp/matching
        _write_test_image(shot_path, flat=shot_flat)

        bbox = [0, 0, 40, 40]
        # Call order: ref-frame detection, then frame_idx=1 (ref image
        # again, via the loop), then frame_idx=2 (the shot under test).
        face_app = _FakeFaceApp([
            [_FakeFace(E_REF, bbox)],
            [_FakeFace(E_MATCH, bbox)],
            [_FakeFace(shot_embedding, bbox)],
        ])

        state._analysis_jobs[job_id] = {"status": "running", "results": [], "error": None}
        self._job_ids.append(job_id)

        with mock.patch.object(folder_analysis, "get_face_app", return_value=face_app):
            folder_analysis.run_folder_analysis(
                job_id, [str(ref_path), str(shot_path)],
                sim_threshold, blur_threshold,
                ref_index=1, cache_format="jpg", always_cache=always_cache)

        job = state._analysis_jobs[job_id]
        self.assertEqual(job["status"], "done", job.get("error"))
        return job["results"][-1]  # the shot's result entry

    def test_passing_shot_is_cached_regardless_of_always_cache(self):
        result = self._run("job_pass", shot_flat=False, shot_embedding=E_MATCH,
                            always_cache=False)
        self.assertTrue(result["passed"])
        self.assertIsNotNone(result["frameId"])
        data, _ = video_analysis.find_cache_frame(result["frameId"])
        self.assertIsNotNone(data, "a passing frame should always be cached")

    def test_failing_shot_not_cached_by_default(self):
        # Low similarity -- e.g. a genuine side/three-quarter angle
        # against a front-facing reference, exactly the case that was
        # getting silently dropped before this change.
        result = self._run("job_fail_default", shot_flat=False,
                            shot_embedding=E_MISMATCH, always_cache=False)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failReason"], "sim")
        self.assertIsNone(result["frameId"],
                           "default behavior: a failed frame has no viewable image")
        data, _ = video_analysis.find_cache_frame(f"job_fail_default_{result['frame']:05d}")
        self.assertIsNone(data)

    def test_failing_shot_is_cached_with_always_cache(self):
        result = self._run("job_fail_always", shot_flat=False,
                            shot_embedding=E_MISMATCH, always_cache=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failReason"], "sim")
        self.assertIsNotNone(
            result["frameId"],
            "always_cache=True: a failed shot must still be viewable")
        data, _ = video_analysis.find_cache_frame(result["frameId"])
        self.assertIsNotNone(
            data, "always_cache=True: the actual bytes must be cached too, "
                  "not just a non-null frameId")

    def test_blur_failure_also_respects_always_cache(self):
        # A flat/blurry shot fails on blur_threshold instead of sim --
        # always_cache should still make it viewable.
        result = self._run("job_blur_always", shot_flat=True,
                            shot_embedding=E_MATCH, always_cache=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failReason"], "blur")
        self.assertIsNotNone(result["frameId"])
        data, _ = video_analysis.find_cache_frame(result["frameId"])
        self.assertIsNotNone(data)


if __name__ == "__main__":
    unittest.main()
