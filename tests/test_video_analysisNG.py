"""Unit tests for video_analysisNG.py -- the NG twin of video_analysis.py:
pure sizing helpers, the in-memory frame cache, MemoryVideo over a real
(tiny, synthetic) clip, the frame-split job and the face-analysis job.

InsightFace is stubbed; the video is built in memory so no fixture files
are needed (same approach as tests/test_video_analysis.py).

Run with: python3 -m unittest tests.test_video_analysisNG -v
"""

import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import stateNG
import video_analysisNG as va


def _video_bytes(n_frames=10, width=64, height=48, fps=10):
    import av
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for i in range(n_frames):
        arr = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


class StateNGTests(unittest.TestCase):
    def test_stores_are_independent_dicts_and_locks(self):
        stores = [stateNG._analysis_jobs_ng, stateNG._split_jobs_ng, stateNG._hdmulti_jobs_ng,
                  stateNG._comfy_extracts_ng, stateNG._comfy_jobs_ng, stateNG._preview_jobs_ng,
                  stateNG._frame_cache_ng]
        self.assertTrue(all(isinstance(s, dict) for s in stores))
        self.assertEqual(len({id(s) for s in stores}), len(stores))
        self.assertIsNot(stateNG._frame_cache_lock_ng, stateNG._face_app_lock_ng)
        for lock in (stateNG._frame_cache_lock_ng, stateNG._face_app_lock_ng):
            with lock:
                pass

    def test_ng_state_is_separate_from_the_original_app(self):
        import state
        self.assertIsNot(stateNG._analysis_jobs_ng, state._analysis_jobs)
        self.assertIsNot(stateNG._frame_cache_ng, state._frame_cache)


class SizingHelperTests(unittest.TestCase):
    def test_summarize_resolutions_uniform_and_mixed(self):
        self.assertIsNone(va.summarize_resolutions_ng([]))
        self.assertIsNone(va.summarize_resolutions_ng([{"frame": 1}]))
        uniform = va.summarize_resolutions_ng([{"width": 64, "height": 48}] * 3)
        self.assertTrue(uniform["uniform"])
        self.assertEqual((uniform["modeWidth"], uniform["totalCount"]), (64, 3))
        mixed = va.summarize_resolutions_ng([
            {"width": 64, "height": 48}, {"width": 64, "height": 48}, {"width": 32, "height": 96}])
        self.assertFalse(mixed["uniform"])
        self.assertEqual((mixed["modeWidth"], mixed["modeCount"]), (64, 2))
        self.assertEqual((mixed["minWidth"], mixed["maxHeight"]), (32, 96))

    def test_bbox_ratios(self):
        self.assertAlmostEqual(va.bbox_frame_ratio_ng([0, 0, 50, 50], 100, 100), 0.25)
        self.assertAlmostEqual(va.vert_fill_ratio_ng([0, 0, 10, 50], 200, 100), 0.5)  # short side
        for fn in (va.bbox_frame_ratio_ng, va.vert_fill_ratio_ng):
            self.assertEqual(fn(None, 100, 100), 0.0)
            self.assertEqual(fn([0, 0, 10, 10], 0, 100), 0.0)
        self.assertEqual(va.bbox_frame_ratio_ng([20, 20, 10, 10], 100, 100), 0.0)  # inverted box


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(va.clear_frame_cache_ng, "ngcache")
        self.img = np.full((8, 8, 3), 200, dtype=np.uint8)

    def test_write_find_roundtrip_with_mimetypes(self):
        fid = va.write_cache_frame_ng("ngcache", 7, self.img)
        self.assertEqual(fid, "ngcache_00007")
        data, mime = va.find_cache_frame_ng(fid)
        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(data.startswith(b"\xff\xd8"))
        pid = va.write_cache_frame_ng("ngcache", 8, self.img, "png")
        data, mime = va.find_cache_frame_ng(pid)
        self.assertEqual(mime, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_anchor_key_and_job_ids_containing_underscores(self):
        self.addCleanup(va.clear_frame_cache_ng, "my_job_1")
        self.assertEqual(va.write_cache_frame_ng("my_job_1", "anchor", self.img), "my_job_1_anchor")
        self.assertIsNotNone(va.find_cache_frame_ng("my_job_1_anchor")[0])

    def test_find_misses_return_none_pair(self):
        self.assertEqual(va.find_cache_frame_ng("nounderscore"), (None, None))
        self.assertEqual(va.find_cache_frame_ng("nojob_00001"), (None, None))

    def test_clear_drops_only_that_job(self):
        self.addCleanup(va.clear_frame_cache_ng, "other")
        a = va.write_cache_frame_ng("ngcache", 1, self.img)
        b = va.write_cache_frame_ng("other", 1, self.img)
        va.clear_frame_cache_ng("ngcache")
        self.assertEqual(va.find_cache_frame_ng(a), (None, None))
        self.assertIsNotNone(va.find_cache_frame_ng(b)[0])
        va.clear_frame_cache_ng("never-existed")  # no error


class MemoryVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _video_bytes()

    def test_probe_reads_real_clip(self):
        mv = va.MemoryVideo(self.data)
        self.assertEqual((mv.width, mv.height), (64, 48))
        self.assertAlmostEqual(mv.fps, 10.0)
        self.assertEqual(mv.frame_count, 10)

    def test_garbage_bytes_raise(self):
        with self.assertRaises(Exception):
            va.MemoryVideo(b"not a video")

    def test_seek_and_iter(self):
        mv = va.MemoryVideo(self.data)
        self.assertEqual(mv.seek_frame(3).shape, (48, 64, 3))
        self.assertIsNone(mv.seek_frame(999))
        self.assertEqual([i for i, _ in mv.iter_frames()], list(range(1, 11)))
        self.assertEqual([i for i, _ in mv.iter_frames(start_frame=4, end_frame=6)], [4, 5, 6])
        self.assertEqual([i for i, _ in mv.iter_frames(end_frame=2)], [1, 2])


class _Face:
    def __init__(self, emb):
        self.normed_embedding, self.bbox, self.pose = emb, [0, 0, 32, 32], (0.0, 0.0, 0.0)


class _FaceApp:
    def __init__(self, faces_for):
        self.faces_for = faces_for  # callable(call_number) -> face list
        self.calls = 0

    def get(self, frame):
        self.calls += 1
        return self.faces_for(self.calls)


class JobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _video_bytes()

    def setUp(self):
        self._ids = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for jid in self._ids:
            for d in (stateNG._analysis_jobs_ng, stateNG._split_jobs_ng):
                d.pop(jid, None)
            va.clear_frame_cache_ng(jid)

    def _job(self, store, jid):
        store[jid] = {"status": "running", "results": [], "error": None}
        self._ids.append(jid)
        return store[jid]

    def test_split_extracts_every_frame_in_range(self):
        job = self._job(stateNG._split_jobs_ng, "split1")
        va.run_video_split_ng("split1", self.data, start_sec=0.3, end_sec=0.6)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual([r["frame"] for r in job["results"]], [4, 5, 6, 7])
        r = job["results"][0]
        self.assertEqual((r["width"], r["height"]), (64, 48))
        self.assertIsNotNone(va.find_cache_frame_ng(r["frameId"])[0])

    def test_split_refuses_over_the_frame_cap(self):
        job = self._job(stateNG._split_jobs_ng, "split2")
        with mock.patch.object(va, "MAX_SPLIT_FRAMES", 5):
            va.run_video_split_ng("split2", self.data)
        self.assertEqual(job["status"], "error")
        self.assertIn("~10 frames", job["error"])
        self.assertEqual(job["results"], [])

    def test_split_bad_video_errors_the_job(self):
        job = self._job(stateNG._split_jobs_ng, "split3")
        va.run_video_split_ng("split3", b"junk")
        self.assertEqual(job["status"], "error")

    def test_analysis_flags_mismatched_and_faceless_frames(self):
        job = self._job(stateNG._analysis_jobs_ng, "an1")
        ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        other = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        def faces_for(n):  # n=1 is the reference detection, then frame 1, 2, 3 ...
            if n == 4:
                return []              # frame 3: no face
            return [_Face(other if n == 3 else ref)]   # frame 2 mismatches
        app = _FaceApp(faces_for)
        with mock.patch.object(va, "get_face_app_ng", return_value=app), \
             mock.patch.object(va, "get_blur_score_ng", return_value=99.0):
            va.run_video_analysis_ng("an1", self.data, 0.5, 10.0, end_sec=0.2)
        self.assertEqual(job["status"], "done", job.get("error"))
        first, second, third = job["results"]
        self.assertTrue(first["passed"])
        self.assertEqual((second["passed"], second["failReason"]), (False, "sim"))
        self.assertIsNone(second["frameId"])
        self.assertFalse(third["hasFace"])
        self.assertIsNotNone(job["anchorFrameId"])
        self.assertIn(1, job["frame_embeddings"])
        self.assertNotIn(2, job["frame_embeddings"])
        self.assertTrue(job["resolutionSummary"]["uniform"])

    def test_analysis_blur_failure(self):
        job = self._job(stateNG._analysis_jobs_ng, "an2")
        ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with mock.patch.object(va, "get_face_app_ng", return_value=_FaceApp(lambda n: [_Face(ref)])), \
             mock.patch.object(va, "get_blur_score_ng", return_value=1.0):
            va.run_video_analysis_ng("an2", self.data, 0.5, 10.0, end_sec=0.1)
        self.assertEqual(job["results"][0]["failReason"], "blur")

    def test_analysis_errors_on_unreadable_or_faceless_reference(self):
        job = self._job(stateNG._analysis_jobs_ng, "an3")
        with mock.patch.object(va, "get_face_app_ng", return_value=_FaceApp(lambda n: [])):
            va.run_video_analysis_ng("an3", self.data, 0.5, 10.0)
        self.assertEqual(job["status"], "error")
        self.assertIn("No face detected in reference frame 1", job["error"])

        job = self._job(stateNG._analysis_jobs_ng, "an4")
        with mock.patch.object(va, "get_face_app_ng", return_value=_FaceApp(lambda n: [])):
            va.run_video_analysis_ng("an4", self.data, 0.5, 10.0, ref_frame_idx=999)
        self.assertIn("Could not read reference frame 999", job["error"])


if __name__ == "__main__":
    unittest.main()
