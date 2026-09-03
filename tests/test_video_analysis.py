"""Unit tests for video_analysis.py's MemoryVideo (in-memory PyAV wrapper).

This module exists because MemoryVideo._open_container() shipped without
`import io` / `import av` for about a week (commit 7c26ad9, 2026-08-27)
without anything catching it -- nothing in the repo ever actually
constructed a MemoryVideo from real bytes. A plain "does the module import"
smoke test would NOT have caught that regression either: Python doesn't
validate names used inside a function body until the function is called,
so these tests build a real (tiny, synthetic) video in memory and actually
call into MemoryVideo, the way routes/video.py's real requests do.

Run with:

    python3 -m unittest tests.test_video_analysis -v

from the repo root. Requires the `av` package (listed in requirements.txt).
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import video_analysis
from video_analysis import MemoryVideo


def _make_test_video_bytes(n_frames=10, width=64, height=48, fps=10):
    """Encode a tiny synthetic H.264/mp4 clip entirely in memory and
    return its bytes, so tests don't depend on any fixture file on disk.
    Each frame is a solid color that shifts slightly, purely so frames
    aren't byte-identical (not required by any test here, just realistic)."""
    import av

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(n_frames):
        arr = np.full((height, width, 3), fill_value=(i * 20) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return buf.getvalue()


class MemoryVideoTestCase(unittest.TestCase):
    def setUp(self):
        self.video_bytes = _make_test_video_bytes(n_frames=10, width=64, height=48, fps=10)

    # ---- the actual regression: does MemoryVideo work at all ----------

    def test_open_real_video_succeeds(self):
        # This is the test that would have caught the missing io/av
        # imports: it fails with NameError before the fix, and it's the
        # exact code path routes/video.py's preview-video endpoint hits.
        mv = MemoryVideo(self.video_bytes)
        self.assertGreater(mv.fps, 0)
        self.assertGreater(mv.frame_count, 0)
        self.assertEqual(mv.width, 64)
        self.assertEqual(mv.height, 48)

    def test_open_garbage_bytes_raises(self):
        with self.assertRaises(Exception):
            MemoryVideo(b"not a real video, just garbage bytes")

    def test_seek_frame_returns_correct_shape(self):
        mv = MemoryVideo(self.video_bytes)
        frame = mv.seek_frame(1)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[0], 48)
        self.assertEqual(frame.shape[1], 64)
        self.assertEqual(frame.shape[2], 3)

    def test_seek_frame_last_frame_in_range(self):
        mv = MemoryVideo(self.video_bytes)
        frame = mv.seek_frame(mv.frame_count)
        self.assertIsNotNone(frame)

    def test_iter_frames_yields_expected_count(self):
        mv = MemoryVideo(self.video_bytes)
        frames = list(mv.iter_frames())
        # exact count can vary slightly with encoder behavior on very
        # short clips -- assert it's in the right ballpark rather than
        # pinning an exact number that could make this test brittle.
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 10)

    def test_reopen_for_second_pass(self):
        # MemoryVideo keeps the source bytes so a fresh container can be
        # reopened for a second seek/iteration pass (its own docstring
        # says PyAV containers are single-pass) -- verify that actually
        # works rather than just trusting the comment.
        mv = MemoryVideo(self.video_bytes)
        first = mv.seek_frame(1)
        second = mv.seek_frame(1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
