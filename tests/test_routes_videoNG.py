"""Flask-test-client tests for routes/videoNG.py.

MemoryVideo, the analysis/split workers, yt-dlp, ffmpeg and cv2.VideoWriter
are faked, so no real video is decoded or encoded. FRAME_STORE points at a
temp dir. Job-store entries created here are removed on cleanup.

Run with: python3 -m unittest tests.test_routes_videoNG -v
"""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask

import routes.videoNG as video
from stateNG import _analysis_jobs_ng, _preview_jobs_ng, _split_jobs_ng


def _client():
    app = Flask(__name__)
    app.register_blueprint(video.videoNG_bp)
    app.testing = True
    return app.test_client()


class _FakeVideo:
    def __init__(self, fps=24.0, frame_count=10, width=8, height=6, frames=None, seek=None):
        self.fps, self.frame_count, self.width, self.height = fps, frame_count, width, height
        self._frames = frames or []
        self._seek = seek
        self.iter_kwargs = None

    def seek_frame(self, n):
        self.seeked = n
        return self._seek

    def iter_frames(self, **kw):
        self.iter_kwargs = kw
        yield from self._frames


class _StoreCleanup(unittest.TestCase):
    def _track(self, store, key):
        self.addCleanup(store.pop, key, None)


class PreviewVideoTests(_StoreCleanup):
    def _post(self, files=None, mv=None, raises=None):
        patch = mock.patch.object(video, "MemoryVideo", side_effect=raises) if raises \
            else mock.patch.object(video, "MemoryVideo", return_value=mv)
        with patch:
            return _client().post("/api/ng/preview-video", data=files or {}, content_type="multipart/form-data")

    def test_requires_video(self):
        self.assertEqual(self._post().status_code, 400)

    def test_unreadable_video(self):
        r = self._post({"video": (io.BytesIO(b"x"), "a.mp4")}, raises=RuntimeError("bad"))
        self.assertEqual(r.status_code, 400)
        self.assertIn("bad", r.get_json()["error"])

    def test_zero_fps_or_frames(self):
        for mv in (_FakeVideo(fps=0), _FakeVideo(frame_count=0)):
            self.assertEqual(self._post({"video": (io.BytesIO(b"x"), "a.mp4")}, mv=mv).status_code, 400)

    def test_stores_preview_job(self):
        r = self._post({"video": (io.BytesIO(b"VID"), "a.mp4")}, mv=_FakeVideo(fps=10, frame_count=50))
        body = r.get_json()
        self._track(_preview_jobs_ng, body["previewId"])
        self.assertEqual((body["fps"], body["totalFrames"], body["duration"]), (10, 50, 5.0))
        self.assertEqual(_preview_jobs_ng[body["previewId"]]["videoBytes"], b"VID")


class DownloadVideoUrlTests(unittest.TestCase):
    def _post(self, body, run=None, files=("My Clip.mp4",), payload=b"MP4", codec="h264", conversions=None):
        conversions = [] if conversions is None else conversions
        def fake_run(cmd, **kw):
            if cmd[0].endswith("ffprobe"):
                return SimpleNamespace(returncode=0, stdout=codec + "\n", stderr="")
            if cmd[0].endswith("ffmpeg"):
                conversions.append(cmd)
                with open(cmd[-1], "wb") as fh:
                    fh.write(b"H264")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            outdir = os.path.dirname(cmd[cmd.index("-o") + 1])
            for name in files:
                with open(os.path.join(outdir, name), "wb") as fh:
                    fh.write(payload)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(video.subprocess, "run", **(run or {"side_effect": fake_run})) as m:
            return _client().post("/api/ng/download-video-url", json=body), m

    def test_url_required(self):
        r, m = self._post({"url": "  "})
        self.assertEqual(r.status_code, 400)
        m.assert_not_called()

    def test_success_returns_bytes_and_quoted_filename(self):
        r, m = self._post({"url": "http://v"})
        self.assertEqual((r.status_code, r.mimetype, r.data), (200, "video/mp4", b"MP4"))
        self.assertEqual(r.headers["X-Video-Filename"], "My%20Clip.mp4")
        cmd = m.call_args_list[0].args[0]
        self.assertIn("--no-playlist", cmd)
        self.assertEqual(cmd[-1], "http://v")

    def test_av1_is_converted_to_h264(self):
        conversions = []
        r, _ = self._post({"url": "u"}, codec="av1", conversions=conversions)
        self.assertEqual((r.status_code, r.data), (200, b"H264"))
        self.assertEqual(r.headers["X-Video-Filename"], "My%20Clip.mp4")
        self.assertIn("libx264", conversions[0])

    def test_h264_is_left_alone(self):
        conversions = []
        r, _ = self._post({"url": "u"}, codec="h264", conversions=conversions)
        self.assertEqual((r.data, conversions), (b"MP4", []))

    def test_ytdlp_failure_returns_stderr_tail(self):
        res = SimpleNamespace(returncode=1, stdout="", stderr="\n".join(f"e{i}" for i in range(20)))
        r, _ = self._post({"url": "u"}, run={"return_value": res})
        self.assertEqual(r.status_code, 500)
        self.assertIn("e19", r.get_json()["error"])
        self.assertNotIn("e5\n", r.get_json()["error"])

    def test_timeout_and_exception(self):
        r, _ = self._post({"url": "u"}, run={"side_effect": subprocess.TimeoutExpired("y", 600)})
        self.assertEqual(r.status_code, 504)
        r, _ = self._post({"url": "u"}, run={"side_effect": OSError("x")})
        self.assertEqual(r.status_code, 500)

    def test_success_without_output_file(self):
        r, _ = self._post({"url": "u"}, files=())
        self.assertEqual(r.status_code, 500)


class PreviewFrameTests(_StoreCleanup):
    def _job(self, frames=10):
        _preview_jobs_ng["pv"] = {"videoBytes": b"v", "frames": frames}
        self._track(_preview_jobs_ng, "pv")

    def test_unknown_preview(self):
        self.assertEqual(_client().get("/api/ng/preview-frame/zzz/1").status_code, 404)

    def test_frame_number_is_clamped(self):
        self._job(frames=10)
        fv = _FakeVideo(seek=np.zeros((4, 4, 3), dtype=np.uint8))
        with mock.patch.object(video, "MemoryVideo", return_value=fv):
            r = _client().get("/api/ng/preview-frame/pv/999")
            self.assertEqual((r.status_code, r.mimetype), (200, "image/jpeg"))
            self.assertEqual(fv.seeked, 10)
            _client().get("/api/ng/preview-frame/pv/0")
            self.assertEqual(fv.seeked, 1)

    def test_missing_frame_is_404(self):
        self._job()
        with mock.patch.object(video, "MemoryVideo", return_value=_FakeVideo(seek=None)):
            self.assertEqual(_client().get("/api/ng/preview-frame/pv/1").status_code, 404)


class AnalyzeAndSplitTests(_StoreCleanup):
    def _post(self, route, form, store, worker):
        data = {"video": (io.BytesIO(b"VID"), "My Take.mp4"), **form}
        with mock.patch.object(video.threading, "Thread") as thread, \
             mock.patch.object(video, worker) as w:
            r = _client().post(route, data=data, content_type="multipart/form-data")
        body = r.get_json()
        if "jobId" in body:
            self._track(store, body["jobId"])
        return r, body, thread, w

    def test_analyze_requires_video(self):
        r = _client().post("/api/ng/analyze-video", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_analyze_defaults(self):
        r, body, thread, w = self._post("/api/ng/analyze-video", {}, _analysis_jobs_ng, "run_video_analysis_ng")
        job = _analysis_jobs_ng[body["jobId"]]
        self.assertEqual((job["status"], job["sourceName"], job["cacheFormat"]), ("running", "My Take", "jpg"))
        self.assertEqual((job["simThreshold"], job["blurThreshold"], job["startSec"], job["endSec"]),
                         (0.1, 1.0, None, None))
        self.assertEqual(thread.call_args.kwargs["args"],
                         (body["jobId"], b"VID", 0.1, 1.0, 1, "jpg", None, None))
        self.assertTrue(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once()

    def test_analyze_form_values(self):
        form = {"simThreshold": "0.4", "blurThreshold": "3", "refFrame": "7", "cacheFormat": "png",
                "startSec": "1.5", "endSec": " 12 "}
        _, body, thread, _ = self._post("/api/ng/analyze-video", form, _analysis_jobs_ng, "run_video_analysis_ng")
        self.assertEqual(thread.call_args.kwargs["args"],
                         (body["jobId"], b"VID", 0.4, 3.0, 7, "png", 1.5, 12.0))

    def test_split_requires_video(self):
        r = _client().post("/api/ng/video-split", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_split_job(self):
        _, body, thread, _ = self._post("/api/ng/video-split", {"startSec": "2", "cacheFormat": "png"},
                                        _split_jobs_ng, "run_video_split_ng")
        job = _split_jobs_ng[body["jobId"]]
        self.assertEqual((job["sourceName"], job["cacheFormat"], job["startSec"], job["endSec"]),
                         ("My Take", "png", 2.0, None))
        self.assertEqual(thread.call_args.kwargs["args"], (body["jobId"], b"VID", "png", 2.0, None))


class StatusAndFrameFileTests(_StoreCleanup):
    def test_analysis_status(self):
        c = _client()
        self.assertEqual(c.get("/api/ng/analysis-status/zzz").status_code, 404)
        _analysis_jobs_ng["st"] = {"status": "running", "results": [{"frame": 1}], "error": None}
        self._track(_analysis_jobs_ng, "st")
        body = c.get("/api/ng/analysis-status/st").get_json()
        self.assertEqual((body["status"], body["frameCount"], body["simThreshold"], body["blurThreshold"]),
                         ("running", 1, 0.1, 1))
        self.assertIsNone(body["resolutionSummary"])

    def test_split_status(self):
        c = _client()
        self.assertEqual(c.get("/api/ng/video-split-status/zzz").status_code, 404)
        _split_jobs_ng["ss"] = {"status": "done", "results": [1, 2], "error": None}
        self._track(_split_jobs_ng, "ss")
        self.assertEqual(c.get("/api/ng/video-split-status/ss").get_json()["frameCount"], 2)

    def test_framefile(self):
        c = _client()
        with mock.patch.object(video, "find_cache_frame_ng", return_value=(b"IMG", "image/png")):
            r = c.get("/api/ng/framefile/f1")
        self.assertEqual((r.data, r.mimetype), (b"IMG", "image/png"))
        with mock.patch.object(video, "find_cache_frame_ng", return_value=(None, None)):
            self.assertEqual(c.get("/api/ng/framefile/f1").status_code, 404)


class _FakeWriter:
    instances = []

    def __init__(self, path, fourcc, fps, size):
        self.path, self.fps, self.size, self.frames = path, fps, size, []
        _FakeWriter.instances.append(self)

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        with open(self.path, "wb") as fh:
            fh.write(b"RAWVIDEO")


class BuildPlaybackTests(_StoreCleanup):
    def setUp(self):
        _FakeWriter.instances = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for p in (mock.patch.object(video, "FRAME_STORE", self._tmp.name),
                  mock.patch.object(video.cv2, "VideoWriter", _FakeWriter),
                  mock.patch.object(video.cv2, "putText")):
            p.start()
            self.addCleanup(p.stop)

    def _job(self, **kw):
        job = {"status": "done", "results": [], "videoBytes": b"SRC", **kw}
        _analysis_jobs_ng["bp"] = job
        self._track(_analysis_jobs_ng, "bp")
        return job

    def _post(self, fv=None, which=None, run=None):
        fv = fv or _FakeVideo()
        which = which or {}
        with mock.patch.object(video, "MemoryVideo", return_value=fv), \
             mock.patch("shutil.which", side_effect=lambda n: which.get(n)), \
             mock.patch.object(video.subprocess, "run", **(run or {"side_effect": self._fake_ffmpeg})) as m:
            r = _client().post("/api/ng/build-playback/bp")
        return r, fv, m

    @staticmethod
    def _fake_ffmpeg(cmd, **kw):
        if "-select_streams" in cmd:
            return SimpleNamespace(stdout="1\n", returncode=0)
        with open(cmd[-1], "wb") as fh:
            fh.write(b"FINAL")
        return SimpleNamespace(returncode=0)

    def test_guards(self):
        c = _client()
        self.assertEqual(c.post("/api/ng/build-playback/zzz").status_code, 404)
        job = self._job(status="running")
        self.assertEqual(c.post("/api/ng/build-playback/bp").status_code, 400)
        job.update(status="done", videoBytes=None)
        r = c.post("/api/ng/build-playback/bp")
        self.assertEqual((r.status_code, "no longer" in r.get_json()["error"]), (400, True))

    def test_without_ffmpeg_serves_raw_and_cleans_up(self):
        job = self._job()
        frames = [(1, np.ones((6, 8, 3), np.uint8)), (2, np.ones((6, 8, 3), np.uint8))]
        r, _, run = self._post(fv=_FakeVideo(frames=frames), which={})
        self.assertEqual(r.get_json(), {"ready": True, "url": "/api/ng/playback-file/bp", "fps": 24.0})
        self.assertEqual(job["playbackBytes"], b"RAWVIDEO")
        run.assert_not_called()
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_passed_frames_kept_rejected_blanked_with_label(self):
        results = [{"frame": 1, "passed": True}, {"frame": 2, "passed": False, "failReason": "blur"},
                   {"frame": 3, "passed": False, "failReason": "sim"}, {"frame": 4, "passed": False}]
        self._job(results=results)
        src = [(i, np.full((6, 8, 3), 9, np.uint8)) for i in (1, 2, 3, 4, 5)]
        with mock.patch.object(video.cv2, "putText") as put:
            self._post(fv=_FakeVideo(frames=src))
        written = _FakeWriter.instances[0].frames
        self.assertEqual([int(f.max()) for f in written], [9, 0, 0, 0, 0])
        labels = [c.args[1] for c in put.call_args_list]
        self.assertEqual(labels, ["BLURRY  (frame 2)", "LOW MATCH  (frame 3)",
                                  "NO FACE  (frame 4)", "NO FACE  (frame 5)"])

    def test_trims_to_analysed_range(self):
        self._job(startSec=1.0, endSec=2.0)
        _, fv, _ = self._post(fv=_FakeVideo(fps=10.0))
        self.assertEqual(fv.iter_kwargs, {"start_frame": 11, "end_frame": 21})

    def test_ffmpeg_with_audio_maps_both_streams(self):
        job = self._job()
        r, _, run = self._post(which={"ffmpeg": "/ff", "ffprobe": "/fp"})
        self.assertEqual(job["playbackBytes"], b"FINAL")
        encode_cmd = run.call_args_list[-1].args[0]
        self.assertIn("1:a:0", encode_cmd)
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_ffmpeg_without_audio_skips_audio_mapping(self):
        def run(cmd, **kw):
            if "-select_streams" in cmd:
                return SimpleNamespace(stdout="", returncode=0)
            return self._fake_ffmpeg(cmd)

        self._job()
        _, _, m = self._post(which={"ffmpeg": "/ff", "ffprobe": "/fp"}, run={"side_effect": run})
        self.assertNotIn("1:a:0", m.call_args_list[-1].args[0])

    def test_ffmpeg_failure_is_500_and_cleans_up(self):
        def run(cmd, **kw):
            if "-select_streams" in cmd:
                return SimpleNamespace(stdout="", returncode=0)
            raise subprocess.CalledProcessError(1, cmd, stderr=b"boom")

        self._job()
        r, _, _ = self._post(which={"ffmpeg": "/ff", "ffprobe": "/fp"}, run={"side_effect": run})
        self.assertEqual(r.status_code, 500)
        self.assertIn("boom", r.get_json()["error"])
        self.assertEqual(os.listdir(self._tmp.name), [])


class PlaybackFileTests(_StoreCleanup):
    def test_serves_bytes_with_range_support(self):
        c = _client()
        self.assertEqual(c.get("/api/ng/playback-file/zzz").status_code, 404)
        _analysis_jobs_ng["pf"] = {"playbackBytes": b"0123456789"}
        self._track(_analysis_jobs_ng, "pf")
        r = c.get("/api/ng/playback-file/pf", headers={"Range": "bytes=2-4"})
        self.assertEqual((r.status_code, r.data, r.mimetype), (206, b"234", "video/mp4"))
        r.close()
        _analysis_jobs_ng["pf"] = {"playbackBytes": b""}
        self.assertEqual(c.get("/api/ng/playback-file/pf").status_code, 404)


if __name__ == "__main__":
    unittest.main()
