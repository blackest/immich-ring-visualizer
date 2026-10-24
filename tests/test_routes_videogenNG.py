"""Flask-test-client tests for routes/videogenNG.py (Animate view).

ltx_engineNG, video_jobsNG, engine_installNG, job_logsNG and the frame cache
are mocked at the module-attribute level; their own logic has its own tests.
This checks the HTTP layer: validation, reference-image resolution, status
code mapping and plumbing of arguments.

Run with: python3 -m unittest tests.test_routes_videogenNG -v
"""

import base64
import io
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.videogenNG as vg


def _client():
    app = Flask(__name__)
    app.register_blueprint(vg.videogenNG_bp)
    app.testing = True
    return app.test_client()


class StatusTests(unittest.TestCase):
    def _get(self, ready, ok=True, reason=None):
        with mock.patch.object(vg.ltx_engineNG, "ltx_health_ng", return_value={"ready": ready}), \
             mock.patch.object(vg.engine_installNG, "mac_apple_silicon_ok", return_value=(ok, reason)):
            return _client().get("/api/ng/videogen/status").get_json()

    def test_not_ready_is_installable(self):
        body = self._get(False, True, None)
        self.assertEqual((body["reachable"], body["installable"]), (False, True))
        self.assertEqual(body["default_width"], vg.ltx_engineNG.LTX_WIDTH)

    def test_ready_is_not_installable_and_hides_reason(self):
        body = self._get(True, False, "intel")
        self.assertEqual((body["reachable"], body["installable"], body["install_blocked_reason"]),
                         (True, False, None))

    def test_blocked_reason_surfaces_when_not_ready(self):
        body = self._get(False, False, "intel")
        self.assertEqual((body["installable"], body["install_blocked_reason"]), (False, "intel"))


class InstallTests(unittest.TestCase):
    def test_start_and_status(self):
        c = _client()
        with mock.patch.object(vg.ltx_installNG, "build_steps", return_value=["s"]), \
             mock.patch.object(vg.engine_installNG, "start_install_job",
                               return_value=SimpleNamespace(job_id="i1")) as start:
            r = c.post("/api/ng/videogen/install")
        start.assert_called_once_with("ltx", ["s"])
        self.assertEqual((r.status_code, r.get_json()["poll_url"]), (202, "/api/ng/videogen/install/i1"))

        with mock.patch.object(vg.engine_installNG, "install_job_status", return_value={"s": 1}):
            self.assertEqual(c.get("/api/ng/videogen/install/i1").get_json(), {"s": 1})
        with mock.patch.object(vg.engine_installNG, "install_job_status", side_effect=LookupError("x")):
            self.assertEqual(c.get("/api/ng/videogen/install/i1").status_code, 404)

    def test_start_value_error(self):
        with mock.patch.object(vg.ltx_installNG, "build_steps", return_value=[]), \
             mock.patch.object(vg.engine_installNG, "start_install_job", side_effect=ValueError("busy")):
            self.assertEqual(_client().post("/api/ng/videogen/install").status_code, 400)


class LorasTests(unittest.TestCase):
    def test_list(self):
        with mock.patch.object(vg, "list_ltx_loras_ng", return_value=["a"]):
            self.assertEqual(_client().get("/api/ng/videogen/loras").get_json(), {"loras": ["a"]})


class EnhanceTests(unittest.TestCase):
    def _post(self, data, **patch):
        with mock.patch.object(vg.ltx_engineNG, "enhance_ltx_prompt_ng", **patch) as f, \
             mock.patch.object(vg.ltx_engineNG, "LtxConfig"):
            r = _client().post("/api/ng/videogen/enhance", data=data, content_type="multipart/form-data")
        return r, f

    def test_prompt_required(self):
        r, f = self._post({"prompt": "  "}, return_value="x")
        self.assertEqual(r.status_code, 400)
        f.assert_not_called()

    def test_no_reference(self):
        r, f = self._post({"prompt": "walk", "seed": "4"}, return_value="enhanced")
        self.assertEqual(r.get_json(), {"ok": True, "prompt": "enhanced"})
        self.assertEqual((f.call_args.kwargs["seed"], f.call_args.kwargs["image_b64"]), (4, None))

    def test_uploaded_file_is_base64_encoded(self):
        r, f = self._post({"prompt": "walk", "file": (io.BytesIO(b"IMG"), "r.png")}, return_value="e")
        self.assertEqual(f.call_args.kwargs["image_b64"], base64.b64encode(b"IMG").decode())

    def test_ref_frame_id_uses_cache(self):
        with mock.patch.object(vg, "find_cache_frame_ng", return_value=(b"CACHED", "image/png")) as cache:
            r, f = self._post({"prompt": "walk", "ref_frame_id": "j_1"}, return_value="e")
        cache.assert_called_once_with("j_1")
        self.assertEqual(f.call_args.kwargs["image_b64"], base64.b64encode(b"CACHED").decode())

    def test_evicted_ref_frame_is_404(self):
        with mock.patch.object(vg, "find_cache_frame_ng", return_value=(None, None)):
            r, f = self._post({"prompt": "walk", "ref_frame_id": "j_1"}, return_value="e")
        self.assertEqual(r.status_code, 404)
        f.assert_not_called()

    def test_error_mapping(self):
        cases = [(FileNotFoundError("m"), 404), (subprocess.TimeoutExpired("c", 1), 504),
                 (RuntimeError("r"), 502)]
        for exc, code in cases:
            r, _ = self._post({"prompt": "walk"}, side_effect=exc)
            self.assertEqual(r.status_code, code, exc)


class GemmaJsonRoutesTests(unittest.TestCase):
    """estimate-duration, discuss and chat share validation and error mapping."""

    def _post(self, route, body, fn, **patch):
        with mock.patch.object(vg.ltx_engineNG, fn, **patch) as f, \
             mock.patch.object(vg.ltx_engineNG, "LtxConfig"):
            r = _client().post(f"/api/ng/videogen/{route}", json=body)
        return r, f

    def test_estimate_duration(self):
        r, f = self._post("estimate-duration", {"prompt": "p", "seed": 3.0}, "estimate_scene_seconds_ng",
                          return_value=4.5)
        self.assertEqual(r.get_json(), {"ok": True, "seconds": 4.5})
        self.assertEqual(f.call_args.kwargs["seed"], 3)
        r, f = self._post("estimate-duration", {"prompt": ""}, "estimate_scene_seconds_ng", return_value=1)
        self.assertEqual(r.status_code, 400)

    def test_estimate_duration_ignores_non_numeric_seed(self):
        _, f = self._post("estimate-duration", {"prompt": "p", "seed": "x"}, "estimate_scene_seconds_ng",
                          return_value=1)
        self.assertIsNone(f.call_args.kwargs["seed"])

    def test_estimate_error_mapping(self):
        for exc, code in [(FileNotFoundError("m"), 404), (subprocess.TimeoutExpired("c", 1), 504),
                          (RuntimeError("r"), 502)]:
            r, _ = self._post("estimate-duration", {"prompt": "p"}, "estimate_scene_seconds_ng", side_effect=exc)
            self.assertEqual(r.status_code, code)

    def test_message_validation_shared_by_discuss_and_chat(self):
        bad = [None, [], "x", [{"role": "system", "content": "x"}], [{"role": "user", "content": 5}], ["str"]]
        for route, fn in (("discuss", "discuss_next_scene_ng"), ("chat", "chat_with_gemma_ng")):
            for messages in bad:
                r, f = self._post(route, {"messages": messages}, fn, return_value={})
                self.assertEqual(r.status_code, 400, (route, messages))
                f.assert_not_called()

    def test_discuss_and_chat_pass_clean_turns_and_last_turn_images(self):
        msgs = [{"role": "user", "content": "a", "images": ["OLD"], "extra": 1},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c", "images": ["NEW", "", 5]}]
        for route, fn in (("discuss", "discuss_next_scene_ng"), ("chat", "chat_with_gemma_ng")):
            r, f = self._post(route, {"messages": msgs, "seed": 9}, fn, return_value={"reply": "hi"})
            self.assertEqual(r.get_json(), {"ok": True, "reply": "hi"})
            turns = f.call_args.args[0]
            self.assertEqual(turns[0], {"role": "user", "content": "a"})
            self.assertEqual(f.call_args.kwargs["images"], ["NEW"])
            self.assertEqual(f.call_args.kwargs["seed"], 9)

    def test_images_none_when_last_turn_is_assistant_or_empty(self):
        msgs = [{"role": "user", "content": "a", "images": ["X"]}, {"role": "assistant", "content": "b"}]
        _, f = self._post("chat", {"messages": msgs}, "chat_with_gemma_ng", return_value={})
        self.assertIsNone(f.call_args.kwargs["images"])
        msgs = [{"role": "user", "content": "a", "images": ["", 1]}]
        _, f = self._post("chat", {"messages": msgs}, "chat_with_gemma_ng", return_value={})
        self.assertIsNone(f.call_args.kwargs["images"])

    def test_discuss_chat_error_mapping(self):
        for route, fn in (("discuss", "discuss_next_scene_ng"), ("chat", "chat_with_gemma_ng")):
            for exc, code in [(FileNotFoundError("m"), 404), (subprocess.TimeoutExpired("c", 1), 504),
                              (RuntimeError("r"), 502)]:
                r, _ = self._post(route, {"messages": [{"role": "user", "content": "x"}]}, fn, side_effect=exc)
                self.assertEqual(r.status_code, code, (route, exc))


class GenerateTests(unittest.TestCase):
    def _post(self, data, lora_path="unset", **start):
        start = start or {"return_value": SimpleNamespace(job_id="v1")}
        patches = [mock.patch.object(vg.video_jobs, "start_video_job_ng", **start)]
        if lora_path != "unset":
            patches.append(mock.patch.object(vg, "resolve_ltx_lora_path_ng", return_value=lora_path))
        with patches[0] as f:
            for p in patches[1:]:
                p.start()
                self.addCleanup(p.stop)
            r = _client().post("/api/ng/videogen/generate", data=data, content_type="multipart/form-data")
        return r, f

    def test_prompt_required(self):
        r, f = self._post({"prompt": ""})
        self.assertEqual(r.status_code, 400)
        f.assert_not_called()

    def test_text_to_video_defaults(self):
        r, f = self._post({"prompt": "a dog"})
        self.assertEqual((r.status_code, r.get_json()["poll_url"]), (202, "/api/ng/videogen/jobs/v1"))
        args, kw = f.call_args
        self.assertEqual(args, (None, ".jpg", "a dog", 3.0, None))
        self.assertEqual((kw["width"], kw["height"], kw["frame_rate"], kw["lora_path"], kw["lora_strength"]),
                         (None, None, None, None, 1.0))

    def test_all_fields_and_upload(self):
        r, f = self._post({"prompt": "p", "duration_s": "5.5", "seed": "11", "width": "640", "height": "384",
                           "frame_rate": "24", "file": (io.BytesIO(b"IMG"), "ref.WEBP")})
        args, kw = f.call_args
        self.assertEqual(args, (b"IMG", ".webp", "p", 5.5, 11))
        self.assertEqual((kw["width"], kw["height"], kw["frame_rate"]), (640, 384, 24.0))

    def test_cached_ref_frame_mimetype_sets_extension(self):
        with mock.patch.object(vg, "find_cache_frame_ng", return_value=(b"C", "image/png")):
            _, f = self._post({"prompt": "p", "ref_frame_id": "j_a"})
        self.assertEqual(f.call_args.args[:2], (b"C", ".png"))

    def test_evicted_ref_frame_is_404(self):
        with mock.patch.object(vg, "find_cache_frame_ng", return_value=(None, None)):
            r, f = self._post({"prompt": "p", "ref_frame_id": "j_a"})
        self.assertEqual(r.status_code, 404)
        f.assert_not_called()

    def test_lora_resolution(self):
        r, f = self._post({"prompt": "p", "lora_name": "nope"}, lora_path=None)
        self.assertEqual(r.status_code, 400)
        f.assert_not_called()
        r, f = self._post({"prompt": "p", "lora_name": "style", "lora_strength": "0.6"}, lora_path="/l.safetensors")
        self.assertEqual((f.call_args.kwargs["lora_path"], f.call_args.kwargs["lora_strength"]),
                         ("/l.safetensors", 0.6))

    def test_start_error_mapping(self):
        for exc, code in [(LookupError("a"), 404), (FileNotFoundError("b"), 404), (ValueError("c"), 400)]:
            r, _ = self._post({"prompt": "p"}, side_effect=exc)
            self.assertEqual(r.status_code, code, exc)


class JobRoutesTests(unittest.TestCase):
    def test_status(self):
        c = _client()
        with mock.patch.object(vg.video_jobs, "job_status_ng", return_value={"status": "done"}):
            self.assertEqual(c.get("/api/ng/videogen/jobs/j").get_json(), {"status": "done"})
        with mock.patch.object(vg.video_jobs, "job_status_ng", side_effect=LookupError("x")):
            self.assertEqual(c.get("/api/ng/videogen/jobs/j").status_code, 404)

    def test_cancel(self):
        with mock.patch.object(vg.video_jobs, "cancel_video_job_ng", return_value=True):
            self.assertEqual(_client().delete("/api/ng/videogen/jobs/j").get_json(), {"ok": True})


class ServeVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mp4 = os.path.join(self._tmp.name, "out.mp4")
        with open(self.mp4, "wb") as fh:
            fh.write(b"MP4DATA")

    def _get(self, suffix, job):
        with mock.patch.object(vg.video_jobs, "get_job_ng", return_value=job):
            return _client().get(f"/api/ng/videogen/jobs/j1/{suffix}")

    def test_inline_and_download(self):
        job = SimpleNamespace(mp4_path=self.mp4)
        r = self._get("video", job)
        self.assertEqual((r.status_code, r.mimetype, r.data), (200, "video/mp4", b"MP4DATA"))
        self.assertNotIn("attachment", r.headers.get("Content-Disposition", ""))
        r.close()
        r = self._get("download", job)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        self.assertIn("animate-j1.mp4", r.headers["Content-Disposition"])
        r.close()

    def test_unknown_job_and_unready_video(self):
        for suffix in ("video", "download"):
            self.assertEqual(self._get(suffix, None).status_code, 404)
            self.assertEqual(self._get(suffix, SimpleNamespace(mp4_path=None)).status_code, 404)
            self.assertEqual(self._get(suffix, SimpleNamespace(mp4_path="/missing.mp4")).status_code, 404)


class JobLogRoutesTests(unittest.TestCase):
    def _get(self, url, fn, **patch):
        with mock.patch.object(vg.job_logsNG, fn, **patch) as f:
            return _client().get(url), f

    def test_listings(self):
        r, _ = self._get("/api/ng/videogen/logs", "list_recent_logs_ng", return_value=[1])
        self.assertEqual(r.get_json(), {"ok": True, "entries": [1]})
        r, _ = self._get("/api/ng/videogen/logs/archive", "list_archive_years_ng", return_value=["2026"])
        self.assertEqual(r.get_json(), {"ok": True, "years": ["2026"]})
        r, f = self._get("/api/ng/videogen/logs/archive/2026", "list_archive_months_ng", return_value=["10"])
        f.assert_called_once_with("2026")
        self.assertEqual(r.get_json()["months"], ["10"])
        r, f = self._get("/api/ng/videogen/logs/archive/2026/10", "list_archive_day_entries_ng", return_value=[2])
        f.assert_called_once_with("2026", "10")
        self.assertEqual(r.get_json()["entries"], [2])

    def test_ref_and_video_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.bin")
            with open(p, "wb") as fh:
                fh.write(b"X")
            r, f = self._get("/api/ng/videogen/logs/2026-10-24/j1/ref", "log_ref_path_ng", return_value=p)
            f.assert_called_once_with("2026-10-24", "j1")
            self.assertEqual(r.data, b"X")
            r.close()
            r, _ = self._get("/api/ng/videogen/logs/2026-10-24/j1/video", "log_video_path_ng", return_value=p)
            self.assertEqual(r.mimetype, "video/mp4")
            r.close()
        r, _ = self._get("/api/ng/videogen/logs/d/j/ref", "log_ref_path_ng", return_value=None)
        self.assertEqual(r.status_code, 404)
        r, _ = self._get("/api/ng/videogen/logs/d/j/video", "log_video_path_ng", return_value=None)
        self.assertEqual(r.status_code, 404)

    def test_update_notes(self):
        c = _client()
        with mock.patch.object(vg.job_logsNG, "update_log_notes_ng", return_value={"notes": "hi"}) as f:
            r = c.patch("/api/ng/videogen/logs/d/j/notes", json={"notes": "hi"})
        f.assert_called_once_with("d", "j", "hi")
        self.assertEqual(r.get_json(), {"ok": True, "entry": {"notes": "hi"}})
        with mock.patch.object(vg.job_logsNG, "update_log_notes_ng", return_value=None):
            self.assertEqual(c.patch("/api/ng/videogen/logs/d/j/notes", json={}).status_code, 404)

    def test_delete(self):
        c = _client()
        with mock.patch.object(vg.job_logsNG, "delete_log_ng", return_value=True):
            self.assertEqual(c.delete("/api/ng/videogen/logs/d/j").get_json(), {"ok": True})
        with mock.patch.object(vg.job_logsNG, "delete_log_ng", return_value=False):
            self.assertEqual(c.delete("/api/ng/videogen/logs/d/j").status_code, 404)


if __name__ == "__main__":
    unittest.main()
