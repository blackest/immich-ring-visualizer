"""immichring MCP server: the client and the tool layer, against a stub that answers like the app does.

No GPU and no running app: a throwaway HTTP server on a free port returns the same JSON shapes the NG
routes return (read from routes/h3NG.py, routes/kleinNG.py, routes/enhancedh3NG.py).

Run with: python3 -m unittest tests.test_immichring_mcp -v
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

MCP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp_server")
sys.path.insert(0, MCP_DIR)
import immichring_client as api  # noqa: E402
import immichring_mcp as mcp  # noqa: E402

SEEN = []   # (method, path, raw body) for every request the stub got


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        body = self._body()
        SEEN.append(("POST", self.path, body))
        if self.path == "/api/ng/enhanced-h3/brief":
            if b"empty-brief" in body:
                return self._send(201, {"windows": [], "question": "Which character?"})
            return self._send(201, {"id": "b1", "windows": [{"continues": False, "prompt": "shot one", "shots": [1]}, {"continues": False, "prompt": "shot two", "shots": [2]}], "note": "ok"})
        if self.path == "/api/ng/h3/generate":
            if b"toolong" in body:
                return self._send(400, {"error": "duration_s must be at most 15"})
            return self._send(202, {"ok": True, "job_id": "abc123", "poll_url": "/api/ng/h3/jobs/abc123"})
        if self.path == "/api/ng/music/generate":
            if b"toolong" in body:
                return self._send(400, {"error": "duration_s too long"})
            return self._send(202, {"ok": True, "job_id": "m1", "poll_url": "/api/ng/music/jobs/m1"})
        if self.path == "/api/ng/speech/generate":
            return self._send(202, {"ok": True, "job_id": "s1", "poll_url": "/api/ng/speech/jobs/s1"})
        if self.path == "/api/ng/klein/generate":
            return self._send(200, {"jobId": "k1", "job": {"id": "k1", "status": "queued"}})
        self._send(404, {"error": "nope"})

    def do_GET(self):
        SEEN.append(("GET", self.path, b""))
        p = self.path
        if p == "/api/ng/h3/jobs/abc123":
            return self._send(200, {"job_id": "abc123", "status": "completed", "prompt": "a hill", "error": None})
        if p == "/api/ng/h3/jobs/bad":
            return self._send(200, {"job_id": "bad", "status": "failed", "prompt": "x", "error": "out of memory"})
        if p == "/api/ng/h3/jobs/run":
            return self._send(200, {"job_id": "run", "status": "rendering", "prompt": "x", "error": None})
        if p == "/api/ng/h3/jobs/abc123/download":
            return self._send(200, b"MP4DATA", "video/mp4")
        if p == "/api/ng/h3/jobs/none":
            return self._send(404, {"error": "no such job 'none'"})
        if p == "/api/ng/h3/jobs":
            return self._send(200, {"jobs": [{"job_id": "abc123", "status": "completed", "prompt": "a hill",
                                              "queued_at": 20}]})
        if p == "/api/ng/music/jobs/m1":
            return self._send(200, {"job_id": "m1", "status": "completed", "style": "slow blues", "lyrics": "la la",
                                    "audio_seconds": 61.234, "error": None})
        if p == "/api/ng/music/logs":
            return self._send(200, {"ok": True, "entries": [
                {"job_id": "m9", "date": "2026-11-18", "status": "completed", "style": "slow blues", "lyrics": "",
                 "audio_seconds": 30.0, "error": None, "has_audio": True, "finished_at": 50, "queued_at": 10},
                {"job_id": "m8", "date": "2026-11-17", "status": "failed", "style": "polka", "lyrics": "",
                 "audio_seconds": None, "error": "out of memory", "has_audio": False, "finished_at": 5, "queued_at": 1}]})
        if p == "/api/ng/music/logs/2026-11-18/m9/audio":
            return self._send(200, b"LOGFLAC", "audio/flac")
        if p == "/api/ng/music/jobs/m1/download":
            return self._send(200, b"FLACDATA", "audio/flac")
        if p == "/api/ng/speech/voices":
            return self._send(200, {"presets": [{"id": "Aiden", "label": "Aiden"}, {"id": "Serena", "label": "Serena"}],
                                    "characters": [{"id": "knight", "label": "knight"}], "styles": [], "languages": []})
        if p == "/api/ng/speech/jobs/s1":
            return self._send(200, {"job_id": "s1", "status": "failed", "text": "hello", "error": "voice model missing",
                                    "audio_seconds": None})
        if p == "/api/ng/speech/jobs/s2":
            return self._send(200, {"job_id": "s2", "status": "completed", "text": "hello", "error": None,
                                    "audio_seconds": 2.0})
        if p == "/api/ng/speech/jobs/s2/download":
            return self._send(200, b"WAVDATA", "audio/wav")
        if p == "/api/ng/speech/jobs":
            return self._send(200, {"jobs": [{"job_id": "s2", "status": "completed", "text": "hello",
                                              "audio_seconds": 2.0, "queued_at": 40}]})
        if p == "/api/ng/klein/status/k1":
            return self._send(200, {"id": "k1", "status": "done", "prompt": "an apple", "error": None,
                                    "results": [{"file": "out1.png"}, {"file": "out2.png"}], "created_at": 30})
        if p == "/api/ng/klein/image/k1/out1.png":
            return self._send(200, b"PNGDATA", "image/png")
        if p.startswith("/api/ng/klein/jobs"):
            return self._send(200, {"jobs": [{"id": "k1", "status": "done", "prompt": "an apple",
                                              "results": [{"file": "out1.png"}], "created_at": 30}], "total": 1})
        self._send(404, {"error": "nope"})


def run(name, args):
    res = mcp.call_tool(name, args)
    text = res["content"][0]["text"]
    return res["isError"], (text if res["isError"] else json.loads(text))


class StubbedCase(unittest.TestCase):
    def setUp(self):
        SEEN.clear()
        self.server = HTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        patch = mock.patch.object(api, "BASE", f"http://127.0.0.1:{self.server.server_address[1]}")
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def pic(self, name, data=b"IMG"):
        p = self.tmp / name
        p.write_bytes(data)
        return str(p)


class ToolLayerTests(StubbedCase):
    def test_eight_tools_with_schemas(self):
        names = [t["name"] for t in mcp._tool_list()]
        self.assertEqual(names, ["compile_brief", "render_video", "make_image", "make_song", "make_speech",
                                 "job_status", "fetch", "recent_jobs"])
        self.assertTrue(all("run" not in t and t["inputSchema"]["type"] == "object" for t in mcp._tool_list()))

    def test_protocol_handshake_and_list(self):
        init = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(init["result"]["serverInfo"]["name"], "immichring")
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(len(mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]), 8)
        self.assertEqual(mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"], -32601)

    def test_missing_and_unknown_arguments_are_sentences(self):
        self.assertEqual(run("compile_brief", {}), (True, "compile_brief needs: intent."))
        err, text = run("job_status", {"job_id": "h3:abc123", "kind": "h3"})
        self.assertTrue(err)
        self.assertIn("does not take: kind", text)
        self.assertTrue(run("nope", {})[0])

    def test_compile_brief_returns_windows_or_the_compilers_question(self):
        err, out = run("compile_brief", {"intent": "a man on a hill"})
        self.assertFalse(err)
        self.assertEqual(out["windows"], ["shot one", "shot two"])
        self.assertEqual(run("compile_brief", {"intent": "empty-brief"}), (True, "Which character?"))


class RenderTests(StubbedCase):
    def test_defaults_are_h3_fast_draft(self):
        err, out = run("render_video", {"prompt": "a hill"})
        self.assertFalse(err)
        self.assertEqual(out, {"job_id": "h3:abc123", "status": "queued"})
        body = SEEN[-1][2].decode()
        for want in ('name="model"\r\n\r\nh3q8', 'name="quality"\r\n\r\ndraft', 'name="tristep"\r\n\r\n1',
                     'name="prompt"\r\n\r\na hill'):
            self.assertIn(want, body)

    def test_windows_go_as_chain_prompts_and_replace_prompt(self):
        run("render_video", {"windows": ["one", "two"]})
        body = SEEN[-1][2].decode()
        self.assertIn('name="chain_prompts"\r\n\r\n["one", "two"]', body)
        self.assertNotIn('name="prompt"', body)
        self.assertIn('name="duration_s"\r\n\r\n10.0', body)

    def test_compiler_objects_are_reduced_to_strings_and_one_window_becomes_a_prompt(self):
        run("render_video", {"windows": [{"prompt": "one"}, {"prompt": "two"}]})
        self.assertIn('name="chain_prompts"\r\n\r\n["one", "two"]', SEEN[-1][2].decode())
        run("render_video", {"windows": ["only"]})
        body = SEEN[-1][2].decode()
        self.assertIn('name="prompt"\r\n\r\nonly', body)
        self.assertNotIn("chain_prompts", body)

    def test_needs_something_to_render_and_a_known_quality(self):
        self.assertTrue(run("render_video", {})[0])
        self.assertIn("quality must be one of", run("render_video", {"prompt": "x", "quality": "ultra"})[1])

    def test_app_error_comes_back_as_its_own_sentence(self):
        self.assertEqual(run("render_video", {"prompt": "toolong"}), (True, "duration_s must be at most 15"))

    def test_first_frame_is_uploaded_and_missing_files_are_named(self):
        run("render_video", {"prompt": "x", "first_frame": self.pic("start.png")})
        self.assertIn(b'name="file"; filename="start.png"', SEEN[-1][2])
        err, text = run("render_video", {"prompt": "x", "first_frame": str(self.tmp / "gone.png")})
        self.assertTrue(err)
        self.assertIn("No picture at", text)


class ImageTests(StubbedCase):
    def test_references_are_uploaded_in_slot_order(self):
        err, out = run("make_image", {"prompt": "an apple", "pictures": [self.pic("a.png"), self.pic("b.jpg")]})
        self.assertFalse(err)
        self.assertEqual(out["job_id"], "klein:k1")
        body = SEEN[-1][2]
        self.assertIn(b'name="ref0"; filename="a.png"', body)
        self.assertIn(b'name="ref1"; filename="b.jpg"', body)
        self.assertIn(b'name="width"\r\n\r\n1024', body)

    def test_more_than_three_pictures_is_refused(self):
        err, text = run("make_image", {"prompt": "x", "pictures": [self.pic(f"{i}.png") for i in range(4)]})
        self.assertTrue(err)
        self.assertIn("At most 3", text)


class SongSpeechTests(StubbedCase):
    def test_a_song_goes_as_json_and_comes_back_as_a_music_id(self):
        self.assertEqual(run("make_song", {"style": "slow blues", "lyrics": "la la"}),
                         (False, {"job_id": "music:m1", "status": "queued"}))
        method, path, body = SEEN[-1]
        self.assertEqual((method, path), ("POST", "/api/ng/music/generate"))
        self.assertEqual(json.loads(body), {"style": "slow blues", "lyrics": "la la", "instrumental": False})

    def test_seconds_is_only_sent_when_given_and_an_instrumental_needs_no_words(self):
        run("make_song", {"style": "ambient", "instrumental": True, "seconds": 90})
        self.assertEqual(json.loads(SEEN[-1][2]), {"style": "ambient", "lyrics": "", "instrumental": True,
                                                   "duration_s": 90.0})

    def test_a_song_needs_a_style_or_lyrics_and_app_errors_come_through(self):
        self.assertIn("style", run("make_song", {})[1])
        self.assertEqual(run("make_song", {"style": "x", "seconds": 5}), (False, {"job_id": "music:m1", "status": "queued"}))
        self.assertEqual(run("make_song", {"style": "toolong"}), (True, "duration_s too long"))

    def test_speech_uses_the_first_preset_when_no_voice_is_named(self):
        err, out = run("make_speech", {"text": "hello"})
        self.assertEqual((err, out), (False, {"job_id": "speech:s1", "status": "queued", "voice": "Aiden"}))
        body = SEEN[-1][2].decode()
        for want in ('name="kind"\r\n\r\npreset', 'name="voice"\r\n\r\nAiden', 'name="text"\r\n\r\nhello',
                     'name="language"\r\n\r\nenglish'):
            self.assertIn(want, body)

    def test_a_saved_character_voice_goes_as_a_character_and_names_ignore_case(self):
        err, out = run("make_speech", {"text": "hi", "voice": "KNIGHT"})
        self.assertFalse(err)
        self.assertEqual(out["voice"], "knight")
        body = SEEN[-1][2].decode()
        self.assertIn('name="kind"\r\n\r\ncharacter', body)
        self.assertIn('name="character"\r\n\r\nknight', body)
        self.assertNotIn('name="voice"', body)

    def test_an_unknown_voice_lists_the_choices_and_sends_nothing(self):
        before = len([s for s in SEEN if s[1] == "/api/ng/speech/generate"])
        err, text = run("make_speech", {"text": "hi", "voice": "Zed"})
        self.assertTrue(err)
        self.assertIn("Aiden, Serena, knight", text)
        self.assertEqual(len([s for s in SEEN if s[1] == "/api/ng/speech/generate"]), before)

    def test_status_for_music_and_speech_uses_the_shared_words(self):
        self.assertEqual(run("job_status", {"job_id": "music:m1"})[1],
                         {"job_id": "music:m1", "kind": "song", "status": "done", "prompt": "la la", "seconds": 61.2})
        err, out = run("job_status", {"job_id": "speech:s1"})
        self.assertEqual((out["status"], out["error"]), ("failed", "voice model missing"))

    def test_songs_and_speech_are_saved_with_the_right_extension(self):
        out = run("fetch", {"job_id": "music:m1", "dest": str(self.tmp)})[1]
        self.assertTrue(out["saved"].endswith("music-m1.flac"))
        self.assertEqual(Path(out["saved"]).read_bytes(), b"FLACDATA")
        out = run("fetch", {"job_id": "speech:s2", "dest": str(self.tmp / "hello.wav")})[1]
        self.assertEqual(Path(out["saved"]).read_bytes(), b"WAVDATA")

    def test_recent_jobs_includes_speech(self):
        kinds = {j["kind"] for j in run("recent_jobs", {})[1]["jobs"]}
        self.assertEqual(kinds, {"video", "picture", "speech", "song"})

    def test_a_song_the_app_forgot_is_still_found_and_fetched_through_the_log(self):
        err, out = run("job_status", {"job_id": "music:m9"})
        self.assertEqual((err, out["status"], out["seconds"]), (False, "done", 30.0))
        saved = run("fetch", {"job_id": "music:m9", "dest": str(self.tmp)})[1]["saved"]
        self.assertEqual(Path(saved).read_bytes(), b"LOGFLAC")
        self.assertEqual(run("job_status", {"job_id": "music:m8"})[1]["error"], "out of memory")
        self.assertTrue(run("job_status", {"job_id": "music:nothere"})[0])


class StatusFetchRecentTests(StubbedCase):
    def test_states_are_named_the_same_for_both_kinds(self):
        self.assertEqual(run("job_status", {"job_id": "h3:abc123"})[1]["status"], "done")
        self.assertEqual(run("job_status", {"job_id": "h3:run"})[1]["status"], "rendering")
        self.assertEqual(run("job_status", {"job_id": "klein:k1"})[1], {
            "job_id": "klein:k1", "kind": "picture", "status": "done", "prompt": "an apple", "pictures": 2})
        failed = run("job_status", {"job_id": "h3:bad"})[1]
        self.assertEqual((failed["status"], failed["error"]), ("failed", "out of memory"))

    def test_klein_words_are_translated_so_waiting_ends_on_a_failure(self):
        # Klein says "error" for a failed job and "running"/"waiting" while it works; a wait that only
        # knew the shared words would never stop on a failed picture.
        for theirs, ours in (("error", "failed"), ("running", "rendering"), ("waiting", "queued"),
                             ("queued", "queued"), ("cancelled", "cancelled"), ("done", "done")):
            self.assertEqual(api._klein_view({"id": "k", "status": theirs, "prompt": "x"})["status"], ours)
        self.assertEqual(api._klein_view({"id": "k", "status": "error", "error": "no memory"})["error"], "no memory")

    def test_bad_ids_and_unknown_jobs(self):
        self.assertIn("is not a job id", run("job_status", {"job_id": "abc123"})[1])
        self.assertIn("is not a job id", run("job_status", {"job_id": "foo:1"})[1])
        self.assertEqual(run("job_status", {"job_id": "h3:none"}), (True, "no such job 'none'"))

    def test_waiting_gives_up_at_the_deadline(self):
        self.assertEqual(api.job_status("h3:run", wait_s=0)["status"], "rendering")

    def test_fetch_saves_video_and_first_picture(self):
        err, _ = run("fetch", {"job_id": "h3:abc123", "dest": str(self.tmp / "clip.mp4")})
        self.assertFalse(err)
        self.assertEqual((self.tmp / "clip.mp4").read_bytes(), b"MP4DATA")
        err, out = run("fetch", {"job_id": "klein:k1", "dest": str(self.tmp)})
        self.assertFalse(err)
        self.assertEqual((self.tmp / "klein-k1.png").read_bytes(), b"PNGDATA")
        self.assertIn("1 more picture", out["note"])

    def test_fetch_refuses_a_job_that_is_not_done(self):
        err, text = run("fetch", {"job_id": "h3:run", "dest": str(self.tmp / "x.mp4")})
        self.assertTrue(err)
        self.assertIn("rendering, not done yet", text)
        self.assertFalse((self.tmp / "x.mp4").exists())

    def test_recent_jobs_merges_all_kinds_newest_first(self):
        err, out = run("recent_jobs", {})
        self.assertFalse(err)
        self.assertEqual([j["job_id"] for j in out["jobs"]], ["music:m9", "speech:s2", "klein:k1", "h3:abc123", "music:m8"])


class OutsideTests(unittest.TestCase):
    def test_unreachable_app_says_so(self):
        with mock.patch.object(api, "BASE", "http://127.0.0.1:9"):
            err, text = run("recent_jobs", {})
        self.assertTrue(err)
        self.assertIn("Cannot reach immichring", text)

    def test_stdio_server_round_trip(self):
        """The real script, over a pipe, the way a client talks to it."""
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "job_status", "arguments": {}}}]
        p = subprocess.run([sys.executable, os.path.join(MCP_DIR, "immichring_mcp.py")],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                           timeout=20)
        replies = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual([r["id"] for r in replies], [1, 2])
        self.assertTrue(replies[1]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
