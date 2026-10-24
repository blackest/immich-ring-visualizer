"""Flask-test-client tests for routes/chatNG.py (Ollama proxy) and
routes/rachelNG.py (Hermes gateway proxy, SSE -> NDJSON).

`requests.get` / `requests.post` are mocked (both modules import requests
inside the handlers), so nothing touches the network.

Run with: python3 -m unittest tests.test_routes_chat_rachelNG -v
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.chatNG as chat
import routes.rachelNG as rachel


class _Resp:
    def __init__(self, status=200, body=None, chunks=(), lines=(), text="", raise_in_stream=None):
        self.status_code = status
        self._body = body
        self.content = b"x" if body is not None else b""
        self.text = text
        self._chunks, self._lines = list(chunks), list(lines)
        self._raise = raise_in_stream
        self.closed = False

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, chunk_size=None):
        yield from self._chunks
        if self._raise:
            raise self._raise

    def iter_lines(self, decode_unicode=False):
        yield from self._lines
        if self._raise:
            raise self._raise

    def close(self):
        self.closed = True


def _client(bp):
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.testing = True
    return app.test_client()


def _ndjson(resp):
    return [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l]


class ChatStatusModelTests(unittest.TestCase):
    def setUp(self):
        self.c = _client(chat.chatNG_bp)
        p = mock.patch.object(chat, "get_ollama_base_url", return_value="http://ollama:11434")
        p.start()
        self.addCleanup(p.stop)

    def test_status_reachable_unreachable_and_http_error(self):
        with mock.patch("requests.get", return_value=_Resp(200, {"version": "0.9"})) as g:
            body = self.c.get("/api/ng/chat/status").get_json()
        self.assertEqual(body, {"reachable": True, "base_url": "http://ollama:11434", "version": "0.9"})
        self.assertEqual(g.call_args.args[0], "http://ollama:11434/api/version")
        with mock.patch("requests.get", side_effect=ConnectionError("refused")):
            body = self.c.get("/api/ng/chat/status").get_json()
        self.assertEqual((body["reachable"], body["error"]), (False, "refused"))
        with mock.patch("requests.get", return_value=_Resp(503)):
            self.assertEqual(self.c.get("/api/ng/chat/status").get_json()["error"], "HTTP 503")

    def test_models_sorted_case_insensitively_with_loaded_list(self):
        tags = _Resp(200, {"models": [
            {"name": "zeta:1b", "size": 5, "details": {"parameter_size": "1B"}},
            {"name": "Alpha:7b", "size": 9, "details": {"parameter_size": "7B", "quantization_level": "Q4",
                                                       "family": "llama"}},
            {"name": None},
        ]})
        ps = _Resp(200, {"models": [{"name": "Alpha:7b"}]})
        with mock.patch("requests.get", side_effect=[tags, ps]):
            body = self.c.get("/api/ng/chat/models").get_json()
        self.assertEqual([m["name"] for m in body["models"]], [None, "Alpha:7b", "zeta:1b"])
        alpha = body["models"][1]
        self.assertEqual((alpha["parameter_size"], alpha["quantization"], alpha["family"]), ("7B", "Q4", "llama"))
        self.assertEqual(body["loaded"], ["Alpha:7b"])

    def test_models_survive_a_failing_ps_endpoint(self):
        for ps in (ConnectionError("x"), _Resp(500)):
            with self.subTest(ps=ps):
                side = [_Resp(200, {"models": []}), ps]
                with mock.patch("requests.get", side_effect=side):
                    self.assertEqual(self.c.get("/api/ng/chat/models").get_json(), {"models": [], "loaded": []})

    def test_models_502_when_tags_fails(self):
        with mock.patch("requests.get", side_effect=ConnectionError("down")):
            r = self.c.get("/api/ng/chat/models")
        self.assertEqual(r.status_code, 502)
        self.assertIn("could not reach Ollama", r.get_json()["error"])
        with mock.patch("requests.get", return_value=_Resp(500)):
            r = self.c.get("/api/ng/chat/models")
        self.assertEqual(r.status_code, 502)
        self.assertIn("HTTP 500", r.get_json()["error"])


class ChatSendTests(unittest.TestCase):
    def setUp(self):
        self.c = _client(chat.chatNG_bp)
        p = mock.patch.object(chat, "get_ollama_base_url", return_value="http://ollama:11434")
        p.start()
        self.addCleanup(p.stop)

    def _send(self, body):
        return self.c.post("/api/ng/chat/send", json=body)

    def test_validation(self):
        msgs = [{"role": "user", "content": "hi"}]
        for body in ({}, {"model": "  ", "messages": msgs}, {"model": "m"}, {"model": "m", "messages": []},
                     {"model": "m", "messages": "hi"}):
            with self.subTest(body=body):
                with mock.patch("requests.post") as post:
                    self.assertEqual(self._send(body).status_code, 400)
                post.assert_not_called()

    def test_streams_upstream_chunks_verbatim_with_no_buffering_headers(self):
        up = _Resp(200, chunks=[b'{"message":{"content":"he"}}\n', b"", b'{"done":true}\n'])
        with mock.patch("requests.post", return_value=up) as post:
            r = self._send({"model": " m ", "messages": [{"role": "user", "content": "hi"}],
                            "options": {"temperature": 0.2}})
            data = r.get_data()
        self.assertEqual(data, b'{"message":{"content":"he"}}\n{"done":true}\n')
        self.assertEqual(r.mimetype, "application/x-ndjson")
        self.assertEqual((r.headers["Cache-Control"], r.headers["X-Accel-Buffering"]), ("no-cache", "no"))
        kw = post.call_args.kwargs
        self.assertEqual(kw["json"], {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                      "stream": True, "options": {"temperature": 0.2}})
        self.assertEqual(kw["timeout"], (5, 300))
        self.assertTrue(up.closed)

    def test_empty_or_non_dict_options_are_not_forwarded(self):
        for opts in ({}, "hot", None):
            with self.subTest(opts=opts):
                with mock.patch("requests.post", return_value=_Resp(200)) as post:
                    self._send({"model": "m", "messages": [{"role": "user", "content": "x"}],
                                "options": opts}).get_data()
                self.assertNotIn("options", post.call_args.kwargs["json"])

    def test_upstream_errors_become_502_with_detail(self):
        msgs = [{"role": "user", "content": "x"}]
        with mock.patch("requests.post", return_value=_Resp(404, {"error": "model not found"})):
            r = self._send({"model": "m", "messages": msgs})
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.get_json()["error"], "Ollama returned HTTP 404: model not found")
        with mock.patch("requests.post", return_value=_Resp(500, text="boom " * 100)):
            r = self._send({"model": "m", "messages": msgs})
        self.assertLessEqual(len(r.get_json()["error"]), len("Ollama returned HTTP 500: ") + 200)
        with mock.patch("requests.post", side_effect=ConnectionError("down")):
            r = self._send({"model": "m", "messages": msgs})
        self.assertIn("could not reach Ollama: down", r.get_json()["error"])

    def test_mid_stream_break_becomes_a_final_error_line(self):
        up = _Resp(200, chunks=[b'{"message":{"content":"a"}}\n'], raise_in_stream=ConnectionError("cut"))
        with mock.patch("requests.post", return_value=up):
            lines = _ndjson(self._send({"model": "m", "messages": [{"role": "user", "content": "x"}]}))
        self.assertEqual(lines[-1], {"error": "cut", "done": True})
        self.assertTrue(up.closed)


class RachelTests(unittest.TestCase):
    def setUp(self):
        self.c = _client(rachel.rachelNG_bp)
        for name, val in (("get_hermes_api_key", "KEY"), ("get_hermes_base_url", "http://hermes:8642"),
                          ("get_hermes_model", "rachel-1"), ("get_hermes_session_id", "SID"),
                          ("get_hermes_session_key", "SKEY")):
            p = mock.patch.object(rachel, name, return_value=val)
            p.start()
            self.addCleanup(p.stop)

    def _send(self, body):
        return self.c.post("/api/ng/rachel/send", json=body)

    @staticmethod
    def _sse(*objs, done=True):
        lines = [": keepalive", ""]
        for o in objs:
            lines += ["data: " + (o if isinstance(o, str) else json.dumps(o)), ""]
        if done:
            lines.append("data: [DONE]")
        return lines

    @staticmethod
    def _delta(text):
        return {"choices": [{"delta": {"content": text}}]}

    def test_status(self):
        with mock.patch("requests.get", return_value=_Resp(200, {"platform": "mac", "version": "1", "status": "ok"})) as g:
            body = self.c.get("/api/ng/rachel/status").get_json()
        self.assertEqual(g.call_args.args[0], "http://hermes:8642/health")
        self.assertEqual(body, {"reachable": True, "base_url": "http://hermes:8642", "model": "rachel-1",
                                "platform": "mac", "version": "1", "status": "ok"})
        with mock.patch("requests.get", side_effect=ConnectionError("nope")):
            self.assertEqual(self.c.get("/api/ng/rachel/status").get_json()["reachable"], False)
        with mock.patch("requests.get", return_value=_Resp(401)):
            self.assertEqual(self.c.get("/api/ng/rachel/status").get_json()["error"], "HTTP 401")

    def test_send_validation(self):
        for body in ({}, {"messages": []}, {"messages": "hi"}):
            with self.subTest(body=body):
                self.assertEqual(self._send(body).status_code, 400)

    def test_sse_is_translated_to_ndjson_and_headers_attached(self):
        up = _Resp(200, lines=self._sse(self._delta("Hel"), {"choices": [{"delta": {}}]},
                                        self._delta("lo"), "not json", {"id": "x"}))
        with mock.patch("requests.post", return_value=up) as post:
            r = self._send({"messages": [{"role": "user", "content": "hi"}], "options": {"temperature": 1}})
            lines = _ndjson(r)
        self.assertEqual(lines, [{"message": {"content": "Hel"}}, {"message": {"content": "lo"}}, {"done": True}])
        kw = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "http://hermes:8642/v1/chat/completions")
        self.assertEqual(kw["json"], {"model": "rachel-1", "messages": [{"role": "user", "content": "hi"}],
                                      "stream": True, "temperature": 1.0})
        self.assertEqual(kw["headers"], {"Authorization": "Bearer KEY", "X-Hermes-Session-Id": "SID",
                                         "X-Hermes-Session-Key": "SKEY", "Accept": "text/event-stream"})
        self.assertEqual(kw["timeout"], (10, 600))
        self.assertEqual(r.headers["X-Accel-Buffering"], "no")
        self.assertTrue(up.closed)

    def test_temperature_only_forwarded_when_numeric(self):
        for opts in ({"temperature": "hot"}, {}, "x", None):
            with self.subTest(opts=opts):
                with mock.patch("requests.post", return_value=_Resp(200, lines=[])) as post:
                    self._send({"messages": [{"role": "user", "content": "x"}], "options": opts}).get_data()
                self.assertNotIn("temperature", post.call_args.kwargs["json"])

    def test_stream_stops_at_done_marker(self):
        up = _Resp(200, lines=self._sse(self._delta("a")) + ["data: " + json.dumps(self._delta("late"))])
        with mock.patch("requests.post", return_value=up):
            lines = _ndjson(self._send({"messages": [{"role": "user", "content": "x"}]}))
        self.assertEqual(lines, [{"message": {"content": "a"}}, {"done": True}])

    def test_error_chunk_ends_the_stream_with_message(self):
        for err, want in (({"message": "rate limited"}, "rate limited"), ("plain", "plain")):
            with self.subTest(err=err):
                up = _Resp(200, lines=self._sse(self._delta("a"), {"error": err}, self._delta("never")))
                with mock.patch("requests.post", return_value=up):
                    lines = _ndjson(self._send({"messages": [{"role": "user", "content": "x"}]}))
                self.assertEqual(lines, [{"message": {"content": "a"}}, {"error": want, "done": True}])

    def test_mid_stream_exception_yields_error_line(self):
        up = _Resp(200, lines=self._sse(self._delta("a"), done=False), raise_in_stream=OSError("reset"))
        with mock.patch("requests.post", return_value=up):
            lines = _ndjson(self._send({"messages": [{"role": "user", "content": "x"}]}))
        self.assertEqual(lines[-1], {"error": "reset", "done": True})
        self.assertTrue(up.closed)

    def test_upstream_http_errors(self):
        msgs = [{"role": "user", "content": "x"}]
        cases = [
            (_Resp(401, {"error": {"message": "bad key"}}), "HTTP 401: bad key"),
            # a string "error" makes .get("message") raise, so the detail falls back to the raw body text
            (_Resp(500, {"error": "oops"}, text='{"error": "oops"}'), 'HTTP 500: {"error": "oops"}'),
            (_Resp(502, text="gateway down"), "HTTP 502: gateway down"),
        ]
        for up, want in cases:
            with self.subTest(want=want):
                with mock.patch("requests.post", return_value=up):
                    r = self._send({"messages": msgs})
                self.assertEqual(r.status_code, 502)
                self.assertIn(want, r.get_json()["error"])
                self.assertTrue(up.closed)
        with mock.patch("requests.post", side_effect=ConnectionError("down")):
            r = self._send({"messages": msgs})
        self.assertIn("could not reach the Hermes gateway", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
