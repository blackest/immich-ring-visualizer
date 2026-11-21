"""Unit tests for which H3 compiler an Enhanced H3 brief goes to: Gemma's for text and pictures,
the Qwen one when any attached asset is a video, and a refine back to whichever wrote the brief.
Nothing is contacted: requests.request is mocked.

Run with: python3 -m unittest tests.test_enhancedh3_routingNG -v
"""

import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import routes.enhancedh3NG as routes

GEMMA = "http://127.0.0.1:8700"
QWEN = "http://127.0.0.1:8701"


class _Res:
    def __init__(self, status=201, body=None):
        self.status_code, self._body = status, body or {}
        self.ok = status < 400

    def json(self):
        return self._body


def _client():
    app = Flask(__name__)
    app.register_blueprint(routes.enhancedh3NG_bp)
    return app.test_client()


class CompilerBase(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(routes.configNG, "get_h3ir_base_url", return_value=GEMMA)
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(routes.configNG, "get_h3ir_video_base_url", return_value=QWEN)
        patch.start(); self.addCleanup(patch.stop)

    def test_text_and_pictures_use_the_default_compiler(self):
        self.assertEqual(routes._compiler_base([]), GEMMA)
        self.assertEqual(routes._compiler_base([{"kind": "image", "role": "subject"}]), GEMMA)

    def test_any_video_uses_the_video_compiler(self):
        items = [{"kind": "image", "role": "subject"}, {"kind": "video", "role": "edit_source"}]
        self.assertEqual(routes._compiler_base(items), QWEN)


class BriefRouting(CompilerBase):
    def setUp(self):
        super().setUp()
        routes._brief_base.clear()
        self.calls = []

        def fake(method, url, **kw):
            self.calls.append((method, url))
            if method == "POST":
                return _Res(201, {"id": "b1", "plan": {"shots": []}, "ir": {"prompt": "p", "sections": {}}})
            return _Res(200, {"id": "b1", "plan": {"shots": []}, "ir": {"prompt": "p", "sections": {}}})
        patch = mock.patch.object(routes.requests, "request", side_effect=fake)
        patch.start(); self.addCleanup(patch.stop)

    def _brief(self, assets=None, files=None):
        data = {"intent": "a woman sits", "seconds": "5"}
        if assets:
            data["assets"] = json.dumps(assets)
        for field, name in (files or {}).items():
            data[field] = (io.BytesIO(b"bytes-" + name.encode()), name)
        return _client().post("/api/ng/enhanced-h3/brief", data=data, content_type="multipart/form-data")

    def test_text_only_goes_to_gemma_compiler(self):
        self._brief()
        self.assertEqual([c[1] for c in self.calls], [GEMMA + "/v1/briefs"])

    def test_video_upload_and_brief_both_go_to_the_video_compiler(self):
        self._brief([{"field": "clip", "kind": "video", "role": "edit_source"}], {"clip": "a.mp4"})
        self.assertTrue(self.calls)
        self.assertTrue(all(url.startswith(QWEN) for _, url in self.calls), self.calls)

    def test_refine_returns_to_the_compiler_that_wrote_the_brief(self):
        self._brief([{"field": "clip", "kind": "video", "role": "edit_source"}], {"clip": "a.mp4"})
        self.calls.clear()
        _client().post("/api/ng/enhanced-h3/refine", json={"id": "b1", "change": "darker"})
        self.assertEqual(self.calls, [("PATCH", QWEN + "/v1/briefs/b1")])

    def test_refine_of_an_unknown_brief_uses_the_default_compiler(self):
        _client().post("/api/ng/enhanced-h3/refine", json={"id": "zzz", "change": "darker"})
        self.assertEqual(self.calls, [("PATCH", GEMMA + "/v1/briefs/zzz")])


class SpokenLine(CompilerBase):
    """A spoken line must not vanish quietly: the brief is checked for its <d> tag and asked again once."""

    LINE = [{"text": "Good evening princess", "speaker": "old man", "language": "English"}]

    def setUp(self):
        super().setUp()
        routes._brief_base.clear()
        self.payloads = []
        self.answers = []

        def fake(method, url, **kw):
            body = dict(kw.get("json") or {})   # a copy: the route edits its payload after sending it
            self.payloads.append(body)
            prompt = self.answers.pop(0) if self.answers else "p"
            status = 500 if prompt == "CRASH" else 201
            return _Res(status, {"id": "b1", "plan": {"shots": []}, "ir": {"prompt": prompt, "sections": {}}})
        patch = mock.patch.object(routes.requests, "request", side_effect=fake)
        patch.start(); self.addCleanup(patch.stop)

    def _brief(self, dialogue=True, silent=False):
        data = {"intent": "a fairy and an old man", "seconds": "5"}
        if dialogue:
            data["dialogue"] = json.dumps(self.LINE)
        if silent:
            data["silent"] = "true"
        r = _client().post("/api/ng/enhanced-h3/brief", data=data, content_type="multipart/form-data")
        return r.status_code, r.get_json()

    def test_a_brief_that_already_speaks_is_left_alone(self):
        self.answers = ["x <d>[English] Good evening</d>"]
        status, body = self._brief()
        self.assertEqual(status, 201)
        self.assertEqual(len(self.payloads), 1)
        self.assertNotIn("note", body)

    def test_a_line_the_compiler_dropped_is_written_into_the_sentence_and_asked_again(self):
        self.answers = ["reference layout, room tone", "x <d>[English] Good evening princess</d>"]
        status, body = self._brief()
        self.assertEqual(status, 201)
        self.assertEqual(len(self.payloads), 2)
        self.assertIn("dialogue", self.payloads[0])
        self.assertNotIn("dialogue", self.payloads[1])
        self.assertIn('Old man says: "Good evening princess".', self.payloads[1]["intent"])
        self.assertIn("<d>", body["prompt"])
        self.assertIn("written into the sentence", body["note"])

    def test_when_the_second_try_also_drops_it_the_note_says_so_and_the_first_brief_stays(self):
        self.answers = ["first", "second, still nothing said"]
        status, body = self._brief()
        self.assertEqual(status, 201)
        self.assertEqual(body["prompt"], "first")
        self.assertIn("no spoken line", body["note"])
        self.assertIn("<d>", body["note"])

    def test_no_dialogue_asked_or_a_silent_clip_makes_no_second_call(self):
        self._brief(dialogue=False)
        self._brief(silent=True)
        self.assertEqual(len(self.payloads), 2)

    def test_the_older_fallback_for_a_crash_still_writes_the_line_and_is_not_asked_a_third_time(self):
        self.answers = ["CRASH", "x <d>[English] Good evening princess</d>"]
        status, body = self._brief()
        self.assertEqual((status, len(self.payloads)), (201, 2))
        self.assertIn("can't take a spoken line", body["note"])


class ErrorText(unittest.TestCase):
    def test_a_nested_detail_shows_its_message_not_a_dict(self):
        text = routes._error_text(_Res(422, {"detail": {"code": "replacement-target-unnamed",
                                                       "message": "Say who it replaces."}}))
        self.assertEqual(text, "Say who it replaces.")

    def test_flat_forms_still_work(self):
        self.assertEqual(routes._error_text(_Res(400, {"detail": "plain"})), "plain")
        self.assertEqual(routes._error_text(_Res(400, {"message": "m", "detail": "d"})), "m")


if __name__ == "__main__":
    unittest.main()
