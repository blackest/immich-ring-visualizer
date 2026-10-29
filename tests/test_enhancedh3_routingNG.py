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


if __name__ == "__main__":
    unittest.main()
