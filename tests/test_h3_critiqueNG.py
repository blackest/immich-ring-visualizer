"""Unit tests for "Ask Gemma" about a window (h3_critiqueNG.py and its route): the contact sheet, reading
Gemma's reply, what is sent to the server, and the refusals. The Gemma server is faked; ffmpeg is real.

Run with: python3 -m unittest tests.test_h3_critiqueNG -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_critiqueNG as cr
import routes.h3_rerollNG as routes
from tests.test_h3_rerollNG import RerollBase
from tests.test_h3_stitchNG import HAVE_FFMPEG

GOOD = {"anchor_look": "pale shirt", "window_look": "navy shirt",
        "differences": [{"what": "shirt is navy now", "kind": "wardrobe", "seen_in": "last 3", "likely_cause": "no shirt in prompt"},
                        "hair not visible", {"what": "  "}, {"what": "odd", "kind": "weird"}],
        "rewrite": "integrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: x\n\nnon_diegetic_music: y"}


class Parse(unittest.TestCase):
    def test_a_good_reply_is_cleaned_up(self):
        out = cr.parse_reply_ng("Sure! ```json\n" + json.dumps(GOOD) + "\n```")
        self.assertEqual([d["kind"] for d in out["differences"]], ["wardrobe", "other", "other"])
        self.assertEqual(out["differences"][1]["what"], "hair not visible")
        self.assertTrue(out["rewrite"].startswith("integrated_multimodal_description"))
        self.assertIsNone(out["warning"])

    def test_a_rewrite_without_the_prompt_format_is_dropped_with_a_warning(self):
        out = cr.parse_reply_ng(json.dumps({**GOOD, "rewrite": "just some prose"}))
        self.assertIsNone(out["rewrite"])
        self.assertIn("format", out["warning"])

    def test_an_unreadable_reply_is_refused(self):
        for bad in ("", "no json here", "{not json}", "[1, 2]"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "expected form"):
                cr.parse_reply_ng(bad)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Window(RerollBase):
    def setUp(self):
        super().setUp()
        self.sent = {}

        def fake_post(url, json=None, timeout=None):
            self.sent = dict(url=url, body=json)
            return SimpleNamespace(json=lambda: {"choices": [{"message": {"content": __import__("json").dumps(GOOD)}}]})

        def fake_get(url, timeout=None):
            return SimpleNamespace(json=lambda: {"data": [{"id": "whisper-large"}, {"id": "/m/gemma4-26b"}]})

        for target, name, val in ((cr.requests, "post", fake_post), (cr.requests, "get", fake_get),
                                  (cr.h3_jobs, "busy_ng", lambda: None),
                                  (cr.configNG, "get_gemma_review_base_url", lambda: "http://gemma")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)

    def test_the_sheet_has_the_right_number_of_tiles(self):
        from PIL import Image
        import io
        clip = self.job_dir / "h3_100_w2.mp4"
        two = Image.open(io.BytesIO(cr.sheet_ng(clip)))
        full = Image.open(io.BytesIO(cr.sheet_ng(clip, self.job_dir / "h3_100_w1.mp4", self.job_dir / "h3_100_w1.mp4")))
        self.assertEqual(two.width, cr.COLUMNS * cr.TILE_WIDTH)
        self.assertGreater(full.height, 0)
        self.assertEqual(round(full.height / two.height), 1)  # 6 tiles and 9 tiles both fit two rows of 5

    def test_what_is_sent_and_what_comes_back(self):
        out = cr.critique_window_ng("job1", 2, note="the shirt should stay pale")
        body = self.sent["body"]
        self.assertEqual((self.sent["url"], body["model"]), ("http://gemma/v1/chat/completions", "/m/gemma4-26b"))
        text, image = body["messages"][1]["content"]
        self.assertIn("prompt 1", text["text"])
        self.assertIn("prompt 2", text["text"])
        self.assertIn("the shirt should stay pale", text["text"])
        self.assertTrue(image["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual((out["window"], out["note_given"], len(out["differences"])), (2, True, 3))
        self.assertIsInstance(out["motion"], float)

    def test_window_one_has_no_previous_or_anchor_and_no_note_means_none_sent(self):
        out = cr.critique_window_ng("job1", 1)
        self.assertNotIn("The person says", self.sent["body"]["messages"][1]["content"][0]["text"])
        self.assertFalse(out["note_given"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "window must be"):
            cr.critique_window_ng("job1", 9)
        with self.assertRaises(LookupError):
            cr.critique_window_ng("nope", 1)
        with mock.patch.object(cr.h3_jobs, "busy_ng", return_value=object()):
            with self.assertRaises(cr.GpuBusy):
                cr.critique_window_ng("job1", 2)
        with mock.patch.object(cr.requests, "get", side_effect=cr.requests.exceptions.ConnectionError("refused")):
            with self.assertRaisesRegex(cr.GemmaUnavailable, "h3ir_serveNG.sh gemma"):
                cr.critique_window_ng("job1", 2)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Route(Window):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(routes.h3RerollNG_bp)
        self.client = app.test_client()

    def test_statuses(self):
        r = self.client.post("/api/ng/h3/jobs/job1/windows/2/critique", json={"note": "x"})
        self.assertEqual((r.status_code, r.get_json()["window"]), (200, 2))
        self.assertEqual(self.client.post("/api/ng/h3/jobs/job1/windows/9/critique").status_code, 400)
        self.assertEqual(self.client.post("/api/ng/h3/jobs/zz/windows/1/critique").status_code, 404)
        with mock.patch.object(cr.h3_jobs, "busy_ng", return_value=object()):
            self.assertEqual(self.client.post("/api/ng/h3/jobs/job1/windows/2/critique").status_code, 409)
        with mock.patch.object(cr.requests, "get", side_effect=cr.requests.exceptions.ConnectionError("x")):
            self.assertEqual(self.client.post("/api/ng/h3/jobs/job1/windows/2/critique").status_code, 503)


if __name__ == "__main__":
    unittest.main()
