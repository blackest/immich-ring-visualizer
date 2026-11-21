"""Unit tests for the H3 prompt writer option (h3_writerNG.py and its route). The model is never run."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_writerNG as w
import routes.enhancedh3NG as route

WINDOW = "integrated_multimodal_description: [Shot 1] A fairy (S1) says: <d>[English] Hi</d>\n\noverall_soundscape: x\n\nnon_diegetic_music: y"


class Pure(unittest.TestCase):
    def test_a_leading_reference_paragraph_is_dropped(self):
        self.assertEqual(w.clean_reply_ng("How the reference pictures align...\n\n" + WINDOW), WINDOW)

    def test_a_reply_without_a_description_is_refused(self):
        with self.assertRaises(w.WriterError):
            w.clean_reply_ng("nothing useful")

    def test_speaker_tags_follow_the_pane(self):
        self.assertIn("(S2)", w.with_speaker_tags_ng(WINDOW, "s2"))
        self.assertEqual(w.with_speaker_tags_ng(WINDOW, ""), WINDOW)
        self.assertEqual(w.with_speaker_tags_ng(WINDOW, "the woman"), WINDOW)

    def test_the_request_is_one_shot_text_to_video(self):
        r = w.request_ng(" a fairy ", 5.1)
        self.assertEqual((r["prompt"], r["duration"], r["one_shot"], r["task"]), ("a fairy", 5, True, "T2VA"))


class Write(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "phosphene_rewriter.json").write_text("{}")
        for target, name, val in ((w, "PYTHON", Path(__file__)), (w, "WORKER", Path(__file__))):
            p = mock.patch.object(target, name, val); p.start(); self.addCleanup(p.stop)
        p = mock.patch.dict(os.environ, {"RINGVIZ_H3_WRITER_MODEL": str(self.tmp)}); p.start(); self.addCleanup(p.stop)

    def run_with(self, reply, err=""):
        out = SimpleNamespace(stdout=("noise\n" + w.SENTINEL + json.dumps(reply) + "\n") if reply is not None else "", stderr=err)
        return lambda *a, **k: out

    def test_a_good_reply_comes_back_as_one_window(self):
        text = w.write_window_ng("a fairy", 5.1, "S2", runner=self.run_with({"rewrite": "preamble\n" + WINDOW}))
        self.assertTrue(text.startswith("integrated_multimodal_description:"))
        self.assertIn("(S2)", text)

    def test_errors_are_plain(self):
        for reply, text in (({"error": "boom"}, "boom"), (None, "returned nothing")):
            with self.subTest(text=text), self.assertRaisesRegex(w.WriterError, text):
                w.write_window_ng("a fairy", runner=self.run_with(reply))
        with self.assertRaisesRegex(w.WriterError, "describe"):
            w.write_window_ng("  ")

    def test_a_missing_model_says_how_to_get_it(self):
        with mock.patch.dict(os.environ, {"RINGVIZ_H3_WRITER_MODEL": str(self.tmp / "none")}), \
                self.assertRaisesRegex(w.WriterError, "Write it for me"):
            w.write_window_ng("a fairy")


class Route(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(route.enhancedh3NG_bp)
        self.client = app.test_client()

    def test_shaped_like_a_brief_and_seconds_capped(self):
        with mock.patch.object(route.h3_writer, "write_window_ng", return_value=WINDOW) as m:
            r = self.client.post("/api/ng/enhanced-h3/write-window", data={"intent": "x", "seconds": "10", "speaker": "S2"})
        b = r.get_json()
        self.assertEqual((r.status_code, b["windows"][0]["prompt"], b["target"]["effective_seconds"]), (201, WINDOW, 5.1))
        self.assertIn("5.1", b["note"])
        self.assertEqual(m.call_args.args[2], "S2")

    def test_extension_adds_cast_and_look(self):
        with mock.patch.object(route.h3_writer, "write_window_ng", return_value=WINDOW) as m, \
                mock.patch.object(route.h3_extend_look, "established_ng", return_value={"cast": "An old man.", "look": "Painterly."}):
            b = self.client.post("/api/ng/enhanced-h3/write-window", data={"intent": "x", "extend_job": "j1"}).get_json()
        self.assertIn("An old man.", b["windows"][0]["prompt"])
        self.assertIn("Style: Painterly.", m.call_args.args[0])          # the writer is told the look first
        self.assertIn("An old man", m.call_args.args[0])
        self.assertIn("Painterly.", b["windows"][0]["prompt"])

    def test_empty_intent_and_writer_failure(self):
        self.assertEqual(self.client.post("/api/ng/enhanced-h3/write-window", data={}).status_code, 400)
        with mock.patch.object(route.h3_writer, "write_window_ng", side_effect=w.WriterError("not set up")):
            r = self.client.post("/api/ng/enhanced-h3/write-window", data={"intent": "x"})
        self.assertEqual((r.status_code, r.get_json()["error"]), (502, "not set up"))


if __name__ == "__main__":
    unittest.main()
