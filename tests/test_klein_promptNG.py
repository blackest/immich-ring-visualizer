"""Klein prompt improver: input checks, tag protection, Ollama failures, and the route.

Run with: python3 -m unittest tests.test_klein_promptNG -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import klein_promptNG as kp


def reply(text, status=200):
    r = mock.Mock(status_code=status)
    r.json.return_value = {"message": {"content": text}}
    return lambda *a, **k: r


class Improve(unittest.TestCase):
    def test_returns_the_caption_and_sends_the_system_prompt(self):
        seen = {}
        def post(url, json=None, timeout=None):
            seen.update(json); return reply("A red vase on a table.")()
        out = kp.improve_prompt_ng("red vase", "gemma", post=post)
        self.assertEqual(out, "A red vase on a table.")
        self.assertEqual(seen["messages"][0]["content"], kp.DEFAULT_SYSTEM)
        self.assertFalse(seen["stream"])

    def test_custom_system_prompt_and_quotes_stripped(self):
        out = kp.improve_prompt_ng("x", "m", system="be brief", post=reply('"Short."'))
        self.assertEqual(out, "Short.")

    def test_blank_prompt_or_model_refused(self):
        with self.assertRaises(ValueError): kp.improve_prompt_ng("  ", "m")
        with self.assertRaises(ValueError): kp.improve_prompt_ng("x", "")

    def test_a_rewrite_that_drops_a_tag_is_refused(self):
        with self.assertRaises(RuntimeError):
            kp.improve_prompt_ng("a vase <vase_1> here", "m", post=reply("A vase here."))
        self.assertIn("<vase_1>", kp.improve_prompt_ng("a vase <vase_1>", "m", post=reply("A tall vase <vase_1>.")))

    def test_ollama_errors(self):
        with self.assertRaises(RuntimeError): kp.improve_prompt_ng("x", "m", post=reply("", 500))
        with self.assertRaises(RuntimeError): kp.improve_prompt_ng("x", "m", post=reply("   "))


class Route(unittest.TestCase):
    def setUp(self):
        from flask import Flask
        from routes.kleinNG import kleinNG_bp
        app = Flask(__name__); app.register_blueprint(kleinNG_bp); self.c = app.test_client()

    def test_info_and_improve(self):
        self.assertEqual(self.c.get("/api/ng/klein/improve").get_json()["system"], kp.DEFAULT_SYSTEM)
        with mock.patch.object(kp, "improve_prompt_ng", return_value="Done."):
            r = self.c.post("/api/ng/klein/improve", json={"prompt": "p", "model": "m"})
        self.assertEqual(r.get_json(), {"prompt": "Done."})

    def test_errors_map_to_status(self):
        self.assertEqual(self.c.post("/api/ng/klein/improve", json={}).status_code, 400)
        with mock.patch.object(kp, "improve_prompt_ng", side_effect=RuntimeError("down")):
            self.assertEqual(self.c.post("/api/ng/klein/improve", json={"prompt": "p", "model": "m"}).status_code, 502)


if __name__ == "__main__":
    unittest.main()
