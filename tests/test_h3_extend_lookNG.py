"""Extending a film: its established cast and look are read from the film's own window prompts and carried into the new
windows (h3_extend_lookNG.py), and the brief route uses them. Gemma and the compiler are stubbed.

Run with: venv/bin/python3 -m unittest tests.test_h3_extend_lookNG -v
"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_continueNG as cont
import h3_extend_lookNG as ext
import routes.enhancedh3NG as routes

TAIL = "\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: Cello."
EST = {"cast": "An old man, bald with a white beard; a fairy with blonde hair and pointed ears",
       "look": "Cinematic digital painting at night, a warm golden glow"}


def win(text):
    return {"prompt": f"integrated_multimodal_description: [Shot 1] {text}{TAIL}", "shots": [1], "continues": False}


class Previous(unittest.TestCase):
    def test_the_film_s_window_prompts_come_from_its_manifest_newest_last(self):
        man = {"windows": [{"prompt": f"w{i}"} for i in range(1, 7)]}
        with mock.patch.object(ext.reroll, "resolve_source_ng", return_value=SimpleNamespace(prompt="x")), \
                mock.patch.object(ext.reroll, "load_manifest_ng", return_value=man):
            self.assertEqual(ext.previous_prompts_ng("j"), ["w3", "w4", "w5", "w6"])

    def test_a_single_window_render_falls_back_to_its_own_prompt(self):
        with mock.patch.object(ext.reroll, "resolve_source_ng", return_value=SimpleNamespace(prompt="the only prompt")), \
                mock.patch.object(ext.reroll, "load_manifest_ng", side_effect=ValueError("single window")):
            self.assertEqual(ext.previous_prompts_ng("j"), ["the only prompt"])

    def test_a_film_with_no_saved_prompt_is_refused(self):
        with mock.patch.object(ext.reroll, "resolve_source_ng", return_value=SimpleNamespace(prompt="")), \
                mock.patch.object(ext.reroll, "load_manifest_ng", side_effect=ValueError("single window")):
            with self.assertRaises(ValueError):
                ext.previous_prompts_ng("j")

    def test_an_unknown_film_is_a_lookup_error(self):
        with mock.patch.object(ext.reroll, "resolve_source_ng", side_effect=LookupError("no such job")):
            with self.assertRaises(LookupError):
                ext.previous_prompts_ng("nope")


class Established(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(ext, "previous_prompts_ng", return_value=[win("A man and a fairy")["prompt"]])
        p.start()
        self.addCleanup(p.stop)

    def test_the_cast_and_look_are_read_from_gemma_s_reply_with_no_window_to_write(self):
        seen = {}

        def ask(text):
            seen["text"] = text
            return json.dumps({"cast": EST["cast"], "look": EST["look"], "windows": []})
        self.assertEqual(ext.established_ng("j", "a new ending", ask=ask), EST)
        self.assertIn("No window is empty", seen["text"])
        self.assertIn("A man and a fairy", seen["text"])

    def test_a_reply_with_neither_is_an_error(self):
        with self.assertRaises(cont.ContinueError):
            ext.established_ng("j", ask=lambda _t: json.dumps({"cast": "", "look": "", "windows": [
                {"window": 1, "beat": "x", "soundscape": "", "dialogue": ""}]}))

    def test_gemma_down_is_a_continue_error(self):
        def down(_t):
            raise cont.ContinueError("refused")
        with self.assertRaises(cont.ContinueError):
            ext.established_ng("j", ask=down)


class Lines(unittest.TestCase):
    def test_the_intent_states_style_and_cast_up_front(self):
        out = ext.intent_with_ng("The camera pulls back.", EST)
        self.assertTrue(out.startswith("Style: Cinematic digital painting at night, a warm golden glow. The same people"))
        self.assertTrue(out.endswith("The camera pulls back."))
        self.assertEqual(ext.intent_with_ng("x", {"cast": "", "look": ""}), "x")

    def test_every_new_window_gets_both_lines_including_the_first(self):
        out = ext.add_lines_ng([win("The camera pulls back."), win("The door opens.")], EST)
        for w in out:
            self.assertIn("[Shot 1] An old man, bald with a white beard; a fairy with blonde hair and pointed ears. "
                          "Cinematic digital painting at night, a warm golden glow. ", w["prompt"])
            self.assertIn("non_diegetic_music: Cello.", w["prompt"])           # the sound sections are kept
        self.assertIn("The camera pulls back.", out[0]["prompt"])

    def test_a_line_already_in_a_window_is_not_added_again(self):
        w = win("An old man, bald with a white beard; a fairy with blonde hair and pointed ears. The door opens.")
        out = ext.add_lines_ng([w], EST)
        self.assertEqual(out[0]["prompt"].count("An old man, bald with a white beard"), 1)

    def test_lines_without_a_full_stop_get_one_and_do_not_run_together(self):
        out = cont._with_lines(win("The camera cuts.")["prompt"], ["the old man in a robe", "Night, warm glow"])
        self.assertIn("[Shot 1] the old man in a robe. Night, warm glow. The camera cuts.", out)


class BriefRoute(unittest.TestCase):
    """The route reads the established look, states it in the intent, and adds it to every window."""

    def setUp(self):
        routes._brief_base.clear()
        self.payloads = []

        def fake(method, url, **kw):
            self.payloads.append(dict(kw.get("json") or {}))
            body = {"id": "b1", "plan": {"shots": []}, "ir": {"prompt": "p", "sections": {}}}
            return SimpleNamespace(status_code=201, ok=True, json=lambda: body)
        for target, name, val in ((routes.requests, "request", fake),
                                  (routes.configNG, "get_h3ir_base_url", lambda: "http://c"),
                                  (routes.configNG, "get_h3ir_video_base_url", lambda: "http://v"),
                                  (routes, "_shape", lambda brief, seconds: {"id": "b1", "windows": [win("a"), win("b")]})):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        app = Flask(__name__)
        app.register_blueprint(routes.enhancedh3NG_bp)
        self.client = app.test_client()

    def post(self, **extra):
        data = {"intent": "The door opens.", "seconds": "10"}
        data.update(extra)
        r = self.client.post("/api/ng/enhanced-h3/brief", data=data, content_type="multipart/form-data")
        return r.status_code, r.get_json()

    def test_an_extension_states_the_look_and_adds_it_to_every_window(self):
        with mock.patch.object(routes.h3_extend_look, "established_ng", return_value=EST) as est:
            status, body = self.post(extend_job="film1")
        est.assert_called_once()
        self.assertEqual(status, 201)
        self.assertIn("Style: Cinematic digital painting at night", self.payloads[0]["intent"])
        self.assertTrue(self.payloads[0]["intent"].endswith("The door opens."))
        for w in body["windows"]:
            self.assertIn("An old man, bald with a white beard", w["prompt"])

    def test_without_extend_job_nothing_changes(self):
        with mock.patch.object(routes.h3_extend_look, "established_ng") as est:
            status, body = self.post()
        est.assert_not_called()
        self.assertEqual((status, self.payloads[0]["intent"]), (201, "The door opens."))
        self.assertNotIn("An old man", body["windows"][0]["prompt"])

    def test_an_unreadable_earlier_clip_leaves_the_brief_alone_and_says_so(self):
        for exc in (cont.ContinueError("gemma down"), LookupError("no such job"), ValueError("no prompt")):
            with self.subTest(exc=exc), mock.patch.object(routes.h3_extend_look, "established_ng", side_effect=exc):
                status, body = self.post(extend_job="film1")
            self.assertEqual(status, 201)
            self.assertIn("could not be read", body["note"])
            self.assertNotIn("An old man", body["windows"][0]["prompt"])


if __name__ == "__main__":
    unittest.main()
