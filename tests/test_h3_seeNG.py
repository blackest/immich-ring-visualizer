"""Unit tests for h3_seeNG.py (Gemma looks at the last frame for the H3 writer) and its use by the write-window route. Gemma and ffmpeg are stubbed."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import h3_seeNG as see
import routes.enhancedh3NG as route

DESC = "An old man with a white beard in a grey robe stands in a dark forest at night, lit by a warm golden glow. Painted style."


class Pure(unittest.TestCase):
    def test_description_is_one_clean_paragraph(self):
        self.assertEqual(see.clean_ng("**An old man**\n\n# stands   here in the dark forest at night."),
                         "An old man stands here in the dark forest at night.")
        self.assertEqual(len(see.clean_ng("x" * 2000)), 700)

    def test_an_empty_description_is_refused(self):
        with self.assertRaises(see.SeeError):
            see.clean_ng("ok")

    def test_seen_asks_gemma_with_the_frame(self):
        asked = []
        out = see.seen_ng("clip.mp4", ask=lambda jpeg: asked.append(jpeg) or DESC, frame=lambda p: b"JPEG")
        self.assertEqual((out, asked), (DESC, [b"JPEG"]))

    def test_the_sentence_states_what_was_seen_first(self):
        self.assertTrue(see.intent_with_seen_ng("He laughs.", "A man.").startswith("The last frame of the earlier clip shows: A man. "))
        self.assertEqual(see.intent_with_seen_ng("He laughs.", None), "He laughs.")


class Route(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(route.enhancedh3NG_bp)
        self.client = app.test_client()
        self.win = "integrated_multimodal_description: [Shot 1] x\n\noverall_soundscape: y\n\nnon_diegetic_music: z"

    def post(self, **extra):
        data = {"intent": "He laughs.", "extend_job": "j1", **extra}
        with mock.patch.object(route.h3_extend_look, "established_ng", return_value={"cast": "A man.", "look": "Painted."}), \
                mock.patch.object(route.h3_reroll, "resolve_source_ng", return_value=mock.Mock(mp4_path="/x.mp4")), \
                mock.patch.object(route.h3_writer, "write_window_ng", return_value=self.win) as m:
            with mock.patch.object(route.h3_see, "seen_ng", return_value=DESC):
                r = self.client.post("/api/ng/enhanced-h3/write-window", data=data)
        return r.get_json(), m

    def test_with_look_the_writer_is_told_what_the_frame_shows(self):
        _, m = self.post(look="true")
        self.assertIn("The last frame of the earlier clip shows: An old man", m.call_args.args[0])

    def test_without_look_gemma_is_not_asked(self):
        _, m = self.post()
        self.assertNotIn("last frame", m.call_args.args[0])

    def test_a_gemma_failure_is_a_note_not_an_error(self):
        with mock.patch.object(route.h3_extend_look, "established_ng", return_value={"cast": "A man.", "look": "Painted."}), \
                mock.patch.object(route.h3_reroll, "resolve_source_ng", return_value=mock.Mock(mp4_path="/x.mp4")), \
                mock.patch.object(route.h3_see, "seen_ng", side_effect=see.SeeError("down")), \
                mock.patch.object(route.h3_writer, "write_window_ng", return_value=self.win):
            r = self.client.post("/api/ng/enhanced-h3/write-window", data={"intent": "x", "extend_job": "j1", "look": "true"})
        self.assertEqual(r.status_code, 201)
        self.assertIn("could not look", r.get_json()["note"])


if __name__ == "__main__":
    unittest.main()
