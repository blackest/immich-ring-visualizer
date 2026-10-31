"""Unit tests for h3_continueNG: windows a brief leaves empty are written, not copied. Gemma is stubbed.

Run with: python3 -m unittest tests.test_h3_continueNG -v
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_continueNG as cont

SPEECH = "says in a low, gravelly voice (S1): <d>[English] The kitten deserves a chance.</d>"
TAIL = "\n\noverall_soundscape: Wind on stone.\n\nnon_diegetic_music: One cello note."


def _w(text, continues=False):
    return {"prompt": f"integrated_multimodal_description: [Shot 1] {text}{TAIL}", "shots": [1], "continues": continues}


def _windows(speech=True):
    shot1 = "A bald man with a grey beard in a brown tunic leans on a wall" + (f" and {SPEECH}" if speech else ".")
    return [_w(shot1), _w(shot1, True), _w("Extreme close-up of his eyes behind glasses."), _w("Extreme close-up of his eyes behind glasses.", True)]


def _reply(**over):
    wins = [{"window": 2, "beat": "He turns toward the stairs.", "soundscape": "Boots on gravel.", "dialogue": "Then I will go."},
            {"window": 4, "beat": "He lowers his gaze.", "soundscape": "Wind dies.", "dialogue": "Stay."}]
    return json.dumps({"cast": "Aldric, a bald man with a grey beard in a brown tunic.", "windows": wins, **over})


class Parsing(unittest.TestCase):
    def test_speech_voice_and_language_are_read(self):
        self.assertEqual(cont.speech_in_ng(SPEECH), ("a low, gravelly voice (S1)", "English"))
        self.assertIsNone(cont.speech_in_ng("no speech here"))

    def test_json_inside_prose_is_found(self):
        out = cont.parse_reply_ng("Sure:\n```json\n" + _reply() + "\n```")
        self.assertEqual(sorted(out["windows"]), [2, 4])

    def test_no_json_or_no_beats_is_an_error(self):
        for bad in ("nothing", "{}", '{"windows": [{"window": 2, "beat": ""}]}', "{not json}"):
            with self.assertRaises(cont.ContinueError):
                cont.parse_reply_ng(bad)

    def test_request_marks_empty_windows_and_speech_rule(self):
        text = cont.build_request_ng(_windows(), "a man on a wall", True)
        self.assertIn("Window 2: EMPTY", text)
        self.assertIn("window(s): 2, 4", text)
        self.assertIn("Speech is allowed", text)
        self.assertIn("NOT allowed", cont.build_request_ng(_windows(), "x", False))


class Applying(unittest.TestCase):
    def run_it(self, windows=None, improvise=True):
        return cont.continue_windows_ng(windows or _windows(), "a man", improvise, ask=lambda _t: _reply())

    def test_carry_on_windows_are_rewritten_and_not_copies(self):
        out, note = self.run_it()
        self.assertIsNone(note)
        self.assertNotEqual(out[1]["prompt"], out[0]["prompt"])
        self.assertTrue(out[1]["generated"] and out[3]["generated"])
        self.assertIn("He turns toward the stairs.", out[1]["prompt"])
        self.assertNotIn("kitten", out[1]["prompt"])

    def test_first_window_is_untouched_and_later_ones_get_the_cast(self):
        out, _ = self.run_it()
        self.assertEqual(out[0], _windows()[0])
        for w in out[1:]:
            self.assertIn("Aldric, a bald man", w["prompt"])
        self.assertEqual(out[2]["prompt"].count("Aldric"), 1)

    def test_new_line_uses_the_same_voice_and_language(self):
        out, _ = self.run_it()
        self.assertIn("says in a low, gravelly voice (S1): <d>[English] Then I will go.</d>", out[1]["prompt"])

    def test_gemmas_own_language_tag_and_quotes_are_not_doubled(self):
        reply = _reply()
        reply = reply.replace("Then I will go.", "[English] \\\"Then I will go.\\\"")
        out, _ = cont.continue_windows_ng(_windows(), "a man", True, ask=lambda _t: reply)
        self.assertIn("<d>[English] Then I will go.</d>", out[1]["prompt"])

    def test_improvise_off_writes_no_dialogue(self):
        out, _ = self.run_it(improvise=False)
        self.assertNotIn("<d>", out[1]["prompt"])

    def test_silent_brief_gets_no_dialogue_even_if_gemma_wrote_some(self):
        out, _ = self.run_it(_windows(speech=False))
        self.assertNotIn("<d>", out[1]["prompt"] + out[3]["prompt"])

    def test_soundscape_is_varied_and_music_kept(self):
        p = self.run_it()[0][1]["prompt"]
        self.assertIn("overall_soundscape: Boots on gravel.", p)
        self.assertIn("non_diegetic_music: One cello note.", p)

    def test_nothing_to_write_returns_the_windows_as_is(self):
        wins = [_w("a"), _w("b")]
        self.assertEqual(cont.continue_windows_ng(wins, "x", ask=lambda _t: 1 / 0), (wins, None))
        single = [_w("a", True)]
        self.assertEqual(cont.continue_windows_ng(single, "x", ask=lambda _t: 1 / 0), (single, None))

    def test_gemma_failure_keeps_the_windows_and_says_so(self):
        def down(_t):
            raise cont.ContinueError("connection refused")
        wins = _windows()
        out, note = cont.continue_windows_ng(wins, "x", ask=down)
        self.assertEqual(out, wins)
        self.assertIn("connection refused", note)


if __name__ == "__main__":
    unittest.main()
