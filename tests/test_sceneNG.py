"""Unit tests for the Scene feature's pure parts so far: the script format (speakers, settings, limits) and cutting one
voice take into per-line audio at its pauses. Synthetic audio only; nothing renders and no model is read.

Run with: python3 -m unittest tests.test_sceneNG -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scene_composeNG as comp
import scene_scriptNG as sc
import scene_voiceNG as sv


class ScriptTests(unittest.TestCase):
    def test_speakers_lines_and_wrapped_continuations(self):
        L = sc.parse_script_ng("KNIGHT: One, two.\nWOMAN: Three.\nknight: Four\n  five six.\n")
        self.assertEqual([(l.speaker, l.text) for l in L],
                         [("KNIGHT", "One, two."), ("WOMAN", "Three."), ("KNIGHT", "Four five six.")])
        self.assertEqual(sc.speakers_ng(L), ["KNIGHT", "WOMAN"])            # the first spelling is kept
        self.assertEqual(len(sc.lines_of_ng(L, "knight")), 2)

    def test_settings_are_parsed_and_untagged_lines_keep_their_speakers_last_setting(self):
        L = sc.parse_script_ng("KNIGHT [on the battlements]: A.\nWOMAN [in the great hall]: B.\nKNIGHT: C.\nWOMAN: D.")
        self.assertEqual([l.setting for l in L], ["on the battlements", "in the great hall", "", ""])
        R = sc.resolve_settings_ng(L)
        self.assertEqual([l.setting for l in R], ["on the battlements", "in the great hall"] * 2)
        same = sc.parse_script_ng("A [hall]: x.\nA: y.\nA [yard]: z.\nA: w.")
        self.assertEqual(sc.moves_ng(same), [2])                             # only the real change of place

    def test_unknown_speaker_overlong_line_empty_and_too_many_lines_are_refused_in_plain_words(self):
        with self.assertRaisesRegex(sc.SceneScriptError, "no face and voice"):
            sc.parse_script_ng("GHOST: Boo.", known=["Knight"])
        with self.assertRaisesRegex(sc.SceneScriptError, "Split it into two"):
            sc.parse_script_ng("A: " + "word " * 25)
        with self.assertRaisesRegex(sc.SceneScriptError, "start with a speaker"):
            sc.parse_script_ng("no speaker here")
        with self.assertRaises(sc.SceneScriptError):
            sc.parse_script_ng("   \n \n")
        with self.assertRaisesRegex(sc.SceneScriptError, "at most"):
            sc.parse_script_ng("\n".join(f"A: line {i}." for i in range(sc.MAX_LINES + 1)))

    def test_sentences_and_the_single_paragraph_a_speaker_is_voiced_in(self):
        self.assertEqual(sc.sentences_ng("Today, I think, I have enough. Come in out of the wind."),
                         ["Today, I think, I have enough.", "Come in out of the wind."])
        self.assertEqual(sc.sentences_ng("no full stop"), ["no full stop."])
        L = sc.parse_script_ng("A: One. Two.\nB: Other.\nA: Three")
        text, counts = sc.speakable_text_ng(sc.lines_of_ng(L, "a"))
        self.assertEqual((text, counts), ("One. Two. Three.", [2, 1]))

    def test_a_line_starting_with_an_ampersand_shares_the_previous_shot(self):
        L = sc.parse_script_ng("A: Where are you?\n& B [kitchen]: Right here.\nA: Good.")
        self.assertEqual([l.joined for l in L], [False, True, False])
        self.assertEqual([[l.speaker for l in shot] for shot in sc.shots_ng(L)], [["A", "B"], ["A"]])
        self.assertEqual(L[1].setting, "kitchen")

    def test_shared_frame_rules_are_refused_in_plain_words(self):
        for bad, why in (("& A: hi.", "none yet"), ("A: hi.\n& A: again.", "two different speakers"),
                         ("A: hi.\n& B: yo.\n& C: hey.", "two speakers at most"),
                         ("A: one two three four five six seven eight.\n& B: nine ten eleven twelve thirteen fourteen fifteen sixteen.",
                          "share one 5 second window"), ("A: hi.\n& no colon here", "write  NAME")):
            with self.subTest(why=why), self.assertRaisesRegex(sc.SceneScriptError, why):
                sc.parse_script_ng(bad)

    def test_the_known_check_is_case_insensitive(self):
        self.assertEqual(len(sc.parse_script_ng("knight: Hi.", known=["KNIGHT"])), 1)


SR = 16000


def speech(spec, sr=SR, seed=3):
    """spec = [(kind, seconds, level)] with kind 'v' (voiced noise-tone) or 's' (silence); returns float32."""
    rng = np.random.default_rng(seed)
    parts = []
    for kind, secs, level in spec:
        n = int(secs * sr)
        if kind == "s":
            parts.append(np.zeros(n, dtype=np.float32))
        else:
            t = np.arange(n) / sr
            parts.append((level * (np.sin(2 * np.pi * 120 * t) + 0.3 * rng.standard_normal(n))).astype(np.float32))
    return np.concatenate(parts)


class ComposeTests(unittest.TestCase):
    def test_two_pictures_side_by_side_each_centre_cropped_with_a_seam(self):
        import io
        from PIL import Image

        def pic(color, size):
            b = io.BytesIO()
            Image.new("RGB", size, color).save(b, "PNG")
            return b.getvalue()
        out = comp.split_screen_ng(pic((250, 0, 0), (400, 200)), pic((0, 0, 250), (100, 300)), 640, 384)
        im = Image.open(io.BytesIO(out)).convert("RGB")
        self.assertEqual(im.size, (640, 384))
        self.assertGreater(im.getpixel((100, 100))[0], 200)
        self.assertGreater(im.getpixel((540, 100))[2], 200)
        self.assertEqual(im.getpixel((320, 100)), (0, 0, 0))              # the seam
        for bad in ((b"junk", pic((1, 1, 1), (50, 50)), 640, 384), (pic((1, 1, 1), (50, 50)),) * 2 + (63, 384),
                    (pic((1, 1, 1), (50, 50)),) * 2 + (641, 384)):
            with self.assertRaises(comp.ComposeError):
                comp.split_screen_ng(*bad)


class VoiceTests(unittest.TestCase):
    # three "sentences" with short comma gaps inside and real sentence gaps between
    TAKE = [("s", 0.1, 0), ("v", 0.8, .2), ("s", 0.06, 0), ("v", 0.9, .2), ("s", 0.50, 0),
            ("v", 1.5, .2), ("s", 0.06, 0), ("v", 1.0, .2), ("s", 0.35, 0),
            ("v", 1.2, .2), ("s", 0.1, 0)]

    def test_the_cuts_are_the_longest_gaps_not_the_commas(self):
        x = speech(self.TAKE)
        cuts = sv.cut_points_ng(x, SR, 3)
        self.assertEqual(len(cuts), 2)
        self.assertAlmostEqual(cuts[0], 0.1 + 0.8 + 0.06 + 0.9 + 0.25, delta=0.04)
        self.assertAlmostEqual(cuts[1], 0.1 + 0.8 + 0.06 + 0.9 + 0.5 + 1.5 + 0.06 + 1.0 + 0.175, delta=0.04)

    def test_a_pause_with_a_breathy_hiss_is_found_by_the_looser_levels(self):
        # a clone's pauses are not silent: here the sentence gap carries a hiss at ~10% of the voice level (real takes: 5-15%)
        rng = np.random.default_rng(9)
        hiss = (0.016 * rng.standard_normal(int(0.5 * SR))).astype(np.float32)
        x = np.concatenate([speech([("s", .1, 0), ("v", 1.2, .2)]), hiss, speech([("v", 1.2, .2), ("s", .1, 0)])])
        self.assertEqual(sv.find_gaps_ng(x, SR), [])                       # the strict level sees no gap at all
        (cut,) = sv.cut_points_ng(x, SR, 2)
        self.assertAlmostEqual(cut, 0.1 + 1.2 + 0.25, delta=0.06)
        pieces = sv.split_take_ng(x, SR, 2)
        self.assertEqual(len(pieces), 2)

    def test_split_gives_one_trimmed_piece_per_line_that_keeps_every_word(self):
        x = speech(self.TAKE)
        pieces = sv.split_take_ng(x, SR, 3)
        self.assertEqual(len(pieces), 3)
        secs = [len(p) / SR for p in pieces]
        for got, want in zip(secs, (1.76, 2.56, 1.2)):                         # voiced time plus a little breathing room
            self.assertAlmostEqual(got, want + 0.24, delta=0.08)
        self.assertAlmostEqual(sum(float((p ** 2).sum()) for p in pieces), float((x ** 2).sum()), delta=float((x ** 2).sum()) * 0.01)

    def test_a_line_with_two_sentences_keeps_its_inner_pause_and_cuts_only_between_lines(self):
        # line 1 = two sentences (0.45 s pause inside), line 2 = one sentence; the gap between the lines is the SHORTEST
        x = speech([("s", .1, 0), ("v", 1.0, .2), ("s", .45, 0), ("v", 1.0, .2), ("s", .30, 0), ("v", 1.2, .2), ("s", .1, 0)])
        wrong = sv.split_take_ng(x, SR, 2)                          # the plain rule cuts at the longest gap: inside line 1
        right = sv.split_take_ng(x, SR, 2, group_sizes=[2, 1])
        self.assertAlmostEqual(len(wrong[0]) / SR, 1.0 + 0.24, delta=0.08)
        self.assertAlmostEqual(len(right[0]) / SR, 1.0 + 0.45 + 1.0 + 0.24, delta=0.08)
        self.assertAlmostEqual(len(right[1]) / SR, 1.2 + 0.24, delta=0.08)
        with self.assertRaises(sv.SceneAudioError):
            sv.split_take_ng(x, SR, 2, group_sizes=[2])

    def test_one_line_is_the_whole_take_trimmed(self):
        x = speech([("s", 0.4, 0), ("v", 2.0, .2), ("s", 0.4, 0)])
        (p,) = sv.split_take_ng(x, SR, 1)
        self.assertAlmostEqual(len(p) / SR, 2.0 + 0.24, delta=0.08)

    def test_too_few_pauses_and_a_piece_over_the_window_are_refused_with_the_fix(self):
        run_together = speech([("s", .1, 0), ("v", 1, .2), ("s", .06, 0), ("v", 1, .2), ("s", .1, 0)])
        with self.assertRaisesRegex(sv.SceneAudioError, "full stop"):
            sv.cut_points_ng(run_together, SR, 3)
        long_take = speech([("s", .1, 0), ("v", 6.0, .2), ("s", .5, 0), ("v", 1.0, .2), ("s", .1, 0)])
        with self.assertRaisesRegex(sv.SceneAudioError, "line 1 runs"):
            sv.split_take_ng(long_take, SR, 2)

    def test_loudness_matching_brings_pieces_to_one_level_with_a_limit_on_the_boost(self):
        a, b = speech([("v", 1, .1)]), speech([("v", 1, .3)], seed=5)
        out = sv.match_loudness_ng([a, b])
        la, lb = (float(np.sqrt((p ** 2).mean())) for p in out)
        self.assertAlmostEqual(la, lb, delta=0.01 * max(la, lb) + 0.01)
        quiet = speech([("v", 1, .01)])
        boosted = sv.match_loudness_ng([quiet, b], target_rms=0.5)[0]
        self.assertLessEqual(float(np.sqrt((boosted ** 2).mean())) / float(np.sqrt((quiet ** 2).mean())), 4.0 + 1e-3)

    def test_wav_round_trip(self):
        x = speech([("v", 0.5, .2)])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.wav")
            sv.write_wav_ng(p, x, SR)
            y, sr = sv.read_wav_ng(p)
        self.assertEqual(sr, SR)
        self.assertLess(float(np.abs(x - y).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
