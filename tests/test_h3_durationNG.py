"""Unit tests for the H3 length limit (5 minutes): it is a cap on frames, so the 300.542 s the compiler
sends for a 300 s brief (7213 frames, 59 windows, the same as 300.0) is accepted and 301 s (7230 frames) is not.

Run with: python3 -m unittest tests.test_h3_durationNG -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as eng


class DurationLimit(unittest.TestCase):
    def test_the_compilers_snapped_cap_is_the_same_render_as_the_cap(self):
        snapped = eng._align_h3_frames_ng(eng.H3_MAX_DURATION_S) / eng.H3_FRAME_RATE  # 300 s -> 300.542 s
        self.assertEqual(eng._h3_chain_plan_ng(snapped), eng._h3_chain_plan_ng(eng.H3_MAX_DURATION_S))
        eng._validate_h3_duration_ng(eng.H3_MAX_DURATION_S)
        eng._validate_h3_duration_ng(snapped)

    def test_a_clip_that_needs_more_frames_is_refused(self):
        with self.assertRaisesRegex(ValueError, f"between 3.0 and {eng.H3_MAX_DURATION_S}"):
            eng._validate_h3_duration_ng(eng.H3_MAX_DURATION_S + 1.0)
        with self.assertRaises(ValueError):
            eng._validate_h3_duration_ng(eng.H3_MAX_DURATION_S * 2)

    def test_the_cap_is_five_minutes_and_a_long_chain_gets_a_proportional_watchdog(self):
        self.assertEqual(eng.H3_MAX_DURATION_S, 300.0)
        self.assertEqual(eng._h3_chain_plan_ng(300.0)[1], 59)
        one = eng.resolve_h3_quality_timeout_ng("draft", 5.0)
        self.assertEqual(eng.resolve_h3_quality_timeout_ng("draft", 300.0), one * 59)

    def test_the_minimum_is_unchanged(self):
        eng._validate_h3_duration_ng(3.0)
        with self.assertRaises(ValueError):
            eng._validate_h3_duration_ng(2.9)


if __name__ == "__main__":
    unittest.main()
