"""Unit tests for the H3 length limit: it is a cap on frames, so the 30.677 s the compiler sends for
a 30 s brief (736 frames, 6 windows, the same as 30.0) is accepted and 30.7 s (753 frames) is not.

Run with: python3 -m unittest tests.test_h3_durationNG -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as eng


class DurationLimit(unittest.TestCase):
    def test_the_compilers_snapped_thirty_seconds_is_the_same_render_as_thirty(self):
        self.assertEqual(eng._h3_chain_plan_ng(30.677), eng._h3_chain_plan_ng(30.0))
        eng._validate_h3_duration_ng(30.0)
        eng._validate_h3_duration_ng(30.677)

    def test_one_more_window_of_frames_is_refused(self):
        with self.assertRaisesRegex(ValueError, "between 3.0 and 30.0"):
            eng._validate_h3_duration_ng(30.7)
        with self.assertRaises(ValueError):
            eng._validate_h3_duration_ng(45.0)

    def test_the_minimum_is_unchanged(self):
        eng._validate_h3_duration_ng(3.0)
        with self.assertRaises(ValueError):
            eng._validate_h3_duration_ng(2.9)


if __name__ == "__main__":
    unittest.main()
