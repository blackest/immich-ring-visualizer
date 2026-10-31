"""The H3 runner the app uses is the copy vendored in this repository, and it carries every flag the app
passes that Phosphene's own pinned copy of the engine does not have.

Run with: python3 -m unittest tests.test_h3_vendoredNG -v
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as h3
import h3_loraNG
import h3_refmod_makeNG as make

NEEDED_FLAGS = ("--ref2va-reference", "--ref2va-reference-size", "--refmod", "--lazy-adaln", "--adaln-cache-dir",
                "--keyframe", "--sigma-subset", "--extend-from", "--chain-context-frames")


class Vendored(unittest.TestCase):
    def test_the_runner_is_the_one_in_this_repository(self):
        repo = Path(h3.__file__).resolve().parent
        self.assertEqual(Path(h3.H3_RUNNER), repo / "vendor" / "minimax_h3_mlx" / "scripts" / "generate_staged.py")
        self.assertTrue(Path(h3.H3_RUNNER).is_file())

    def test_it_takes_every_flag_the_app_passes(self):
        for flag in NEEDED_FLAGS:
            with self.subTest(flag=flag):
                self.assertTrue(h3_loraNG.runner_has_flag_ng(h3.H3_RUNNER, flag), flag)

    def test_the_licence_and_notes_travel_with_it(self):
        for name in ("LICENSE", "VENDORED.md"):
            self.assertTrue((h3.H3_VENDOR_DIR / name).is_file(), name)

    def test_the_refmod_script_sits_beside_the_runner(self):
        self.assertTrue(make.script_path_ng().is_file())


if __name__ == "__main__":
    unittest.main()
