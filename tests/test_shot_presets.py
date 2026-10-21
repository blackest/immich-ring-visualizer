"""Unit tests for shot_presets.py and its NG twin shot_presetsNG.py.

Run with: python3 -m unittest tests.test_shot_presets -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shot_presets
import shot_presetsNG

TWINS = (("shot_presets", shot_presets, shot_presets.resolve_preset),
         ("shot_presetsNG", shot_presetsNG, shot_presetsNG.resolve_preset_ng))


class ShotPresetTests(unittest.TestCase):
    def each(self):
        return [(mod, resolve) for _name, mod, resolve in TWINS]

    def test_resolve_known_and_unknown(self):
        for mod, resolve in self.each():
            self.assertIs(resolve("default"), mod.DEFAULT_PRESET)
            self.assertIs(resolve("extended"), mod.EXTENDED_PRESET)
            with self.assertRaisesRegex(ValueError, "unknown preset 'nope'.*default"):
                resolve("nope")

    def test_preset_sizes_and_unique_keys(self):
        for mod, resolve in self.each():
            self.assertEqual(len(resolve("default")), 3)
            self.assertEqual(len(resolve("extended")), 15)
            for preset in ("default", "extended"):
                keys = [s.key for s in resolve(preset)]
                self.assertEqual(len(keys), len(set(keys)), preset)

    def test_every_shot_has_a_framing_phrase(self):
        for mod, resolve in self.each():
            for preset in ("default", "extended"):
                for shot in resolve(preset):
                    self.assertTrue(shot.pose_phrase.strip(), shot.key)

    def test_angled_shots_do_not_take_the_anchor(self):
        # Profile/three-quarter shots must opt out of the anchor image, or
        # the anchor's near-frontal pose overrides the text instruction.
        for mod, resolve in self.each():
            for shot in resolve("extended"):
                if "profile" in shot.key or "three_quarter" in shot.key:
                    self.assertFalse(shot.use_anchor, shot.key)

    def test_extended_backgrounds_come_from_scenes(self):
        for mod, resolve in self.each():
            scenes = set(mod._BACKGROUNDS)
            for shot in resolve("extended"):
                self.assertIn(shot.background, scenes, shot.key)

    def test_style_presets_include_none(self):
        for mod, resolve in self.each():
            self.assertEqual(mod.STYLE_PRESETS["none"], "")
            for k, v in mod.STYLE_PRESETS.items():
                if k != "none":
                    self.assertTrue(v)

    def test_twins_stay_in_sync(self):
        a, b = shot_presets, shot_presetsNG
        self.assertEqual([vars(s) for s in a.EXTENDED_PRESET],
                         [vars(s) for s in b.EXTENDED_PRESET])
        self.assertEqual([vars(s) for s in a.DEFAULT_PRESET],
                         [vars(s) for s in b.DEFAULT_PRESET])
        self.assertEqual(a._BACKGROUNDS, b._BACKGROUNDS)
        self.assertEqual(list(b.SCENES.values()), b._BACKGROUNDS)
        self.assertEqual(a.STYLE_PRESETS, b.STYLE_PRESETS)


if __name__ == "__main__":
    unittest.main()
