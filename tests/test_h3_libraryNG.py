"""Unit tests for the H3 prompt library parser, filters and path safety. Small generated folders only.

Run with: python3 -m unittest tests.test_h3_libraryNG -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_libraryNG as lib


def txt(desc, sound="Rain.", music="N/A"):
    return ("integrated_multimodal_description: %s\noverall_soundscape: %s\nnon_diegetic_music: %s\n"
            % (desc, sound, music))


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        rows = {
            "000001": txt("[Shot 1] Live-action, cinematic film, a thief runs: <d>[English] Go!</d> [Shot 2] At 00:03.000, a cut.",
                          music="Taiko drums."),
            "000002": txt("[Shot 1] 2D-animated in action-anime style, a fox leaps."),
            "000003": txt("[Shot 1] Grainy VHS home video, a dog barks at a cat."),
        }
        for k, v in rows.items():
            open(os.path.join(self.dir, k + ".txt"), "w").write(v)
            open(os.path.join(self.dir, k + ".mp4"), "wb").write(b"x")
        open(os.path.join(self.dir, "README.md"), "w").write("not a clip")

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_fields(self):
        f = lib.parse_prompt_text(txt("[Shot 1] a", "snd", "mus"))
        self.assertEqual((f["overall_soundscape"], f["non_diegetic_music"]), ("snd", "mus"))

    def test_records(self):
        r = lib.get_clip(self.dir, "000001")
        self.assertEqual((r["shots"], r["dialogue"], r["music"], r["look"]), (2, True, True, "Live-action"))
        self.assertTrue(r["full"].startswith("integrated_multimodal_description:"))
        self.assertIn("\n\noverall_soundscape: Rain.\n\nnon_diegetic_music: Taiko", r["full"])
        self.assertEqual(lib.get_clip(self.dir, "000002")["look"], "Animation")
        self.assertEqual(lib.get_clip(self.dir, "000003")["look"], "Vintage")
        self.assertFalse(lib.get_clip(self.dir, "000002")["music"])

    def test_filters(self):
        self.assertEqual(lib.list_clips(self.dir)["total"], 3)
        self.assertEqual(lib.list_clips(self.dir, query="fox LEAPS")["total"], 1)
        self.assertEqual(lib.list_clips(self.dir, shots=2)["items"][0]["id"], "000001")
        self.assertEqual(lib.list_clips(self.dir, dialogue=False)["total"], 2)
        self.assertEqual(lib.list_clips(self.dir, music=True)["total"], 1)
        out = lib.list_clips(self.dir, look="Animation")
        self.assertEqual(out["total"], 1)
        self.assertEqual(sum(out["looks"].values()), 3)  # look counts ignore the look filter itself

    def test_paging_and_short_items(self):
        out = lib.list_clips(self.dir, offset=1, limit=1)
        self.assertEqual([i["id"] for i in out["items"]], ["000002"])
        self.assertNotIn("full", out["items"][0])

    def test_path_safety(self):
        self.assertIsNone(lib.video_path(self.dir, "../etc/passwd"))
        self.assertIsNone(lib.video_path(self.dir, "000999"))
        self.assertTrue(lib.video_path(self.dir, "000001").endswith("000001.mp4"))
        self.assertIsNone(lib.get_clip(self.dir, "README"))

    def test_real_dataset_if_present(self):
        real = "/Users/john/immich-app/repo/minimax_h3_1k"
        if not os.path.isdir(real):
            self.skipTest("dataset not here")
        out = lib.list_clips(real, limit=1)
        self.assertEqual(out["total"], 1000)
        self.assertNotIn("Other", [k for k, v in out["looks"].items() if v > 100])


if __name__ == "__main__":
    unittest.main()
