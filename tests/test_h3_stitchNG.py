"""Unit tests for joining window clips (h3_stitchNG.py): which joins are accepted, how many frames
each one leaves, and a real ffmpeg run on tiny synthetic clips.

Run with: python3 -m unittest tests.test_h3_stitchNG -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_stitchNG as st

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def make_clip(path, frames=48, freq=440):
    """A tiny clip with picture and a tone, `frames` frames at 24 fps."""
    secs = frames / 24
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size=64x48:rate=24:duration={secs}", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:sample_rate=32000:duration={secs}", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def stream_info(path):
    """{"video": {"frames", "seconds"}, "audio": {...}} read from the file."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_read_packets,duration",
                          "-count_packets", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return {s["codec_type"]: {"frames": int(s["nb_read_packets"]), "seconds": float(s["duration"])}
            for s in json.loads(out)["streams"]}


class JoinChoice(unittest.TestCase):
    def test_default_is_a_cut_everywhere(self):
        joins = st.normalise_joins_ng(None, 3)
        self.assertEqual([j["type"] for j in joins], ["cut"] * 3)
        self.assertEqual([j["seam"] for j in joins], [1, 2, 3])

    def test_a_chosen_join_is_kept_with_its_frames(self):
        joins = st.normalise_joins_ng([{"seam": 2, "type": "dissolve", "seconds": 0.5}], 3)
        self.assertEqual((joins[1]["type"], joins[1]["frames"]), ("dissolve", 12))
        self.assertEqual(joins[0]["type"], "cut")

    def test_refusals(self):
        for bad, text in (([{"seam": 4, "type": "cut"}], "seam must be 1 to 3"),
                          ([{"seam": 0, "type": "cut"}], "seam must be 1 to 3"),
                          ([{"seam": "x"}], "seam must be a number"),
                          ([{"seam": 1, "type": "wipe"}], "join type"),
                          ([{"seam": 1, "type": "fade", "seconds": 9}], "seconds must be"),
                          (["cut"], "each join")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, text):
                st.normalise_joins_ng(bad, 3)

    def test_a_cut_ignores_seconds(self):
        self.assertEqual(st.normalise_joins_ng([{"seam": 1, "type": "cut", "seconds": 99}], 1)[0]["frames"], 0)


class FrameMath(unittest.TestCase):
    def total(self, kind, seconds=0.5, windows=3, frames=124):
        joins = st.normalise_joins_ng([{"seam": s, "type": kind, "seconds": seconds} for s in range(1, windows)],
                                      windows - 1)
        return st.build_graph_ng([frames] * windows, joins)[3]

    def test_cuts_match_the_runners_own_chain(self):
        self.assertEqual(self.total("cut"), 124 + 123 * 2)

    def test_dissolve_and_audio_lead_each_shorten_by_their_length(self):
        self.assertEqual(self.total("dissolve"), 124 + (123 - 12) * 2)
        self.assertEqual(self.total("audio_lead"), 124 + (123 - 12) * 2)

    def test_fade_keeps_the_length_of_a_cut(self):
        self.assertEqual(self.total("fade"), 124 + 123 * 2)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class RealJoin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clips = []
        for n, freq in enumerate((330, 440, 550)):
            c = self.tmp / f"c{n}.mp4"
            make_clip(c, 48, freq)
            self.clips.append(c)

    def join(self, joins, **kw):
        return st.stitch_windows_ng(self.clips, joins, self.tmp / "out.mp4", **kw)

    def test_cuts_drop_one_frame_per_seam_and_picture_and_sound_agree(self):
        done = self.join([])
        self.assertEqual(done["frames"], 48 + 47 * 2)
        info = stream_info(self.tmp / "out.mp4")
        self.assertEqual(info["video"]["frames"], done["frames"])
        self.assertAlmostEqual(info["audio"]["seconds"], done["seconds"], delta=0.05)

    def test_every_join_type_runs_and_has_the_length_the_math_says(self):
        for kind in ("dissolve", "fade", "audio_lead"):
            with self.subTest(kind=kind):
                done = self.join([{"seam": 1, "type": kind, "seconds": 0.5},
                                  {"seam": 2, "type": kind, "seconds": 0.5}])
                expected = st.build_graph_ng([48] * 3, done["joins"])[3]
                self.assertEqual(done["frames"], expected)
                info = stream_info(self.tmp / "out.mp4")
                self.assertEqual(info["video"]["frames"], expected)
                self.assertAlmostEqual(info["audio"]["seconds"], expected / 24, delta=0.06)

    def test_mixed_joins_and_trim_tail(self):
        done = self.join([{"seam": 1, "type": "dissolve", "seconds": 0.5}], trim_tail=5)
        self.assertEqual(done["frames"], 48 + 47 - 12 + 47 - 5)

    def test_a_failure_leaves_no_half_written_file(self):
        with self.assertRaises(FileNotFoundError):
            st.stitch_windows_ng([self.tmp / "gone.mp4", self.clips[0]], [], self.tmp / "x.mp4")
        self.assertFalse((self.tmp / "x.mp4").exists())


if __name__ == "__main__":
    unittest.main()
