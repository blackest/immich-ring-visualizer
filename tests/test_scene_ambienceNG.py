"""Unit tests for scene ambience: the bed prompt, pulling the sound out of a render, and mixing a bed under a clip's speech
(level, length, picture untouched). Small generated clips only; no model runs.

Run with: python3 -m unittest tests.test_scene_ambienceNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scene_ambienceNG as amb


def make_clip(path, seconds=3, audio="sine=frequency=440:duration={s},volume=0.5"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=24:duration={seconds}",
           "-f", "lavfi", "-i", audio.format(s=seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    subprocess.run(cmd, check=True)


def probe(path, entries, stream="a"):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries, "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True, check=True)
    return r.stdout.strip()


class PromptTests(unittest.TestCase):
    def test_the_prompt_asks_for_an_empty_place_and_its_sound_with_no_people_speech_or_music(self):
        p = amb.ambience_prompt_ng("on the castle battlements")
        for part in ("empty place", "on the castle battlements", "no people", "overall_soundscape:", "no speech",
                     "non_diegetic_music: N/A"):
            self.assertIn(part, p)

    def test_an_empty_setting_is_refused(self):
        for bad in ("", "   "):
            with self.assertRaises(amb.AmbienceError):
                amb.ambience_prompt_ng(bad)


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)

    def test_the_sound_is_extracted_as_a_mono_32k_wav_and_a_silent_or_missing_render_is_refused(self):
        clip = self.t / "c.mp4"
        make_clip(clip)
        wav = amb.extract_audio_ng(str(clip), str(self.t / "bed.wav"))
        self.assertEqual(probe(wav, "stream=channels,sample_rate").split(","), ["32000", "1"])
        with self.assertRaisesRegex(amb.AmbienceError, "no file"):
            amb.extract_audio_ng(str(self.t / "nope.mp4"), str(self.t / "x.wav"))
        silent = self.t / "s.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=24:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)], check=True)
        with self.assertRaises(amb.AmbienceError):
            amb.extract_audio_ng(str(silent), str(self.t / "y.wav"))

    def test_the_bed_sits_below_the_speech_by_the_asked_amount_and_the_picture_is_untouched(self):
        voice = self.t / "voice.mp4"
        make_clip(voice, 3)                                                  # a steady tone standing in for speech
        bed_src = self.t / "bed_src.mp4"
        make_clip(bed_src, 5, audio="anoisesrc=color=pink:duration={s}:amplitude=0.4")
        bed = amb.extract_audio_ng(str(bed_src), str(self.t / "bed.wav"))
        speech = amb.level_ng(str(voice))
        out = amb.mix_ambience_ng(str(voice), bed, str(self.t / "mixed.mp4"), keep_s=2.5)
        self.assertAlmostEqual(float(probe(out, "format=duration", stream="v")), 2.5, delta=0.35)
        self.assertEqual(probe(out, "stream=codec_name", stream="v"), "h264")
        self.assertEqual(probe(out, "stream=width,height", stream="v"), "96,64")
        mixed = amb.level_ng(out)
        db = 20 * __import__("math").log10(mixed / speech)
        self.assertLess(abs(db), 1.0)                                         # a bed 22 dB down barely moves the level
        loud = amb.mix_ambience_ng(str(voice), bed, str(self.t / "loud.mp4"), keep_s=2.5, bed_db=-3.0)
        self.assertGreater(amb.level_ng(loud), mixed * 1.1)                  # and a louder bed does

    def test_nothing_to_measure_is_an_error(self):
        empty = self.t / "e.wav"
        empty.write_bytes(b"")
        with self.assertRaises(amb.AmbienceError):
            amb.level_ng(str(empty))


if __name__ == "__main__":
    unittest.main()
