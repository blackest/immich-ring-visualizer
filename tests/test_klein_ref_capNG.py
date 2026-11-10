"""A big reference picture is shrunk before Klein sees it; the upload is kept and small or unreadable ones pass through.

Run with: python3 -m unittest tests.test_klein_ref_capNG -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import klein_engineNG as engine
import klein_jobsNG as klein


class RefCap(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _pic(self, name, size, fmt="PNG", exif_orientation=None):
        path = self.dir / name
        im = Image.new("RGB", size, (120, 80, 40))
        if exif_orientation:
            exif = Image.Exif()
            exif[0x0112] = exif_orientation
            im.save(path, fmt, exif=exif)
        else:
            im.save(path, fmt)
        return path

    def test_big_picture_is_shrunk_to_the_long_side(self):
        big = self._pic("subject.jpeg", (4032, 3024), "JPEG")
        out, notes = klein._capped_refs_ng([{"slot": "subject", "path": str(big)}], self.dir)
        self.assertNotEqual(out[0], str(big))
        self.assertEqual(notes, ["subject shrunk from 4032x3024 to 1024x768"])
        with Image.open(out[0]) as im:
            self.assertEqual(im.size, (engine.REF_MAX_SIDE, 768))
        self.assertTrue(big.exists())                       # the upload is kept as it came

    def test_small_picture_passes_through(self):
        small = self._pic("location.png", (800, 600))
        out, notes = klein._capped_refs_ng([{"slot": "location", "path": str(small)}], self.dir)
        self.assertEqual((out, notes), ([str(small)], []))
        self.assertFalse((self.dir / "refs_capped").exists())

    def test_rotation_tag_is_applied(self):
        # stored landscape, tagged "rotate 90": upright it is 3000 wide by 4000 tall
        tall = self._pic("subject.jpeg", (4000, 3000), "JPEG", exif_orientation=6)
        out, _ = klein._capped_refs_ng([{"slot": "subject", "path": str(tall)}], self.dir)
        with Image.open(out[0]) as im:
            self.assertEqual(im.size, (768, engine.REF_MAX_SIDE))

    def test_unreadable_file_passes_through(self):
        junk = self.dir / "extra.png"
        junk.write_bytes(b"not a picture")
        out, notes = klein._capped_refs_ng([{"slot": "extra", "path": str(junk)}], self.dir)
        self.assertEqual((out, notes), ([str(junk)], []))

    def test_order_is_kept(self):
        a = self._pic("subject.jpeg", (3000, 2000), "JPEG")
        b = self._pic("location.png", (500, 500))
        out, _ = klein._capped_refs_ng([{"slot": "subject", "path": str(a)}, {"slot": "location", "path": str(b)}], self.dir)
        self.assertTrue(out[0].endswith("subject.png"))
        self.assertEqual(out[1], str(b))


if __name__ == "__main__":
    unittest.main()
