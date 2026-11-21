"""Sharp 2x for a Klein picture (klein_sharpNG.py and POST /api/ng/klein/sharp): where the new picture goes, that asking
twice does not run the model twice, and what is refused or reported. The PiperSR runner is faked.

Run with: venv/bin/python3 -m unittest tests.test_klein_sharpNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

import klein_sharpNG as ks
import routes.kleinNG as routes

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
       b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7\x9a\xa0\xa0\x00\x00\x00\x00IEND\xaeB`\x82")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "job1" / "out"
        self.out.mkdir(parents=True)
        self.pic = self.out / "klein_1.png"
        self.pic.write_bytes(PNG)
        self.runs = []
        for target, name, val in ((routes, "KLEIN_DIR", self.tmp.name),
                                  (ks.h3_sharp, "pipersr_python_ng", lambda: "/venv/bin/python")):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        self.behaviour = "ok"
        p = mock.patch.object(ks.subprocess, "run", self.fake_run)
        p.start()
        self.addCleanup(p.stop)

    def fake_run(self, cmd, **kw):
        self.runs.append(cmd)
        if self.behaviour == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        part = Path(cmd[cmd.index("-o") + 1])
        if self.behaviour == "ok":
            part.write_bytes(PNG)
            return SimpleNamespace(returncode=0, stdout="[pipersr] done 0.8s", stderr="")
        part.write_bytes(b"half")     # a failed run may leave a partial file behind
        return SimpleNamespace(returncode=2, stdout="[pipersr] error cannot read that picture: boom", stderr="")


class Module(Base):
    def test_the_new_picture_sits_beside_the_original_and_the_original_is_untouched(self):
        dest = ks.sharpen_picture_ng(self.pic)
        self.assertEqual(dest, self.out / "klein_1_sr2x100.png")
        self.assertTrue(dest.is_file())
        self.assertEqual(self.pic.read_bytes(), PNG)
        self.assertEqual(self.runs[0][:3], ["/venv/bin/python", str(ks.h3_sharp.SCRIPT), str(self.pic)])
        self.assertEqual(self.runs[0][self.runs[0].index("--strength") + 1], "1")
        self.assertFalse(list(self.out.glob("*.partial.png")))

    def test_asking_again_returns_the_existing_one_without_running_it_again(self):
        ks.sharpen_picture_ng(self.pic)
        ks.sharpen_picture_ng(self.pic)
        self.assertEqual(len(self.runs), 1)

    def test_a_failed_run_says_why_and_leaves_nothing_behind(self):
        self.behaviour = "fail"
        with self.assertRaisesRegex(RuntimeError, "cannot read that picture"):
            ks.sharpen_picture_ng(self.pic)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["klein_1.png"])

    def test_a_run_that_takes_too_long_is_reported(self):
        self.behaviour = "timeout"
        with self.assertRaisesRegex(RuntimeError, "too long"):
            ks.sharpen_picture_ng(self.pic)

    def test_a_missing_picture_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            ks.sharpen_picture_ng(self.out / "nope.png")

    def test_each_strength_is_its_own_file_and_is_passed_on(self):
        full = ks.sharpen_picture_ng(self.pic)
        half = ks.sharpen_picture_ng(self.pic, 0.5)
        self.assertEqual((full.name, half.name), ("klein_1_sr2x100.png", "klein_1_sr2x50.png"))
        self.assertEqual(len(self.runs), 2)
        self.assertEqual(self.runs[1][self.runs[1].index("--strength") + 1], "0.5")

    def test_a_strength_out_of_range_is_refused_before_anything_runs(self):
        for bad in (-0.1, 1.5, "x", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ks.sharpen_picture_ng(self.pic, bad)
        self.assertEqual(self.runs, [])


class Route(Base):
    def setUp(self):
        super().setUp()
        app = Flask(__name__)
        app.register_blueprint(routes.kleinNG_bp)
        self.client = app.test_client()

    def post(self, ref):
        return self.client.post("/api/ng/klein/sharp", json={"ref": ref})

    def test_a_picture_comes_back_as_a_ref_the_image_route_can_serve(self):
        r = self.post("job1/klein_1.png")
        body = r.get_json()
        self.assertEqual((r.status_code, body["ref"], body["file"]), (200, "job1/klein_1_sr2x100.png", "klein_1_sr2x100.png"))
        self.assertEqual((body["width"], body["height"]), (1, 1))
        self.assertEqual(self.client.get("/api/ng/klein/image/job1/klein_1_sr2x100.png").status_code, 200)

    def test_unknown_pictures_and_odd_paths_are_refused(self):
        for ref in ("job1/nope.png", "nojob/klein_1.png", "../job1/out/klein_1.png", "job1/../klein_1.png", "", "x"):
            with self.subTest(ref=ref):
                self.assertEqual(self.post(ref).status_code, 404)
        self.assertEqual(self.runs, [])

    def test_a_failed_run_is_a_500_with_the_reason(self):
        self.behaviour = "fail"
        r = self.post("job1/klein_1.png")
        self.assertEqual(r.status_code, 500)
        self.assertIn("cannot read that picture", r.get_json()["error"])

    def test_a_strength_can_be_asked_for_and_a_bad_one_is_a_400(self):
        r = self.client.post("/api/ng/klein/sharp", json={"ref": "job1/klein_1.png", "strength": 0.5})
        self.assertEqual((r.status_code, r.get_json()["file"]), (200, "klein_1_sr2x50.png"))
        r = self.client.post("/api/ng/klein/sharp", json={"ref": "job1/klein_1.png", "strength": 3})
        self.assertEqual(r.status_code, 400)
        self.assertIn("strength", r.get_json()["error"])

    def test_a_missing_env_says_how_to_build_it(self):
        with mock.patch.object(ks.h3_sharp, "pipersr_python_ng",
                               side_effect=FileNotFoundError("no PiperSR env (run vendor/pipersr_x2/build_venv.sh)")):
            r = self.post("job1/klein_1.png")
        self.assertEqual(r.status_code, 400)
        self.assertIn("build_venv.sh", r.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
