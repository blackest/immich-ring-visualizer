"""Unit tests for making a RefMod from pictures (h3_refmod_makeNG.py): the name and picture checks, which
pictures are kept, and the command the task job runs. No model runs; the subprocess is faked.

Run with: python3 -m unittest tests.test_h3_refmod_makeNG -v
"""

import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_refmod_makeNG as mk


def png(size=(256, 256), color=(10, 20, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue(), ".png"


class FakeProc:
    def __init__(self, lines, code, make=None):
        self.stdout = io.StringIO("".join(l + "\n" for l in lines))
        self.code, self.make = code, make

    def wait(self):
        if self.make:
            Path(self.make).write_bytes(b"safetensors")
        return self.code


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "refmods").mkdir()
        (self.tmp / "make_refmod.py").write_text("#")
        self.queued = {}

        def fake_start(task, label, run, meta=None):
            self.queued = dict(task=task, label=label, run=run, meta=meta)
            return SimpleNamespace(job_id="m1", meta=meta)

        for target, name, val in (
                (mk, "refmods_dir_ng", lambda: self.tmp / "refmods"),
                (mk, "script_path_ng", lambda: self.tmp / "scripts" / "make_refmod.py"),
                (mk.h3, "_resolve_h3_python_ng", lambda c: "/venv/bin/python"),
                (mk.h3, "_resolve_h3_compact_root_ng", lambda c: "/models/ddalcu-q8"),
                (mk.h3_jobs, "start_task_job_ng", fake_start)):
            p = mock.patch.object(target, name, val)
            p.start()
            self.addCleanup(p.stop)
        (self.tmp / "scripts").mkdir()
        (self.tmp / "scripts" / "make_refmod.py").write_text("#")


class Spacing(unittest.TestCase):
    def test_the_librarys_eight_of_twenty_two(self):
        self.assertEqual(mk.evenly_spaced_ng(22, 8), [0, 3, 6, 9, 12, 15, 18, 21])

    def test_edges(self):
        self.assertEqual(mk.evenly_spaced_ng(5, 9), [0, 1, 2, 3, 4])
        self.assertEqual(mk.evenly_spaced_ng(5, 1), [0])
        self.assertEqual(mk.evenly_spaced_ng(10, 2), [0, 9])

    def test_rows_per_picture(self):
        self.assertEqual(mk.rows_per_picture_ng("square"), 256)   # the library's 22 pictures = 5,632 rows
        self.assertEqual(mk.rows_per_picture_ng("landscape"), 336)


class Validate(Base):
    def test_good_input_keeps_everything_up_to_the_limit(self):
        name, kept, rows = mk.validate_ng("lake_v1", [png()] * 3, "square")
        self.assertEqual((name, len(kept), rows), ("lake_v1", 3, 768))

    def test_too_many_pictures_are_thinned_evenly(self):
        pics = [png(color=(i, 0, 0)) for i in range(30)]
        _n, kept, rows = mk.validate_ng("many", pics, "square")
        self.assertEqual((len(kept), rows), (mk.MAX_PICTURES, 24 * 256))
        self.assertEqual(kept[0], pics[0])
        self.assertEqual(kept[-1], pics[-1])
        _n, kept, _r = mk.validate_ng("many", pics, "square", max_pictures=8)
        self.assertEqual(len(kept), 8)

    def test_refusals(self):
        (self.tmp / "refmods" / "taken.safetensors").write_bytes(b"x")
        for args, text in ((("", [png()], "square"), "name must be"),
                           (("../x", [png()], "square"), "name must be"),
                           (("ok", [png()], "wide"), "canvas must be"),
                           (("taken", [png()], "square"), "already exists"),
                           (("ok", [], "square"), "at least one picture"),
                           (("ok", [(b"x", ".png")], "square"), "not a readable picture"),
                           (("ok", [(png()[0], ".gif")], "square"), "must be .png"),
                           (("ok", [png((64, 64))], "square"), "at least 128 px")):
            with self.subTest(args=args[0]), self.assertRaisesRegex(ValueError, text):
                mk.validate_ng(*args)
        with self.assertRaisesRegex(ValueError, "whole number"):
            mk.validate_ng("ok", [png()], "square", max_pictures="x")


class Run(Base):
    def run_job(self, proc, **kw):
        mk.start_refmod_job_ng("lake_v1", [png(), png()], **kw)
        log = []
        job = SimpleNamespace(job_dir=self.tmp / "job", cancelled=False, _proc=None,
                              append_log=log.append, log_tail=lambda n: log[-n:])
        job.job_dir.mkdir(exist_ok=True)
        with mock.patch.object(mk.subprocess, "Popen", return_value=proc) as popen:
            try:
                return self.queued["run"](job), popen, job
            except RuntimeError as e:
                return e, popen, job

    def test_queues_a_refmod_task_with_the_row_count(self):
        mk.start_refmod_job_ng("lake_v1", [png()] * 4, canvas="landscape")
        self.assertEqual((self.queued["task"], self.queued["meta"]["rows"], self.queued["meta"]["canvas"]),
                         ("refmod", 4 * 336, "landscape"))

    def test_a_clean_run_writes_the_pictures_and_names_the_output(self):
        out = self.tmp / "refmods" / "lake_v1.safetensors"
        result, popen, job = self.run_job(FakeProc(["wrote x"], 0, make=out))
        self.assertIsNone(result)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "/venv/bin/python")
        self.assertEqual(cmd[cmd.index("--name") + 1], "lake_v1")
        self.assertEqual((cmd[cmd.index("--width") + 1], cmd[cmd.index("--height") + 1]), ("512", "512"))
        self.assertEqual(sorted(p.name for p in (job.job_dir / "pictures").iterdir()), ["000.png", "001.png"])

    def test_a_failed_run_says_why_and_exit_zero_without_a_file_fails(self):
        result, _p, _j = self.run_job(FakeProc(["boom"], 2))
        self.assertIn("exit 2", str(result))
        result, _p, _j = self.run_job(FakeProc([], 0))
        self.assertIsInstance(result, RuntimeError)

    def test_a_missing_script_is_reported_at_once(self):
        (self.tmp / "scripts" / "make_refmod.py").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "make_refmod"):
            mk.start_refmod_job_ng("lake_v1", [png()])


class Route(Base):
    def setUp(self):
        super().setUp()
        import routes.h3_toolsNG as tools
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(tools.h3ToolsNG_bp)
        self.client = app.test_client()

    def post(self, **data):
        files = [(io.BytesIO(png()[0]), f"p{i}.png") for i in range(3)]
        return self.client.post("/api/ng/h3/refmods/create", data={"files": files, **data},
                                content_type="multipart/form-data")

    def test_accepts_pictures_and_reports_the_rows(self):
        r = self.post(name="lake_v1", canvas="landscape")
        self.assertEqual((r.status_code, r.get_json()["job_id"], r.get_json()["rows"]), (202, "m1", 3 * 336))

    def test_a_refusal_comes_back_as_400(self):
        self.assertEqual(self.post(name="bad name").status_code, 400)
        self.assertEqual(self.client.post("/api/ng/h3/refmods/create", data={"name": "ok"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
