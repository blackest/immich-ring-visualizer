"""h3_image_engineNG: checks, the command line, the restart after macOS's GPU watchdog, cancel. No model is launched."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3_image_engineNG as eng  # noqa: E402


class _Proc:
    """Stands in for subprocess.Popen: prints `lines`, exits with `rc`, and writes the picture when it succeeds."""
    def __init__(self, rc=0, lines=(), on_ok=None):
        self.stdout, self.pid, self._rc, self._on_ok = iter(list(lines)), 4242, rc, on_ok

    def wait(self, timeout=None):
        if self._rc == 0 and self._on_ok:
            self._on_ok()
        return self._rc

    def poll(self):
        return self._rc


class HealthAndValidationTests(unittest.TestCase):
    def test_health_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(eng, "PACK_DIR", Path(d)):
            health = eng.h3_image_health_ng()
        self.assertFalse(health["ok"])
        self.assertTrue(any("not downloaded" in p for p in health["problems"]))

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            ref = Path(d) / "a.png"
            ref.write_bytes(b"x")
            eng.validate_ng("make it", 1, "3:2", "draft", [str(ref)])          # fine
            bad = [("", 1, "3:2", "draft", []), ("x", 0, "3:2", "draft", []), ("x", 9, "3:2", "draft", []),
                   ("x", 1, "3:2", "ultra", []), ("x", 1, "5:7", "draft", []),
                   ("x", 1, "match", "draft", []),                                # 'same as picture' needs a picture
                   ("x", 1, "3:2", "draft", [str(ref)] * 7),
                   ("x", 1, "3:2", "draft", [str(Path(d) / "gone.png")])]
            for args in bad:
                with self.assertRaises(ValueError, msg=str(args[:4])):
                    eng.validate_ng(*args)

    def test_command_edit_in_the_shape_of_picture_one(self):
        cmd = eng._command("fix it", Path("/o/a.png"), Path("/o/a.json"), "match", 4.0, 7, ["/r/1.jpg", "/r/2.jpg"])
        self.assertEqual(cmd.count("--reference"), 2)
        self.assertIn("--match-reference", cmd)
        self.assertEqual(cmd[cmd.index("--mp") + 1], "4")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")

    def test_command_text_to_picture_has_the_shape_and_no_reference_flags(self):
        cmd = eng._command("a cat", Path("/o/a.png"), Path("/o/a.json"), "3:2", 8.0, 1, [])
        self.assertEqual(cmd[cmd.index("--aspect") + 1], "3:2")
        self.assertNotIn("--reference", cmd)
        self.assertNotIn("--match-reference", cmd)


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        self.popens, self.logs = [], []
        patcher = mock.patch.object(eng, "h3_image_health_ng", return_value={"ok": True, "problems": []})
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.object(eng.h3, "_clean_subprocess_env_ng", return_value={})
        env.start()
        self.addCleanup(env.stop)

    def _gen(self, procs, **kw):
        procs = iter(procs)

        def popen(cmd, **kwargs):
            self.popens.append(cmd)
            proc = next(procs)
            png, metrics = Path(cmd[cmd.index("-o") + 1]), Path(cmd[cmd.index("--metrics") + 1])
            proc._on_ok = lambda: (png.write_bytes(b"png"), metrics.write_text(json.dumps({"width": 64, "height": 32})))
            return proc
        args = dict(prompt="a cat", n=1, shape="1:1", tier="draft", output_dir=self.out, seed=5, refs=())
        args.update(kw)
        with mock.patch("subprocess.Popen", popen):
            return eng.generate_h3_image_ng(on_log=self.logs.append, **args)

    def test_pictures_and_their_seeds(self):
        got = []
        res = self._gen([_Proc(), _Proc()], n=2, on_result=got.append)
        self.assertEqual([r["seed"] for r in res], [5, 6])
        self.assertEqual((res[0]["width"], res[0]["height"]), (64, 32))
        self.assertEqual(len(got), 2)
        self.assertTrue(all(Path(r["png_path"]).is_file() for r in res))

    def test_a_render_stopped_by_macos_is_started_again(self):
        stopped = _Proc(rc=-6, lines=["[METAL] Command buffer execution failed: Impacting Interactivity"])
        res = self._gen([stopped, _Proc()])
        self.assertEqual(len(self.popens), 2)
        self.assertEqual(len(res), 1)
        self.assertTrue(any("macOS stopped the render" in l for l in self.logs))

    def test_three_stops_give_up_with_the_reason(self):
        dead = lambda: _Proc(rc=-6, lines=["Impacting Interactivity"])
        with self.assertRaisesRegex(RuntimeError, "Impacting Interactivity"):
            self._gen([dead(), dead(), dead()])
        self.assertEqual(len(self.popens), 3)

    def test_other_failures_are_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "bad picture"):
            self._gen([_Proc(rc=1, lines=["bad picture"]), _Proc()])
        self.assertEqual(len(self.popens), 1)

    def test_stop_cancels(self):
        with self.assertRaises(eng.H3ImageCancelled):
            self._gen([_Proc(lines=["working"])], should_stop=lambda: True)

    def test_a_finished_render_with_no_file_is_an_error(self):
        proc = _Proc()
        procs = iter([proc])
        with mock.patch("subprocess.Popen", lambda cmd, **kw: next(procs)):
            with self.assertRaisesRegex(RuntimeError, "wrote no picture"):
                eng.generate_h3_image_ng("a cat", 1, "1:1", "draft", self.out, seed=1)


if __name__ == "__main__":
    unittest.main()
