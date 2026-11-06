"""The install buttons must never re-pin, reset or reinstall into a checkout that belongs to Phosphene.

Phosphene keeps ltx-2-mlx, yue2-mlx and minimax-h3-mlx next to its panel file and moves them itself on
every Update. Our engine resolvers fall back to those folders, so the install recipes recognise one
(engine_installNG.phosphene_owned) and only fill in what is missing. An app-managed checkout, with no
Phosphene panel file beside it, installs exactly as before.

Nothing here touches the network or a real venv: run_logged and subprocess are replaced.

Run with: python3 -m unittest tests.test_install_phosphene_guardNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_installNG
import h3_installNG
import ltx_installNG
import music_installNG


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def checkout(self, name, *, owned, venv=None, extra=()):
        """A fake engine checkout (with a .git folder); `owned` puts Phosphene's panel file beside it."""
        parent = self.root / ("phosphene" if owned else "app")
        dest = parent / name
        (dest / ".git").mkdir(parents=True)
        if owned:
            (parent / "mlx_ltx_panel.py").write_text("")
        if venv:
            (dest / venv).parent.mkdir(parents=True, exist_ok=True)
            (dest / venv).write_text("")
        for rel in extra:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_text("")
        return dest

    def run_step(self, module, repo_attr, dest, step, **extra_patches):
        """Run `module._step_<step>` with every command recorded and none executed; returns (commands, logs)."""
        cmds, logs = [], []
        with mock.patch.object(module, repo_attr, dest), \
             mock.patch.object(module, "run_logged", side_effect=lambda cmd, **kw: cmds.append(cmd)), \
             mock.patch.object(module.subprocess if hasattr(module, "subprocess") else __import__("subprocess"),
                               "run", side_effect=lambda cmd, **kw: cmds.append(cmd) or mock.Mock(returncode=0, stdout="")):
            getattr(module, f"_step_{step}")(logs.append)
        return cmds, logs


class PhospheneOwnedHelper(Base):
    def test_recognises_a_checkout_beside_the_panel_file(self):
        self.assertTrue(engine_installNG.phosphene_owned(self.checkout("minimax-h3-mlx", owned=True)))
        self.assertFalse(engine_installNG.phosphene_owned(self.checkout("minimax-h3-mlx", owned=False)))


class H3(Base):
    def test_owned_checkout_keeps_its_pin_and_venv(self):
        dest = self.checkout("minimax-h3-mlx", owned=True, venv=".venv/bin/python",
                             extra=("scripts/quantize_stream.py",))
        for step in ("clone", "deps"):
            cmds, logs = self.run_step(h3_installNG, "H3_REPO_DIR", dest, step)
            self.assertEqual(cmds, [], f"{step} ran {cmds}")
            self.assertTrue(any("Phosphene" in line for line in logs))

    def test_owned_checkout_without_the_quantizer_is_a_plain_error(self):
        dest = self.checkout("minimax-h3-mlx", owned=True)
        with mock.patch.object(h3_installNG, "H3_REPO_DIR", dest):
            with self.assertRaisesRegex(RuntimeError, "Phosphene's Update"):
                h3_installNG._step_clone(lambda _: None)

    def test_app_managed_venv_still_installs_deps(self):
        dest = self.checkout("minimax-h3-mlx", owned=False, venv=".venv/bin/python")
        cmds, _ = self.run_step(h3_installNG, "H3_REPO_DIR", dest, "deps")
        self.assertEqual(len(cmds), 1)
        self.assertIn("install", cmds[0])


class LTX(Base):
    def test_owned_checkout_keeps_its_pin_and_venv(self):
        dest = self.checkout("ltx-2-mlx", owned=True, venv="env/bin/python")
        for step in ("clone", "deps"):
            cmds, logs = self.run_step(ltx_installNG, "LTX_REPO_DIR", dest, step)
            self.assertEqual(cmds, [], f"{step} ran {cmds}")
            self.assertTrue(any("Phosphene" in line for line in logs))

    def test_app_managed_venv_still_installs_deps(self):
        dest = self.checkout("ltx-2-mlx", owned=False, venv="env/bin/python")
        cmds, _ = self.run_step(ltx_installNG, "LTX_REPO_DIR", dest, "deps")
        self.assertTrue(cmds)


class Music(Base):
    def test_owned_checkout_keeps_its_pin_and_venv(self):
        dest = self.checkout("yue2-mlx", owned=True, venv=".venv/bin/python")
        for step in ("clone", "deps"):
            cmds, logs = self.run_step(music_installNG, "MUSIC_REPO_DIR", dest, step)
            self.assertEqual(cmds, [], f"{step} ran {cmds}")
            self.assertTrue(any("Phosphene" in line for line in logs))

    def test_owned_venv_that_is_not_python_312_is_never_cleared(self):
        dest = self.checkout("yue2-mlx", owned=True, venv=".venv/bin/python")
        cmds = []
        with mock.patch.object(music_installNG, "MUSIC_REPO_DIR", dest), \
             mock.patch.object(music_installNG, "run_logged", side_effect=lambda c, **k: cmds.append(c)), \
             mock.patch("subprocess.run", return_value=mock.Mock(returncode=1)):
            with self.assertRaisesRegex(RuntimeError, "Phosphene"):
                music_installNG._step_venv(lambda _: None)
        self.assertEqual(cmds, [])

    def test_app_managed_venv_still_syncs_deps(self):
        dest = self.checkout("yue2-mlx", owned=False, venv=".venv/bin/python")
        cmds, _ = self.run_step(music_installNG, "MUSIC_REPO_DIR", dest, "deps")
        self.assertTrue(cmds)


if __name__ == "__main__":
    unittest.main()
