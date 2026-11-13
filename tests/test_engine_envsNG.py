"""H3 and speech engine envs owned by this repository: the resolvers prefer vendor/<engine> once its env is built, and
the H3 installer never runs git or pip in a vendored folder. Nothing is run.

Run with: python3 -m unittest tests.test_engine_envsNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG
import h3_installNG
import speech_engineNG

REPO = Path(__file__).resolve().parent.parent


class _Root(unittest.TestCase):
    module = None
    env_vars: tuple = ()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()          # /var is a symlink to /private/var on macOS
        for p in (mock.patch.object(self.module, "__file__", str(self.root / "engine.py")), mock.patch.dict(os.environ)):
            p.start()
            self.addCleanup(p.stop)
        for key in self.env_vars:
            os.environ.pop(key, None)

    def _venv(self, folder):
        (folder / ".venv" / "bin").mkdir(parents=True)
        (folder / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")


class H3ResolverTests(_Root):
    module = h3_engineNG
    env_vars = ("LTX_H3_ROOT",)

    def setUp(self):
        super().setUp()
        self.vendored = self.root / "vendor" / "minimax_h3_mlx"

    def test_prefers_the_vendored_copy_once_its_venv_is_built(self):
        self._venv(self.vendored)
        self.assertEqual(h3_engineNG._resolve_h3_repo_dir_ng(), self.vendored)

    def test_ignores_it_while_the_venv_is_missing(self):
        self.vendored.mkdir(parents=True)                  # source is committed, venv not built yet
        self.assertNotEqual(h3_engineNG._resolve_h3_repo_dir_ng(), self.vendored)

    def test_the_environment_variable_wins(self):
        self._venv(self.vendored)
        os.environ["LTX_H3_ROOT"] = str(self.root / "elsewhere")
        self.assertEqual(h3_engineNG._resolve_h3_repo_dir_ng(), self.root / "elsewhere")


class SpeechResolverTests(_Root):
    module = speech_engineNG
    env_vars = ("LTX_SPEECH_ROOT",)

    def setUp(self):
        super().setUp()
        self.vendored = self.root / "vendor" / "speech_runner"

    def test_prefers_the_vendored_env_once_it_is_built(self):
        self._venv(self.vendored)
        self.assertEqual(speech_engineNG._resolve_speech_repo_dir_ng(), self.vendored)

    def test_ignores_it_while_the_venv_is_missing(self):
        self.vendored.mkdir(parents=True)
        self.assertNotEqual(speech_engineNG._resolve_speech_repo_dir_ng(), self.vendored)

    def test_the_environment_variable_wins(self):
        self._venv(self.vendored)
        os.environ["LTX_SPEECH_ROOT"] = str(self.root / "elsewhere")
        self.assertEqual(speech_engineNG._resolve_speech_repo_dir_ng(), self.root / "elsewhere")


class H3InstallerGuardTests(unittest.TestCase):
    """With H3_REPO_DIR pointing at the vendored folder, the git and pip steps must run nothing."""

    def setUp(self):
        patches = [
            mock.patch.object(h3_installNG, "H3_REPO_DIR", REPO / "vendor" / "minimax_h3_mlx"),
            mock.patch.object(h3_installNG, "run_logged"),
            mock.patch.object(h3_installNG.subprocess, "run"),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.logs = []

    def test_clone_venv_and_deps_steps_do_nothing_and_say_so(self):
        for step in (h3_installNG._step_clone, h3_installNG._step_venv, h3_installNG._step_deps):
            with self.subTest(step=step.__name__):
                step(self.logs.append)
        self.mocks[1].assert_not_called()   # run_logged: no git, no uv
        self.mocks[2].assert_not_called()   # subprocess.run: no probe of the venv
        self.assertEqual(len(self.logs), 3)
        self.assertTrue(all("vendor" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
