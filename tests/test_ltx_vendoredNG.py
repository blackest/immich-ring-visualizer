"""The vendored ltx-2-mlx (vendor/ltx_2_mlx): the engine resolver prefers it once its env is built, and the
installer never runs git or pip in it. Nothing is rendered and no command is run.

Run with: python3 -m unittest tests.test_ltx_vendoredNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_installNG
import ltx_engineNG
import ltx_installNG

REPO = Path(__file__).resolve().parent.parent


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()   # /var is a symlink to /private/var on macOS
        # pretend ltx_engineNG.py lives in this temp "repo"
        patch = mock.patch.object(ltx_engineNG, "__file__", str(self.root / "ltx_engineNG.py"))
        patch.start()
        self.addCleanup(patch.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LTX_MLX_DIR", None)
        self.vendored = self.root / "vendor" / "ltx_2_mlx"

    def _build_env(self, binary=True, python=True):
        bin_dir = self.vendored / "env" / "bin"
        bin_dir.mkdir(parents=True)
        if binary:
            (bin_dir / "ltx-2-mlx").write_text("#!/bin/sh\n")
        if python:
            (bin_dir / "python").write_text("#!/bin/sh\n")

    def test_prefers_the_vendored_copy_once_its_env_exists(self):
        self._build_env()
        self.assertEqual(ltx_engineNG._resolve_ltx_repo_dir_ng(), self.vendored)

    def test_ignores_it_while_the_env_is_missing_or_half_built(self):
        self.vendored.mkdir(parents=True)  # source is committed, env not built yet
        self.assertNotEqual(ltx_engineNG._resolve_ltx_repo_dir_ng(), self.vendored)

    def test_half_built_env_is_not_enough(self):
        for kw in ({"binary": False}, {"python": False}):
            with self.subTest(kw=kw):
                with tempfile.TemporaryDirectory() as d:
                    d = Path(d).resolve()
                    self.vendored = d / "vendor" / "ltx_2_mlx"
                    with mock.patch.object(ltx_engineNG, "__file__", str(d / "ltx_engineNG.py")):
                        self._build_env(**kw)
                        self.assertNotEqual(ltx_engineNG._resolve_ltx_repo_dir_ng(), self.vendored)

    def test_the_environment_variable_still_wins(self):
        self._build_env()
        os.environ["LTX_MLX_DIR"] = str(self.root / "somewhere-else")
        self.assertEqual(ltx_engineNG._resolve_ltx_repo_dir_ng(), self.root / "somewhere-else")


class VendoredCheckTests(unittest.TestCase):
    def test_a_folder_under_vendor_is_vendored(self):
        self.assertTrue(engine_installNG.app_vendored(REPO / "vendor" / "ltx_2_mlx"))
        self.assertTrue(engine_installNG.app_vendored(str(REPO / "vendor" / "yue2_mlx" / "src")))

    def test_other_folders_are_not(self):
        for p in (REPO, REPO / "tests", "/Volumes/AI/Pinkio/api/phosphene.git/ltx-2-mlx",
                  Path.home() / "ltx-2-mlx"):
            with self.subTest(p=str(p)):
                self.assertFalse(engine_installNG.app_vendored(p))

    def test_a_sibling_with_a_similar_name_is_not(self):
        self.assertFalse(engine_installNG.app_vendored(REPO / "vendor_other" / "ltx_2_mlx"))


class InstallerGuardTests(unittest.TestCase):
    """With LTX_REPO_DIR pointing at the vendored folder, the git and pip steps must run nothing."""

    def setUp(self):
        patches = [
            mock.patch.object(ltx_installNG, "LTX_REPO_DIR", REPO / "vendor" / "ltx_2_mlx"),
            mock.patch.object(ltx_installNG, "run_logged"),
            mock.patch.object(ltx_installNG.subprocess, "run"),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        self.logs = []

    def test_clone_venv_and_deps_steps_do_nothing_and_say_so(self):
        for step in (ltx_installNG._step_clone, ltx_installNG._step_venv, ltx_installNG._step_deps):
            with self.subTest(step=step.__name__):
                step(self.logs.append)
        self.mocks[1].assert_not_called()   # run_logged: no git, no uv
        self.mocks[2].assert_not_called()   # subprocess.run: no git rev-parse
        self.assertEqual(len(self.logs), 3)
        self.assertTrue(all("vendor" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
