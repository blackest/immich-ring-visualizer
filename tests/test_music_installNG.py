"""Non-network unit tests for music_installNG.py / engine_installNG.py.

_step_clone is the one piece worth testing for correctness rather than
style -- it's ported from Phosphene's music_checkout.sh specifically to
avoid the "checkout reports success but HEAD is somewhere else" failure
class that file's own comments document. Exercises it against a real,
local, throwaway git repo (a tempdir, never the network) standing in for
GitHub, so the actual git plumbing (fetch --force + checkout --force
--detach + the pin assertion) runs for real without any dependency on
vanch007/mlx-Yue being reachable.

Run with: python3 -m unittest tests.test_music_installNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_installNG
import music_installNG


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _make_origin_repo(tmp: Path) -> tuple[str, str]:
    """A tiny local git repo with two commits -- returns (path, first_commit_sha).
    Pinning to the FIRST commit (not the branch tip) is what actually proves
    the checkout follows the pin instead of just "whatever's on the branch"."""
    origin = tmp / "origin"
    origin.mkdir()
    _run(["git", "init", "-q"], cwd=origin)
    _run(["git", "config", "user.email", "test@example.com"], cwd=origin)
    _run(["git", "config", "user.name", "test"], cwd=origin)
    (origin / "a.txt").write_text("first\n")
    _run(["git", "add", "a.txt"], cwd=origin)
    _run(["git", "commit", "-q", "-m", "first"], cwd=origin)
    first_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=origin,
        capture_output=True, text=True, check=True).stdout.strip()
    (origin / "a.txt").write_text("second\n")
    _run(["git", "add", "a.txt"], cwd=origin)
    _run(["git", "commit", "-q", "-m", "second"], cwd=origin)
    return str(origin), first_sha


class StepCloneTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.origin_url, self.first_sha = _make_origin_repo(self.tmp)
        self.dest = self.tmp / "checkout"

    def _clone(self):
        logs = []
        with mock.patch.object(music_installNG, "MUSIC_CLONE_URL", self.origin_url), \
             mock.patch.object(music_installNG, "MUSIC_PIN_SHA", self.first_sha), \
             mock.patch.object(music_installNG, "MUSIC_REPO_DIR", self.dest):
            music_installNG._step_clone(logs.append)
        return logs

    def test_fresh_clone_lands_on_pinned_commit_not_branch_tip(self):
        self._clone()
        head = music_installNG._git_head(self.dest)
        self.assertEqual(head, self.first_sha)
        # The branch tip (second commit) must NOT be what's checked out.
        self.assertEqual((self.dest / "a.txt").read_text(), "first\n")

    def test_rerun_on_existing_checkout_is_idempotent(self):
        self._clone()
        self._clone()  # re-run, same pin -- must not fail or move HEAD
        self.assertEqual(music_installNG._git_head(self.dest), self.first_sha)

    def test_wrong_pin_raises_instead_of_silently_landing_elsewhere(self):
        bogus = "0" * 40
        logs = []
        with mock.patch.object(music_installNG, "MUSIC_CLONE_URL", self.origin_url), \
             mock.patch.object(music_installNG, "MUSIC_PIN_SHA", bogus), \
             mock.patch.object(music_installNG, "MUSIC_REPO_DIR", self.dest):
            with self.assertRaises(Exception):
                music_installNG._step_clone(logs.append)


class MacApplesiliconGuardTestCase(unittest.TestCase):
    def test_blocks_on_non_darwin(self):
        with mock.patch("platform.system", return_value="Linux"):
            ok, reason = engine_installNG.mac_apple_silicon_ok()
        self.assertFalse(ok)
        self.assertIn("Linux", reason)

    def test_blocks_on_intel_mac(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="x86_64"):
            ok, reason = engine_installNG.mac_apple_silicon_ok()
        self.assertFalse(ok)
        self.assertIn("Intel", reason)

    def test_allows_apple_silicon(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="arm64"):
            ok, reason = engine_installNG.mac_apple_silicon_ok()
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_start_install_job_raises_synchronously_when_blocked(self):
        with mock.patch("platform.system", return_value="Linux"):
            with self.assertRaises(ValueError):
                engine_installNG.start_install_job("music", [])


if __name__ == "__main__":
    unittest.main()
