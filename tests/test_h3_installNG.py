"""Non-network unit tests for h3_installNG.py's clone/pin logic and RAM
preflight.

Run with: python3 -m unittest tests.test_h3_installNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_installNG


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _make_origin_repo(tmp: Path) -> tuple[str, str]:
    """A repo with a branch matching H3_BRANCH's shape -- returns
    (path, tip_sha) after committing scripts/quantize_stream.py, which
    _step_clone requires to exist post-checkout."""
    origin = tmp / "origin"
    origin.mkdir()
    _run(["git", "init", "-q", "-b", "codex/h3-engine-v2"], cwd=origin)
    _run(["git", "config", "user.email", "test@example.com"], cwd=origin)
    _run(["git", "config", "user.name", "test"], cwd=origin)
    (origin / "scripts").mkdir()
    (origin / "scripts" / "quantize_stream.py").write_text("# stub\n")
    (origin / "minimax_h3_mlx").mkdir()
    (origin / "minimax_h3_mlx" / "__init__.py").write_text("")
    _run(["git", "add", "-A"], cwd=origin)
    _run(["git", "commit", "-q", "-m", "initial"], cwd=origin)
    tip = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin,
                          capture_output=True, text=True, check=True).stdout.strip()
    return str(origin), tip


class StepCloneTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.origin_url, self.tip_sha = _make_origin_repo(self.tmp)
        self.dest = self.tmp / "checkout"

    def _clone(self):
        logs = []
        with mock.patch.object(h3_installNG, "H3_URL", self.origin_url), \
             mock.patch.object(h3_installNG, "H3_REPO_DIR", self.dest):
            h3_installNG._step_clone(logs.append)
        return logs

    def test_fresh_clone_lands_on_branch_tip(self):
        self._clone()
        self.assertEqual(h3_installNG._git_head(self.dest), self.tip_sha)
        self.assertTrue((self.dest / "scripts" / "quantize_stream.py").is_file())

    def test_rerun_after_local_edit_is_idempotent_and_resets(self):
        self._clone()
        (self.dest / "minimax_h3_mlx" / "__init__.py").write_text("dirtied\n")
        self._clone()
        self.assertEqual(h3_installNG._git_head(self.dest), self.tip_sha)
        self.assertEqual((self.dest / "minimax_h3_mlx" / "__init__.py").read_text(), "")

    def test_missing_quantize_script_raises(self):
        # A branch tip without scripts/quantize_stream.py must be refused
        # loudly (see h3_checkout.sh's own WARN for this exact case)
        # rather than silently leaving the Q8 build unrunnable.
        bare = self.tmp / "bare_no_quant"
        bare.mkdir()
        _run(["git", "init", "-q", "-b", "codex/h3-engine-v2"], cwd=bare)
        _run(["git", "config", "user.email", "t@example.com"], cwd=bare)
        _run(["git", "config", "user.name", "t"], cwd=bare)
        (bare / "readme.txt").write_text("no quantizer here\n")
        _run(["git", "add", "-A"], cwd=bare)
        _run(["git", "commit", "-q", "-m", "no quantizer"], cwd=bare)

        logs = []
        with mock.patch.object(h3_installNG, "H3_URL", str(bare)), \
             mock.patch.object(h3_installNG, "H3_REPO_DIR", self.dest):
            with self.assertRaises(RuntimeError):
                h3_installNG._step_clone(logs.append)


class RamGuardTestCase(unittest.TestCase):
    def test_blocks_below_36gb(self):
        fake = mock.Mock(returncode=0, stdout="20000000000\n")
        with mock.patch("subprocess.run", return_value=fake):
            ok, reason = h3_installNG.ram_ok()
        self.assertFalse(ok)
        self.assertIn("36 GB", reason)

    def test_allows_above_36gb(self):
        fake = mock.Mock(returncode=0, stdout="64000000000\n")
        with mock.patch("subprocess.run", return_value=fake):
            ok, reason = h3_installNG.ram_ok()
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_fails_open_when_unreadable(self):
        with mock.patch("subprocess.run", side_effect=OSError("no sysctl")):
            ok, reason = h3_installNG.ram_ok()
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
