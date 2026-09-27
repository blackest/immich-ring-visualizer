"""Non-network unit tests for ltx_installNG.py's clone/pin logic.

Same reasoning as tests/test_music_installNG.py: _step_clone is ported
specifically to avoid a documented Phosphene failure class (a retagged
release leaving a stale local tag, a dirty tree blocking the checkout), so
it's worth exercising against a real local git repo rather than trusting
the port by inspection.

Run with: python3 -m unittest tests.test_ltx_installNG -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ltx_installNG


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _make_origin_repo(tmp: Path) -> tuple[str, str, str]:
    """Local repo with a tagged commit followed by an UNTAGGED later commit
    -- returns (path, tag_name, tagged_commit_sha). Landing on the tagged
    commit (not the branch tip) is the property worth proving."""
    origin = tmp / "origin"
    origin.mkdir()
    _run(["git", "init", "-q"], cwd=origin)
    _run(["git", "config", "user.email", "test@example.com"], cwd=origin)
    _run(["git", "config", "user.name", "test"], cwd=origin)
    (origin / "a.txt").write_text("tagged\n")
    _run(["git", "add", "a.txt"], cwd=origin)
    _run(["git", "commit", "-q", "-m", "tagged commit"], cwd=origin)
    tag = "v1.0.0+test"
    _run(["git", "tag", tag], cwd=origin)
    tagged_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=origin,
        capture_output=True, text=True, check=True).stdout.strip()
    (origin / "a.txt").write_text("newer, untagged\n")
    _run(["git", "add", "a.txt"], cwd=origin)
    _run(["git", "commit", "-q", "-m", "later commit, no tag"], cwd=origin)
    return str(origin), tag, tagged_sha


class StepCloneTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.origin_url, self.tag, self.tagged_sha = _make_origin_repo(self.tmp)
        self.dest = self.tmp / "checkout"

    def _clone(self):
        logs = []
        with mock.patch.object(ltx_installNG, "LTX_FORK_URL", self.origin_url), \
             mock.patch.object(ltx_installNG, "LTX_PIN_TAG", self.tag), \
             mock.patch.object(ltx_installNG, "LTX_REPO_DIR", self.dest):
            ltx_installNG._step_clone(logs.append)
        return logs

    def test_fresh_clone_lands_on_tagged_commit_not_branch_tip(self):
        self._clone()
        self.assertEqual(ltx_installNG._git_head(self.dest), self.tagged_sha)
        self.assertEqual((self.dest / "a.txt").read_text(), "tagged\n")

    def test_rerun_is_idempotent(self):
        self._clone()
        self._clone()
        self.assertEqual(ltx_installNG._git_head(self.dest), self.tagged_sha)

    def test_dirty_tree_does_not_block_the_pin_move(self):
        self._clone()
        (self.dest / "a.txt").write_text("locally dirtied\n")
        self._clone()
        self.assertEqual(ltx_installNG._git_head(self.dest), self.tagged_sha)
        self.assertEqual((self.dest / "a.txt").read_text(), "tagged\n")

    def test_missing_tag_raises(self):
        logs = []
        with mock.patch.object(ltx_installNG, "LTX_FORK_URL", self.origin_url), \
             mock.patch.object(ltx_installNG, "LTX_PIN_TAG", "v9.9.9-does-not-exist"), \
             mock.patch.object(ltx_installNG, "LTX_REPO_DIR", self.dest):
            with self.assertRaises(Exception):
                ltx_installNG._step_clone(logs.append)


if __name__ == "__main__":
    unittest.main()
