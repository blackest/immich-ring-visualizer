"""Unit tests for hidream_engine.generate_hidream()'s PNG-validity check.

Regression coverage for a real bug seen in production: the HiDream
subprocess can exit with rc==0 (a "clean" exit) while having written a
truncated/incomplete PNG to disk -- e.g. if something interrupts it
mid-write (an OOM kill, the Metal "Impacting Interactivity" watchdog,
or a killed/orphaned subprocess). Before this fix, generate_hidream()
only checked that the output file *existed*, so a truncated file with
only its top few scanlines actually decoded (the rest reading back as
black) would silently be reported as a successful candidate and
composited straight into the character sheet.

These tests fake the subprocess layer (no real HiDream/model/venv
needed) by patching subprocess.Popen with a stand-in that "writes" the
files the real script would have written, so the test can control
whether that file ends up valid, truncated, or the wrong size.

Run with:

    python3 -m unittest tests.test_hidream_engine -v

from the repo root.
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

import hidream_engine


def _real_png_bytes(width, height, color=(120, 60, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _extract_output_paths(cmd):
    idx = cmd.index("--output")
    j = idx + 1
    paths = []
    while j < len(cmd) and not str(cmd[j]).startswith("--"):
        paths.append(cmd[j])
        j += 1
    return paths


class _FakeProc:
    """Stands in for the subprocess.Popen(...) object generate_hidream()
    drives -- iterable (empty) stdout, and wait() returns a fixed rc."""

    def __init__(self, rc=0):
        self.stdout = iter(())
        self._rc = rc
        self.pid = 99999

    def wait(self):
        return self._rc


def _make_fake_popen(write_output):
    """Returns a fake replacing subprocess.Popen. `write_output(paths)`
    is called with the list of --output paths pulled out of the launched
    cmd, so the test can decide what ends up on disk at each of them."""

    def _fake_popen(cmd, **kwargs):
        paths = _extract_output_paths(cmd)
        write_output(paths)
        return _FakeProc(rc=0)

    return _fake_popen


class GenerateHidreamValidityCheckTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

        # generate_hidream() bails out early unless it thinks the venv
        # python, the model dir, and the generator script all exist --
        # none of that matters for these tests (Popen is faked), so
        # stub the resolvers and point the script path at a real
        # (empty, never executed) file.
        self._fake_script = self.output_dir / "fake_generate.py"
        self._fake_script.write_text("# not executed, just needs to exist\n")

        patchers = [
            mock.patch.object(hidream_engine, "_resolve_hidream_python",
                               return_value="/usr/bin/python3"),
            mock.patch.object(hidream_engine, "_resolve_hidream_model",
                               return_value="/fake/model/dir"),
            mock.patch.object(hidream_engine, "HIDREAM_GENERATE_SCRIPT",
                               self._fake_script),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

        self.config = hidream_engine.HiDreamConfig()
        # Resolve what width/height generate_hidream() will actually
        # expect on disk for a simple square request, so tests can write
        # PNGs that either match it (valid case) or don't (mismatch case).
        snap_w, snap_h = hidream_engine._snap_to_trained_resolution(1024, 1024)
        self.expected_w = hidream_engine._patch_align(snap_w)
        self.expected_h = hidream_engine._patch_align(snap_h)

    def _run(self, write_output):
        with mock.patch("subprocess.Popen", _make_fake_popen(write_output)):
            return hidream_engine.generate_hidream(
                prompt="a test prompt", n=1, width=1024, height=1024,
                output_dir=self.output_dir, base_seed=42, config=self.config)

    def _run_with_popen(self, fake_popen):
        # For the rc-based-branch tests, which need to control the fake
        # Popen directly rather than going through the write_output hook.
        with mock.patch("subprocess.Popen", fake_popen):
            return hidream_engine.generate_hidream(
                prompt="a test prompt", n=1, width=1024, height=1024,
                output_dir=self.output_dir, base_seed=42, config=self.config)

    # ---- the actual regression -------------------------------------

    def test_truncated_png_raises_instead_of_succeeding(self):
        # This is the case that would have silently succeeded before the
        # fix: rc==0, the file exists, but it's only a partial write.
        full_bytes = _real_png_bytes(self.expected_w, self.expected_h)

        def write_output(paths):
            # Simulate a write interrupted partway through -- only the
            # first ~15% of the real PNG's bytes land on disk (past the
            # header, into the compressed pixel data, then nothing).
            cutoff = max(64, len(full_bytes) // 7)
            Path(paths[0]).write_bytes(full_bytes[:cutoff])

        with self.assertRaises(RuntimeError) as ctx:
            self._run(write_output)
        self.assertIn("incomplete/corrupt", str(ctx.exception))

    def test_wrong_dimensions_raises(self):
        # rc==0, the file is a perfectly valid PNG -- just not the size
        # that was actually requested (a different kind of partial/wrong
        # write than outright truncation).
        def write_output(paths):
            Path(paths[0]).write_bytes(_real_png_bytes(64, 64))

        with self.assertRaises(RuntimeError) as ctx:
            self._run(write_output)
        self.assertIn("64x64", str(ctx.exception))

    def test_valid_full_png_succeeds(self):
        def write_output(paths):
            Path(paths[0]).write_bytes(
                _real_png_bytes(self.expected_w, self.expected_h))

        results = self._run(write_output)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["seed"], 42)
        self.assertTrue(Path(results[0]["png_path"]).is_file())

    def test_missing_file_still_raises_as_before(self):
        # Pre-existing behavior (not new): if the subprocess "forgets" to
        # write the file at all, that's still a clean, distinct error.
        def write_output(paths):
            pass  # write nothing

        with self.assertRaises(RuntimeError) as ctx:
            self._run(write_output)
        self.assertIn("no PNG at", str(ctx.exception))

    # ---- rc-based branches (not new, but undocumented by any existing
    # test -- cheap to cover while already exercising this function) ---

    def test_killed_subprocess_raises_image_job_cancelled(self):
        def _fake_popen(cmd, **kwargs):
            return _FakeProc(rc=-9)

        with self.assertRaises(hidream_engine.ImageJobCancelled):
            self._run_with_popen(_fake_popen)

    def test_nonzero_rc_raises_runtime_error(self):
        def _fake_popen(cmd, **kwargs):
            return _FakeProc(rc=1)

        with self.assertRaises(RuntimeError):
            self._run_with_popen(_fake_popen)


if __name__ == "__main__":
    unittest.main()
