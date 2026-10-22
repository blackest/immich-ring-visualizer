"""Unit tests for h3_engineNG's turbo adapter resolution and argv.

Run with: python3 -m unittest tests.test_h3_turbo_resolveNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG as eng


class TurboResolveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        self.v1, self.old, self.adaln = d / "v1.st", d / "old.st", d / "adaln.st"
        for name, p in (("H3_TURBO_V1_LORA_PATH", self.v1), ("H3_TURBO_LORA_PATH", self.old),
                        ("H3_TURBO_ADALN_PATH", self.adaln)):
            patcher = mock.patch.object(eng, name, p)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_nothing_on_disk(self):
        self.assertEqual(eng._resolve_h3_turbo_ng(), (None, None))

    def test_v1_preferred_and_needs_no_adaln(self):
        self.v1.write_bytes(b"x")
        self.old.write_bytes(b"x")
        self.adaln.write_bytes(b"x")
        self.assertEqual(eng._resolve_h3_turbo_ng(), (str(self.v1), None))

    def test_legacy_pair_is_all_or_nothing(self):
        self.old.write_bytes(b"x")
        self.assertEqual(eng._resolve_h3_turbo_ng(), (None, None))
        self.adaln.write_bytes(b"x")
        self.assertEqual(eng._resolve_h3_turbo_ng(), (str(self.old), str(self.adaln)))

    def test_argv_shapes(self):
        self.assertEqual(eng._h3_turbo_argv_ng("/a.st", None), ["--lora", "/a.st:1.0"])
        self.assertEqual(eng._h3_turbo_argv_ng("/a.st", "/b.st"),
                         ["--lora", "/a.st", "--lora-adaln", "/b.st"])


if __name__ == "__main__":
    unittest.main()
