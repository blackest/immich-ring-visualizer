"""Non-network unit tests for ltx_pack_fetchNG.py's assembly/verification
logic -- the part ported faithfully from Phosphene's fetch_pack_release.py
specifically because a truncated/corrupt multi-GB download silently
producing a broken model is the failure class it exists to prevent.

_download_shard is stubbed everywhere here (writes deterministic bytes
instead of making a real HTTP request) so these run instantly and offline.

Run with: python3 -m unittest tests.test_ltx_pack_fetchNG -v
"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ltx_pack_fetchNG as fetcher


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AssembleFileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest_dir = Path(self._tmp.name)

    def test_skips_already_verified_file(self):
        data = b"already here, correct"
        target = self.dest_dir / "model.bin"
        target.write_bytes(data)
        spec = {"bytes": len(data), "sha256": _sha256(data), "shards": [{"asset": "should-not-be-fetched"}]}

        with mock.patch.object(fetcher, "_download_shard") as dl:
            fetcher._assemble_file("model.bin", spec, self.dest_dir, lambda m: None)
        dl.assert_not_called()

    def test_reassembles_shards_in_order_and_verifies(self):
        part_a, part_b = b"first-half-", b"second-half"
        whole = part_a + part_b

        def fake_download(url, dest, expect_bytes, expect_sha, label, on_log):
            data = part_a if "shard0" in dest.name else part_b
            dest.write_bytes(data)

        spec = {
            "bytes": len(whole), "sha256": _sha256(whole),
            "shards": [
                {"asset": "shard0", "bytes": len(part_a), "sha256": _sha256(part_a)},
                {"asset": "shard1", "bytes": len(part_b), "sha256": _sha256(part_b)},
            ],
        }
        with mock.patch.object(fetcher, "_download_shard", side_effect=fake_download):
            fetcher._assemble_file("model.bin", spec, self.dest_dir, lambda m: None)

        target = self.dest_dir / "model.bin"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), whole)

    def test_raises_on_final_assembly_sha_mismatch(self):
        def fake_download(url, dest, expect_bytes, expect_sha, label, on_log):
            dest.write_bytes(b"x" * expect_bytes)  # right size, wrong content

        spec = {
            "bytes": 4, "sha256": _sha256(b"real"),
            "shards": [{"asset": "shard0", "bytes": 4, "sha256": _sha256(b"x" * 4)}],
        }
        with mock.patch.object(fetcher, "_download_shard", side_effect=fake_download):
            with self.assertRaises(RuntimeError):
                fetcher._assemble_file("model.bin", spec, self.dest_dir, lambda m: None)
        self.assertFalse((self.dest_dir / "model.bin").exists())


if __name__ == "__main__":
    unittest.main()
