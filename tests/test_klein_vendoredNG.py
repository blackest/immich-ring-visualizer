"""Klein's worker interpreter: PHOSPHENE_FLUX3_ROOT first, then vendor/flux3_worker once its env is built, then
Phosphene's install. Nothing is run.

Run with: python3 -m unittest tests.test_klein_vendoredNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import klein_engineNG


class WorkerPythonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()          # /var is a symlink to /private/var on macOS
        for p in (mock.patch.object(klein_engineNG, "__file__", str(self.root / "klein_engineNG.py")),
                  mock.patch.dict(os.environ)):
            p.start()
            self.addCleanup(p.stop)
        for key in ("PHOSPHENE_FLUX3_ROOT", "PHOSPHENE_ROOT"):
            os.environ.pop(key, None)
        self.vendored = self.root / "vendor" / "flux3_worker"

    def _build_env(self):
        (self.vendored / "env" / "bin").mkdir(parents=True)
        (self.vendored / "env" / "bin" / "python").write_text("#!/bin/sh\n")

    def test_prefers_the_vendored_env_once_it_is_built(self):
        self._build_env()
        self.assertEqual(klein_engineNG.worker_python(), self.vendored / "env" / "bin" / "python")

    def test_falls_back_to_phosphene_while_it_is_not_built(self):
        self.vendored.mkdir(parents=True)                  # lock file committed, env not built yet
        got = klein_engineNG.worker_python()
        self.assertNotEqual(got, self.vendored / "env" / "bin" / "python")
        self.assertEqual(got.parts[-4:], ("flux3-worker", "env", "bin", "python"))

    def test_phosphene_root_is_still_honoured_for_the_fallback(self):
        os.environ["PHOSPHENE_ROOT"] = str(self.root / "phos")
        self.assertEqual(klein_engineNG.worker_python(), self.root / "phos" / "flux3-worker" / "env" / "bin" / "python")

    def test_the_environment_variable_wins_over_everything(self):
        self._build_env()
        os.environ["PHOSPHENE_FLUX3_ROOT"] = str(self.root / "elsewhere")
        self.assertEqual(klein_engineNG.worker_python(), self.root / "elsewhere" / "env" / "bin" / "python")


if __name__ == "__main__":
    unittest.main()
