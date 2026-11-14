"""The app's own model folder (models_rootNG) and the resolvers that look in it first. Nothing is run or downloaded.

Run with: python3 -m unittest tests.test_models_rootNG -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h3_engineNG
import klein_engineNG
import klein_lorasNG
import ltx_engineNG
import models_rootNG
import music_engineNG
import speech_engineNG


class ModelsRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        p = mock.patch.dict(os.environ, {"IMMICHRING_MODELS_ROOT": str(self.root)})
        p.start()
        self.addCleanup(p.stop)
        for key in ("LTX_H3_MODELS", "LTX_MLX_MODELS_DIR", "LTX_MUSIC_MODELS", "LTX_SPEECH_MODELS", "KLEIN_LORAS_DIR"):
            os.environ.pop(key, None)

    def test_a_missing_folder_reads_as_not_there(self):
        self.assertIsNone(models_rootNG.mlx_models())
        self.assertIsNone(models_rootNG.hf_hub())

    def test_the_folders_are_found_once_they_exist(self):
        (self.root / "mlx_models").mkdir()
        (self.root / "HF_HOME" / "hub").mkdir(parents=True)
        self.assertEqual(models_rootNG.mlx_models(), self.root / "mlx_models")
        self.assertEqual(models_rootNG.hf_hub(), self.root / "HF_HOME" / "hub")

    def test_every_engine_looks_there_first(self):
        mm = self.root / "mlx_models"
        for sub in ("hailuo-h3/models", "yue2", "loras/flux3"):
            (mm / sub).mkdir(parents=True)
        (mm / "speech" / "customvoice").mkdir(parents=True)
        (mm / "speech" / "customvoice" / "config.json").write_text("{}")
        (self.root / "HF_HOME" / "hub").mkdir(parents=True)
        self.assertEqual(h3_engineNG._resolve_h3_models_root_ng(), mm / "hailuo-h3" / "models")
        self.assertEqual(ltx_engineNG._resolve_ltx_weights_root_ng(), mm)
        self.assertEqual(music_engineNG._resolve_music_models_root_ng(), mm / "yue2")
        self.assertEqual(speech_engineNG._resolve_speech_pack_ng(), mm / "speech")
        self.assertEqual(klein_engineNG.cache_roots()[0], self.root / "HF_HOME" / "hub")
        self.assertEqual(klein_lorasNG.loras_dir(), mm / "loras" / "flux3")

    def test_the_engine_variables_still_win(self):
        (self.root / "mlx_models").mkdir()
        os.environ["LTX_MLX_MODELS_DIR"] = str(self.root / "elsewhere")
        self.assertEqual(ltx_engineNG._resolve_ltx_weights_root_ng(), self.root / "elsewhere")


class VendoredH3ScriptsTests(unittest.TestCase):
    def test_the_install_steps_find_their_scripts_in_the_vendored_folder(self):
        scripts = Path(models_rootNG.__file__).resolve().parent / "vendor" / "minimax_h3_mlx" / "scripts"
        for name in ("download_selected.py", "quantize_stream.py"):
            self.assertTrue((scripts / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
