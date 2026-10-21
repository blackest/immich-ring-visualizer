"""Unit tests for detection.py and detectionNG.py's pure helpers and lazy
model loader (InsightFace itself is faked -- no model download).

Run with: python3 -m unittest tests.test_detection -v
"""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detection
import detectionNG

# (module, cache attr, loader, best, blur, largest)
TWINS = (
    (detection, "_face_app", "get_face_app", "pick_best_face", "get_blur_score", "pick_largest_face"),
    (detectionNG, "_face_app_ng", "get_face_app_ng", "pick_best_face_ng", "get_blur_score_ng", "pick_largest_face_ng"),
)


def _face(bbox, emb=None):
    return SimpleNamespace(bbox=np.array(bbox), normed_embedding=np.array(emb if emb is not None else [0, 0]))


class DetectionTests(unittest.TestCase):
    def test_pick_largest_face(self):
        small, big = _face([0, 0, 10, 10]), _face([0, 0, 50, 20])
        for mod, _a, _l, _b, _bl, largest in TWINS:
            self.assertIs(getattr(mod, largest)([small, big]), big)

    def test_pick_best_face_uses_embedding_similarity(self):
        ref = np.array([1.0, 0.0])
        near, far = _face([0, 0, 1, 1], [0.9, 0.1]), _face([0, 0, 1, 1], [0.0, 1.0])
        for mod, _a, _l, best, _bl, _lg in TWINS:
            self.assertIs(getattr(mod, best)([far, near], ref), near)

    def test_pick_best_face_single_face_skips_embedding(self):
        only = SimpleNamespace(bbox=None)  # no embedding attribute at all
        for mod, _a, _l, best, _bl, _lg in TWINS:
            self.assertIs(getattr(mod, best)([only], None), only)

    def test_blur_score_sharp_beats_flat(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        flat = np.full((40, 40, 3), 90, dtype=np.uint8)
        for mod, _a, _l, _b, blur, _lg in TWINS:
            f = getattr(mod, blur)
            self.assertGreater(f(noisy), f(flat))
            self.assertEqual(f(flat), 0.0)
            self.assertEqual(f(np.empty((0, 0, 3), dtype=np.uint8)), 0.0)

    def test_face_app_lazy_loaded_once_with_coreml_preferred(self):
        for mod, attr, loader, *_ in TWINS:
            fake_cls = mock.Mock()
            pkg = types.ModuleType("insightface")
            app_mod = types.ModuleType("insightface.app")
            app_mod.FaceAnalysis = fake_cls
            with mock.patch.dict(sys.modules, {"insightface": pkg, "insightface.app": app_mod}), \
                 mock.patch.object(mod, attr, None):
                first = getattr(mod, loader)()
                second = getattr(mod, loader)()
            self.assertIs(first, second)
            fake_cls.assert_called_once()
            self.assertEqual(fake_cls.call_args.kwargs["providers"][0], "CoreMLExecutionProvider")
            fake_cls.return_value.prepare.assert_called_once_with(ctx_id=0, det_size=(640, 640))


if __name__ == "__main__":
    unittest.main()
