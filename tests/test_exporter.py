"""Unit tests for exporter.py and exporterNG.py -- Immich HTTP and the
face detector are mocked; real cv2 encode/decode and file writes are used.

Run with: python3 -m unittest tests.test_exporter -v
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exporter
import exporterNG

# (module, export fn, face-app getter, largest-face fn)
TWINS = (
    (exporter, "export_immich_asset_ids", "get_face_app", "pick_largest_face"),
    (exporterNG, "export_immich_asset_ids_ng", "get_face_app_ng", "pick_largest_face_ng"),
)

ASSET = "0123456789abcdef"


def _png_bytes(w=200, h=100):
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 90, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _params(**kw):
    p = {"out_w": 64, "out_h": 64, "mode": "contain", "margin": 2.2, "interp": "auto",
         "upscale": True, "max_upscale": None, "pad_mode": "none", "min_face_px": 0,
         "native": False}
    p.update(kw)
    return p


def _fake_get(content=b"", status=200, name="pic.jpg"):
    def get(url, **kw):
        if url.endswith("/original"):
            return SimpleNamespace(status_code=status, content=content)
        return SimpleNamespace(json=lambda: {"originalFileName": name})
    return get


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dest = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def run_export(self, mod, fn, get, p, faces=None, face_error=False):
        app = mock.Mock()
        if face_error:
            app.get.side_effect = RuntimeError("boom")
        else:
            app.get.return_value = faces or []
        getter = [t for t in TWINS if t[0] is mod][0][2]
        with mock.patch.object(mod.requests, "get", get), \
             mock.patch.object(mod, getter, return_value=app):
            return getattr(mod, fn)([ASSET], self.dest, p), app

    def test_saves_png_named_from_original_and_id(self):
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                res, app = self.run_export(mod, fn, _fake_get(_png_bytes()), _params())
                self.assertEqual(res["saved"], ["immich_pic_01234567.png"])
                self.assertEqual(res["errors"], [])
                out = cv2.imread(os.path.join(self.dest, res["saved"][0]))
                self.assertEqual(out.shape[:2], (64, 64))
                app.get.assert_not_called()  # contain mode needs no detector

    def test_http_error_and_undecodable_image_reported(self):
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                res, _ = self.run_export(mod, fn, _fake_get(status=404), _params())
                self.assertEqual((res["saved"], res["errors"]), ([], [f"{ASSET}: HTTP 404"]))
                res, _ = self.run_export(mod, fn, _fake_get(b"not an image"), _params())
                self.assertEqual(res["errors"], [f"{ASSET}: could not decode image"])

    def test_one_failure_does_not_stop_the_batch(self):
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                def get(url, **kw):
                    if "bad" in url:
                        raise ConnectionError("down")
                    return _fake_get(_png_bytes())(url, **kw)
                with mock.patch.object(mod.requests, "get", get):
                    res = getattr(mod, fn)(["bad00000", ASSET], self.dest, _params())
                self.assertEqual(len(res["saved"]), 1)
                self.assertIn("bad00000: down", res["errors"])

    def test_face_mode_uses_largest_face_and_survives_detector_error(self):
        faces = [SimpleNamespace(bbox=np.array([10, 10, 30, 30])),
                 SimpleNamespace(bbox=np.array([50, 20, 150, 90]))]
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                res, app = self.run_export(mod, fn, _fake_get(_png_bytes()),
                                           _params(mode="face"), faces=faces)
                app.get.assert_called_once()
                self.assertEqual(len(res["saved"]), 1)
                res, _ = self.run_export(mod, fn, _fake_get(_png_bytes()),
                                         _params(mode="face"), face_error=True)
                self.assertEqual(len(res["saved"]), 1)  # falls back to a center crop

    def test_small_face_is_skipped_with_reason(self):
        face = [SimpleNamespace(bbox=np.array([0, 0, 5, 5]))]
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                res, _ = self.run_export(mod, fn, _fake_get(_png_bytes()),
                                         _params(min_face_px=30), faces=face)
                self.assertEqual(res["saved"], [])
                self.assertIn("skipped (face", res["errors"][0])

    def test_widened_and_padded_counts(self):
        face = [SimpleNamespace(bbox=np.array([95, 45, 105, 55]))]
        for mod, fn, *_ in TWINS:
            with self.subTest(mod=mod.__name__):
                res, _ = self.run_export(mod, fn, _fake_get(_png_bytes()),
                                         _params(mode="face", out_w=256, out_h=256,
                                                 max_upscale=2.0), faces=face)
                self.assertEqual((res["widened"], res["padded"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
