"""Unit tests for image_ops.py and its NG twin image_opsNG.py. Both are
exercised by the same cases so the twins can't silently drift apart.

Run with: python3 -m unittest tests.test_image_ops -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_ops
import image_opsNG


def _api(mod, suffix):
    return {
        "crop": getattr(mod, "crop_resize_export" + suffix),
        "as_bool": getattr(mod, "_as_bool" + suffix),
        "params": getattr(mod, "_export_params_from_body" + suffix),
        "height_px": getattr(mod, "face_export_height_px" + suffix),
        "skip": getattr(mod, "should_skip_for_small_face" + suffix),
        "pick": getattr(mod, "_pick_interp" + suffix),
        "center": getattr(mod, "_center_crop_to_aspect" + suffix),
    }


TWINS = (("image_ops", _api(image_ops, "")), ("image_opsNG", _api(image_opsNG, "_ng")))


def _img(w=400, h=300):
    return np.full((h, w, 3), 128, dtype=np.uint8)


class ImageOpsTwinTests(unittest.TestCase):
    def each(self):
        return [api for _name, api in TWINS]

    def test_as_bool(self):
        for api in self.each():
            f = api["as_bool"]
            self.assertTrue(f(True))
            self.assertFalse(f(False))
            self.assertFalse(f("false"))  # the whole point: bool("false") is True
            for s in ("1", "TRUE", " yes ", "on"):
                self.assertTrue(f(s))
            self.assertTrue(f(None, True))
            self.assertFalse(f(None))

    def test_pick_interp_auto_follows_direction(self):
        import cv2
        for api in self.each():
            self.assertEqual(api["pick"]("auto", 0.5), cv2.INTER_AREA)
            self.assertEqual(api["pick"]("auto", 2.0), cv2.INTER_LANCZOS4)
            self.assertEqual(api["pick"]("cubic", 0.5), cv2.INTER_CUBIC)

    def test_center_crop_to_aspect(self):
        for api in self.each():
            self.assertEqual(api["center"](_img(400, 300), 100, 100).shape[:2], (300, 300))
            self.assertEqual(api["center"](_img(300, 400), 100, 100).shape[:2], (300, 300))

    def test_stretch_and_contain_output_size(self):
        for api in self.each():
            out, info = api["crop"](_img(), None, 128, 64, mode="stretch")
            self.assertEqual(out.shape[:2], (64, 128))
            out, info = api["crop"](_img(), None, 100, 100, mode="contain")
            self.assertEqual(out.shape, (100, 100, 3))
            self.assertAlmostEqual(info["scale"], 0.25)
            self.assertEqual(int(out[0, 0].sum()), 0)  # letterbox bars are black
            self.assertEqual(int(out[50, 50, 0]), 128)

    def test_native_skips_resize(self):
        for api in self.each():
            src = _img()
            out, info = api["crop"](src, None, 100, 100, mode="contain", native=True)
            self.assertIs(out, src)
            out, info = api["crop"](src, None, 100, 100, mode="center", native=True)
            self.assertEqual((out.shape[:2], info["scale"]), ((300, 300), 1.0))

    def test_face_mode_output_size_and_no_bbox_fallback(self):
        for api in self.each():
            out, _ = api["crop"](_img(), [150, 100, 250, 200], 64, 64, mode="face")
            self.assertEqual(out.shape[:2], (64, 64))
            out, _ = api["crop"](_img(), None, 64, 64, mode="face")  # center crop
            self.assertEqual(out.shape[:2], (64, 64))

    def test_face_crop_shifts_back_inside_frame(self):
        for api in self.each():
            out, info = api["crop"](_img(), [0, 0, 60, 60], 64, 64, mode="face")
            self.assertEqual(out.shape[:2], (64, 64))
            self.assertFalse(info["padded"])

    def test_max_upscale_widens_crop(self):
        for api in self.each():
            _, tight = api["crop"](_img(), [190, 140, 210, 160], 256, 256, mode="face")
            _, wide = api["crop"](_img(), [190, 140, 210, 160], 256, 256,
                                  mode="face", max_upscale=2.0)
            self.assertFalse(tight["widened"])
            self.assertTrue(wide["widened"])
            self.assertLess(wide["scale"], tight["scale"])

    def test_pad_mode_pads_when_edge_blocks_full_view(self):
        for api in self.each():
            out, info = api["crop"](_img(100, 100), [0, 0, 40, 40], 64, 64,
                                    mode="face", margin=4.0, pad_mode="black")
            self.assertTrue(info["padded"])
            self.assertEqual(out.shape[:2], (64, 64))

    def test_upscale_off_pads_at_native_resolution(self):
        for api in self.each():
            out, info = api["crop"](_img(50, 50), None, 200, 200, mode="center", upscale=False)
            self.assertEqual((out.shape[:2], info["scale"]), ((200, 200), 1.0))
            self.assertEqual(int(out[0, 0].sum()), 0)
            self.assertEqual(int(out[100, 100, 0]), 128)

    def test_tiny_output_dims_clamped(self):
        for api in self.each():
            out, _ = api["crop"](_img(), None, 1, 1, mode="stretch")
            self.assertEqual(out.shape[:2], (8, 8))

    def test_export_params_defaults_and_coercion(self):
        for api in self.each():
            p = api["params"]({})
            self.assertEqual((p["out_w"], p["out_h"], p["mode"], p["margin"]),
                             (512, 512, "contain", 2.2))
            self.assertTrue(p["upscale"])
            self.assertFalse(p["native"])
            self.assertIsNone(p["max_upscale"])
            p = api["params"]({"width": "256", "cropMode": "face", "upscale": "false",
                               "maxUpscale": "1.5", "minFacePx": "oops", "native": "yes"})
            self.assertEqual((p["out_w"], p["mode"], p["max_upscale"], p["min_face_px"]),
                             (256, "face", 1.5, 0.0))
            self.assertFalse(p["upscale"])
            self.assertTrue(p["native"])

    def test_face_export_height_px_by_mode(self):
        for api in self.each():
            img, bbox = _img(400, 300), [0, 0, 50, 100]
            base = {"out_w": 100, "out_h": 100, "margin": 2.0, "max_upscale": None,
                    "min_face_px": 0}
            self.assertIsNone(api["height_px"](img, None, dict(base, mode="face")))
            self.assertAlmostEqual(api["height_px"](img, bbox, dict(base, mode="stretch")),
                                   100 * (100 / 300))
            self.assertAlmostEqual(api["height_px"](img, bbox, dict(base, mode="contain")),
                                   100 * 0.25)
            self.assertAlmostEqual(api["height_px"](img, bbox, dict(base, mode="face")),
                                   100 * (100 / 200))
            self.assertAlmostEqual(api["height_px"](img, bbox, dict(base, mode="center")),
                                   100 * (100 / 300))

    def test_max_upscale_shrinks_face_height(self):
        for api in self.each():
            p = {"out_w": 100, "out_h": 100, "margin": 1.0, "mode": "face",
                 "min_face_px": 0}
            bbox = [0, 0, 10, 10]
            free = api["height_px"](_img(), bbox, dict(p, max_upscale=None))
            capped = api["height_px"](_img(), bbox, dict(p, max_upscale=2.0))
            self.assertLess(capped, free)

    def test_should_skip_for_small_face(self):
        for api in self.each():
            p = {"out_w": 100, "out_h": 100, "margin": 2.0, "mode": "contain",
                 "max_upscale": None, "min_face_px": 0}
            self.assertEqual(api["skip"](_img(), [0, 0, 5, 5], p), (False, None))
            p["min_face_px"] = 20
            skip, why = api["skip"](_img(), None, p)
            self.assertEqual((skip, why), (True, "no detected face"))
            skip, why = api["skip"](_img(), [0, 0, 10, 40], p)  # 40*0.25 = 10px
            self.assertTrue(skip)
            self.assertIn("below minimum", why)
            self.assertEqual(api["skip"](_img(), [0, 0, 10, 200], p), (False, None))


if __name__ == "__main__":
    unittest.main()
