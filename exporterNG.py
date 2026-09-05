"""NG twin of exporter.py -- exports a list of Immich asset ids straight
to disk (crop/resize applied per Export Settings), independent of any
analysis job. NG-only file, wired to configNG/detectionNG/image_opsNG
instead of the originals, per the NG duplication rule in
APP_ARCHITECTURE_NOTES.md.
"""

import os

import numpy as np
import requests

from configNG import IMMICH_API_KEY, IMMICH_BASE_URL
from detectionNG import get_face_app_ng, pick_largest_face_ng
from image_opsNG import crop_resize_export_ng, should_skip_for_small_face_ng


def export_immich_asset_ids_ng(asset_ids, dest_dir, p):
    import cv2

    face_app = get_face_app_ng() if p["mode"] == "face" or p["min_face_px"] > 0 else None
    saved = []
    errors = []
    widened_count = 0
    padded_count = 0

    for asset_id in asset_ids:
        try:
            meta = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}",
                headers={"x-api-key": IMMICH_API_KEY},
                timeout=20,
            ).json()
            orig_name = meta.get("originalFileName", f"{asset_id}.jpg")

            r = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
                headers={"x-api-key": IMMICH_API_KEY},
                stream=True,
                timeout=60,
            )
            if r.status_code != 200:
                errors.append(f"{asset_id}: HTTP {r.status_code}")
                continue

            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"{asset_id}: could not decode image")
                continue

            bbox = None
            if face_app is not None:
                try:
                    faces = face_app.get(img)
                    if faces:
                        bbox = list(map(int, pick_largest_face_ng(faces).bbox))
                except Exception:
                    bbox = None

            skip, reason = should_skip_for_small_face_ng(img, bbox, p)
            if skip:
                errors.append(f"{asset_id}: skipped ({reason})")
                continue

            out_img, info = crop_resize_export_ng(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"], p["native"])
            if info.get("widened"):
                widened_count += 1
            if info.get("padded"):
                padded_count += 1
            dest_name = f"immich_{os.path.splitext(orig_name)[0]}_{asset_id[:8]}.png"
            dest_path = os.path.join(dest_dir, dest_name)
            cv2.imwrite(dest_path, out_img)
            saved.append(dest_name)
        except Exception as e:
            errors.append(f"{asset_id}: {e}")

    return {
        "saved": saved,
        "errors": errors,
        "widened": widened_count,
        "padded": padded_count,
    }
