
from flask import Flask, request, jsonify, Response, send_file, render_template
import psycopg2
from psycopg2 import pool as pg_pool
import requests
import os
import tempfile
import threading
import uuid
import numpy as np
import zipfile

def _pick_interp(name, scale):
    import cv2
    table = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
    }
    if name in table:
        return table[name]
    # auto: best filter for the resize direction
    return cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4

def _center_crop_to_aspect(img, out_w, out_h):
    h, w = img.shape[:2]
    target_ratio = out_w / out_h
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        new_h = max(1, int(round(w / target_ratio)))
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]

def crop_resize_export(img, bbox, out_w, out_h, mode="face", margin=2.2, interp="auto", upscale=True, max_upscale=None, pad_mode="none", native=False):
    """Crop + resize a BGR image for training-set export.
    mode: 'face' (bbox-centered, keeps the face framed), 'center' (center crop,
    ignores face), 'contain' (letterbox, no cropping), 'stretch' (naive resize).
    bbox: [x1, y1, x2, y2] in source-image pixels, or None.
    max_upscale: if the tight face crop would need more than this much
    magnification to fill out_w x out_h, the crop is widened (zoomed out to
    include more of the body/background) instead of blowing up the pixels.
    pad_mode: what to do when the widened crop hits a frame edge and the
    source frame itself is too small to deliver the full widened field of
    view: 'none' lets the final scale exceed max_upscale as a last resort,
    'black' pads with black bars, 'edge' pads by extending the border pixels.
    native: skip the final resize entirely and keep the crop at whatever
    real pixel size it comes out to. out_w/out_h still set the *aspect
    ratio* the crop is cut to for 'face'/'center' modes; 'stretch' and
    'contain' just return the source image untouched, since a fixed target
    box is meaningless once nothing is being resized to fit one. Avoids
    silently downsampling (or, worse, upsampling) source images that don't
    match your usual export size.
    Returns (image, info) where info has 'scale' (final resize factor),
    'widened' (crop was pulled back to respect max_upscale) and 'padded'
    (frame edge forced padding to avoid exceeding max_upscale).
    """
    import cv2

    h, w = img.shape[:2]
    out_w, out_h = max(8, int(out_w)), max(8, int(out_h))

    if native and mode in ("stretch", "contain"):
        return img, {"scale": 1.0, "widened": False, "padded": False}

    if mode == "stretch":
        s = out_w / max(1, w)
        interp_flag = _pick_interp(interp, s)
        return cv2.resize(img, (out_w, out_h), interpolation=interp_flag), {"scale": s, "widened": False, "padded": False}

    if mode == "contain":
        scale = min(out_w / w, out_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp_flag = _pick_interp(interp, scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)
        canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)
        oy, ox = (out_h - new_h) // 2, (out_w - new_w) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        return canvas, {"scale": scale, "widened": False, "padded": False}

    widened = False
    padded = False
    if mode == "face" and bbox:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        target_ratio = out_w / out_h
        side = max(bw, bh) * margin
        crop_h = side
        crop_w = side * target_ratio

        # If a tight crop would need heavy magnification, widen it (zoom out)
        # instead of upscaling a small patch of pixels.
        if max_upscale and max_upscale > 0:
            needed_scale = out_w / crop_w
            if needed_scale > max_upscale:
                grow = needed_scale / max_upscale
                crop_w *= grow
                crop_h *= grow
                widened = True

        x1c, y1c = cx - crop_w / 2, cy - crop_h / 2
        x2c, y2c = cx + crop_w / 2, cy + crop_h / 2
        # shift (not shrink) back into bounds where possible
        if x1c < 0:
            x2c -= x1c; x1c = 0
        if y1c < 0:
            y2c -= y1c; y1c = 0
        if x2c > w:
            x1c -= (x2c - w); x2c = w
        if y2c > h:
            y1c -= (y2c - h); y2c = h
        x1c, y1c = max(0, x1c), max(0, y1c)
        x2c, y2c = min(w, x2c), min(h, y2c)

        cw0, ch0 = x2c - x1c, y2c - y1c
        if pad_mode != "none":
            # Keep the full desired field of view: pad the gap left by the
            # frame edge rather than losing coverage or distorting aspect.
            crop = img[int(y1c):int(y2c), int(x1c):int(x2c)]
            pad_w = crop_w - cw0
            pad_h = crop_h - ch0
            if pad_w > 1 or pad_h > 1:
                left = max(0, int(round(pad_w / 2)))
                right = max(0, int(round(pad_w - left)))
                top = max(0, int(round(pad_h / 2)))
                bottom = max(0, int(round(pad_h - top)))
                border = cv2.BORDER_REPLICATE if pad_mode == "edge" else cv2.BORDER_CONSTANT
                crop = cv2.copyMakeBorder(crop, top, bottom, left, right, border, value=[0, 0, 0])
                padded = True
        else:
            # No padding: if the frame edge left a crop whose aspect no longer
            # matches the target, re-tighten it to target_ratio so the final
            # resize doesn't distort (squish/stretch) the image. This trades a
            # bit of field-of-view for correct proportions.
            if cw0 > 0 and ch0 > 0 and abs((cw0 / ch0) - target_ratio) > 1e-3:
                cur_ratio = cw0 / ch0
                cxm, cym = (x1c + x2c) / 2, (y1c + y2c) / 2
                if cur_ratio > target_ratio:
                    new_w = ch0 * target_ratio
                    x1c, x2c = cxm - new_w / 2, cxm + new_w / 2
                else:
                    new_h = cw0 / target_ratio
                    y1c, y2c = cym - new_h / 2, cym + new_h / 2
            crop = img[int(y1c):int(y2c), int(x1c):int(x2c)]
    else:
        crop = _center_crop_to_aspect(img, out_w, out_h)

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        crop = img
        ch, cw = crop.shape[:2]

    if native:
        return crop, {"scale": 1.0, "widened": widened, "padded": padded}

    scale = out_w / cw
    if not upscale and scale > 1:
        # pad at native resolution instead of upscaling past 1:1
        canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)
        oy, ox = max(0, (out_h - ch) // 2), max(0, (out_w - cw) // 2)
        ch2, cw2 = min(ch, out_h), min(cw, out_w)
        canvas[oy:oy + ch2, ox:ox + cw2] = crop[:ch2, :cw2]
        return canvas, {"scale": 1.0, "widened": widened, "padded": padded}

    interp_flag = _pick_interp(interp, scale)
    return cv2.resize(crop, (out_w, out_h), interpolation=interp_flag), {"scale": scale, "widened": widened, "padded": padded}

def _as_bool(value, default=False):
    """Coerces a value that may be a real bool (JSON body) or a string
    (query string, e.g. the preview endpoints) into a proper bool. Plain
    bool(...) silently breaks on query strings since bool('false') is
    True - any non-empty string is truthy."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")

def _export_params_from_body(body):
    max_upscale = body.get("maxUpscale")
    min_face_px = body.get("minFacePx")
    try:
        max_upscale = float(max_upscale) if max_upscale not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        max_upscale = None
    try:
        min_face_px = float(min_face_px) if min_face_px not in (None, "", 0, "0") else 0.0
    except (TypeError, ValueError):
        min_face_px = 0.0
    return {
        "out_w": int(body.get("width") or 512),
        "out_h": int(body.get("height") or 512),
        "mode": body.get("cropMode") or "contain",
        "margin": float(body.get("margin") or 2.2),
        "interp": body.get("interp") or "auto",
        "upscale": _as_bool(body.get("upscale"), True),
        "max_upscale": max_upscale,
        "pad_mode": body.get("padMode") or "none",
        "min_face_px": min_face_px,
        "native": _as_bool(body.get("native"), False),
    }

def face_export_height_px(img, bbox, p):
    if not bbox:
        return None

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bh = max(0, y2 - y1)
    if p["mode"] == "stretch":
        return bh * (p["out_h"] / max(1, h))
    if p["mode"] == "contain":
        return bh * min(p["out_w"] / max(1, w), p["out_h"] / max(1, h))
    if p["mode"] == "face":
        side = max(x2 - x1, bh) * p["margin"]
        if p["max_upscale"] and p["max_upscale"] > 0:
            target_ratio = p["out_w"] / p["out_h"]
            crop_w = side * target_ratio
            needed_scale = p["out_w"] / max(1, crop_w)
            if needed_scale > p["max_upscale"]:
                side *= needed_scale / p["max_upscale"]
        return bh * (p["out_h"] / max(1, side))

    return bh * (p["out_h"] / max(1, h))

def should_skip_for_small_face(img, bbox, p):
    if p["min_face_px"] <= 0:
        return False, None
    height_px = face_export_height_px(img, bbox, p)
    if height_px is None:
        return True, "no detected face"
    if height_px < p["min_face_px"]:
        return True, f"face {height_px:.1f}px below minimum {p['min_face_px']:.1f}px"
    return False, None

