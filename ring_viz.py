#!/usr/bin/env python3
"""
Immich Ring Visualizer
-----------------------
A small local tool: pick a reference face, see nearest matches arranged
in confidence rings around it. Click any thumbnail to recenter.

Run:
    pip install flask psycopg2-binary requests --break-system-packages
    python3 ring_viz.py

Then open http://localhost:5050
"""

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
import shutil

# ---- CONFIG: edit these to match your setup ----
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"          # from your .env DB_PASSWORD
PG_DB = "immich"

IMMICH_BASE_URL = "http://localhost:2283"
IMMICH_API_KEY = "L4mP37A5kNWHPME0024ms2SGep7KR8xP4oAB9UNGqOM"
# --------------------------------------------------

app = Flask(__name__)

# ---- connection pool (avoids opening/closing a new connection per request) ----
db_pool = None
try:
    db_pool = pg_pool.SimpleConnectionPool(
        1, 10, host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )
except Exception as e:
    print(f"Warning: could not initialize DB pool yet ({e}). Will retry on first request.")


def get_conn():
    global db_pool
    if db_pool is None:
        db_pool = pg_pool.SimpleConnectionPool(
            1, 10, host=PG_HOST, port=PG_PORT, user=PG_USER,
            password=PG_PASSWORD, dbname=PG_DB
        )
    return db_pool.getconn()


def release_conn(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

# ---- video-clip analysis (independent of Immich, uses InsightFace directly) ----
FRAME_STORE = tempfile.mkdtemp(prefix="ringviz_frames_")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
_face_app = None
_analysis_jobs = {}  # jobId -> {"status": ..., "results": [...], "video": ..., "anchor": ...}
_preview_jobs = {}  # previewId -> {"video": path, "fps": float, "frames": int}


def get_face_app():
    """Lazy-load InsightFace, preferring Apple Silicon CoreML acceleration."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        _face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def pick_best_face(faces, ref_embedding):
    if len(faces) == 1:
        return faces[0]
    return max(faces, key=lambda f: float(np.dot(ref_embedding, f.normed_embedding)))


def get_blur_score(crop):
    import cv2
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


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


def crop_resize_export(img, bbox, out_w, out_h, mode="face", margin=2.2, interp="auto", upscale=True, max_upscale=None, pad_mode="none"):
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
    Returns (image, info) where info has 'scale' (final resize factor),
    'widened' (crop was pulled back to respect max_upscale) and 'padded'
    (frame edge forced padding to avoid exceeding max_upscale).
    """
    import cv2

    h, w = img.shape[:2]
    out_w, out_h = max(8, int(out_w)), max(8, int(out_h))

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
        "upscale": bool(body.get("upscale", True)),
        "max_upscale": max_upscale,
        "pad_mode": body.get("padMode") or "none",
        "min_face_px": min_face_px,
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


def export_immich_asset_ids(asset_ids, dest_dir, p):
    import cv2

    face_app = get_face_app() if p["mode"] == "face" or p["min_face_px"] > 0 else None
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
                        bbox = list(map(int, pick_largest_face(faces).bbox))
                except Exception:
                    bbox = None

            skip, reason = should_skip_for_small_face(img, bbox, p)
            if skip:
                errors.append(f"{asset_id}: skipped ({reason})")
                continue

            out_img, info = crop_resize_export(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"])
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


def run_video_analysis(job_id, video_path, anchor_path, sim_threshold, blur_threshold, ref_frame_idx=1):
    import cv2

    job = _analysis_jobs[job_id]
    try:
        face_app = get_face_app()

        cap = cv2.VideoCapture(video_path)
        
        target_frame = max(1, ref_frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        
        ret, ref_frame = cap.read()
        if not ret:
            job["status"] = "error"
            job["error"] = f"Could not read reference frame {target_frame} from video"
            cap.release()
            return

        cv2.imwrite(anchor_path, ref_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference frame {target_frame}"
            cap.release()
            return
        ref_embedding = ref_faces[0].normed_embedding

        cap.release()
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            faces = face_app.get(frame)
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None
                })
                job["results"] = results
                continue

            face = pick_best_face(faces, ref_embedding)
            sim_score = float(np.dot(ref_embedding, face.normed_embedding))

            # InsightFace face.pose is [pitch, yaw, roll] in degrees.
            pitch, yaw, roll = (float(p) for p in face.pose)

            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            blur_score = get_blur_score(crop)

            fail_reason = None
            if not (sim_score > sim_threshold):
                fail_reason = "sim"
            elif not (blur_score > blur_threshold):
                fail_reason = "blur"
            
            passed = (fail_reason is None)
            frame_id = f"{job_id}_{frame_idx:05d}"

            if passed:
                out_path = os.path.join(FRAME_STORE, f"{frame_id}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                job.setdefault("frame_embeddings", {})[frame_idx] = face.normed_embedding.tolist()

            results.append({
                "frame": frame_idx, "sim": sim_score, "blur": blur_score,
                "passed": passed, "failReason": fail_reason, "hasFace": True,
                "frameId": frame_id if passed else None,
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "bbox": [x1, y1, x2, y2]
            })
            job["results"] = results

        cap.release()
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_folder_analysis(job_id, image_paths, anchor_path, sim_threshold, blur_threshold, ref_index=1):
    """Same pipeline as run_video_analysis, but the 'frames' are a set of
    still images from a folder/zip upload instead of decoded video frames.
    image_paths is a pre-sorted list; frame numbering follows that order so
    the rest of the app (ring, pose strip, export, cross-check) can treat
    this exactly like a video analysis job with zero changes."""
    import cv2

    job = _analysis_jobs[job_id]
    try:
        face_app = get_face_app()

        ref_idx = max(1, min(ref_index, len(image_paths))) - 1
        ref_frame = cv2.imread(image_paths[ref_idx])
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference image {os.path.basename(image_paths[ref_idx])}"
            return

        cv2.imwrite(anchor_path, ref_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference image {os.path.basename(image_paths[ref_idx])}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []
        for i, img_path in enumerate(image_paths):
            frame_idx = i + 1
            orig_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name
                })
                job["results"] = results
                continue

            faces = face_app.get(frame)
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name
                })
                job["results"] = results
                continue

            face = pick_best_face(faces, ref_embedding)
            sim_score = float(np.dot(ref_embedding, face.normed_embedding))

            # InsightFace face.pose is [pitch, yaw, roll] in degrees.
            pitch, yaw, roll = (float(p) for p in face.pose)

            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            blur_score = get_blur_score(crop)

            fail_reason = None
            if not (sim_score > sim_threshold):
                fail_reason = "sim"
            elif not (blur_score > blur_threshold):
                fail_reason = "blur"

            passed = (fail_reason is None)
            frame_id = f"{job_id}_{frame_idx:05d}"

            if passed:
                out_path = os.path.join(FRAME_STORE, f"{frame_id}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                job.setdefault("frame_embeddings", {})[frame_idx] = face.normed_embedding.tolist()

            results.append({
                "frame": frame_idx, "sim": sim_score, "blur": blur_score,
                "passed": passed, "failReason": fail_reason, "hasFace": True,
                "frameId": frame_id if passed else None,
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "bbox": [x1, y1, x2, y2], "origName": orig_name
            })
            job["results"] = results

        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/api/analyze-folder", methods=["POST"])
def analyze_folder():
    """Accepts either multiple image files (folder picker / multi-select)
    under the 'images' field, or a single 'zip' file containing images.
    Runs the exact same analysis pipeline as video, just over stills."""
    sim_threshold = float(request.form.get("simThreshold", 0.65))
    blur_threshold = float(request.form.get("blurThreshold", 50))
    ref_index = int(request.form.get("refIndex", 1))

    job_id = uuid.uuid4().hex[:12]
    src_dir = os.path.join(FRAME_STORE, f"{job_id}_srcimgs")
    os.makedirs(src_dir, exist_ok=True)
    anchor_path = os.path.join(FRAME_STORE, f"{job_id}_anchor.jpg")

    saved_paths = []

    zip_file = request.files.get("zip")
    if zip_file and zip_file.filename:
        zip_path = os.path.join(src_dir, "_upload.zip")
        zip_file.save(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    ext = os.path.splitext(member)[1].lower()
                    if ext not in IMAGE_EXTS or member.startswith("__MACOSX"):
                        continue
                    # flatten any subfolder structure from the zip
                    safe_name = os.path.basename(member)
                    if not safe_name:
                        continue
                    out_path = os.path.join(src_dir, safe_name)
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    saved_paths.append(out_path)
        except zipfile.BadZipFile:
            shutil.rmtree(src_dir, ignore_errors=True)
            return jsonify({"error": "uploaded file is not a valid zip"}), 400
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        source_name = os.path.splitext(zip_file.filename)[0]
    else:
        images = request.files.getlist("images")
        if not images:
            shutil.rmtree(src_dir, ignore_errors=True)
            return jsonify({"error": "provide 'images' (multiple files) or a 'zip' file"}), 400
        for f in images:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            safe_name = os.path.basename(f.filename)
            out_path = os.path.join(src_dir, safe_name)
            f.save(out_path)
            saved_paths.append(out_path)
        source_name = request.form.get("sourceName") or "folder_set"

    if not saved_paths:
        shutil.rmtree(src_dir, ignore_errors=True)
        return jsonify({"error": "no valid images found (jpg/jpeg/png/webp/bmp)"}), 400

    saved_paths.sort(key=lambda p: os.path.basename(p).lower())

    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "sourceType": "folder",
        "srcDir": src_dir,
        "simThreshold": sim_threshold,
    }

    t = threading.Thread(
        target=run_folder_analysis,
        args=(job_id, saved_paths, anchor_path, sim_threshold, blur_threshold, ref_index),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(saved_paths)})


@app.route("/api/analyze-immich", methods=["POST"])
def analyze_immich():
    """Same analysis pipeline as analyze-folder (pose extraction, sim/blur
    gating), but sourced from Immich asset IDs already known to the app -
    a selection, a person cluster, or cross-check matches - instead of
    files uploaded from the local filesystem. Saves the round trip of
    downloading images out of Immich by hand just to re-upload them."""
    body = request.get_json(force=True) or {}
    asset_ids = body.get("assetIds") or []
    sim_threshold = float(body.get("simThreshold", 0.65))
    blur_threshold = float(body.get("blurThreshold", 50))
    ref_index = int(body.get("refIndex", 1))

    if not asset_ids:
        return jsonify({"error": "provide 'assetIds' (non-empty list)"}), 400

    job_id = uuid.uuid4().hex[:12]
    src_dir = os.path.join(FRAME_STORE, f"{job_id}_srcimgs")
    os.makedirs(src_dir, exist_ok=True)
    anchor_path = os.path.join(FRAME_STORE, f"{job_id}_anchor.jpg")

    saved_paths = []
    fetch_errors = []
    for asset_id in asset_ids:
        try:
            meta = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}",
                headers={"x-api-key": IMMICH_API_KEY},
                timeout=20,
            ).json()
            orig_name = meta.get("originalFileName") or f"{asset_id}.jpg"
            ext = os.path.splitext(orig_name)[1].lower()
            if ext not in IMAGE_EXTS:
                ext = ".jpg"

            r = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
                headers={"x-api-key": IMMICH_API_KEY},
                stream=True,
                timeout=60,
            )
            if r.status_code != 200:
                fetch_errors.append(f"{asset_id}: HTTP {r.status_code}")
                continue

            # prefix with the asset id so filenames can't collide across
            # assets and so the exported frame stays traceable back to the
            # Immich library item it came from
            safe_name = f"{asset_id}_{os.path.basename(orig_name)}"
            out_path = os.path.join(src_dir, safe_name)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            saved_paths.append(out_path)
        except Exception as e:
            fetch_errors.append(f"{asset_id}: {e}")

    if not saved_paths:
        shutil.rmtree(src_dir, ignore_errors=True)
        return jsonify({"error": "could not fetch any of the requested Immich assets", "fetchErrors": fetch_errors}), 400

    saved_paths.sort(key=lambda p: os.path.basename(p).lower())

    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": f"immich_selection_{len(saved_paths)}", "sourceType": "immich",
        "srcDir": src_dir,
        "simThreshold": sim_threshold,
    }

    t = threading.Thread(
        target=run_folder_analysis,
        args=(job_id, saved_paths, anchor_path, sim_threshold, blur_threshold, ref_index),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(saved_paths), "fetchErrors": fetch_errors})


@app.route("/api/preview-video", methods=["POST"])
def preview_video():
    """Prepare a video for exact frame-by-frame preview."""
    import cv2
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    preview_id = uuid.uuid4().hex[:12]
    video_path = os.path.join(FRAME_STORE, f"{preview_id}_preview.mp4")
    request.files["video"].save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 400

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0 or total_frames <= 0:
        return jsonify({"error": "Could not determine video FPS/frame count"}), 400

    _preview_jobs[preview_id] = {
        "video": video_path,
        "fps": fps,
        "frames": total_frames,
    }

    return jsonify({
        "previewId": preview_id,
        "fps": fps,
        "totalFrames": total_frames,
        "duration": total_frames / fps,
    })


@app.route("/api/preview-frame/<preview_id>/<int:frame_no>")
def preview_frame(preview_id, frame_no):
    """Return one exact decoded video frame, 1-based."""
    import cv2

    job = _preview_jobs.get(preview_id)
    if not job:
        return "", 404

    frame_no = max(1, min(job["frames"], frame_no))
    cap = cv2.VideoCapture(job["video"])
    if not cap.isOpened():
        return "", 404

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "", 404

    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "", 500

    return Response(encoded.tobytes(), mimetype="image/jpeg")


@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    sim_threshold = float(request.form.get("simThreshold", 0.65))
    blur_threshold = float(request.form.get("blurThreshold", 100))
    ref_frame = int(request.form.get("refFrame", 1))

    job_id = uuid.uuid4().hex[:12]
    video_path = os.path.join(FRAME_STORE, f"{job_id}_source.mp4")
    anchor_path = os.path.join(FRAME_STORE, f"{job_id}_anchor.jpg")
    request.files["video"].save(video_path)

    source_name = os.path.splitext(request.files["video"].filename or "clip")[0]
    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "videoPath": video_path,
        "simThreshold": sim_threshold,
    }

    t = threading.Thread(
        target=run_video_analysis,
        args=(job_id, video_path, anchor_path, sim_threshold, blur_threshold, ref_frame),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id})


@app.route("/api/analysis-status/<job_id>")
def analysis_status(job_id):
    job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "error": job.get("error"),
        "frameCount": len(job["results"]),
        "results": job["results"],
        "simThreshold": job.get("simThreshold", 0.65),
    })


@app.route("/api/framefile/<frame_id>")
def frame_file(frame_id):
    path = os.path.join(FRAME_STORE, f"{frame_id}.jpg")
    if not os.path.exists(path):
        return "", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/export-job/<job_id>", methods=["POST"])
def export_job(job_id):
    import cv2

    job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    source_name = job.get("sourceName", job_id)
    dest_dir = os.path.join(EXPORT_DIR, source_name)
    os.makedirs(dest_dir, exist_ok=True)

    body = request.get_json(silent=True) or {}
    selected_frames = body.get("frames")
    selected_set = set(selected_frames) if "frames" in body else None
    selected_asset_ids = body.get("assetIds") or []
    p = _export_params_from_body(body)

    face_app = None
    saved = []
    errors = []
    skipped_count = 0
    widened_count = 0
    padded_count = 0
    for r in job["results"]:
        if not (r.get("passed") and r.get("frameId")):
            continue
        if selected_set is not None and r["frame"] not in selected_set:
            continue
        src = os.path.join(FRAME_STORE, f"{r['frameId']}.jpg")
        if not os.path.exists(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue

        bbox = r.get("bbox")
        if (p["mode"] == "face" or p["min_face_px"] > 0) and not bbox:
            # legacy job predating stored bbox: detect on demand
            try:
                if face_app is None:
                    face_app = get_face_app()
                faces = face_app.get(img)
                if faces:
                    ref = job.get("frame_embeddings", {}).get(r["frame"])
                    face = pick_best_face(faces, np.array(ref, dtype=np.float32)) if ref else pick_largest_face(faces)
                    bbox = list(map(int, face.bbox))
            except Exception:
                bbox = None

        skip, reason = should_skip_for_small_face(img, bbox, p)
        if skip:
            skipped_count += 1
            errors.append(f"frame {r['frame']}: skipped ({reason})")
            continue

        out_img, info = crop_resize_export(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"])
        if info.get("widened"):
            widened_count += 1
        if info.get("padded"):
            padded_count += 1
        dest_name = f"fr{r['frame']:05d}_sim{r['sim']:.2f}.png"
        dest_path = os.path.join(dest_dir, dest_name)
        cv2.imwrite(dest_path, out_img)
        saved.append(dest_name)

    immich_result = {"saved": [], "errors": [], "widened": 0, "padded": 0}
    if selected_asset_ids:
        immich_result = export_immich_asset_ids(selected_asset_ids, dest_dir, p)
        widened_count += immich_result["widened"]
        padded_count += immich_result["padded"]

    all_saved = saved + immich_result["saved"]
    return jsonify({
        "exported": len(all_saved),
        "frameExported": len(saved),
        "immichExported": len(immich_result["saved"]),
        "path": dest_dir,
        "files": all_saved,
        "errors": errors + immich_result["errors"],
        "skipped": skipped_count + len([e for e in immich_result["errors"] if ": skipped (" in e]),
        "widened": widened_count,
        "padded": padded_count,
    })


@app.route("/api/build-playback/<job_id>", methods=["POST"])
def build_playback(job_id):
    import cv2

    job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": "analysis not finished yet"}), 400

    video_path = job.get("videoPath")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "source video no longer available"}), 400

    by_frame = {r["frame"]: r for r in job["results"]}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_path = os.path.join(FRAME_STORE, f"{job_id}_playback_raw.mp4")
    out_path = os.path.join(FRAME_STORE, f"{job_id}_playback.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    frame_idx = 0
    blank = np.zeros((height, width, 3), dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        r = by_frame.get(frame_idx)

        if r and r.get("passed"):
            writer.write(frame)
        else:
            canvas = blank.copy()
            reason = "NO FACE"
            if r:
                reason = "BLURRY" if r.get("failReason") == "blur" else "LOW MATCH" if r.get("failReason") == "sim" else "NO FACE"
            label = f"{reason}  (frame {frame_idx})"
            cv2.putText(canvas, label, (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 60, 200), 2, cv2.LINE_AA)
            writer.write(canvas)

    cap.release()
    writer.release()

    import subprocess
    import shutil as _shutil
    ffmpeg_bin = _shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", raw_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out_path],
                check=True, capture_output=True
            )
            os.remove(raw_path)
        except subprocess.CalledProcessError as e:
            job["playbackPath"] = raw_path
            return jsonify({
                "error": f"ffmpeg transcode failed: {e.stderr.decode(errors='ignore')[-400:]}"
            }), 500
    else:
        out_path = raw_path

    job["playbackPath"] = out_path
    return jsonify({"ready": True, "url": f"/api/playback-file/{job_id}", "fps": fps})


@app.route("/api/playback-file/<job_id>")
def playback_file(job_id):
    job = _analysis_jobs.get(job_id)
    if not job or not job.get("playbackPath") or not os.path.exists(job["playbackPath"]):
        return "", 404
    return send_file(job["playbackPath"], mimetype="video/mp4")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/random-face")
def random_face():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.id, a."originalFileName"
            FROM asset_face af
            JOIN asset a ON a.id = af."assetId"
            WHERE af."personId" IS NOT NULL
            ORDER BY random()
            LIMIT 1;
        """)
        row = cur.fetchone()
        cur.close()
        if not row:
            return jsonify({"error": "no faces found"}), 404
        return jsonify({"assetId": row[0], "filename": row[1]})
    finally:
        release_conn(conn)


@app.route("/api/find-by-filename")
def find_by_filename():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, "originalFileName"
            FROM asset
            WHERE "originalFileName" ILIKE %s
            ORDER BY LENGTH("originalFileName") ASC
            LIMIT 8;
        """, (f"%{name}%",))
        rows = cur.fetchall()
        cur.close()
        return jsonify([{"assetId": r[0], "filename": r[1]} for r in rows])
    finally:
        release_conn(conn)


@app.route("/api/asset-face-pose/<asset_id>")
def asset_face_pose(asset_id):
    """Return yaw/pitch/roll for the largest face in an Immich asset preview."""
    import cv2

    try:
        r = requests.get(
            f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
            headers={"x-api-key": IMMICH_API_KEY},
            params={"size": "preview"},
            timeout=15,
        )
        if r.status_code != 200:
            return jsonify({"error": f"Immich thumbnail HTTP {r.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    arr = np.frombuffer(r.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode Immich thumbnail"}), 500

    try:
        faces = get_face_app().get(frame)
    except Exception as e:
        return jsonify({"error": f"face detection failed: {e}"}), 500

    if not faces:
        return jsonify({"error": "no face detected in Immich thumbnail"}), 404

    face = pick_largest_face(faces)
    pitch, yaw, roll = (float(p) for p in face.pose)
    return jsonify({"assetId": asset_id, "yaw": yaw, "pitch": pitch, "roll": roll})


@app.route("/api/neighbors")
def neighbors():
    asset_id = request.args.get("assetId")
    if not asset_id:
        return jsonify({"error": "assetId required"}), 400

    limit = int(request.args.get("limit", 30))

    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT a.id, a."originalFileName", 1 - (fs.embedding <=> ref.embedding) AS similarity
            FROM face_search fs
            JOIN asset_face af ON af.id = fs."faceId"
            JOIN asset a ON a.id = af."assetId"
            CROSS JOIN (
                SELECT fs2.embedding FROM face_search fs2
                JOIN asset_face af2 ON af2.id = fs2."faceId"
                WHERE af2."assetId" = %s
                LIMIT 1
            ) ref
            ORDER BY fs.embedding <=> ref.embedding
            LIMIT %s;
        """, (asset_id, limit))
        rows = cur.fetchall()
        mode = "face"

        if not rows:
            cur.execute("""
                SELECT a.id, a."originalFileName", 1 - (s.embedding <=> ref.embedding) AS similarity
                FROM smart_search s
                JOIN asset a ON a.id = s."assetId"
                CROSS JOIN (SELECT embedding FROM smart_search WHERE "assetId" = %s) ref
                ORDER BY s.embedding <=> ref.embedding
                LIMIT %s;
            """, (asset_id, limit))
            rows = cur.fetchall()
            mode = "clip"

        cur.close()
    finally:
        release_conn(conn)

    results = [
        {"assetId": r[0], "filename": r[1], "similarity": float(r[2])}
        for r in rows
    ]
    return jsonify({"mode": mode, "results": results})


@app.route("/api/immich-cross-check/<job_id>/<int:frame_no>")
def immich_cross_check(job_id, frame_no):
    job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    embedding = job.get("frame_embeddings", {}).get(frame_no)
    if embedding is None:
        return jsonify({"error": "no stored embedding for that frame (it may not have passed the sim/blur gates)"}), 404

    limit = int(request.args.get("limit", 12))
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.id, a."originalFileName",
                   1 - (fs.embedding <=> %s::vector) AS similarity
            FROM face_search fs
            JOIN asset_face af ON af.id = fs."faceId"
            JOIN asset a ON a.id = af."assetId"
            ORDER BY fs.embedding <=> %s::vector
            LIMIT %s;
        """, (vec_literal, vec_literal, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_conn(conn)

    results = [
        {"assetId": r[0], "filename": r[1], "similarity": float(r[2])}
        for r in rows
    ]
    return jsonify({"frame": frame_no, "results": results})


@app.route("/api/immich-face-pose/<job_id>/<int:frame_no>/<asset_id>")
def immich_face_pose(job_id, frame_no, asset_id):
    """Lazily compute yaw/pitch/roll for one Immich match, only called when
    the match is actually added to the ring (not for every cross-check row)."""
    import cv2

    job = _analysis_jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    ref_embedding = job.get("frame_embeddings", {}).get(frame_no)

    try:
        r = requests.get(
            f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
            headers={"x-api-key": IMMICH_API_KEY},
            params={"size": "preview"},
            timeout=15,
        )
        if r.status_code != 200:
            return jsonify({"error": f"Immich thumbnail HTTP {r.status_code}"}), 502
        img_bytes = r.content
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode Immich thumbnail"}), 500

    try:
        face_app = get_face_app()
        faces = face_app.get(frame)
    except Exception as e:
        return jsonify({"error": f"face detection failed: {e}"}), 500

    if not faces:
        return jsonify({"error": "no face detected in Immich thumbnail"}), 404

    if ref_embedding is not None:
        face = pick_best_face(faces, np.array(ref_embedding, dtype=np.float32))
    else:
        face = faces[0]

    pitch, yaw, roll = (float(p) for p in face.pose)
    return jsonify({"assetId": asset_id, "yaw": yaw, "pitch": pitch, "roll": roll})


@app.route("/api/person-clusters")
def person_clusters():
    """Rank named Immich persons by how tightly their faces cluster in
    embedding space. High avg_sim usually means either very consistent
    real-world photos of that person, or a pile of near-duplicate stills
    (e.g. screenshotted from video) — worth eyeballing before using as a
    LoRA source. No video job or upload required, pure Immich DB query."""
    min_faces = int(request.args.get("minFaces", 5))
    limit = int(request.args.get("limit", 30))

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT af."personId", p.name,
                   COUNT(*) AS face_count,
                   AVG(1 - (fs.embedding <=> centroid.emb)) AS avg_sim
            FROM asset_face af
            JOIN face_search fs ON fs."faceId" = af.id
            JOIN person p ON p.id = af."personId"
            CROSS JOIN LATERAL (
                SELECT AVG(fs2.embedding) AS emb
                FROM face_search fs2
                JOIN asset_face af2 ON af2.id = fs2."faceId"
                WHERE af2."personId" = af."personId"
            ) centroid
            WHERE af."personId" IS NOT NULL
            GROUP BY af."personId", p.name
            HAVING COUNT(*) >= %s
            ORDER BY avg_sim DESC
            LIMIT %s;
        """, (min_faces, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_conn(conn)

    return jsonify([
        {
            "personId": r[0],
            "name": r[1] or "(unnamed)",
            "faceCount": r[2],
            "avgSim": float(r[3]),
        }
        for r in rows
    ])


@app.route("/api/person-assets/<person_id>")
def person_assets(person_id):
    """All (or up to `limit`) assets Immich has tagged for a given personId.
    Used to populate the thumbnail grid when a person-cluster row is opened,
    and as the source list for building a selection to export."""
    limit = int(request.args.get("limit", 200))

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.id, a."originalFileName"
            FROM asset_face af
            JOIN asset a ON a.id = af."assetId"
            WHERE af."personId" = %s
            LIMIT %s;
        """, (person_id, limit))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_conn(conn)

    return jsonify([{"assetId": r[0], "filename": r[1]} for r in rows])


@app.route("/api/export-immich-assets", methods=["POST"])
def export_immich_assets():
    body = request.get_json(silent=True) or {}
    asset_ids = body.get("assetIds") or []
    if not asset_ids:
        return jsonify({"error": "no assetIds given"}), 400
    p = _export_params_from_body(body)

    dest_dir = os.path.join(EXPORT_DIR, "immich_selected")
    os.makedirs(dest_dir, exist_ok=True)

    result = export_immich_asset_ids(asset_ids, dest_dir, p)
    return jsonify({
        "exported": len(result["saved"]),
        "path": dest_dir,
        "files": result["saved"],
        "errors": result["errors"],
        "widened": result["widened"],
        "padded": result["padded"],
    })


@app.route("/api/thumb/<asset_id>")
def thumb(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "thumbnail"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))


@app.route("/api/preview/<asset_id>")
def preview_size(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "preview"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
