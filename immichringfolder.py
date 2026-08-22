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

from flask import Flask, request, jsonify, Response, send_file
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
    return Response(HTML, mimetype="text/html")


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


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ring Visualizer</title>
<style>
  :root {
    --bg: #0b0b0e;
    --ring-line: #232329;
    --text: #e8e8ec;
    --dim: #6b6b76;
    --accent: #7cc4ff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    overflow: hidden;
  }
  #preview-hover-panel {
    display: none;
    position: fixed;
    top: 16px;
    right: 356px;
    z-index: 30;
    width: 340px;
    height: 355px;
    background: #101015;
    border: 1px solid #2a2a32;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    pointer-events: none;
  }
  #preview-hover-panel.active { display: block; }
  #preview-hover-panel img {
    width: 100%;
    height: calc(100% - 38px);
    object-fit: contain;
    border-radius: 6px;
    background: #050507;
  }
  #preview-hover-panel .preview-caption {
    margin-top: 4px;
    font-size: 11px;
    color: var(--dim);
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
  }
  #left-panel {
    position: fixed;
    top: 16px;
    left: 16px;
    bottom: 16px;
    z-index: 10;
    font-size: 13px;
    color: var(--dim);
    background: rgba(10, 10, 14, 0.72);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid #2a2a32;
    border-radius: 10px;
    width: 340px;
    max-width: calc(100vw - 32px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #left-panel.collapsed-all {
    width: 44px !important;
  }
  #left-panel-content {
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 0;
    flex: 1;
  }
  #left-panel.collapsed-all #left-panel-content {
    display: none;
  }
  #left-controls-pane,
  #left-preview-pane {
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  #left-controls-pane {
    flex: 0 0 46%;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 2px;
  }
  #left-preview-pane {
    flex: 1 1 auto;
    overflow: hidden;
  }
  #left-splitter {
    height: 9px;
    flex: 0 0 9px;
    cursor: ns-resize;
    position: relative;
    margin: -2px 0;
  }
  #left-splitter::before {
    content: "";
    position: absolute;
    left: 36%;
    right: 36%;
    top: 4px;
    height: 1px;
    background: #32323e;
  }
  #left-splitter:hover::before,
  #left-splitter.dragging::before {
    background: var(--accent);
  }
  #left-resize-handle {
    position: absolute;
    top: 0;
    right: -4px;
    width: 8px;
    height: 100%;
    cursor: ew-resize;
    z-index: 11;
  }
  #left-resize-handle:hover, #left-resize-handle.dragging {
    background: rgba(124,196,255,0.25);
  }
  #left-panel-collapse-all {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #16161c;
    border: 1px solid #2a2a32;
    border-radius: 5px;
    color: var(--dim);
    cursor: pointer;
    font-size: 11px;
    z-index: 12;
  }
  #left-panel-collapse-all:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  .panel-section {
    border: 1px solid #212127;
    border-radius: 8px;
    background: rgba(255,255,255,0.02);
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex-shrink: 0;
  }
  .panel-section.expanded.grow {
    flex: 1;
    min-height: 160px;
  }
  #left-preview-pane .panel-section {
    flex: 1;
  }
  #left-preview-pane .panel-section.expanded {
    min-height: 0;
  }
  .panel-section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    cursor: pointer;
    font-size: 11px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--dim);
    user-select: none;
    flex-shrink: 0;
  }
  .panel-section-header:hover {
    color: var(--text);
  }
  .panel-section-header .chev {
    display: inline-block;
    transition: transform 0.15s ease;
    font-size: 10px;
    width: 10px;
  }
  .panel-section.expanded .panel-section-header .chev {
    transform: rotate(90deg);
  }
  .panel-section-header .sec-title {
    flex: 1;
  }
  .panel-section-body {
    display: none;
    padding: 0 10px 10px 10px;
    min-height: 0;
    flex: 1;
    flex-direction: column;
  }
  .panel-section.expanded .panel-section-body {
    display: flex;
  }
  #left-panel .fname {
    color: var(--text);
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 2px;
  }
  #left-panel .mode {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: #1a1a20;
    border: 1px solid #2a2a32;
    font-size: 11px;
    letter-spacing: 0.03em;
    margin-top: 6px;
  }
  #search-box {
    margin-top: 10px;
    position: relative;
    width: 260px;
  }
  #search-input {
    width: 100%;
    background: #16161c;
    border: 1px solid #2a2a32;
    color: var(--text);
    font-size: 12px;
    padding: 7px 10px;
    border-radius: 6px;
    outline: none;
    font-family: inherit;
  }
  #search-input:focus {
    border-color: var(--accent);
  }
  #search-results {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    width: 100%;
    background: #16161c;
    border: 1px solid #2a2a32;
    border-radius: 6px;
    overflow: hidden;
    display: none;
    max-height: 240px;
    overflow-y: auto;
  }
  #search-results.open { display: block; }
  .search-result-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 12px;
  }
  .search-result-row:hover { background: #1f1f27; }
  .search-result-row img {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    object-fit: cover;
    background: #111;
  }
  #video-drop {
    margin-top: 10px;
    width: 260px;
  }
  #video-drop-zone {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    padding: 6px 10px;
    border: 1px solid #2a4a6a;
    border-radius: 6px;
    background: #1a2a3a;
    color: var(--accent);
    font-size: 11px;
    cursor: pointer;
  }
  #video-drop-zone > div:last-child {
    display: none;
  }
  #video-drop-zone.dragover {
    border-color: var(--accent);
    background: #22384d;
    color: var(--text);
  }
  #preview-container {
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
    overflow: hidden;
  }
  #preview-canvas {
    width: 100%;
    height: auto;
    flex: 0 1 auto;
    min-height: 80px;
    max-height: calc(100% - 82px);
    object-fit: contain;
    border-radius: 6px;
    background: #000;
    display: block;
  }
  #preview-controls-scroll {
    overflow-y: auto;
    overflow-x: hidden;
    flex: 1;
    min-height: 0;
    margin-top: 6px;
    padding-right: 4px;
  }
  #preview-help {
    font-size: 10px;
    color: var(--dim);
    text-align: center;
    margin-top: 5px;
  }
  .preview-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 11px;
    color: var(--text);
  }
  .btn-seek {
    background: #1e1e26;
    border: 1px solid #32323e;
    color: var(--text);
    border-radius: 4px;
    padding: 3px 8px;
    cursor: pointer;
    font-size: 11px;
  }
  .btn-seek:hover {
    background: #282834;
    border-color: var(--accent);
  }
  .btn-analyze {
    width: 100%;
    margin-top: 8px;
    padding: 6px;
    background: #1a2a3a;
    border: 1px solid #2a4a6a;
    color: var(--accent);
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
  }
  .btn-analyze:hover {
    background: #22384d;
  }
  #video-status {
    margin-top: 6px;
    font-size: 11px;
    color: var(--dim);
  }
  #video-status .progress-track {
    height: 4px;
    background: #1c1c22;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 4px;
  }
  #video-status .progress-fill {
    height: 100%;
    background: var(--accent);
  }
  #stage {
    position: absolute;
    top: 0;
    left: 0;
    right: 340px;
    height: 100vh;
  }
  #sidebar {
    position: fixed;
    top: 0;
    right: 0;
    width: 340px;
    height: 100vh;
    background: #0e0e12;
    border-left: 1px solid #212127;
    overflow: hidden;
    z-index: 10;
    display: flex;
    flex-direction: column;
  }
  #sidebar-current {
    flex-shrink: 0;
    padding: 14px 16px;
    border-bottom: 1px solid #212127;
    background: #12121a;
    display: flex;
    gap: 12px;
    align-items: center;
  }
  #sidebar-current img {
    width: 56px;
    height: 56px;
    border-radius: 8px;
    object-fit: cover;
    flex-shrink: 0;
    background: #1a1a20;
    border: 2px solid var(--accent);
  }
  #sidebar-current .info {
    flex: 1;
    min-width: 0;
  }
  #sidebar-current .label {
    font-size: 10px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 3px;
  }
  #sidebar-current .fname {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  #sidebar-current .mode {
    font-size: 10px;
    color: var(--dim);
    margin-top: 3px;
  }
  #sidebar-current .detail {
    font-size: 10px;
    color: var(--dim);
    margin-top: 3px;
    line-height: 1.3;
  }
  #sidebar-scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  #sidebar-header {
    padding: 10px 16px;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: var(--dim);
    text-transform: uppercase;
    border-bottom: 1px solid #1c1c22;
    position: sticky;
    top: 0;
    background: #0e0e12;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  #sidebar-header-title {
    flex: 1;
  }
  #toggle-list-btn {
    background: #16161c;
    border: 1px solid #2a2a32;
    color: var(--dim);
    border-radius: 5px;
    padding: 3px 7px;
    cursor: pointer;
    font-size: 10px;
    text-transform: none;
    letter-spacing: 0;
  }
  #toggle-list-btn:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  #sidebar.list-hidden #immich-selection-bar,
  #sidebar.list-hidden #list-body {
    display: none !important;
  }
  .list-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 16px;
    cursor: pointer;
    border-bottom: 1px solid #16161b;
  }
  .list-row:hover {
    background: #16161c;
  }
  .list-row.active {
    background: #172230;
    border-left: 2px solid var(--accent);
  }
  .list-row img {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
    background: #1a1a20;
  }
  .list-row .info {
    flex: 1;
    min-width: 0;
  }
  .list-row .fname {
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .list-row .simbar-track {
    height: 3px;
    background: #1c1c22;
    border-radius: 2px;
    margin-top: 5px;
    overflow: hidden;
  }
  .list-row .simbar-fill {
    height: 100%;
    background: var(--accent);
  }
  .list-row .simpct {
    font-size: 10px;
    color: var(--dim);
    margin-left: 8px;
    flex-shrink: 0;
    width: 34px;
    text-align: right;
  }
  .ring {
    position: absolute;
    top: 50%;
    left: 50%;
    border: 1px solid var(--ring-line);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
  }
  .node {
    position: absolute;
    top: 50%;
    left: 50%;
    border-radius: 50%;
    overflow: hidden;
    cursor: pointer;
    background: #111;
    border: 2px solid #26262e;
    transition: box-shadow 0.15s ease;
    will-change: transform, z-index;
  }
  #pose-list-view {
    position: absolute;
    top: 0;
    left: 0;
    right: 340px;
    height: 100vh;
    overflow-x: auto;
    overflow-y: hidden;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 24px 40px;
    box-sizing: border-box;
  }
  .pose-list-item {
    flex: 0 0 auto;
    width: 96px;
    cursor: pointer;
    text-align: center;
    transition: transform 0.15s ease;
  }
  .pose-list-item img {
    width: 96px;
    height: 96px;
    object-fit: cover;
    border-radius: 6px;
    border: 2px solid #26262e;
    display: block;
  }
  .pose-list-item.selected img {
    border-color: var(--accent);
  }
  .pose-list-item .plabel {
    font-size: 9px;
    color: var(--dim);
    margin-top: 3px;
    white-space: nowrap;
  }
  .pose-list-anchor {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-right: 12px;
    padding-right: 12px;
    border-right: 1px solid #26262e;
  }
  .pose-list-anchor img {
    width: 110px;
    height: 110px;
    object-fit: cover;
    border-radius: 6px;
    border: 2px solid var(--accent);
    display: block;
  }
  #pose-list-scrubber {
    position: absolute;
    left: 0;
    right: 340px;
    bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 40px;
    box-sizing: border-box;
    z-index: 6;
  }
  #pose-list-scrubber input[type="range"] {
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    background: #26262e;
    border-radius: 2px;
    outline: none;
  }
  #pose-list-scrubber input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid #0e0e12;
  }
  #pose-list-scrubber input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    border: 2px solid #0e0e12;
  }
  .pose-scrub-label {
    font-size: 10px;
    color: var(--dim);
    white-space: nowrap;
    min-width: 60px;
  }
  #pose-scrub-right {
    text-align: right;
  }
  .node.export-selected {
    outline: 3px solid #7cc4ff;
    outline-offset: -1px;
  }
  .node:hover {
    box-shadow: 0 0 0 3px var(--accent);
    z-index: 5;
  }
  .node img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .node.center {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent), 0 0 24px rgba(124,196,255,0.4);
    cursor: default;
  }
  .sim-label {
    position: absolute;
    bottom: -18px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--dim);
    white-space: nowrap;
  }
  #loading {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--dim);
    font-size: 13px;
  }
</style>
</head>
<body>
<div id="preview-hover-panel"><img id="preview-hover-img"><div class="preview-caption" id="preview-hover-caption"></div></div>
<div id="left-panel">
  <div id="left-resize-handle"></div>
  <div id="left-panel-collapse-all" title="Collapse panel">◂</div>
  <div id="left-panel-content">
    <div id="left-controls-pane">

    <div class="panel-section expanded" data-section="identity">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Anchor</span></div>
      <div class="panel-section-body">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 11px; margin-bottom: 8px;">
          <label for="ring-scale-input">Ring Scale:</label>
          <input id="ring-scale-input" type="range" min="25" max="400" value="100" style="flex: 1;">
          <span id="ring-scale-val">100%</span>
        </div>
        <div id="ring-sort-controls" style="display:flex;gap:10px;font-size:10px;color:var(--dim);margin-bottom:8px;flex-wrap:wrap;">
          <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" class="ring-sort-cb" data-metric="sim" checked> Similarity</label>
          <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" class="ring-sort-cb" data-metric="yaw"> Yaw</label>
          <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" class="ring-sort-cb" data-metric="pitch"> Pitch</label>
          <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input type="checkbox" class="ring-sort-cb" data-metric="roll"> Roll</label>
        </div>
        <div id="ring-squeeze-controls" style="display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim);margin-bottom:8px;">
          <label for="ring-squeeze-slider" style="white-space:nowrap;">Min sim</label>
          <input id="ring-squeeze-slider" type="range" min="0" max="100" value="20" style="flex:1;">
          <span id="ring-squeeze-val" style="min-width:70px;text-align:right;white-space:nowrap;">20% (0/0)</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:10px;margin-bottom:8px;">
          <button type="button" id="find-neutral-btn" style="flex:1;font-size:11px;padding:5px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:4px;cursor:pointer;">Find neutral pose (calibration frame)</button>
        </div>
        <div id="neutral-pose-readout" style="font-size:10px;color:var(--dim);margin-bottom:8px;display:none;"></div>
        <div class="fname" id="hud-filename">loading…</div>
        <div id="hud-sub">Immich Ring Visualizer</div>
        <div class="mode" id="hud-mode"></div>
      </div>
    </div>

    <div class="panel-section expanded" data-section="search">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Search</span></div>
      <div class="panel-section-body">
        <div id="search-box" style="width:100%;">
          <input id="search-input" type="text" placeholder="Type a filename to start from…" autocomplete="off">
          <div id="search-results"></div>
        </div>
      </div>
    </div>

    <div class="panel-section expanded" data-section="video-analysis">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Video / Image-Set Analysis</span></div>
      <div class="panel-section-body">
        <div id="video-drop" style="width:100%;">
          <div style="display: flex; gap: 6px; margin-bottom: 8px;">
            <input id="sim-threshold" type="number" step="0.05" value="0.65" title="Similarity Threshold" style="width: 47%; background: #16161c; border: 1px solid #2a2a32; color: var(--text); padding: 4px; border-radius: 4px;">
            <input id="blur-threshold" type="number" step="1" value="50" title="Blur Threshold" style="width: 47%; background: #16161c; border: 1px solid #2a2a32; color: var(--text); padding: 4px; border-radius: 4px;">
          </div>
          <input type="file" id="video-file-input" accept="video/*" style="display:none">
          <input type="file" id="folder-images-input" accept="image/*" multiple style="display:none">
          <input type="file" id="folder-zip-input" accept=".zip" style="display:none">
          <div style="display:flex;gap:6px;">
            <div id="video-drop-zone" style="flex:1;">
              Select video
            </div>
            <button type="button" id="folder-images-btn" style="flex:1;font-size:11px;padding:6px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:4px;cursor:pointer;">Load folder</button>
          </div>
          <div style="display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim);margin-top:6px;">
            <button type="button" id="folder-zip-btn" style="font-size:10px;padding:4px 8px;background:#16161c;border:1px solid #2a2a32;color:var(--dim);border-radius:4px;cursor:pointer;white-space:nowrap;">or a .zip</button>
            <label for="folder-ref-index" style="white-space:nowrap;margin-left:6px;">Ref position</label>
            <input id="folder-ref-index" type="number" min="1" value="1" style="width:50px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <span style="white-space:nowrap;">(image-set only, 1st alphabetically by default)</span>
          </div>
          <div id="video-status"></div>
          <div id="folder-status" style="font-size:10px;color:var(--dim);"></div>
        </div>
      </div>
    </div>

    <div class="panel-section" data-section="person-clusters">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Person Clusters</span></div>
      <div class="panel-section-body">
        <div style="display:flex;gap:6px;align-items:center;margin-bottom:8px;font-size:11px;">
          <label style="color:var(--dim);">Min faces</label>
          <input id="pc-min-faces" type="number" step="1" min="1" value="5" style="width:56px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
          <label style="color:var(--dim);">Tight ≥</label>
          <input id="pc-tight-threshold" type="number" step="0.01" min="0" max="1" value="0.85" title="Avg similarity above which a cluster is flagged as suspiciously tight (e.g. duplicate stills)" style="width:56px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
        </div>
        <div style="margin-bottom:8px;">
          <button type="button" id="pc-load-btn" style="width:100%;font-size:11px;padding:5px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:4px;cursor:pointer;">Load clusters</button>
        </div>
        <div id="pc-status" style="font-size:10px;color:var(--dim);margin-bottom:6px;"></div>
        <div id="pc-cluster-list" style="display:flex;flex-direction:column;gap:2px;max-height:220px;overflow-y:auto;"></div>
        <div id="pc-thumb-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(56px,1fr));gap:4px;margin-top:8px;"></div>
      </div>
    </div>

    <div class="panel-section" data-section="export-settings">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Export Settings</span></div>
      <div class="panel-section-body">
        <div style="display:flex;flex-direction:column;gap:6px;font-size:11px;">
          <div style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);">Size</label>
            <input id="export-width" type="number" step="8" value="512" style="width:70px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <span style="color:var(--dim);">×</span>
            <input id="export-height" type="number" step="8" value="512" style="width:70px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <div style="display:flex;gap:3px;margin-left:4px;">
              <button type="button" class="export-preset-btn" data-w="512" data-h="512" style="font-size:9px;padding:3px 5px;background:#16161c;border:1px solid #2a2a32;color:var(--dim);border-radius:4px;cursor:pointer;">512</button>
              <button type="button" class="export-preset-btn" data-w="576" data-h="576" style="font-size:9px;padding:3px 5px;background:#16161c;border:1px solid #2a2a32;color:var(--dim);border-radius:4px;cursor:pointer;">576</button>
              <button type="button" class="export-preset-btn" data-w="1024" data-h="1024" style="font-size:9px;padding:3px 5px;background:#16161c;border:1px solid #2a2a32;color:var(--dim);border-radius:4px;cursor:pointer;">1024</button>
            </div>
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);">Crop</label>
            <select id="export-crop-mode" style="flex:1;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
              <option value="face">Face-centered (keep face framed)</option>
              <option value="center">Center crop (ignore face)</option>
              <option value="contain" selected>Fit, no crop (black bars)</option>
              <option value="stretch">Stretch to fill</option>
            </select>
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);" title="Skip exports where the detected face would be smaller than this in the final image. Set 0 to disable.">Min face</label>
            <input id="export-min-face" type="number" step="4" min="0" value="48" style="width:70px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <span style="color:var(--dim);font-size:10px;">px in export</span>
          </div>
          <div id="export-margin-row" style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);" title="How much room around the face to include. 1.0 = tight to the detected box, higher = more head/shoulders.">Margin</label>
            <input id="export-margin" type="number" step="0.1" min="1" value="2.2" style="width:70px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <span style="color:var(--dim);font-size:10px;">× face box</span>
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);">Scaling</label>
            <select id="export-interp" style="flex:1;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
              <option value="auto" selected>Auto (Lanczos up / Area down)</option>
              <option value="lanczos">Lanczos4</option>
              <option value="cubic">Cubic</option>
              <option value="area">Area</option>
              <option value="linear">Linear</option>
              <option value="nearest">Nearest</option>
            </select>
          </div>
          <label style="display:flex;align-items:center;gap:6px;color:var(--dim);">
            <input id="export-upscale" type="checkbox" checked>
            Allow scaling up (fill frame with face even if source crop is smaller than target)
          </label>
          <div id="export-max-upscale-row" style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);" title="If the face is small relative to the frame (e.g. full-body shots), cap magnification and widen the crop to include more of the body/background instead of blowing up a few pixels.">Cap at</label>
            <input id="export-max-upscale" type="number" step="0.1" min="1" value="2.5" style="width:70px;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
            <span style="color:var(--dim);font-size:10px;">× — beyond this, zoom out instead of upscaling (don't invent detail)</span>
          </div>
          <div id="export-pad-row" style="display:flex;gap:6px;align-items:center;">
            <label style="width:56px;color:var(--dim);" title="If the widened crop hits a frame edge (source too small in one dimension), what to do about the remaining gap to stay under the cap.">If clipped</label>
            <select id="export-pad-mode" style="flex:1;background:#16161c;border:1px solid #2a2a32;color:var(--text);padding:4px;border-radius:4px;">
              <option value="none" selected>Zoom in anyway (may exceed cap)</option>
              <option value="black">Pad with black bars</option>
              <option value="edge">Pad by extending edge pixels</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    </div>
    <div id="left-splitter" title="Resize controls and frame preview"></div>
    <div id="left-preview-pane">

    <div class="panel-section grow" data-section="frame-preview">
      <div class="panel-section-header"><span class="chev">▸</span><span class="sec-title">Frame Preview</span></div>
      <div class="panel-section-body">
        <div id="preview-container">
          <div style="font-size:11px;color:var(--dim);margin-bottom:7px;">Reference frame — use ← / → to step one actual video frame</div>
          <canvas id="preview-canvas"></canvas>
          <audio id="reference-audio" preload="auto" style="display:none"></audio>
          
          <div id="preview-controls-scroll">
            <div class="preview-controls">
              <button class="btn-seek" id="btn-prev-frame">◀ -1</button>
              <button class="btn-seek" id="btn-play-frames">▶ Play</button>
              <button class="btn-seek" id="btn-stop-frames">■ Stop</button>
              <span id="frame-counter">Frame: 1</span>
              <button class="btn-seek" id="btn-next-frame">+1 ▶</button>
            </div>
            <div id="preview-help">Use ← / → keys to scrub one actual decoded frame at a time.</div>
            <button class="btn-analyze" id="btn-start-analysis">Run Analysis with Selected Frame</button>
          </div>
        </div>
      </div>
    </div>

    </div>
  </div>
</div>

<div id="stage"><div id="loading">loading…</div></div>
<div id="pose-list-view" style="display:none;"></div>
<div id="pose-list-scrubber" style="display:none;">
  <span id="pose-scrub-left" class="pose-scrub-label"></span>
  <input id="pose-scrub-slider" type="range" min="0" max="1000" value="0">
  <span id="pose-scrub-right" class="pose-scrub-label"></span>
</div>
<div id="sidebar">
  <div id="sidebar-current">
    <img id="sidebar-current-img" src="" alt="">
    <div class="info">
      <div class="label">Currently selected</div>
      <div class="fname" id="sidebar-current-fname">—</div>
      <div class="mode" id="sidebar-current-mode"></div>
      <div class="detail" id="sidebar-current-detail"></div>
    </div>
  </div>
  <div id="sidebar-scroll">
    <div id="sidebar-header"><span id="sidebar-header-title">Ranked matches</span><button id="toggle-list-btn" title="Hide ranked match list">Hide</button></div>
    <div id="immich-selection-bar" style="display:none;padding:8px 16px;border-bottom:1px solid #212127;background:#12121a;">
      <div style="font-size:10px;color:var(--dim);margin-bottom:5px;">
        <span id="immich-selection-count">0</span> Immich image(s) selected — sticks across browsing
      </div>
      <div style="display:flex;gap:6px;">
        <button id="immich-export-selected-btn" style="flex:1;padding:5px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:10px;">Export selected to disk</button>
        <button id="immich-view-selected-btn" style="padding:5px 8px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:6px;cursor:pointer;font-size:10px;">View</button>
        <button id="immich-clear-selected-btn" style="padding:5px 8px;background:#2a1a1a;border:1px solid #4a2a2a;color:#d98080;border-radius:6px;cursor:pointer;font-size:10px;">Clear</button>
      </div>
    </div>
    <div id="list-body"></div>
  </div>
</div>

<video id="hidden-video" style="display:none;" muted playsinline></video>

<div id="selection-modal-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:100;align-items:center;justify-content:center;">
  <div id="selection-modal" style="background:#16161c;border:1px solid #2a2a32;border-radius:10px;width:min(720px,90vw);max-height:80vh;display:flex;flex-direction:column;overflow:hidden;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #26262e;">
      <div style="font-size:12px;color:var(--text);">Export selection — <span id="selection-modal-count">0</span> item(s). Double-click to remove.</div>
      <button id="selection-modal-close" style="background:none;border:none;color:var(--dim);font-size:16px;cursor:pointer;line-height:1;">✕</button>
    </div>
    <div id="selection-modal-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:8px;padding:14px;overflow-y:auto;"></div>
  </div>
</div>

<script>
const stage = document.getElementById('stage');
const CENTER_SIZE = 120;
const MIN_SIZE = 34;
const MAX_RADIUS_VW = 42;

function sizeForSim(sim) {
  const t = Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
  return MIN_SIZE + t * (CENTER_SIZE - MIN_SIZE);
}

let ringScale = 1.0;

function radiusForSim(sim) {
  const minDim = Math.min(window.innerWidth, window.innerHeight);
  const maxR = minDim * (MAX_RADIUS_VW / 100) * ringScale;
  const t = 1 - Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
  return (90 * ringScale) + t * (maxR - (90 * ringScale));
}

let lastRenderArgs = null;

const originalRender = render;
render = function(centerId, data, centerThumbUrl) {
  lastRenderArgs = { centerId, data, centerThumbUrl };
  originalRender(centerId, data, centerThumbUrl);
};

document.getElementById('ring-scale-input').addEventListener('input', (e) => {
  const val = e.target.value;
  ringScale = val / 100;
  document.getElementById('ring-scale-val').textContent = `${val}%`;
  if (lastRenderArgs) {
    render(lastRenderArgs.centerId, lastRenderArgs.data, lastRenderArgs.centerThumbUrl);
  }
});

async function loadNeighbors(assetId) {
  stage.innerHTML = '<div id="loading">loading neighbors…</div>';
  const res = await fetch(`/api/neighbors?assetId=${assetId}&limit=36`);
  const data = await res.json();
  render(assetId, data);
}

function thumbUrlFor(r) {
  return r.thumbUrl || `/api/thumb/${r.assetId}`;
}

function previewUrlFor(r) {
  return r.thumbUrl || `/api/preview/${r.assetId}`;
}

let currentPoseRequestToken = 0;

function setSidebarCurrentDetail(centerId, centerResult) {
  const detailEl = document.getElementById('sidebar-current-detail');
  if (!detailEl) return;
  const requestToken = ++currentPoseRequestToken;

  const sim = centerResult && typeof centerResult.similarity === 'number'
    ? `match: ${(centerResult.similarity * 100).toFixed(1)}%`
    : '';
  detailEl.textContent = sim;

  const canShowInlinePose = centerResult &&
    centerResult.pitch !== undefined &&
    centerResult.yaw !== undefined &&
    centerResult.roll !== undefined &&
    centerResult.pitch !== null;
  if (canShowInlinePose) {
    detailEl.textContent = `${sim}${sim ? ' · ' : ''}pitch: ${centerResult.pitch.toFixed(1)} yaw: ${centerResult.yaw.toFixed(1)} roll: ${centerResult.roll.toFixed(1)}`;
    return;
  }

  if (!centerId || centerId === '__anchor__') return;

  fetch(`/api/asset-face-pose/${centerId}`)
    .then(res => res.json())
    .then(pose => {
      if (requestToken !== currentPoseRequestToken || pose.error) return;
      detailEl.textContent = `${sim}${sim ? ' · ' : ''}pitch: ${pose.pitch.toFixed(1)} yaw: ${pose.yaw.toFixed(1)} roll: ${pose.roll.toFixed(1)}`;
    })
    .catch(() => {});
}

const hoverPanel = document.getElementById('preview-hover-panel');
const hoverImg = document.getElementById('preview-hover-img');
const hoverCaption = document.getElementById('preview-hover-caption');
let hoverTimer = null;

function showHoverPreview(r) {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    hoverImg.src = previewUrlFor(r);
    const pct = (r.similarity * 100).toFixed(1);
    
    let poseText = '';
    if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
      poseText = `<br>pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
    }
    
    hoverCaption.innerHTML = `${r.filename} — ${pct}%${poseText}`;
    hoverPanel.classList.add('active');
  }, 80);
}

function hideHoverPreview() {
  clearTimeout(hoverTimer);
  hoverPanel.classList.remove('active');
}

function render(centerId, data, centerThumbUrl) {
  stage.innerHTML = '';
  const modeLabel = data.mode === 'face' ? 'FACE SIMILARITY' : 'CLIP (WHOLE-IMAGE) SIMILARITY';
  document.getElementById('hud-mode').textContent = modeLabel;

  const results = data.results.filter(r => r.assetId !== centerId);
  const centerResult = data.results.find(r => r.assetId === centerId) || data.results[0];
  document.getElementById('hud-filename').textContent = centerResult ? centerResult.filename : '';

  const curImg = document.getElementById('sidebar-current-img');
  const curFname = document.getElementById('sidebar-current-fname');
  const curMode = document.getElementById('sidebar-current-mode');
  if (curImg) curImg.src = centerThumbUrl || ('/api/thumb/' + centerId);
  if (curFname) curFname.textContent = centerResult ? centerResult.filename : '—';
  if (curMode) curMode.textContent = modeLabel;
  setSidebarCurrentDetail(centerId, centerResult);

  [0.9, 0.7, 0.5, 0.35].forEach(band => {
    const r = radiusForSim(band);
    const ring = document.createElement('div');
    ring.className = 'ring';
    ring.style.width = (r * 2) + 'px';
    ring.style.height = (r * 2) + 'px';
    stage.appendChild(ring);
  });

  const center = document.createElement('div');
  center.className = 'node center';
  center.style.width = CENTER_SIZE + 'px';
  center.style.height = CENTER_SIZE + 'px';
  center.dataset.baseX = 0;
  center.dataset.baseY = 0;
  center.style.transform = 'translate(-50%, -50%)';
  center.innerHTML = `<img src="${centerThumbUrl || ('/api/thumb/' + centerId)}" loading="lazy">`;
  
  center.addEventListener('mouseenter', () => showHoverPreview({
    assetId: centerId,
    filename: 'Reference',
    similarity: 1,
    fromImmich: false
  }));
  center.addEventListener('mouseleave', hideHoverPreview);

  stage.appendChild(center);

  const bandCount = 8;
  const bands = Array.from({length: bandCount}, () => []);
  results.forEach(r => {
    const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
    const bandIdx = Math.min(bandCount - 1, Math.floor((1 - t) * bandCount));
    bands[bandIdx].push(r);
  });

  bands.forEach((bandResults, bandIdx) => {
    if (bandResults.length === 0) return;
    const t = 1 - (bandIdx / (bandCount - 1));
    const avgSim = 0.25 + t * 0.75;
    const radius = radiusForSim(avgSim);
    const size = sizeForSim(avgSim);

    const angleOffset = bandIdx * 0.6;
    bandResults.forEach((r, i) => {
      const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      const node = document.createElement('div');
      node.className = 'node';
      node.style.width = size + 'px';
      node.style.height = size + 'px';
      node.style.left = `calc(50% + ${x}px)`;
      node.style.top = `calc(50% + ${y}px)`;
      node.dataset.baseX = x;
      node.dataset.baseY = y;
      node.style.transform = 'translate(-50%, -50%)';
      if (r.fromImmich) {
        node.style.border = '2px solid #d4a544';
        node.style.boxShadow = '0 0 8px rgba(212,165,68,0.5)';
      }
      node.title = `${r.filename} — ${(r.similarity*100).toFixed(1)}%${r.fromImmich ? ' (from Immich library)' : ''}`;
      node.dataset.filename = r.filename;
      node.innerHTML = `<img src="${thumbUrlFor(r)}" loading="lazy">`;
      const isSelected = r.assetId ? selectedAssetIds.has(r.assetId) : (r.frame !== undefined && selectedFrames.has(r.frame));
      if (isSelected) node.classList.add('export-selected');
      // single click (debounced) = recenter the ring on this node; double
      // click = add/remove from the export queue. Same debounce pattern as
      // the person-clusters grid, so a dblclick doesn't also fire recenter.
      let nodeClickTimer = null;
      const toggleNodeExportSelection = () => {
        if (r.assetId) {
          if (selectedAssetIds.has(r.assetId)) { selectedAssetIds.delete(r.assetId); }
          else { selectedAssetIds.add(r.assetId); }
          updateImmichSelectionBar();
        } else if (r.frame !== undefined) {
          if (selectedFrames.has(r.frame)) { selectedFrames.delete(r.frame); }
          else { selectedFrames.add(r.frame); }
          updateSaveSelectedButton();
        }
        node.classList.toggle('export-selected');
      };
      node.onclick = () => {
        if (!r.assetId || r.fromImmich) return;
        clearTimeout(nodeClickTimer);
        nodeClickTimer = setTimeout(() => loadNeighbors(r.assetId), 220);
      };
      node.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearTimeout(nodeClickTimer);
        toggleNodeExportSelection();
      };
      node.addEventListener('mouseenter', () => showHoverPreview(r));
      node.addEventListener('mouseleave', hideHoverPreview);
      stage.appendChild(node);
    });
  });

  const listBody = document.getElementById('list-body');
  listBody.innerHTML = '';
  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'list-row';
    row.dataset.assetId = r.assetId;
    const pct = (r.similarity * 100).toFixed(1);
    
    let checkboxHtml = '';
    if (r.frame !== undefined) {
      checkboxHtml = `<input type="checkbox" class="frame-select-cb" data-frame="${r.frame}" ${selectedFrames.has(r.frame) ? 'checked' : ''} style="margin-right:6px;flex-shrink:0;">`;
    } else if (r.assetId) {
      checkboxHtml = `<input type="checkbox" class="asset-select-cb" data-asset-id="${r.assetId}" ${selectedAssetIds.has(r.assetId) ? 'checked' : ''} style="margin-right:6px;flex-shrink:0;">`;
    }

    let poseHtml = '';
    if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
      poseHtml = `<div style="font-size:10px;color:var(--dim);margin-top:2px;">pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}</div>`;
    }

    row.innerHTML = `
      ${checkboxHtml}
      <img src="${thumbUrlFor(r)}" loading="lazy" style="${r.fromImmich ? 'border:1.5px solid #d4a544;' : ''}">
      <div class="info">
        <div class="fname">${r.filename}${r.fromImmich ? ' <span style="color:#d4a544;font-size:9px;">● Immich</span>' : ''}</div>
        <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
        ${poseHtml}
      </div>
      <div class="simpct">${pct}%</div>
    `;
    row.onclick = (e) => {
      if (e.target.classList.contains('frame-select-cb') || e.target.classList.contains('asset-select-cb')) return;
      if (r.assetId && !r.fromImmich) loadNeighbors(r.assetId);
    };
    const cb = row.querySelector('.frame-select-cb');
    if (cb) {
      cb.addEventListener('change', () => {
        const frame = parseInt(cb.dataset.frame, 10);
        if (cb.checked) selectedFrames.add(frame);
        else selectedFrames.delete(frame);
        updateSaveSelectedButton();
      });
    }
    const acb = row.querySelector('.asset-select-cb');
    if (acb) {
      acb.addEventListener('change', () => {
        if (acb.checked) selectedAssetIds.add(acb.dataset.assetId);
        else selectedAssetIds.delete(acb.dataset.assetId);
        updateImmichSelectionBar();
      });
    }
    row.addEventListener('mouseenter', () => showHoverPreview(r));
    row.addEventListener('mouseleave', hideHoverPreview);
    listBody.appendChild(row);
  });
}

const selectedFrames = new Set();

// ---- selection persistence: pure in-memory state gets wiped by an
// accidental refresh, which is a real cost once you're deliberately
// hand-picking a curated set rather than just poking at the tool. Mirror
// both selection Sets to localStorage on every change and restore on load.
const SELECTION_STORAGE_KEY = 'immichring_selection_v1';

function saveSelectionToStorage() {
  try {
    localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify({
      frames: Array.from(selectedFrames),
      assetIds: Array.from(selectedAssetIds),
      savedAt: Date.now(),
    }));
  } catch (e) {
    console.warn('Could not persist selection:', e);
  }
}

function restoreSelectionFromStorage() {
  try {
    const raw = localStorage.getItem(SELECTION_STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    (data.frames || []).forEach(f => selectedFrames.add(f));
    (data.assetIds || []).forEach(id => selectedAssetIds.add(id));
    if (selectedFrames.size || selectedAssetIds.size) {
      updateSaveSelectedButton();
      updateImmichSelectionBar();
    }
  } catch (e) {
    console.warn('Could not restore selection:', e);
  }
}

// ---- shared export settings (used by all three export-to-disk buttons) ----
function getExportParams() {
  const upscaleOn = document.getElementById('export-upscale').checked;
  return {
    width: parseInt(document.getElementById('export-width').value, 10) || 512,
    height: parseInt(document.getElementById('export-height').value, 10) || 512,
    cropMode: document.getElementById('export-crop-mode').value,
    minFacePx: parseFloat(document.getElementById('export-min-face').value) || 0,
    margin: parseFloat(document.getElementById('export-margin').value) || 2.2,
    interp: document.getElementById('export-interp').value,
    upscale: upscaleOn,
    maxUpscale: upscaleOn ? (parseFloat(document.getElementById('export-max-upscale').value) || null) : null,
    padMode: upscaleOn ? document.getElementById('export-pad-mode').value : 'none',
  };
}

(function () {
  const cropModeSel = document.getElementById('export-crop-mode');
  const marginRow = document.getElementById('export-margin-row');
  const maxUpscaleRow = document.getElementById('export-max-upscale-row');
  const padRow = document.getElementById('export-pad-row');
  const upscaleCb = document.getElementById('export-upscale');
  function syncRows() {
    const isFace = cropModeSel.value === 'face';
    const showCap = isFace && upscaleCb.checked;
    marginRow.style.display = isFace ? 'flex' : 'none';
    maxUpscaleRow.style.display = showCap ? 'flex' : 'none';
    padRow.style.display = showCap ? 'flex' : 'none';
  }
  cropModeSel.addEventListener('change', syncRows);
  upscaleCb.addEventListener('change', syncRows);
  syncRows();

  document.querySelectorAll('.export-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('export-width').value = btn.dataset.w;
      document.getElementById('export-height').value = btn.dataset.h;
    });
  });
})();

function updateSaveSelectedButton() {
  const btn = document.getElementById('save-selected-btn');
  if (!btn) return;
  const n = selectedFrames.size;
  const a = selectedAssetIds.size;
  const frameText = `${n} selected frame${n === 1 ? '' : 's'}`;
  const assetText = a ? ` + ${a} Immich image${a === 1 ? '' : 's'}` : '';
  btn.textContent = `Save ${frameText}${assetText} to disk`;
  btn.disabled = n === 0 && a === 0;
  btn.style.opacity = (n === 0 && a === 0) ? '0.5' : '1';
  refreshSelectionModalIfOpen();
  saveSelectionToStorage();
}

const selectedAssetIds = new Set();

function updateImmichSelectionBar() {
  const bar = document.getElementById('immich-selection-bar');
  const countEl = document.getElementById('immich-selection-count');
  if (!bar || !countEl) return;
  const n = selectedAssetIds.size;
  countEl.textContent = n;
  bar.style.display = n > 0 ? 'block' : 'none';
  updateSaveSelectedButton();
  refreshSelectionModalIfOpen();
  saveSelectionToStorage();
}

// ---- selection review modal: lets you see everything queued for export
// across both Immich assets and local video frames in one place, and pull
// items back out by double-clicking them (same debounce-free dblclick
// convention used elsewhere for a deliberate "remove" action) ----
function findFrameResult(frame) {
  if (!lastVideoRingState) return null;
  return lastVideoRingState.baseResults.find(r => r.frame === frame)
      || extraImmichNodes.find(r => r.frame === frame)
      || null;
}

function openSelectionModal() {
  document.getElementById('selection-modal-overlay').style.display = 'flex';
  renderSelectionModal();
}

function closeSelectionModal() {
  document.getElementById('selection-modal-overlay').style.display = 'none';
}

function refreshSelectionModalIfOpen() {
  const overlay = document.getElementById('selection-modal-overlay');
  if (overlay && overlay.style.display === 'flex') renderSelectionModal();
}

function renderSelectionModal() {
  const grid = document.getElementById('selection-modal-grid');
  const countEl = document.getElementById('selection-modal-count');
  grid.innerHTML = '';

  const assetItems = Array.from(selectedAssetIds).map(id => {
    const known = [...(lastVideoRingState ? lastVideoRingState.baseResults : []), ...extraImmichNodes]
      .find(r => r.assetId === id);
    return { kind: 'asset', assetId: id, filename: known ? known.filename : id, thumb: `/api/thumb/${id}` };
  });
  const frameItems = Array.from(selectedFrames).map(frame => {
    const r = findFrameResult(frame);
    return { kind: 'frame', frame, filename: r ? r.filename : `frame ${frame}`, thumb: r ? thumbUrlFor(r) : '' };
  });
  const items = [...assetItems, ...frameItems];
  countEl.textContent = items.length;

  if (!items.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;color:var(--dim);font-size:11px;text-align:center;padding:20px;">Nothing selected yet.</div>';
    return;
  }

  items.forEach(it => {
    const cell = document.createElement('div');
    cell.style.textAlign = 'center';
    cell.style.cursor = 'pointer';
    cell.innerHTML = `
      <img src="${it.thumb}" loading="lazy" title="Double-click to remove"
           style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;border:2px solid var(--accent);display:block;">
      <div style="font-size:9px;color:var(--dim);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${it.filename}</div>
    `;
    cell.ondblclick = () => {
      if (it.kind === 'asset') {
        selectedAssetIds.delete(it.assetId);
        updateImmichSelectionBar();
      } else {
        selectedFrames.delete(it.frame);
        updateSaveSelectedButton();
      }
      // reflect removal in whichever grid/ring/strip is currently rendered
      document.querySelectorAll('.pose-list-item.selected, .node.export-selected').forEach(el => {
        if (el.dataset.filename === it.filename) el.classList.remove('selected', 'export-selected');
      });
      renderSelectionModal();
    };
    grid.appendChild(cell);
  });
}

document.getElementById('selection-modal-close').addEventListener('click', closeSelectionModal);
document.getElementById('selection-modal-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'selection-modal-overlay') closeSelectionModal();
});
document.getElementById('immich-view-selected-btn').addEventListener('click', openSelectionModal);

document.getElementById('immich-export-selected-btn').addEventListener('click', async () => {
  const btn = document.getElementById('immich-export-selected-btn');
  const assetIds = Array.from(selectedAssetIds);
  if (!assetIds.length) return;
  const prevText = btn.textContent;
  btn.textContent = 'Exporting…';
  btn.disabled = true;
  try {
    const res = await fetch('/api/export-immich-assets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assetIds, ...getExportParams() }),
    });
    const result = await res.json();
    if (result.error) {
      btn.textContent = 'Error: ' + result.error;
    } else {
      const skipped = result.errors ? result.errors.filter(e => e.includes(': skipped (')).length : 0;
      btn.textContent = `Saved ${result.exported} of ${assetIds.length} → ${result.path}` + (skipped ? ` (${skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded to avoid upscaling)` : '');
      if (result.errors && result.errors.length) {
        console.warn('Some exports failed:', result.errors);
      }
    }
  } catch (e) {
    btn.textContent = 'Request failed';
  }
  setTimeout(() => { btn.textContent = prevText; btn.disabled = false; }, 4000);
});

// ---- person clusters panel: standalone Immich browse, no video job needed ----
(function () {
  const loadBtn = document.getElementById('pc-load-btn');
  const minFacesInput = document.getElementById('pc-min-faces');
  const tightThresholdInput = document.getElementById('pc-tight-threshold');
  const statusEl = document.getElementById('pc-status');
  const listEl = document.getElementById('pc-cluster-list');
  const gridEl = document.getElementById('pc-thumb-grid');

  let lastRows = [];

  function renderRows() {
    const threshold = parseFloat(tightThresholdInput.value);
    const tightCutoff = isNaN(threshold) ? 0.85 : threshold;
    listEl.innerHTML = '';
    lastRows.forEach(p => {
      const row = document.createElement('div');
      row.className = 'list-row';
      row.style.cursor = 'pointer';
      const pct = (p.avgSim * 100).toFixed(1);
      const tight = p.avgSim > tightCutoff;
      row.innerHTML = `
        <div class="info" style="flex:1;">
          <div class="fname">${p.name}${tight ? ' <span style="color:#d4a544;font-size:9px;">● tight cluster</span>' : ''}</div>
          <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
        </div>
        <div class="simpct">${pct}%</div>
        <div style="font-size:10px;color:var(--dim);margin-left:6px;">${p.faceCount}</div>
      `;
      row.onclick = () => loadPersonAssets(p.personId, row);
      listEl.appendChild(row);
    });
  }

  async function loadClusters() {
    statusEl.textContent = 'Loading…';
    listEl.innerHTML = '';
    gridEl.innerHTML = '';
    try {
      const minFaces = parseInt(minFacesInput.value, 10) || 5;
      const res = await fetch(`/api/person-clusters?minFaces=${minFaces}&limit=40`);
      const rows = await res.json();
      if (rows.error) {
        statusEl.textContent = 'Error: ' + rows.error;
        return;
      }
      lastRows = rows;
      statusEl.textContent = `${rows.length} person${rows.length === 1 ? '' : 's'} (sorted tightest-cluster first)`;
      renderRows();
    } catch (e) {
      statusEl.textContent = 'Request failed: ' + e;
    }
  }

  tightThresholdInput.addEventListener('change', renderRows);

  async function loadPersonAssets(personId, rowEl) {
    document.querySelectorAll('#pc-cluster-list .list-row').forEach(r => r.style.background = '');
    if (rowEl) rowEl.style.background = 'rgba(212,165,68,0.12)';
    gridEl.innerHTML = 'Loading…';
    try {
      const res = await fetch(`/api/person-assets/${personId}?limit=200`);
      const assets = await res.json();
      gridEl.innerHTML = '';
      assets.forEach(a => {
        const cell = document.createElement('div');
        cell.style.position = 'relative';
        cell.style.cursor = 'pointer';
        let clickTimer = null;
        const checked = selectedAssetIds.has(a.assetId);
        cell.innerHTML = `
          <img src="/api/thumb/${a.assetId}" loading="lazy" title="${a.filename}"
               style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px;${checked ? 'outline:2px solid #d4a544;' : ''}">
        `;
        const toggleExportSelection = () => {
          if (selectedAssetIds.has(a.assetId)) {
            selectedAssetIds.delete(a.assetId);
            cell.querySelector('img').style.outline = 'none';
          } else {
            selectedAssetIds.add(a.assetId);
            cell.querySelector('img').style.outline = '2px solid #d4a544';
          }
          updateImmichSelectionBar();
        };
        cell.onclick = () => {
          clearTimeout(clickTimer);
          clickTimer = setTimeout(toggleExportSelection, 220);
        };
        cell.ondblclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          clearTimeout(clickTimer);
          const input = document.getElementById('search-input');
          const results = document.getElementById('search-results');
          if (input) input.value = a.filename;
          if (results) results.classList.remove('open');
          loadNeighbors(a.assetId);
        };
        gridEl.appendChild(cell);
      });
    } catch (e) {
      gridEl.innerHTML = 'Failed to load assets: ' + e;
    }
  }

  loadBtn.addEventListener('click', loadClusters);
})();

document.getElementById('immich-clear-selected-btn').addEventListener('click', () => {
  selectedAssetIds.clear();
  updateImmichSelectionBar();
  if (lastVideoRingState) renderVideoRing();
  else {
    document.querySelectorAll('.asset-select-cb').forEach(cb => { cb.checked = false; });
  }
});

async function init() {
  restoreSelectionFromStorage();
  const params = new URLSearchParams(window.location.search);
  let assetId = params.get('assetId');
  if (!assetId) {
    const res = await fetch('/api/random-face');
    const data = await res.json();
    assetId = data.assetId;
  }
  loadNeighbors(assetId);
}

const FISHEYE_RADIUS = 160;
const MAX_SCALE = 2.0;
const MAX_PUSH = 46;

let rafPending = false;
let lastMouse = null;

function applyFisheye(mx, my) {
  const rect = stage.getBoundingClientRect();
  const localX = mx - rect.left;
  const localY = my - rect.top;
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;

  document.querySelectorAll('.node').forEach(node => {
    const baseX = parseFloat(node.dataset.baseX || 0);
    const baseY = parseFloat(node.dataset.baseY || 0);
    const nodeScreenX = centerX + baseX;
    const nodeScreenY = centerY + baseY;

    const dx = nodeScreenX - localX;
    const dy = nodeScreenY - localY;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < FISHEYE_RADIUS) {
      const t = 1 - (dist / FISHEYE_RADIUS);
      const eased = t * t * (3 - 2 * t);
      const scale = 1 + eased * (MAX_SCALE - 1);
      const push = eased * MAX_PUSH;

      const angle = Math.atan2(baseY, baseX);
      const pushX = (baseX === 0 && baseY === 0) ? 0 : Math.cos(angle) * push;
      const pushY = (baseX === 0 && baseY === 0) ? 0 : Math.sin(angle) * push;
      node.style.transform = `translate(-50%, -50%) translate(${pushX}px, ${pushY}px) scale(${scale})`;
      node.style.zIndex = Math.round(10 + eased * 50);
    } else {
      node.style.transform = 'translate(-50%, -50%)';
      node.style.zIndex = 1;
    }
  });
  rafPending = false;
}

stage.addEventListener('mousemove', (e) => {
  lastMouse = [e.clientX, e.clientY];
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(() => applyFisheye(...lastMouse));
  }
});

stage.addEventListener('mouseleave', () => {
  document.querySelectorAll('.node').forEach(node => {
    node.style.transform = 'translate(-50%, -50%)';
    node.style.zIndex = 1;
  });
});

// ---- pose list lens effect: dock-style magnify along the horizontal strip,
// composes with each item's pitch-based translateY so the wave and the
// hover-zoom don't fight each other ----
const POSE_LENS_RADIUS = 140;
const POSE_LENS_MAX_SCALE = 1.6;
let poseLensRafPending = false;
let poseLensLastMouse = null;

function applyPoseLens(mx, my) {
  const listEl = document.getElementById('pose-list-view');
  const items = listEl.querySelectorAll('.pose-list-item');
  items.forEach(item => {
    const rect = item.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dist = Math.hypot(mx - cx, my - cy);
    const baseY = parseFloat(item.dataset.pitchY || 0);
    if (dist < POSE_LENS_RADIUS) {
      const t = 1 - (dist / POSE_LENS_RADIUS);
      const eased = t * t * (3 - 2 * t);
      const scale = 1 + eased * (POSE_LENS_MAX_SCALE - 1);
      item.style.transform = `translateY(${baseY}px) scale(${scale})`;
      item.style.zIndex = Math.round(10 + eased * 50);
    } else {
      item.style.transform = `translateY(${baseY}px)`;
      item.style.zIndex = 1;
    }
  });
  poseLensRafPending = false;
}

const poseListView = document.getElementById('pose-list-view');
poseListView.addEventListener('mousemove', (e) => {
  poseLensLastMouse = [e.clientX, e.clientY];
  if (!poseLensRafPending) {
    poseLensRafPending = true;
    requestAnimationFrame(() => applyPoseLens(...poseLensLastMouse));
  }
});
poseListView.addEventListener('mouseleave', () => {
  poseListView.querySelectorAll('.pose-list-item').forEach(item => {
    const baseY = parseFloat(item.dataset.pitchY || 0);
    item.style.transform = `translateY(${baseY}px)`;
    item.style.zIndex = 1;
  });
});

init();

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimer = null;

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (q.length < 2) {
    searchResults.classList.remove('open');
    return;
  }
  searchTimer = setTimeout(async () => {
    const res = await fetch(`/api/find-by-filename?name=${encodeURIComponent(q)}`);
    const matches = await res.json();
    searchResults.innerHTML = '';
    if (matches.length === 0) {
      searchResults.innerHTML = '<div class="search-result-row" style="color:var(--dim)">no matches</div>';
    } else {
      matches.forEach(m => {
        const row = document.createElement('div');
        row.className = 'search-result-row';
        row.innerHTML = `<img src="/api/thumb/${m.assetId}" loading="lazy"><span>${m.filename}</span>`;
        row.onclick = () => {
          searchResults.classList.remove('open');
          searchInput.value = m.filename;
          loadNeighbors(m.assetId);
        };
        searchResults.appendChild(row);
      });
    }
    searchResults.classList.add('open');
  }, 250);
});

document.addEventListener('click', (e) => {
  if (!document.getElementById('search-box').contains(e.target)) {
    searchResults.classList.remove('open');
  }
});

// ---- left panel: accordion sections, drag-resize, collapse-all ----
(function () {
  const panel = document.getElementById('left-panel');
  const handle = document.getElementById('left-resize-handle');
  const splitter = document.getElementById('left-splitter');
  const controlsPane = document.getElementById('left-controls-pane');
  const collapseAllBtn = document.getElementById('left-panel-collapse-all');

  // restore persisted width / collapsed states
  const savedWidth = localStorage.getItem('ringviz.leftPanelWidth');
  if (savedWidth) panel.style.width = savedWidth + 'px';
  const savedCollapsedAll = localStorage.getItem('ringviz.leftPanelCollapsedAll') === '1';
  if (savedCollapsedAll) {
    panel.classList.add('collapsed-all');
    collapseAllBtn.textContent = '▸';
  }
  const savedControlsPct = parseFloat(localStorage.getItem('ringviz.leftControlsPct'));
  if (!Number.isNaN(savedControlsPct)) {
    controlsPane.style.flexBasis = Math.max(24, Math.min(76, savedControlsPct)) + '%';
  }
  document.querySelectorAll('.panel-section').forEach(sec => {
    const key = 'ringviz.section.' + sec.dataset.section;
    const saved = localStorage.getItem(key);
    if (saved === '1') sec.classList.add('expanded');
    if (saved === '0') sec.classList.remove('expanded');
    const header = sec.querySelector('.panel-section-header');
    header.addEventListener('click', () => {
      sec.classList.toggle('expanded');
      localStorage.setItem(key, sec.classList.contains('expanded') ? '1' : '0');
    });
  });

  collapseAllBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const collapsed = panel.classList.toggle('collapsed-all');
    collapseAllBtn.textContent = collapsed ? '▸' : '◂';
    localStorage.setItem('ringviz.leftPanelCollapsedAll', collapsed ? '1' : '0');
  });

  let splitDragging = false;
  splitter.addEventListener('mousedown', (e) => {
    if (panel.classList.contains('collapsed-all')) return;
    splitDragging = true;
    splitter.classList.add('dragging');
    document.body.style.cursor = 'ns-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!splitDragging) return;
    const rect = document.getElementById('left-panel-content').getBoundingClientRect();
    const pct = ((e.clientY - rect.top) / rect.height) * 100;
    controlsPane.style.flexBasis = Math.max(24, Math.min(76, pct)) + '%';
  });
  document.addEventListener('mouseup', () => {
    if (!splitDragging) return;
    splitDragging = false;
    splitter.classList.remove('dragging');
    document.body.style.cursor = '';
    const rect = document.getElementById('left-panel-content').getBoundingClientRect();
    const controlsRect = controlsPane.getBoundingClientRect();
    localStorage.setItem('ringviz.leftControlsPct', Math.round((controlsRect.height / rect.height) * 100));
  });

  let dragging = false;
  let startX = 0;
  let startWidth = 0;
  handle.addEventListener('mousedown', (e) => {
    if (panel.classList.contains('collapsed-all')) return;
    dragging = true;
    handle.classList.add('dragging');
    startX = e.clientX;
    startWidth = panel.getBoundingClientRect().width;
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const newWidth = Math.max(240, Math.min(720, startWidth + (e.clientX - startX)));
    panel.style.width = newWidth + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    localStorage.setItem('ringviz.leftPanelWidth', Math.round(panel.getBoundingClientRect().width));
  });
})();

// ---- right sidebar: optionally hide ranked/Immich match list ----
(function () {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('toggle-list-btn');
  if (!sidebar || !toggle) return;

  function applyHidden(hidden) {
    sidebar.classList.toggle('list-hidden', hidden);
    toggle.textContent = hidden ? 'Show' : 'Hide';
    toggle.title = hidden ? 'Show ranked match list' : 'Hide ranked match list';
  }

  applyHidden(localStorage.getItem('ringviz.listHidden') === '1');
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const hidden = !sidebar.classList.contains('list-hidden');
    applyHidden(hidden);
    localStorage.setItem('ringviz.listHidden', hidden ? '1' : '0');
  });
})();

// ---- right sidebar: drag-resize ----
(function () {
  const sidebar = document.getElementById('sidebar');
  const stage = document.getElementById('stage');
  if (!sidebar || !stage) return;

  const handle = document.createElement('div');
  handle.id = 'right-resize-handle';
  handle.style.cssText = 'position:fixed;top:0;bottom:0;width:8px;cursor:ew-resize;z-index:11;';
  document.body.appendChild(handle);

  function applyWidth(w) {
    sidebar.style.width = w + 'px';
    stage.style.right = w + 'px';
    handle.style.right = (w - 4) + 'px';
  }

  const savedSidebarWidth = parseInt(localStorage.getItem('ringviz.sidebarWidth'), 10);
  applyWidth(savedSidebarWidth && !isNaN(savedSidebarWidth) ? savedSidebarWidth : 340);

  handle.addEventListener('mouseenter', () => handle.style.background = 'rgba(124,196,255,0.25)');
  handle.addEventListener('mouseleave', () => { if (!dragging) handle.style.background = ''; });

  let dragging = false;
  let startX = 0;
  let startWidth = 0;
  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    startX = e.clientX;
    startWidth = sidebar.getBoundingClientRect().width;
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const newWidth = Math.max(260, Math.min(720, startWidth - (e.clientX - startX)));
    applyWidth(newWidth);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    handle.style.background = '';
    localStorage.setItem('ringviz.sidebarWidth', Math.round(sidebar.getBoundingClientRect().width));
  });
})();

const dropZone = document.getElementById('video-drop-zone');
const fileInput = document.getElementById('video-file-input');
const previewContainer = document.getElementById('preview-container');
const previewCanvas = document.getElementById('preview-canvas');
const hiddenVideo = document.getElementById('hidden-video');
const frameCounter = document.getElementById('frame-counter');
const videoStatus = document.getElementById('video-status');

let currentVideoFile = null;
let currentPreviewId = null;
let currentFrameIdx = 1;
let totalFrames = 1;
let fps = 0;

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFileSelect(e.target.files[0]);
});

window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    handleFileSelect(file);
  }
});

async function handleFileSelect(file) {
  currentVideoFile = file;
  if (referenceAudio.src) URL.revokeObjectURL(referenceAudio.src);
  referenceAudio.src = URL.createObjectURL(file);
  referenceAudio.load();
  currentPreviewId = null;
  currentFrameIdx = 1;
  totalFrames = 1;
  fps = 0;

  videoStatus.textContent = "Uploading video for exact frame preview…";

  const form = new FormData();
  form.append('video', file);

  try {
    const res = await fetch('/api/preview-video', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok || data.error) {
      videoStatus.textContent = 'Error: ' + (data.error || 'Could not prepare video');
      return;
    }

    currentPreviewId = data.previewId;
    fps = data.fps;
    totalFrames = data.totalFrames;

    const previewSection = document.querySelector('.panel-section[data-section="frame-preview"]');
    if (previewSection) previewSection.classList.add('expanded');
    videoStatus.innerHTML =
      `${data.totalFrames.toLocaleString()} frames · ${fps.toFixed(3)} fps · ${data.duration.toFixed(2)} sec`;

    await seekToFrame(1);
  } catch (err) {
    videoStatus.textContent = 'Error: ' + err;
  }
}

async function seekToFrame(idx) {
  if (!currentPreviewId) return;

  currentFrameIdx = Math.max(1, Math.min(totalFrames, idx));
  frameCounter.textContent =
    `Frame: ${currentFrameIdx.toLocaleString()} / ${totalFrames.toLocaleString()}`;

  const img = new Image();

  img.onload = () => {
    previewCanvas.width = img.naturalWidth;
    previewCanvas.height = img.naturalHeight;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
  };

  img.onerror = () => {
    videoStatus.textContent = `Could not decode frame ${currentFrameIdx}`;
  };

  img.src =
    `/api/preview-frame/${currentPreviewId}/${currentFrameIdx}?t=${Date.now()}`;
}

function renderSimSparkline(results, threshold) {
  const wrap = document.getElementById('sim-sparkline-wrap');
  if (!wrap || !results || !results.length) return;

  const sorted = [...results].sort((a, b) => a.frame - b.frame);

  const W = wrap.clientWidth || 320;
  const H = 90;
  const padL = 4, padR = 4, padT = 8, padB = 4;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const minFrame = sorted[0].frame;
  const maxFrame = sorted[sorted.length - 1].frame;
  const frameSpan = Math.max(1, maxFrame - minFrame);

  const xFor = (frame) => padL + ((frame - minFrame) / frameSpan) * plotW;
  const yFor = (sim) => padT + (1 - Math.max(0, Math.min(1, sim))) * plotH;

  const linePoints = sorted.map(r => `${xFor(r.frame).toFixed(1)},${yFor(r.sim).toFixed(1)}`).join(' ');
  const thresholdY = yFor(threshold).toFixed(1);

  const dots = sorted.map(r => {
    const cx = xFor(r.frame).toFixed(1);
    const cy = yFor(r.sim).toFixed(1);
    const color = r.passed ? '#7cc4ff' : '#d9534f';
    return `<circle cx="${cx}" cy="${cy}" r="7" fill="transparent" data-frame="${r.frame}" class="spark-hit" style="cursor:pointer;"></circle>` +
           `<circle cx="${cx}" cy="${cy}" r="2" fill="${color}" style="pointer-events:none;"></circle>`;
  }).join('');

  wrap.innerHTML = `
    <div style="font-size:10px;color:var(--dim);margin-bottom:3px;">
      Match confidence by frame — click a point to jump the preview
    </div>
    <svg width="${W}" height="${H}" style="background:#0c0c10;border:1px solid #22222a;border-radius:6px;display:block;">
      <line x1="${padL}" y1="${thresholdY}" x2="${W - padR}" y2="${thresholdY}"
            stroke="#4a4a55" stroke-width="1" stroke-dasharray="3,3"></line>
      <polyline points="${linePoints}" fill="none" stroke="#5a8fc4" stroke-width="1.5"></polyline>
      ${dots}
    </svg>
  `;

  wrap.querySelectorAll('.spark-hit').forEach(el => {
    el.addEventListener('click', () => {
      const frame = parseInt(el.dataset.frame, 10);
      stopFramePlayback();
      seekToFrame(frame);
      syncAudioToFrame(frame);
      previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  });
}

let playTimer = null;
const referenceAudio = document.getElementById('reference-audio');

function frameTime(frameIdx) {
  return Math.max(0, (frameIdx - 1) / (fps > 0 ? fps : 24));
}

function frameFromTime(timeSeconds) {
  const currentFps = fps > 0 ? fps : 24;
  return Math.min(totalFrames, Math.max(1, Math.floor(timeSeconds * currentFps) + 1));
}

function syncAudioToFrame(frameIdx) {
  if (!referenceAudio.src) return;
  const t = frameTime(frameIdx);
  try {
    referenceAudio.currentTime = t;
  } catch (_) {}
}

function stopFramePlayback() {
  if (playTimer) { 
    clearInterval(playTimer); 
    playTimer = null; 
  }
  referenceAudio.pause();
  document.getElementById('btn-play-frames').textContent = '▶ Play';
}

function startFramePlayback() {
  if (!currentPreviewId || playTimer) return;
  if (currentFrameIdx >= totalFrames) {
    currentFrameIdx = 1;
    syncAudioToFrame(1);
  } else {
    syncAudioToFrame(currentFrameIdx);
  }

  document.getElementById('btn-play-frames').textContent = '⏸ Playing…';

  const playPromise = referenceAudio.play();
  if (playPromise && playPromise.catch) playPromise.catch(() => {});

  const intervalMs = fps > 0 ? (1000 / fps) : (1000 / 24);
  playTimer = setInterval(() => {
    if (referenceAudio.paused || referenceAudio.ended) {
      stopFramePlayback();
      return;
    }

    const targetFrame = frameFromTime(referenceAudio.currentTime);
    if (targetFrame !== currentFrameIdx) {
      seekToFrame(targetFrame);
    }

    if (targetFrame >= totalFrames) {
      stopFramePlayback();
    }
  }, intervalMs / 2);
}

document.getElementById('btn-play-frames').addEventListener('click', () => {
  if (playTimer) stopFramePlayback(); else startFramePlayback();
});

document.getElementById('btn-stop-frames').addEventListener('click', stopFramePlayback);

document.getElementById('btn-prev-frame').addEventListener('click', () => { 
  stopFramePlayback(); 
  const nextIdx = Math.max(1, currentFrameIdx - 1);
  seekToFrame(nextIdx); 
  syncAudioToFrame(nextIdx); 
});

document.getElementById('btn-next-frame').addEventListener('click', () => { 
  stopFramePlayback(); 
  const nextIdx = Math.min(totalFrames, currentFrameIdx + 1);
  seekToFrame(nextIdx); 
  syncAudioToFrame(nextIdx); 
});

document.addEventListener('keydown', (e) => {
  if (!currentPreviewId) return;
  if (document.activeElement && ['INPUT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)) return;
  
  if (e.key === 'ArrowLeft') { 
    e.preventDefault(); 
    stopFramePlayback(); 
    const nextIdx = Math.max(1, currentFrameIdx - 1);
    seekToFrame(nextIdx); 
    syncAudioToFrame(nextIdx); 
  } else if (e.key === 'ArrowRight') { 
    e.preventDefault(); 
    stopFramePlayback(); 
    const nextIdx = Math.min(totalFrames, currentFrameIdx + 1);
    seekToFrame(nextIdx); 
    syncAudioToFrame(nextIdx); 
  }
});

document.getElementById('btn-start-analysis').addEventListener('click', () => {
  if (currentVideoFile) {
    startVideoAnalysis(currentVideoFile, currentFrameIdx);
  }
});

async function startVideoAnalysis(videoFile, refFrameIdx) {
  selectedFrames.clear();
  videoStatus.textContent = 'Uploading for analysis…';
  const form = new FormData();
  form.append('video', videoFile);
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refFrame', refFrameIdx);

  const res = await fetch('/api/analyze-video', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) {
    videoStatus.textContent = 'Error: ' + data.error;
    return;
  }
  pollAnalysis(data.jobId, videoFile.name, refFrameIdx, videoStatus, 'video');
}

async function pollAnalysis(jobId, sourceLabel, refFrameIdx, statusEl, sourceType) {
  statusEl = statusEl || videoStatus;
  sourceType = sourceType || 'video';
  const res = await fetch(`/api/analysis-status/${jobId}`);
  const data = await res.json();

  if (data.status === 'error') {
    statusEl.textContent = 'Error: ' + data.error;
    return;
  }

  const passed = data.results.filter(r => r.passed).length;
  const failedSim = data.results.filter(r => !r.passed && r.failReason === 'sim').length;
  const failedBlur = data.results.filter(r => !r.passed && r.failReason === 'blur').length;

  statusEl.innerHTML = `
    Processing… ${data.frameCount} images seen<br>
    ${passed} kept, ${failedSim} low sim, ${failedBlur} blurry
    <div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, (data.frameCount/200)*100)}%"></div></div>
  `;

  if (data.status === 'running') {
    setTimeout(() => pollAnalysis(jobId, sourceLabel, refFrameIdx, statusEl, sourceType), 800);
    return;
  }

  statusEl.innerHTML = `
    Done — ${passed}/${data.frameCount} kept.
    <div id="sim-sparkline-wrap" style="margin-top:8px;"></div>
    <button id="export-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Save kept frames to disk</button>
    <button id="save-selected-btn" disabled style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;opacity:0.5;">Save 0 selected frames to disk</button>
    <button id="view-selected-btn" style="margin-top:6px;width:100%;padding:6px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:6px;cursor:pointer;font-size:11px;">View selected</button>
    ${sourceType === 'folder' ? '' : `
    <button id="playback-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Build playback (rejected frames blanked)</button>
    <video id="playback-video" controls style="width:100%;margin-top:8px;display:none;border-radius:6px;"></video>
    <div id="playback-frame-controls" style="display:none;margin-top:6px;gap:6px;">
      <button id="playback-prev-frame" class="btn-seek" style="flex:1;">◀ -1 frame</button>
      <button id="playback-next-frame" class="btn-seek" style="flex:1;">+1 frame ▶</button>
    </div>
    `}
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #22222a;">
      <div id="crosscheck-header" style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:var(--dim);margin-bottom:8px;user-select:none;">
        <span class="chev" id="crosscheck-chev" style="display:inline-block;transition:transform .15s ease;font-size:10px;width:10px;">▾</span>
        <span style="flex:1;">Cross-check vs Immich library</span>
      </div>
      <div style="display:flex;gap:6px;">
        <input id="crosscheck-frame-input" type="number" min="1" style="width:70px;background:#0c0c10;border:1px solid #2a2a32;color:var(--text);border-radius:6px;padding:4px 6px;font-size:11px;" placeholder="frame #">
        <button id="crosscheck-btn" style="flex:1;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Check vs Immich library</button>
      </div>
      <div id="crosscheck-results" style="margin-top:8px;"></div>
    </div>
  `;
  renderSimSparkline(data.results, data.simThreshold);
  const ccInput = document.getElementById('crosscheck-frame-input');
  if (ccInput) ccInput.value = currentFrameIdx;
  const ccHeader = document.getElementById('crosscheck-header');
  const ccChev = document.getElementById('crosscheck-chev');
  const ccResultsEl = document.getElementById('crosscheck-results');
  let ccCollapsed = localStorage.getItem('ringviz.crosscheckCollapsed') === '1';
  function applyCcCollapsed() {
    ccResultsEl.style.display = ccCollapsed ? 'none' : '';
    ccChev.style.transform = ccCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
  }
  applyCcCollapsed();
  ccHeader.addEventListener('click', () => {
    ccCollapsed = !ccCollapsed;
    localStorage.setItem('ringviz.crosscheckCollapsed', ccCollapsed ? '1' : '0');
    applyCcCollapsed();
  });
  document.getElementById('crosscheck-btn').onclick = async () => {
    const input = document.getElementById('crosscheck-frame-input');
    const frameNo = parseInt(input.value, 10) || currentFrameIdx;
    const btn = document.getElementById('crosscheck-btn');
    const out = document.getElementById('crosscheck-results');
    if (ccCollapsed) { ccCollapsed = false; localStorage.setItem('ringviz.crosscheckCollapsed', '0'); applyCcCollapsed(); }
    btn.textContent = 'Checking…';
    btn.disabled = true;
    out.innerHTML = '';
    try {
      const res = await fetch(`/api/immich-cross-check/${jobId}/${frameNo}`);
      const result = await res.json();
      if (result.error) {
        out.innerHTML = `<div style="font-size:11px;color:#d9534f;">${result.error}</div>`;
      } else if (!result.results.length) {
        out.innerHTML = `<div style="font-size:11px;color:var(--dim);">No faces in Immich library yet to compare against.</div>`;
      } else {
        out.innerHTML = `<div style="font-size:10px;color:var(--dim);margin-bottom:4px;">Closest matches already in Immich:</div>` +
          result.results.slice(0, 8).map((r, i) => `
            <div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #1a1a20;">
              <img src="/api/thumb/${r.assetId}" style="width:36px;height:36px;object-fit:cover;border-radius:4px;">
              <div style="flex:1;min-width:0;font-size:10px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.filename}</div>
              <div style="font-size:10px;color:var(--accent);">${(r.similarity * 100).toFixed(1)}%</div>
              <button class="cc-add-btn" data-idx="${i}" data-asset-id="${r.assetId}" style="font-size:9px;padding:3px 6px;background:#2a1a3a;border:1px solid #4a2a6a;color:#d4a5ff;border-radius:4px;cursor:pointer;flex-shrink:0;">${extraImmichNodes.some(n => n.assetId === r.assetId) ? 'Remove ✓' : '+ Ring'}</button>
            </div>
          `).join('');
        out.querySelectorAll('.cc-add-btn').forEach(btn => {
          btn.addEventListener('click', () => {
            const match = result.results[parseInt(btn.dataset.idx, 10)];
            const isInRing = extraImmichNodes.some(n => n.assetId === match.assetId);
            if (isInRing) {
              removeImmichNodeFromRing(match.assetId);
              btn.textContent = '+ Ring';
            } else {
              addImmichNodeToRing(match, jobId, frameNo);
              btn.textContent = 'Remove ✓';
            }
          });
        });
      }
    } catch (e) {
      out.innerHTML = `<div style="font-size:11px;color:#d9534f;">Request failed: ${e}</div>`;
    }
    btn.textContent = 'Check vs Immich library';
    btn.disabled = false;
  };
  document.getElementById('export-btn').onclick = async () => {
    const btn = document.getElementById('export-btn');
    const assetIds = Array.from(selectedAssetIds);
    btn.textContent = 'Saving…';
    btn.disabled = true;
    const res = await fetch(`/api/export-job/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assetIds, ...getExportParams() }),
    });
    const result = await res.json();
    btn.textContent = `Saved ${result.frameExported ?? result.exported} frames + ${result.immichExported || 0} Immich → ${result.path}` + (result.skipped ? ` (${result.skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded to avoid upscaling)` : '');
    if (result.errors && result.errors.length) console.warn('Some Immich exports failed:', result.errors);
  };
  const viewSelectedBtn = document.getElementById('view-selected-btn');
  if (viewSelectedBtn) viewSelectedBtn.onclick = openSelectionModal;
  document.getElementById('save-selected-btn').onclick = async () => {
    const btn = document.getElementById('save-selected-btn');
    const frames = Array.from(selectedFrames);
    const assetIds = Array.from(selectedAssetIds);
    if (!frames.length && !assetIds.length) return;
    const prevText = btn.textContent;
    btn.textContent = 'Saving…';
    btn.disabled = true;
    const res = await fetch(`/api/export-job/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames, assetIds, ...getExportParams() }),
    });
    const result = await res.json();
    btn.textContent = `Saved ${result.frameExported ?? 0}/${frames.length} frames + ${result.immichExported || 0}/${assetIds.length} Immich → ${result.path}` + (result.skipped ? ` (${result.skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded)` : '');
    if (result.errors && result.errors.length) console.warn('Some Immich exports failed:', result.errors);
    setTimeout(updateSaveSelectedButton, 3000);
  };
  document.getElementById('playback-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('playback-btn');
    btn.textContent = 'Building…';
    btn.disabled = true;
    const res = await fetch(`/api/build-playback/${jobId}`, { method: 'POST' });
    const result = await res.json();
    if (result.error) {
      btn.textContent = 'Error: ' + result.error;
      return;
    }
    btn.textContent = 'Playback ready';
    const vid = document.getElementById('playback-video');
    vid.src = result.url;
    vid.style.display = 'block';

    const playbackFps = result.fps || 24.0;
    const frameStep = 1 / playbackFps;
    const stepControls = document.getElementById('playback-frame-controls');
    stepControls.style.display = 'flex';

    document.getElementById('playback-prev-frame').onclick = () => {
      vid.pause();
      vid.currentTime = Math.max(0, vid.currentTime - frameStep);
    };
    document.getElementById('playback-next-frame').onclick = () => {
      vid.pause();
      vid.currentTime = Math.min(vid.duration || Infinity, vid.currentTime + frameStep);
    };
  });

  const results = data.results
    .filter(r => r.passed)
    .map(r => ({
      filename: r.origName || `frame_${r.frame}`,
      frame: r.frame,
      similarity: r.sim,
      thumbUrl: `/api/framefile/${r.frameId}`,
      pitch: r.pitch,
      yaw: r.yaw,
      roll: r.roll,
    }))
    .sort((a, b) => b.similarity - a.similarity);

  const anchorUrl = `/api/framefile/${jobId}_anchor`;
  const hudLabel = sourceType === 'folder' ? 'IMAGE SET ANALYSIS (local, not in Immich)' : 'VIDEO FRAME ANALYSIS (local, not in Immich)';
  document.getElementById('hud-mode').textContent = hudLabel;
  document.getElementById('hud-filename').textContent = sourceLabel;
  const curModeEl = document.getElementById('sidebar-current-mode');
  if (curModeEl) curModeEl.textContent = hudLabel;

  lastVideoRingState = {
    anchorUrl,
    refFrameIdx,
    baseResults: results,
    sourceType,
  };
  renderVideoRing();
  updateSaveSelectedButton();
}

// ---- image folder / zip loader: runs the exact same analysis pipeline as
// video, just over a variable-count set of still images (e.g. 32 curated
// LoRA reference frames) instead of decoded video frames ----
const folderImagesInput = document.getElementById('folder-images-input');
const folderZipInput = document.getElementById('folder-zip-input');
const folderStatus = document.getElementById('folder-status');
let lastFolderSource = null;

document.getElementById('folder-images-btn').addEventListener('click', () => folderImagesInput.click());
document.getElementById('folder-zip-btn').addEventListener('click', () => folderZipInput.click());

folderImagesInput.addEventListener('change', () => {
  if (folderImagesInput.files.length) startFolderAnalysis({ images: folderImagesInput.files });
});
folderZipInput.addEventListener('change', () => {
  if (folderZipInput.files.length) startFolderAnalysis({ zip: folderZipInput.files[0] });
});

async function startFolderAnalysis({ images, zip }, refIndexOverride) {
  selectedFrames.clear();
  const refIndex = refIndexOverride || (parseInt(document.getElementById('folder-ref-index').value, 10) || 1);
  document.getElementById('folder-ref-index').value = refIndex;
  lastFolderSource = { images, zip };
  const form = new FormData();
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refIndex', refIndex);

  let sourceLabel;
  if (zip) {
    form.append('zip', zip);
    sourceLabel = zip.name;
    folderStatus.textContent = `Uploading ${zip.name}…`;
  } else {
    Array.from(images).forEach(f => form.append('images', f));
    sourceLabel = `${images.length} images`;
    form.append('sourceName', `imgset_${images.length}`);
    folderStatus.textContent = `Uploading ${images.length} images…`;
  }

  const res = await fetch('/api/analyze-folder', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) {
    folderStatus.textContent = 'Error: ' + data.error;
    return;
  }
  folderStatus.textContent = `${data.imageCount} images accepted, analyzing…`;
  pollAnalysis(data.jobId, sourceLabel, refIndex, folderStatus, 'folder');
}

let lastVideoRingState = null;
const extraImmichNodes = [];
let ringSortMetric = 'sim';

// squeeze filter: a live post-hoc min-similarity cutoff applied on top of
// whatever the original video-analysis job already loaded, so you can
// tighten (or loosen) the working set without re-running the whole job.
// Straight-on similarity browsing wants a high bar; pose-diversity browsing
// (yaw/pitch/roll) wants a low bar so marginal-confidence frames stay
// available - defaults switch automatically per metric, but the slider
// always stays user-overridable.
const SQUEEZE_DEFAULTS = { sim: 65, yaw: 20, pitch: 20, roll: 20 };
let squeezeMinPct = SQUEEZE_DEFAULTS.sim;
let squeezeUserOverridden = false;

const squeezeSlider = document.getElementById('ring-squeeze-slider');
const squeezeVal = document.getElementById('ring-squeeze-val');
squeezeSlider.value = squeezeMinPct;

squeezeSlider.addEventListener('input', () => {
  squeezeMinPct = parseFloat(squeezeSlider.value);
  squeezeUserOverridden = true;
  if (lastVideoRingState) renderVideoRing();
});

document.querySelectorAll('.ring-sort-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    if (cb.checked) {
      document.querySelectorAll('.ring-sort-cb').forEach(other => {
        if (other !== cb) other.checked = false;
      });
      ringSortMetric = cb.dataset.metric;
      if (!squeezeUserOverridden) {
        squeezeMinPct = SQUEEZE_DEFAULTS[ringSortMetric];
        squeezeSlider.value = squeezeMinPct;
      }
    } else {
      // don't allow zero selection - fall back to similarity
      cb.checked = true;
      return;
    }
    if (lastVideoRingState) renderVideoRing();
  });
});

function metricValueForRing(r) {
  const raw = r.similarity;
  return typeof raw === 'number' ? raw : 0;
}

function applySqueeze(combined) {
  const cutoff = squeezeMinPct / 100;
  const kept = combined.filter(r => {
    const sim = typeof r.similarity === 'number' ? r.similarity : (typeof r.sim === 'number' ? r.sim : 1);
    return sim >= cutoff;
  });
  squeezeVal.textContent = `${squeezeMinPct}% (${kept.length}/${combined.length})`;
  return kept;
}

document.getElementById('find-neutral-btn').addEventListener('click', () => {
  if (!lastVideoRingState) return;
  const { baseResults } = lastVideoRingState;
  const pool = applySqueeze([...baseResults, ...extraImmichNodes])
    .filter(r => typeof r.yaw === 'number' && typeof r.pitch === 'number' && typeof r.roll === 'number');
  const readout = document.getElementById('neutral-pose-readout');
  if (!pool.length) {
    readout.style.display = 'block';
    readout.textContent = 'No frames with pose data in current working set.';
    return;
  }
  let best = pool[0];
  let bestScore = Math.abs(best.yaw) + Math.abs(best.pitch) + Math.abs(best.roll);
  pool.forEach(r => {
    const score = Math.abs(r.yaw) + Math.abs(r.pitch) + Math.abs(r.roll);
    if (score < bestScore) { best = r; bestScore = score; }
  });
  readout.style.display = 'block';
  readout.innerHTML = `Most neutral: <b>${best.filename}</b> — yaw ${best.yaw.toFixed(1)}° pitch ${best.pitch.toFixed(1)}° roll ${best.roll.toFixed(1)}° (sim ${(best.similarity*100).toFixed(1)}%)
    <button type="button" id="use-as-reference-btn" style="display:block;width:100%;margin-top:6px;font-size:10px;padding:5px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:4px;cursor:pointer;">Use as reference &amp; re-analyze</button>`;
  flashHighlightFrame(best);

  document.getElementById('use-as-reference-btn').onclick = () => {
    const btn = document.getElementById('use-as-reference-btn');
    const sourceType = lastVideoRingState.sourceType;
    if (sourceType === 'folder') {
      if (!lastFolderSource) {
        btn.textContent = 'Original folder/zip no longer available — reload it first';
        return;
      }
      btn.textContent = 'Re-analyzing…';
      startFolderAnalysis(lastFolderSource, best.frame);
    } else {
      if (!currentVideoFile) {
        btn.textContent = 'Original video no longer available — reload it first';
        return;
      }
      btn.textContent = 'Re-analyzing…';
      startVideoAnalysis(currentVideoFile, best.frame);
    }
  };
});

// flashes/scrolls to whichever node or pose-list item represents this frame,
// in whichever view is currently active
function flashHighlightFrame(r) {
  const el = document.querySelector(`.pose-list-item[data-filename="${CSS.escape(r.filename)}"], .node[data-filename="${CSS.escape(r.filename)}"]`);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    el.style.transition = 'box-shadow 0.15s ease';
    const prevShadow = el.style.boxShadow;
    let flashes = 0;
    const flashInterval = setInterval(() => {
      el.style.boxShadow = flashes % 2 === 0 ? '0 0 0 4px #7cc4ff' : prevShadow;
      flashes++;
      if (flashes > 5) { clearInterval(flashInterval); el.style.boxShadow = prevShadow; }
    }, 200);
  }
}

function renderVideoRing() {
  if (!lastVideoRingState) return;
  const { anchorUrl, refFrameIdx, baseResults, sourceType } = lastVideoRingState;
  const combined = applySqueeze([...baseResults, ...extraImmichNodes]);
  const metricLabel = { sim: 'Similarity', yaw: 'Yaw', pitch: 'Pitch', roll: 'Roll' }[ringSortMetric];
  const baseLabel = sourceType === 'folder' ? 'IMAGE SET ANALYSIS (local, not in Immich)' : 'VIDEO FRAME ANALYSIS (local, not in Immich)';
  const anchorLabel = sourceType === 'folder' ? 'Reference (Anchor)' : `Frame ${refFrameIdx} (Anchor)`;

  const stageEl = document.getElementById('stage');
  const listEl = document.getElementById('pose-list-view');
  const scrubberEl = document.getElementById('pose-list-scrubber');

  if (ringSortMetric === 'sim') {
    listEl.style.display = 'none';
    scrubberEl.style.display = 'none';
    stageEl.style.display = '';
    render('__anchor__', {
      mode: 'face',
      results: [{ assetId: '__anchor__', filename: anchorLabel, similarity: 1.0, thumbUrl: anchorUrl }, ...combined],
    }, anchorUrl);
  } else {
    stageEl.style.display = 'none';
    listEl.style.display = 'flex';
    scrubberEl.style.display = 'flex';
    renderPoseList(ringSortMetric, anchorUrl, anchorLabel, combined);
  }

  const hudMode = document.getElementById('hud-mode');
  if (hudMode) hudMode.textContent = `${baseLabel} · sorted by ${metricLabel}`;
}

function renderPoseList(metric, anchorUrl, anchorLabel, combined) {
  const listEl = document.getElementById('pose-list-view');
  listEl.innerHTML = '';

  const anchorWrap = document.createElement('div');
  anchorWrap.className = 'pose-list-anchor';
  anchorWrap.innerHTML = `<img src="${anchorUrl}" loading="lazy"><div class="plabel">${anchorLabel}</div>`;
  listEl.appendChild(anchorWrap);

  // sort by raw signed value, not abs() - so one end is the most-negative
  // extreme (e.g. head turned hard left) and the other end is the
  // most-positive extreme (turned hard right), with near-neutral poses
  // sitting in the middle. Items missing this metric sort to the middle too.
  const withMetric = combined.filter(r => typeof r[metric] === 'number');
  const withoutMetric = combined.filter(r => typeof r[metric] !== 'number');
  withMetric.sort((a, b) => a[metric] - b[metric]);

  const PITCH_PX_PER_DEG = 1.6;
  const PITCH_CLAMP_DEG = 40;

  withMetric.forEach(r => {
    const item = document.createElement('div');
    const isSelected = r.assetId ? selectedAssetIds.has(r.assetId) : (r.frame !== undefined && selectedFrames.has(r.frame));
    item.className = 'pose-list-item' + (isSelected ? ' selected' : '');
    item.dataset.filename = r.filename;
    const thumb = r.thumbUrl || (r.fromImmich ? `/api/thumb/${r.assetId}` : `/api/thumb/${r.assetId}`);
    item.innerHTML = `<img src="${thumb}" loading="lazy"><div class="plabel">${metric}: ${r[metric].toFixed(1)}°</div>`;
    if (typeof r.pitch === 'number') {
      // pitch up (nose up) raises the thumbnail, pitch down lowers it -
      // gives the strip a wavy "head bob" feel that mirrors the pose itself.
      const clamped = Math.max(-PITCH_CLAMP_DEG, Math.min(PITCH_CLAMP_DEG, r.pitch));
      const offsetPx = -clamped * PITCH_PX_PER_DEG;
      item.dataset.pitchY = offsetPx;
      item.style.transform = `translateY(${offsetPx}px)`;
    } else {
      item.dataset.pitchY = 0;
    }

    // single click (debounced) = browse/recenter; double click = add to
    // export queue. Debounce mirrors the person-clusters grid pattern so a
    // dblclick doesn't also fire the single-click action first.
    let clickTimer = null;
    const toggleExportSelection = () => {
      if (r.assetId) {
        if (selectedAssetIds.has(r.assetId)) { selectedAssetIds.delete(r.assetId); }
        else { selectedAssetIds.add(r.assetId); }
        updateImmichSelectionBar();
      } else if (r.frame !== undefined) {
        if (selectedFrames.has(r.frame)) { selectedFrames.delete(r.frame); }
        else { selectedFrames.add(r.frame); }
        updateSaveSelectedButton();
      }
      item.classList.toggle('selected');
    };
    item.onclick = () => {
      if (!r.assetId) return; // no recenter target for local video frames
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => {
        const input = document.getElementById('search-input');
        if (input) input.value = r.filename || '';
        loadNeighbors(r.assetId);
      }, 220);
    };
    item.ondblclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearTimeout(clickTimer);
      toggleExportSelection();
    };
    item.addEventListener('mouseenter', () => showHoverPreview(r));
    item.addEventListener('mouseleave', hideHoverPreview);
    listEl.appendChild(item);
  });

  if (withoutMetric.length) {
    const note = document.createElement('div');
    note.className = 'pose-list-item';
    note.style.opacity = '0.4';
    note.innerHTML = `<div class="plabel">+${withoutMetric.length} no ${metric} data</div>`;
    listEl.appendChild(note);
  }

  setupPoseListScrubber(listEl, metric, withMetric);
}

// ---- horizontal scrubber: drag-anywhere navigation for the pose strip.
// Essential once you're dealing with hundreds of frames - side-scrolling
// (even with a scroll wheel that supports it) is far too slow to browse
// a 700-frame set. Two-way synced with the strip's actual scroll position,
// and a vertical-wheel fallback for anyone without horizontal scroll input.
function setupPoseListScrubber(listEl, metric, withMetric) {
  const slider = document.getElementById('pose-scrub-slider');
  const leftLabel = document.getElementById('pose-scrub-left');
  const rightLabel = document.getElementById('pose-scrub-right');

  if (withMetric.length) {
    leftLabel.textContent = `${metric}: ${withMetric[0][metric].toFixed(1)}°`;
    rightLabel.textContent = `${metric}: ${withMetric[withMetric.length - 1][metric].toFixed(1)}°`;
  } else {
    leftLabel.textContent = '';
    rightLabel.textContent = '';
  }

  const maxScroll = () => Math.max(1, listEl.scrollWidth - listEl.clientWidth);

  let syncingFromScroll = false;
  slider.value = 0;
  slider.oninput = () => {
    syncingFromScroll = true;
    listEl.scrollLeft = (parseFloat(slider.value) / 1000) * maxScroll();
    syncingFromScroll = false;
  };

  listEl.onscroll = () => {
    if (syncingFromScroll) return;
    slider.value = Math.round((listEl.scrollLeft / maxScroll()) * 1000);
  };

  // vertical wheel -> horizontal scroll, so anyone without a side-scroll
  // wheel/trackpad gesture can still move through the strip with a normal mouse
  listEl.onwheel = (e) => {
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      e.preventDefault();
      listEl.scrollLeft += e.deltaY;
    }
  };
}

function addImmichNodeToRing(match, jobId, frameNo) {
  if (extraImmichNodes.some(n => n.assetId === match.assetId)) return;
  const node = {
    assetId: match.assetId,
    filename: `${match.filename} (Immich)`,
    similarity: match.similarity,
    thumbUrl: `/api/thumb/${match.assetId}`,
    fromImmich: true,
  };
  extraImmichNodes.push(node);
  selectedAssetIds.add(match.assetId);
  updateImmichSelectionBar();
  renderVideoRing();

  if (jobId !== undefined && frameNo !== undefined) {
    fetch(`/api/immich-face-pose/${jobId}/${frameNo}/${match.assetId}`)
      .then(res => res.json())
      .then(pose => {
        if (pose.error) return;
        node.yaw = pose.yaw;
        node.pitch = pose.pitch;
        node.roll = pose.roll;
        renderVideoRing();
      })
      .catch(() => {});
  }
}

function removeImmichNodeFromRing(assetId) {
  const idx = extraImmichNodes.findIndex(n => n.assetId === assetId);
  if (idx === -1) return;
  extraImmichNodes.splice(idx, 1);
  selectedAssetIds.delete(assetId);
  updateImmichSelectionBar();
  renderVideoRing();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
