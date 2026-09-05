"""NG twin of routes/export.py -- Export Settings-aware export endpoints.

NG-only file. Per the NG duplication rule in APP_ARCHITECTURE_NOTES.md,
this is a deliberate duplicate wired to configNG/stateNG/image_opsNG/
exporterNG/video_analysisNG/detectionNG, not an import from the original
modules, so nothing here can break index.html if it changes.

Replaces the earlier stub version of export_job_ng that used to live in
routes/videoNG.py (see APP_ARCHITECTURE_NOTES.md) -- that stub wrote
plain PNGs with no crop/resize/margin handling since the Export Settings
panel was still unported. Now that the panel is wired up, this mirrors
routes/export.py's full behavior: real crop/resize/margin/upscale-cap
handling, small-face skipping, and Immich-asset export, all reading the
same _analysis_jobs_ng job store videoNG.py/folderNG.py write into.
"""

import os

import cv2
import numpy as np
import requests
from flask import Blueprint, request, jsonify, Response

from configNG import EXPORT_DIR, IMMICH_API_KEY, IMMICH_BASE_URL
from detectionNG import get_face_app_ng, pick_best_face_ng, pick_largest_face_ng
from exporterNG import export_immich_asset_ids_ng
from image_opsNG import _export_params_from_body_ng, crop_resize_export_ng, should_skip_for_small_face_ng
from stateNG import _analysis_jobs_ng
from video_analysisNG import MemoryVideo, find_cache_frame_ng

exportNG_bp = Blueprint("exportNG", __name__)


@exportNG_bp.route("/api/ng/export-job/<job_id>", methods=["POST"])
def export_job_ng(job_id):
    job = _analysis_jobs_ng.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    source_name = job.get("sourceName", job_id)
    dest_dir = os.path.join(EXPORT_DIR, source_name)
    os.makedirs(dest_dir, exist_ok=True)

    body = request.get_json(silent=True) or {}
    selected_frames = body.get("frames")
    selected_set = set(selected_frames) if "frames" in body else None
    selected_asset_ids = body.get("assetIds") or []
    p = _export_params_from_body_ng(body)

    face_app = None
    mv = None  # lazily opened only if this job has videoBytes - avoids
               # opening a MemoryVideo for folder/immich-sourced jobs
    saved = []
    errors = []
    skipped_count = 0
    widened_count = 0
    padded_count = 0
    reencoded_count = 0  # frames that had to fall back to the analysis-time
                          # JPEG because the true original was unavailable
    for r in job["results"]:
        if not (r.get("passed") and r.get("frameId")):
            continue
        if selected_set is not None and r["frame"] not in selected_set:
            continue

        # export from the real original wherever possible, not the JPEG q88
        # cache written during analysis - see routes/export.py for the full
        # rationale (kept identical here for parity).
        img = None
        used_original = False

        if job.get("videoBytes"):
            if mv is None:
                mv = MemoryVideo(job["videoBytes"])
            raw = mv.seek_frame(r["frame"])
            if raw is not None:
                img = raw
                used_original = True
        elif job.get("srcImages") and r.get("origName"):
            raw_bytes = job["srcImages"].get(r["origName"])
            if raw_bytes is not None:
                raw = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is not None:
                    img = raw
                    used_original = True

        if img is None:
            cache_bytes, _mimetype = find_cache_frame_ng(r["frameId"])
            if cache_bytes is None:
                continue
            img = cv2.imdecode(np.frombuffer(cache_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            reencoded_count += 1

        bbox = r.get("bbox")
        if (p["mode"] == "face" or p["min_face_px"] > 0) and not bbox:
            # legacy job predating stored bbox: detect on demand
            try:
                if face_app is None:
                    face_app = get_face_app_ng()
                faces = face_app.get(img)
                if faces:
                    ref = job.get("frame_embeddings", {}).get(r["frame"])
                    face = pick_best_face_ng(faces, np.array(ref, dtype=np.float32)) if ref else pick_largest_face_ng(faces)
                    bbox = list(map(int, face.bbox))
            except Exception:
                bbox = None

        skip, reason = should_skip_for_small_face_ng(img, bbox, p)
        if skip:
            skipped_count += 1
            errors.append(f"frame {r['frame']}: skipped ({reason})")
            continue

        out_img, info = crop_resize_export_ng(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"], p["native"])
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
        immich_result = export_immich_asset_ids_ng(selected_asset_ids, dest_dir, p)
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
        # frames exported from the analysis-time JPEG cache rather than the
        # true original - should normally be 0; a nonzero count here means
        # the source video/images were no longer on disk at export time
        "reencodedFromCache": reencoded_count,
    })


@exportNG_bp.route("/api/ng/export-immich-assets", methods=["POST"])
def export_immich_assets_ng():
    body = request.get_json(silent=True) or {}
    asset_ids = body.get("assetIds") or []
    if not asset_ids:
        return jsonify({"error": "no assetIds given"}), 400
    p = _export_params_from_body_ng(body)

    dest_dir = os.path.join(EXPORT_DIR, "immich_selected")
    os.makedirs(dest_dir, exist_ok=True)

    result = export_immich_asset_ids_ng(asset_ids, dest_dir, p)
    return jsonify({
        "exported": len(result["saved"]),
        "path": dest_dir,
        "files": result["saved"],
        "errors": result["errors"],
        "widened": result["widened"],
        "padded": result["padded"],
    })


@exportNG_bp.route("/api/ng/export-preview/<job_id>/<int:frame_no>")
def export_preview_frame_ng(job_id, frame_no):
    """Renders a frame exactly as it would be written by
    /api/ng/export-job - same bbox, same crop_resize_export_ng call, same
    params - so a 'View selected' preview can show the real crop/aspect/
    native-resolution result instead of a generic thumbnail."""
    job = _analysis_jobs_ng.get(job_id)
    if not job:
        return "unknown job", 404

    r = next((x for x in job["results"] if x["frame"] == frame_no), None)
    if not r or not (r.get("passed") and r.get("frameId")):
        return "frame not available for preview (not passed / not cached)", 404

    cache_bytes, _mimetype = find_cache_frame_ng(r["frameId"])
    if cache_bytes is None:
        return "source frame no longer cached", 404
    img = cv2.imdecode(np.frombuffer(cache_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "could not read source frame", 404

    p = _export_params_from_body_ng(request.args)
    bbox = r.get("bbox")
    out_img, _info = crop_resize_export_ng(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"], p["native"])
    ok, encoded = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500
    return Response(encoded.tobytes(), mimetype="image/jpeg")


@exportNG_bp.route("/api/ng/export-preview-immich/<asset_id>")
def export_preview_immich_ng(asset_id):
    """Same idea as export_preview_frame_ng but for an Immich asset -
    fetches the original, detects the face fresh (no cached bbox for
    library assets), and applies the identical crop_resize_export_ng
    call the real Immich export path uses."""
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
        headers={"x-api-key": IMMICH_API_KEY},
        stream=True,
        timeout=60,
    )
    if r.status_code != 200:
        return f"could not fetch original: HTTP {r.status_code}", 502
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "could not decode image", 500

    p = _export_params_from_body_ng(request.args)
    bbox = None
    if p["mode"] == "face" or p["min_face_px"] > 0:
        try:
            face_app = get_face_app_ng()
            faces = face_app.get(img)
            if faces:
                bbox = list(map(int, pick_largest_face_ng(faces).bbox))
        except Exception:
            bbox = None

    out_img, _info = crop_resize_export_ng(img, bbox, p["out_w"], p["out_h"], p["mode"], p["margin"], p["interp"], p["upscale"], p["max_upscale"], p["pad_mode"], p["native"])
    ok, encoded = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500
    return Response(encoded.tobytes(), mimetype="image/jpeg")
