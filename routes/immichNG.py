"""NG twin of routes/immich.py -- Immich search/neighbors ingest source.

NG-only file: duplicated from routes/immich.py as it exists today, wired
to configNG/dbNG/detectionNG/video_analysisNG instead of the originals,
per the NG duplication rule in APP_ARCHITECTURE_NOTES.md -- does not
import from or call into routes/immich.py or any non-NG module.

Scope ported so far (per "go ahead with immich next", later "port
person clusters", and later "port analyze-immich"): filename search,
pgvector nearest-neighbors (face embedding, CLIP fallback), lazy
pose/blur for one asset, thumb/preview image proxying, person-clusters,
person-assets, analyze-immich (batch folder-style pose/blur analysis
over a selection of Immich assets). NOT ported yet: random-face,
immich-cross-check / immich-face-pose (job-scoped -- those belong with
the video analysis ring, a later slice).
"""

import os
import threading
import uuid

import numpy as np
import requests
from flask import Blueprint, request, jsonify, Response

from configNG import IMMICH_API_KEY, IMMICH_BASE_URL
from dbNG import get_conn_ng, release_conn_ng
from detectionNG import get_blur_score_ng, get_face_app_ng, pick_largest_face_ng
from folder_analysisNG import run_folder_analysis_ng
from stateNG import _analysis_jobs_ng
from video_analysisNG import bbox_frame_ratio_ng, vert_fill_ratio_ng

immichNG_bp = Blueprint('immichNG', __name__)


@immichNG_bp.route("/api/ng/analyze-immich", methods=["POST"])
def analyze_immich_ng():
    """NG twin of routes/immich.py's analyze_immich -- same analysis
    pipeline as analyze-folder (pose extraction, sim/blur gating), but
    sourced from Immich asset IDs already known to the project (a ticked
    selection built from search-result neighbors or a person-cluster
    grid -- see appNG.js's selectedAssetIds) instead of files uploaded
    from the local filesystem. This is what lets an Immich-sourced set
    feed the Pose Picker / Shot Scale Picker, which read project.ring
    the same way a video or folder/zip analysis does -- until this
    route existed, "immich" task results (immichRing) carried no pose
    data up front and never populated project.ring at all.

    Downloaded assets are held in memory as (orig_name, bytes) and
    handed to run_folder_analysis_ng directly -- nothing here is
    written to disk, matching the no-disk-writes-for-ingest principle
    already used by videoNG.py / folderNG.py.
    """
    body = request.get_json(force=True) or {}
    asset_ids = body.get("assetIds") or []
    sim_threshold = float(body.get("simThreshold", 0.1))
    blur_threshold = float(body.get("blurThreshold", 1))
    ref_index = int(body.get("refIndex", 1))
    cache_format = "png" if body.get("cacheFormat") == "png" else "jpg"

    if not asset_ids:
        return jsonify({"error": "provide 'assetIds' (non-empty list)"}), 400

    job_id = uuid.uuid4().hex[:12]

    images = []  # list of (orig_name, bytes), in fetch order
    fetch_errors = []
    for asset_id in asset_ids:
        try:
            meta = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}",
                headers={"x-api-key": IMMICH_API_KEY},
                timeout=20,
            ).json()
            orig_name = meta.get("originalFileName") or f"{asset_id}.jpg"

            r = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
                headers={"x-api-key": IMMICH_API_KEY},
                timeout=60,
            )
            if r.status_code != 200:
                fetch_errors.append(f"{asset_id}: HTTP {r.status_code}")
                continue

            # prefix with the asset id so filenames can't collide across
            # assets and so the exported frame stays traceable back to
            # the Immich library item it came from
            safe_name = f"{asset_id}_{os.path.basename(orig_name)}"
            images.append((safe_name, r.content))
        except Exception as e:
            fetch_errors.append(f"{asset_id}: {e}")

    if not images:
        return jsonify({"error": "could not fetch any of the requested Immich assets", "fetchErrors": fetch_errors}), 400

    images.sort(key=lambda pair: pair[0].lower())

    _analysis_jobs_ng[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": f"immich_selection_{len(images)}", "sourceType": "immich",
        "srcImages": dict(images),
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
    }

    t = threading.Thread(
        target=run_folder_analysis_ng,
        args=(job_id, images, sim_threshold, blur_threshold, ref_index, cache_format),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(images), "fetchErrors": fetch_errors})


@immichNG_bp.route("/api/ng/find-by-filename")
def find_by_filename_ng():
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400

    conn = get_conn_ng()
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
        release_conn_ng(conn)


@immichNG_bp.route("/api/ng/neighbors")
def neighbors_ng():
    asset_id = request.args.get("assetId")
    if not asset_id:
        return jsonify({"error": "assetId required"}), 400

    limit = int(request.args.get("limit", 30))

    conn = get_conn_ng()
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
        release_conn_ng(conn)

    results = [
        {"assetId": r[0], "filename": r[1], "similarity": float(r[2])}
        for r in rows
    ]
    return jsonify({"mode": mode, "results": results})


@immichNG_bp.route("/api/ng/asset-face-pose/<asset_id>")
def asset_face_pose_ng(asset_id):
    """Pose/blur/frame-fill metrics for the largest face in an Immich
    asset preview -- called lazily, once, for whichever asset is
    currently centered in an Immich ring (see appNG.js's
    loadImmichCenterPose), not for every neighbor thumbnail."""
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
        faces = get_face_app_ng().get(frame)
    except Exception as e:
        return jsonify({"error": f"face detection failed: {e}"}), 500

    if not faces:
        return jsonify({"error": "no face detected in Immich thumbnail"}), 404

    face = pick_largest_face_ng(faces)
    pitch, yaw, roll = (float(p) for p in face.pose)
    x1, y1, x2, y2 = map(int, face.bbox)
    fh, fw = frame.shape[:2]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    blur_score = get_blur_score_ng(crop)

    return jsonify({
        "assetId": asset_id, "yaw": yaw, "pitch": pitch, "roll": roll,
        "blur": blur_score,
        "bboxRatio": bbox_frame_ratio_ng([x1, y1, x2, y2], fw, fh),
        "vertFillPct": vert_fill_ratio_ng([x1, y1, x2, y2], fw, fh),
    })


@immichNG_bp.route("/api/ng/thumb/<asset_id>")
def thumb_ng(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "thumbnail"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))


@immichNG_bp.route("/api/ng/preview/<asset_id>")
def preview_ng(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "preview"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))


@immichNG_bp.route("/api/ng/person-clusters")
def person_clusters_ng():
    """Rank named Immich persons by how tightly their faces cluster in
    embedding space. High avg_sim usually means either very consistent
    real-world photos of that person, or a pile of near-duplicate stills
    (e.g. screenshotted from video) -- worth eyeballing before using as a
    LoRA source. No video job or upload required, pure Immich DB query."""
    min_faces = int(request.args.get("minFaces", 5))
    limit = int(request.args.get("limit", 30))

    conn = get_conn_ng()
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
        release_conn_ng(conn)

    return jsonify([
        {"personId": r[0], "name": r[1] or "(unnamed)", "faceCount": r[2], "avgSim": float(r[3])}
        for r in rows
    ])


@immichNG_bp.route("/api/ng/person-assets/<person_id>")
def person_assets_ng(person_id):
    """All (or up to `limit`) assets Immich has tagged for a given
    personId. Used to populate the thumbnail grid when a person-cluster
    row is opened, and as the source list for building a selection to
    export."""
    limit = int(request.args.get("limit", 200))

    conn = get_conn_ng()
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
        release_conn_ng(conn)

    return jsonify([{"assetId": r[0], "filename": r[1]} for r in rows])

