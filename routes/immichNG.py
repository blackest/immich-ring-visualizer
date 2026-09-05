"""NG twin of routes/immich.py -- Immich search/neighbors ingest source.

NG-only file: duplicated from routes/immich.py as it exists today, wired
to configNG/dbNG/detectionNG/video_analysisNG instead of the originals,
per the NG duplication rule in APP_ARCHITECTURE_NOTES.md -- does not
import from or call into routes/immich.py or any non-NG module.

Scope ported so far (per "go ahead with immich next"): filename search,
pgvector nearest-neighbors (face embedding, CLIP fallback), lazy
pose/blur for one asset, thumb/preview image proxying. NOT ported yet:
analyze-immich (batch folder-style analysis over a selection),
random-face, immich-cross-check / immich-face-pose (job-scoped -- those
belong with the video analysis ring, a later slice), person-clusters,
person-assets.
"""

import numpy as np
import requests
from flask import Blueprint, request, jsonify, Response

from configNG import IMMICH_API_KEY, IMMICH_BASE_URL
from dbNG import get_conn_ng, release_conn_ng
from detectionNG import get_blur_score_ng, get_face_app_ng, pick_largest_face_ng
from video_analysisNG import bbox_frame_ratio_ng, vert_fill_ratio_ng

immichNG_bp = Blueprint('immichNG', __name__)


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
