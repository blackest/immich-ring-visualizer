import os
import uuid
import threading
import requests
import numpy as np
import cv2
import shutil
from flask import Blueprint, request, jsonify, Response, send_file
from config import FRAME_STORE, IMAGE_EXTS, IMMICH_API_KEY, IMMICH_BASE_URL
from db import get_conn, release_conn
from detection import get_blur_score, get_face_app, pick_best_face, pick_largest_face
from folder_analysis import run_folder_analysis
from state import _analysis_jobs
from video_analysis import bbox_frame_ratio, vert_fill_ratio

immich_bp = Blueprint('immich', __name__)

@immich_bp.route("/api/analyze-immich", methods=["POST"])
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
    cache_format = "png" if body.get("cacheFormat") == "png" else "jpg"

    if not asset_ids:
        return jsonify({"error": "provide 'assetIds' (non-empty list)"}), 400

    job_id = uuid.uuid4().hex[:12]
    src_dir = os.path.join(FRAME_STORE, f"{job_id}_srcimgs")
    os.makedirs(src_dir, exist_ok=True)

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
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
    }

    t = threading.Thread(
        target=run_folder_analysis,
        args=(job_id, saved_paths, sim_threshold, blur_threshold, ref_index, cache_format),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(saved_paths), "fetchErrors": fetch_errors})

@immich_bp.route("/api/random-face")
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

@immich_bp.route("/api/find-by-filename")
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

@immich_bp.route("/api/asset-face-pose/<asset_id>")
def asset_face_pose(asset_id):
    """Return pose/blur/frame-fill metrics for the largest face in an
    Immich asset preview. Kept at this same URL even though it now
    returns more than pose - functionally an Immich asset is no
    different from a video frame or folder image once we're running the
    same detector over it, so it gets the same metric set (minus
    similarity, since that's a separate concern already known from
    wherever this asset appeared - e.g. a neighbors/cross-check match)."""
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
    x1, y1, x2, y2 = map(int, face.bbox)
    fh, fw = frame.shape[:2]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    blur_score = get_blur_score(crop)

    return jsonify({
        "assetId": asset_id, "yaw": yaw, "pitch": pitch, "roll": roll,
        "blur": blur_score,
        "bboxRatio": bbox_frame_ratio([x1, y1, x2, y2], fw, fh),
        "vertFillPct": vert_fill_ratio([x1, y1, x2, y2], fw, fh),
    })

@immich_bp.route("/api/neighbors")
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

@immich_bp.route("/api/immich-cross-check/<job_id>/<int:frame_no>")
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

@immich_bp.route("/api/immich-face-pose/<job_id>/<int:frame_no>/<asset_id>")
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

@immich_bp.route("/api/person-clusters")
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

@immich_bp.route("/api/person-assets/<person_id>")
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

@immich_bp.route("/api/thumb/<asset_id>")
def thumb(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "thumbnail"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))

@immich_bp.route("/api/preview/<asset_id>")
def preview_size(asset_id):
    r = requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail",
        headers={"x-api-key": IMMICH_API_KEY},
        params={"size": "preview"},
        stream=True,
    )
    return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))

