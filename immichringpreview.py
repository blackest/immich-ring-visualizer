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

            # face.pose is [yaw, pitch, roll] in degrees
            yaw, pitch, roll = (float(p) for p in face.pose)

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
                "yaw": yaw, "pitch": pitch, "roll": roll
            })
            job["results"] = results

        cap.release()
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


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
    selected_set = set(selected_frames) if selected_frames else None

    saved = []
    for r in job["results"]:
        if not (r.get("passed") and r.get("frameId")):
            continue
        if selected_set is not None and r["frame"] not in selected_set:
            continue
        src = os.path.join(FRAME_STORE, f"{r['frameId']}.jpg")
        if os.path.exists(src):
            dest_name = f"fr{r['frame']:05d}_sim{r['sim']:.2f}.png"
            dest_path = os.path.join(dest_dir, dest_name)
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dest_path, img)
                saved.append(dest_name)

    return jsonify({"exported": len(saved), "path": dest_dir, "files": saved})


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


@app.route("/api/export-immich-assets", methods=["POST"])
def export_immich_assets():
    body = request.get_json(silent=True) or {}
    asset_ids = body.get("assetIds") or []
    if not asset_ids:
        return jsonify({"error": "no assetIds given"}), 400

    dest_dir = os.path.join(EXPORT_DIR, "immich_selected")
    os.makedirs(dest_dir, exist_ok=True)

    saved = []
    errors = []
    for asset_id in asset_ids:
        try:
            meta = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}",
                headers={"x-api-key": IMMICH_API_KEY},
            ).json()
            orig_name = meta.get("originalFileName", f"{asset_id}.jpg")
            ext = os.path.splitext(orig_name)[1] or ".jpg"

            r = requests.get(
                f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
                headers={"x-api-key": IMMICH_API_KEY},
                stream=True,
            )
            if r.status_code != 200:
                errors.append(f"{asset_id}: HTTP {r.status_code}")
                continue

            dest_name = f"{os.path.splitext(orig_name)[0]}_{asset_id[:8]}{ext}"
            dest_path = os.path.join(dest_dir, dest_name)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            saved.append(dest_name)
        except Exception as e:
            errors.append(f"{asset_id}: {e}")

    return jsonify({"exported": len(saved), "path": dest_dir, "files": saved, "errors": errors})


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
  #hud {
    position: fixed;
    top: 16px;
    left: 16px;
    z-index: 10;
    font-size: 13px;
    color: var(--dim);
    background: rgba(10, 10, 14, 0.72);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid #2a2a32;
    border-radius: 10px;
    padding: 10px 12px;
    max-width: min(360px, calc(100vw - 32px));
  }
  #hud .fname {
    color: var(--text);
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 2px;
  }
  #hud .mode {
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
    display: none;
    position: fixed;
    left: 16px;
    bottom: 16px;
    top: auto;
    width: min(480px, calc(100vw - 372px));
    max-height: calc(100vh - 360px);
    overflow: hidden;
    margin: 0;
    background: #101015;
    border: 1px solid #2a2a32;
    border-radius: 10px;
    padding: 10px;
    z-index: 20;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
  }
  #preview-canvas {
    width: 100%;
    height: auto;
    max-height: calc(100vh - 440px);
    object-fit: contain;
    border-radius: 6px;
    background: #000;
    display: block;
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
    overflow-y: auto;
    z-index: 10;
  }
  #sidebar-header {
    padding: 16px 16px 10px 16px;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: var(--dim);
    text-transform: uppercase;
    border-bottom: 1px solid #1c1c22;
    position: sticky;
    top: 0;
    background: #0e0e12;
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
<div id="hud">
  <div style="margin-top: 8px; display: flex; align-items: center; gap: 8px; font-size: 11px;">
    <label for="ring-scale-input">Ring Scale:</label>
    <input id="ring-scale-input" type="range" min="25" max="400" value="100" style="flex: 1;">
    <span id="ring-scale-val">100%</span>
  </div>
  <div class="fname" id="hud-filename">loading…</div>
  <div id="hud-sub">Immich Ring Visualizer</div>
  <div class="mode" id="hud-mode"></div>
  <div id="search-box">
    <input id="search-input" type="text" placeholder="Type a filename to start from…" autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="video-drop">
    <div style="display: flex; gap: 6px; margin-bottom: 8px;">
      <input id="sim-threshold" type="number" step="0.05" value="0.65" title="Similarity Threshold" style="width: 120px; background: #16161c; border: 1px solid #2a2a32; color: var(--text); padding: 4px; border-radius: 4px;">
      <input id="blur-threshold" type="number" step="1" value="50" title="Blur Threshold" style="width: 120px; background: #16161c; border: 1px solid #2a2a32; color: var(--text); padding: 4px; border-radius: 4px;">
    </div>
    <input type="file" id="video-file-input" accept="video/*" style="display:none">
    <div id="video-drop-zone">
      Select video
    </div>
    <div id="video-status"></div>
  </div>
</div>
<div id="preview-container">
  <div style="font-size:11px;color:var(--dim);margin-bottom:7px;">Reference frame — use ← / → to step one actual video frame</div>
  <canvas id="preview-canvas"></canvas>
  <audio id="reference-audio" preload="auto" style="display:none"></audio>
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

<div id="stage"><div id="loading">loading…</div></div>
<div id="sidebar">
  <div id="sidebar-header">Ranked matches</div>
  <div id="immich-selection-bar" style="display:none;padding:8px 16px;border-bottom:1px solid #212127;background:#12121a;">
    <div style="font-size:10px;color:var(--dim);margin-bottom:5px;">
      <span id="immich-selection-count">0</span> Immich image(s) selected — sticks across browsing
    </div>
    <div style="display:flex;gap:6px;">
      <button id="immich-export-selected-btn" style="flex:1;padding:5px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:10px;">Export selected to disk</button>
      <button id="immich-clear-selected-btn" style="padding:5px 8px;background:#2a1a1a;border:1px solid #4a2a2a;color:#d98080;border-radius:6px;cursor:pointer;font-size:10px;">Clear</button>
    </div>
  </div>
  <div id="list-body"></div>
</div>

<video id="hidden-video" style="display:none;" muted playsinline></video>

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
  document.getElementById('hud-mode').textContent = data.mode === 'face' ? 'FACE SIMILARITY' : 'CLIP (WHOLE-IMAGE) SIMILARITY';

  const results = data.results.filter(r => r.assetId !== centerId);
  const centerResult = data.results.find(r => r.assetId === centerId) || data.results[0];
  document.getElementById('hud-filename').textContent = centerResult ? centerResult.filename : '';

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
      node.innerHTML = `<img src="${thumbUrlFor(r)}" loading="lazy">`;
      node.onclick = () => (r.assetId && !r.fromImmich) ? loadNeighbors(r.assetId) : null;
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

function updateSaveSelectedButton() {
  const btn = document.getElementById('save-selected-btn');
  if (!btn) return;
  const n = selectedFrames.size;
  btn.textContent = `Save ${n} selected frame${n === 1 ? '' : 's'} to disk`;
  btn.disabled = n === 0;
  btn.style.opacity = n === 0 ? '0.5' : '1';
}

const selectedAssetIds = new Set();

function updateImmichSelectionBar() {
  const bar = document.getElementById('immich-selection-bar');
  const countEl = document.getElementById('immich-selection-count');
  if (!bar || !countEl) return;
  const n = selectedAssetIds.size;
  countEl.textContent = n;
  bar.style.display = n > 0 ? 'block' : 'none';
}

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
      body: JSON.stringify({ assetIds }),
    });
    const result = await res.json();
    if (result.error) {
      btn.textContent = 'Error: ' + result.error;
    } else {
      btn.textContent = `Saved ${result.exported} of ${assetIds.length} → ${result.path}`;
      if (result.errors && result.errors.length) {
        console.warn('Some exports failed:', result.errors);
      }
    }
  } catch (e) {
    btn.textContent = 'Request failed';
  }
  setTimeout(() => { btn.textContent = prevText; btn.disabled = false; }, 4000);
});

document.getElementById('immich-clear-selected-btn').addEventListener('click', () => {
  selectedAssetIds.clear();
  updateImmichSelectionBar();
  if (lastVideoRingState) renderVideoRing();
  else {
    document.querySelectorAll('.asset-select-cb').forEach(cb => { cb.checked = false; });
  }
});

async function init() {
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

    previewContainer.style.display = 'block';
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
  pollAnalysis(data.jobId, videoFile.name, refFrameIdx);
}

async function pollAnalysis(jobId, videoFileName, refFrameIdx) {
  const res = await fetch(`/api/analysis-status/${jobId}`);
  const data = await res.json();

  if (data.status === 'error') {
    videoStatus.textContent = 'Error: ' + data.error;
    return;
  }

  const passed = data.results.filter(r => r.passed).length;
  const failedSim = data.results.filter(r => !r.passed && r.failReason === 'sim').length;
  const failedBlur = data.results.filter(r => !r.passed && r.failReason === 'blur').length;

  videoStatus.innerHTML = `
    Processing… ${data.frameCount} frames seen<br>
    ${passed} kept, ${failedSim} low sim, ${failedBlur} blurry
    <div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, (data.frameCount/200)*100)}%"></div></div>
  `;

  if (data.status === 'running') {
    setTimeout(() => pollAnalysis(jobId, videoFileName, refFrameIdx), 800);
    return;
  }

  videoStatus.innerHTML = `
    Done — ${passed}/${data.frameCount} frames kept.
    <div id="sim-sparkline-wrap" style="margin-top:8px;"></div>
    <button id="export-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Save kept frames to disk</button>
    <button id="save-selected-btn" disabled style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;opacity:0.5;">Save 0 selected frames to disk</button>
    <button id="playback-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Build playback (rejected frames blanked)</button>
    <video id="playback-video" controls style="width:100%;margin-top:8px;display:none;border-radius:6px;"></video>
    <div id="playback-frame-controls" style="display:none;margin-top:6px;gap:6px;">
      <button id="playback-prev-frame" class="btn-seek" style="flex:1;">◀ -1 frame</button>
      <button id="playback-next-frame" class="btn-seek" style="flex:1;">+1 frame ▶</button>
    </div>
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #22222a;">
      <div style="font-size:11px;color:var(--dim);margin-bottom:6px;">
        Is this kept frame actually the same person as what's already in Immich?
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
  document.getElementById('crosscheck-btn').onclick = async () => {
    const input = document.getElementById('crosscheck-frame-input');
    const frameNo = parseInt(input.value, 10) || currentFrameIdx;
    const btn = document.getElementById('crosscheck-btn');
    const out = document.getElementById('crosscheck-results');
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
              <button class="cc-add-btn" data-idx="${i}" style="font-size:9px;padding:3px 6px;background:#2a1a3a;border:1px solid #4a2a6a;color:#d4a5ff;border-radius:4px;cursor:pointer;flex-shrink:0;">+ Ring</button>
            </div>
          `).join('');
        out.querySelectorAll('.cc-add-btn').forEach(btn => {
          btn.addEventListener('click', () => {
            const match = result.results[parseInt(btn.dataset.idx, 10)];
            addImmichNodeToRing(match);
            btn.textContent = 'Added ✓';
            btn.disabled = true;
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
    btn.textContent = 'Saving…';
    btn.disabled = true;
    const res = await fetch(`/api/export-job/${jobId}`, { method: 'POST' });
    const result = await res.json();
    btn.textContent = `Saved ${result.exported} frames → ${result.path}`;
  };
  document.getElementById('save-selected-btn').onclick = async () => {
    const btn = document.getElementById('save-selected-btn');
    const frames = Array.from(selectedFrames);
    if (!frames.length) return;
    const prevText = btn.textContent;
    btn.textContent = 'Saving…';
    btn.disabled = true;
    const res = await fetch(`/api/export-job/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames }),
    });
    const result = await res.json();
    btn.textContent = `Saved ${result.exported} of ${frames.length} selected → ${result.path}`;
    setTimeout(updateSaveSelectedButton, 3000);
  };
  document.getElementById('playback-btn').onclick = async () => {
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
  };

  const results = data.results
    .filter(r => r.passed)
    .map(r => ({
      filename: `frame_${r.frame}`,
      frame: r.frame,
      similarity: r.sim,
      thumbUrl: `/api/framefile/${r.frameId}`,
      pitch: r.pitch,
      yaw: r.yaw,
      roll: r.roll,
    }))
    .sort((a, b) => b.similarity - a.similarity);

  const anchorUrl = `/api/framefile/${jobId}_anchor`;
  document.getElementById('hud-mode').textContent = 'VIDEO FRAME ANALYSIS (local, not in Immich)';
  document.getElementById('hud-filename').textContent = videoFileName;

  lastVideoRingState = {
    anchorUrl,
    refFrameIdx,
    baseResults: results,
  };
  renderVideoRing();
  updateSaveSelectedButton();
}

let lastVideoRingState = null;
const extraImmichNodes = [];

function renderVideoRing() {
  if (!lastVideoRingState) return;
  const { anchorUrl, refFrameIdx, baseResults } = lastVideoRingState;
  const combined = [...baseResults, ...extraImmichNodes];
  render('__anchor__', {
    mode: 'face',
    results: [{ assetId: '__anchor__', filename: `Frame ${refFrameIdx} (Anchor)`, similarity: 1.0, thumbUrl: anchorUrl }, ...combined],
  }, anchorUrl);
}

function addImmichNodeToRing(match) {
  if (extraImmichNodes.some(n => n.assetId === match.assetId)) return;
  extraImmichNodes.push({
    assetId: match.assetId,
    filename: `${match.filename} (Immich)`,
    similarity: match.similarity,
    thumbUrl: `/api/thumb/${match.assetId}`,
    fromImmich: true,
  });
  renderVideoRing();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)