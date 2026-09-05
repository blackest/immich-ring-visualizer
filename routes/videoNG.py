"""immichRingNG -- video ingest + per-frame face-similarity analysis.

NG-only file. Copied from routes/video.py / video_analysis.py as they
exist today -- per the NG rule, this is a deliberate duplicate, not an
import from the original modules, so nothing here can break index.html
if it changes. Uses video_analysisNG.py / stateNG.py / detectionNG.py,
its own NG-suffixed twins of the originals.

Scope so far: load a video into a project (preview) + run full
face-similarity analysis against a chosen reference frame, poll its
status, and serve cached result frames back to the ring. One analysis
job per project tab, each in its own background thread, so multiple
character tabs can analyze in parallel -- no queuing at this layer
(queuing is for the shared-GPU character-sheet generation step, later).
"""

import os
import threading
import uuid

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, Response, send_file

from configNG import FRAME_STORE
from stateNG import _analysis_jobs_ng, _preview_jobs_ng
from video_analysisNG import MemoryVideo, run_video_analysis_ng, find_cache_frame_ng

videoNG_bp = Blueprint("videoNG", __name__)


@videoNG_bp.route("/api/ng/preview-video", methods=["POST"])
def preview_video_ng():
    """Load a video into memory for the active project. No disk writes --
    matches the no-writes-for-cheap-to-redo-ingestion principle."""
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    preview_id = uuid.uuid4().hex[:12]
    video_bytes = request.files["video"].read()

    try:
        mv = MemoryVideo(video_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not open video: {e}"}), 400

    if mv.fps <= 0 or mv.frame_count <= 0:
        return jsonify({"error": "Could not determine video FPS/frame count"}), 400

    _preview_jobs_ng[preview_id] = {
        "videoBytes": video_bytes,
        "fps": mv.fps,
        "frames": mv.frame_count,
    }

    return jsonify({
        "previewId": preview_id,
        "fps": mv.fps,
        "totalFrames": mv.frame_count,
        "duration": mv.frame_count / mv.fps,
    })


@videoNG_bp.route("/api/ng/preview-frame/<preview_id>/<int:frame_no>")
def preview_frame_ng(preview_id, frame_no):
    """One exact decoded frame (1-based), as JPEG -- used for frame-preview
    scrubbing before committing to a full analysis run."""
    job = _preview_jobs_ng.get(preview_id)
    if not job:
        return "", 404

    frame_no = max(1, min(job["frames"], frame_no))
    mv = MemoryVideo(job["videoBytes"])
    frame = mv.seek_frame(frame_no)
    if frame is None:
        return "", 404

    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return "", 500

    return Response(encoded.tobytes(), mimetype="image/jpeg")


@videoNG_bp.route("/api/ng/analyze-video", methods=["POST"])
def analyze_video_ng():
    """Run a face-similarity pass over an already-loaded video's frames
    against a chosen reference frame. Re-uploads the video bytes (same
    tradeoff routes/video.py makes) rather than requiring the caller to
    keep the preview job's bytes around -- previewId isn't required here,
    the project just needs to already have the File object client-side."""
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    sim_threshold = float(request.form.get("simThreshold", 0.1))
    blur_threshold = float(request.form.get("blurThreshold", 1))
    ref_frame = int(request.form.get("refFrame", 1))
    cache_format = "png" if request.form.get("cacheFormat") == "png" else "jpg"
    start_sec_raw = request.form.get("startSec", "").strip()
    end_sec_raw = request.form.get("endSec", "").strip()
    start_sec = float(start_sec_raw) if start_sec_raw else None
    end_sec = float(end_sec_raw) if end_sec_raw else None

    job_id = uuid.uuid4().hex[:12]
    video_bytes = request.files["video"].read()
    source_name = os.path.splitext(request.files["video"].filename or "clip")[0]

    _analysis_jobs_ng[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "videoBytes": video_bytes,
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
        "startSec": start_sec,
        "endSec": end_sec,
    }

    t = threading.Thread(
        target=run_video_analysis_ng,
        args=(job_id, video_bytes, sim_threshold, blur_threshold, ref_frame, cache_format, start_sec, end_sec),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id})


@videoNG_bp.route("/api/ng/analysis-status/<job_id>")
def analysis_status_ng(job_id):
    job = _analysis_jobs_ng.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "error": job.get("error"),
        "frameCount": len(job["results"]),
        "results": job["results"],
        "simThreshold": job.get("simThreshold", 0.1),
        "blurThreshold": job.get("blurThreshold", 1),
        "resolutionSummary": job.get("resolutionSummary"),
    })


@videoNG_bp.route("/api/ng/framefile/<frame_id>")
def frame_file_ng(frame_id):
    img_bytes, mimetype = find_cache_frame_ng(frame_id)
    if img_bytes is None:
        return "", 404
    return Response(img_bytes, mimetype=mimetype)


@videoNG_bp.route("/api/ng/build-playback/<job_id>", methods=["POST"])
def build_playback_ng(job_id):
    """Reassemble the full clip with rejected frames replaced by a labeled
    blank frame (NO FACE / BLURRY / LOW MATCH), so you can scrub the whole
    take and see exactly where similarity or blur dropped out. Ported
    from routes/video.py's build_playback -- NG-only twin, writes to
    configNG.FRAME_STORE and reads from _analysis_jobs_ng."""
    job = _analysis_jobs_ng.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done":
        return jsonify({"error": "analysis not finished yet"}), 400

    video_bytes = job.get("videoBytes")
    if not video_bytes:
        return jsonify({"error": "source video no longer available"}), 400

    by_frame = {r["frame"]: r for r in job["results"]}

    mv = MemoryVideo(video_bytes)
    fps = mv.fps or 24.0
    width, height = mv.width, mv.height

    raw_path = os.path.join(FRAME_STORE, f"{job_id}_playback_raw.mp4")
    out_path = os.path.join(FRAME_STORE, f"{job_id}_playback.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    blank = np.zeros((height, width, 3), dtype=np.uint8)

    for frame_idx, frame in mv.iter_frames():
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
    return jsonify({"ready": True, "url": f"/api/ng/playback-file/{job_id}", "fps": fps})


@videoNG_bp.route("/api/ng/playback-file/<job_id>")
def playback_file_ng(job_id):
    job = _analysis_jobs_ng.get(job_id)
    if not job or not job.get("playbackPath") or not os.path.exists(job["playbackPath"]):
        return "", 404
    return send_file(job["playbackPath"], mimetype="video/mp4")
