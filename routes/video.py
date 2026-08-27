from flask import Blueprint, request, jsonify, Response, send_file
from config import FRAME_STORE
from ring_viz import app
from state import _analysis_jobs, _preview_jobs
from video_analysis import MemoryVideo, find_cache_frame, run_video_analysis

video_bp = Blueprint('video', __name__)

@video_bp.route("/api/preview-video", methods=["POST"])
def preview_video():
    """Prepare a video for exact frame-by-frame preview - used to scrub
    around and pick a good reference/anchor frame before committing to a
    full analysis run. Held entirely in memory: this upload used to write
    its own separate copy of the video to disk (on top of the second,
    independent copy analyze-video writes if you go on to run analysis on
    the same clip) - real duplicate SSD writes for the same source file."""
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

    _preview_jobs[preview_id] = {
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

@video_bp.route("/api/preview-frame/<preview_id>/<int:frame_no>")
def preview_frame(preview_id, frame_no):
    """Return one exact decoded video frame, 1-based."""
    import cv2

    job = _preview_jobs.get(preview_id)
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

@video_bp.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "video file required"}), 400

    sim_threshold = float(request.form.get("simThreshold", 0.65))
    blur_threshold = float(request.form.get("blurThreshold", 100))
    ref_frame = int(request.form.get("refFrame", 1))
    cache_format = "png" if request.form.get("cacheFormat") == "png" else "jpg"
    # optional analysis window, in seconds into the clip - lets a long
    # video skip straight to the section that matters instead of always
    # decoding from frame 1. Either end can be omitted.
    start_sec_raw = request.form.get("startSec", "").strip()
    end_sec_raw = request.form.get("endSec", "").strip()
    start_sec = float(start_sec_raw) if start_sec_raw else None
    end_sec = float(end_sec_raw) if end_sec_raw else None

    job_id = uuid.uuid4().hex[:12]
    # read the upload straight into memory rather than saving it to disk -
    # this is the one file that used to hit the SSD unconditionally on
    # every video job, whether or not anything ever got exported. It's
    # held in the job dict for the job's lifetime so export/playback/
    # re-analysis can all reopen it via MemoryVideo without a second
    # upload or a disk round-trip.
    video_bytes = request.files["video"].read()

    source_name = os.path.splitext(request.files["video"].filename or "clip")[0]
    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "videoBytes": video_bytes,
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
        "startSec": start_sec,
        "endSec": end_sec,
    }

    t = threading.Thread(
        target=run_video_analysis,
        args=(job_id, video_bytes, sim_threshold, blur_threshold, ref_frame, cache_format, start_sec, end_sec),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id})

@video_bp.route("/api/analysis-status/<job_id>")
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
        "blurThreshold": job.get("blurThreshold", 50),
        "resolutionSummary": job.get("resolutionSummary"),
    })

@video_bp.route("/api/framefile/<frame_id>")
def frame_file(frame_id):
    img_bytes, mimetype = find_cache_frame(frame_id)
    if img_bytes is None:
        return "", 404
    return Response(img_bytes, mimetype=mimetype)

@video_bp.route("/api/build-playback/<job_id>", methods=["POST"])
def build_playback(job_id):
    import cv2

    job = _analysis_jobs.get(job_id)
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
    return jsonify({"ready": True, "url": f"/api/playback-file/{job_id}", "fps": fps})

@video_bp.route("/api/playback-file/<job_id>")
def playback_file(job_id):
    job = _analysis_jobs.get(job_id)
    if not job or not job.get("playbackPath") or not os.path.exists(job["playbackPath"]):
        return "", 404
    return send_file(job["playbackPath"], mimetype="video/mp4")

