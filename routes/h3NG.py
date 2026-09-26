"""immichRingNG -- HTTP layer for the H3 view (MiniMax-H3 prompt/image-
to-video, see h3_engineNG.py / h3_jobsNG.py).

Twin of routes/videogenNG.py's job-queue shape: POST enqueues and
returns a job id immediately, the frontend polls GET .../jobs/<id> for
progress. Trimmed relative to videogenNG.py -- no /enhance, /discuss,
/chat (those are LTX's own Gemma-side features, out of scope here; see
h3_engineNG.py's docstring for what's deliberately not built yet) and
no LoRA dropdown (not wired server-side either, same reason).
"""

import os

from flask import Blueprint, jsonify, request, send_file

import h3_engineNG
import h3_jobsNG as h3_jobs
from video_analysisNG import find_cache_frame_ng

h3NG_bp = Blueprint("h3NG", __name__)

_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


def _resolve_ref_image_bytes():
    """Same shape as routes/videogenNG.py's own helper -- ref_frame_id
    (a live curation-session frame already cached server-side) takes
    priority over an uploaded `file` field, resolved in-process rather
    than round-tripping the browser. Returns (bytes, ext) or (None,
    None) if neither field was sent."""
    ref_frame_id = request.form.get("ref_frame_id")
    if ref_frame_id:
        img_bytes, mimetype = find_cache_frame_ng(ref_frame_id)
        if img_bytes is None:
            raise LookupError(
                f"reference frame {ref_frame_id!r} is no longer cached -- "
                f"pick the reference again")
        return img_bytes, _CONTENT_TYPE_EXT.get((mimetype or "").lower(), ".jpg")
    fld = request.files.get("file")
    if fld is not None:
        ext = os.path.splitext(fld.filename or "")[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = _CONTENT_TYPE_EXT.get((fld.mimetype or "").lower(), ".jpg")
        return fld.read(), ext
    return None, None


@h3NG_bp.route("/api/ng/h3/status", methods=["GET"])
def h3_status_ng():
    """Reports whether the requested model (h3 or h3q8, default h3q8)
    is reachable -- the frontend gates the Generate button on this and
    uses it to grey out whichever model isn't installed on this Mac."""
    model = request.args.get("model", "h3q8")
    if model not in ("h3", "h3q8"):
        return jsonify({"error": f"model must be 'h3' or 'h3q8' (got {model!r})"}), 400
    health = h3_engineNG.h3_health_ng(model)
    return jsonify({
        "reachable": health["ready"],
        **health,
        "default_width": h3_engineNG.H3_WIDTH,
        "default_height": h3_engineNG.H3_HEIGHT,
        "default_steps": h3_engineNG.H3_STEPS,
        "qualities": h3_engineNG.H3_QUALITIES,
        "quality_timeouts_s": h3_engineNG.H3_QUALITY_TIMEOUTS,
        "dim_step": h3_engineNG.H3_DIM_STEP,
        "min_dim": h3_engineNG.H3_MIN_DIM,
        "max_dim": h3_engineNG.H3_MAX_DIM,
        "min_duration_s": h3_engineNG.H3_MIN_DURATION_S,
        "max_duration_s": h3_engineNG.H3_MAX_DURATION_S,
    })


@h3NG_bp.route("/api/ng/h3/generate", methods=["POST"])
def h3_generate_ng():
    """Form fields: file (multipart -- the keyframe image) OR
    ref_frame_id (a live curation-session frame) -- omit both for a
    text-to-video job with no keyframe. Also: prompt (required),
    duration_s (optional float, default 5.0), model (optional,
    "h3"|"h3q8", default "h3q8"), seed (optional int), quality
    (optional, one of h3_engineNG.H3_QUALITIES -- resolved to width/
    height and to that tier's watchdog budget, H3_QUALITY_TIMEOUTS,
    here), width/height (optional ints, multiples of 32, override
    quality's dims when given -- the tier's timeout still applies since
    it's resolved from `quality` alone), steps (optional int).

    Same "no scratch temp file" reasoning as videogenNG_generate_ng:
    h3_jobs.start_h3_job_ng() writes the keyframe bytes straight into
    the job's own permanent-for-its-lifetime directory."""
    prompt = str(request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    duration_s = request.form.get("duration_s", 5.0, type=float)
    model = str(request.form.get("model") or "h3q8").strip()
    seed = request.form.get("seed", type=int)
    width = request.form.get("width", type=int)
    height = request.form.get("height", type=int)
    quality = request.form.get("quality")
    if quality and width is None and height is None:
        try:
            width, height = h3_engineNG.resolve_h3_quality_ng(quality)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    steps = request.form.get("steps", type=int)
    timeout_s = h3_engineNG.resolve_h3_quality_timeout_ng(quality)

    try:
        img_bytes, ext = _resolve_ref_image_bytes()
    except LookupError as e:
        return jsonify({"error": str(e)}), 404

    try:
        job = h3_jobs.start_h3_job_ng(
            img_bytes, ext or ".jpg", prompt, duration_s, seed,
            model=model, width=width, height=height, steps=steps,
            timeout_s=timeout_s)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "poll_url": f"/api/ng/h3/jobs/{job.job_id}",
    }), 202


@h3NG_bp.route("/api/ng/h3/jobs/<job_id>", methods=["GET"])
def h3_job_status_ng(job_id):
    try:
        status = h3_jobs.job_status_ng(job_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(status)


@h3NG_bp.route("/api/ng/h3/jobs/<job_id>", methods=["DELETE"])
def h3_cancel_job_ng(job_id):
    """Same semantics as videogenNG's cancel route -- `ok: false` just
    means there was nothing left to cancel."""
    cancelled = h3_jobs.cancel_h3_job_ng(job_id)
    return jsonify({"ok": cancelled})


@h3NG_bp.route("/api/ng/h3/jobs/<job_id>/video", methods=["GET"])
def h3_serve_video_ng(job_id):
    """Inline playback -- fed to the queue row's <video src>."""
    job = h3_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.mp4_path or not os.path.isfile(job.mp4_path):
        return jsonify({"error": "video not ready"}), 404
    return send_file(job.mp4_path, mimetype="video/mp4")


@h3NG_bp.route("/api/ng/h3/jobs/<job_id>/download", methods=["GET"])
def h3_download_video_ng(job_id):
    """Same file as .../video, forced as an attachment -- same reasoning
    as videogenNG's own download route (iPad Safari's inline-playback
    save affordance is unreliable)."""
    job = h3_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.mp4_path or not os.path.isfile(job.mp4_path):
        return jsonify({"error": "video not ready"}), 404
    return send_file(
        job.mp4_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"h3-{job_id}.mp4",
    )
