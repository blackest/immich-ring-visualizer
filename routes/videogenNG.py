"""immichRingNG -- HTTP layer for the Animate view (image-to-video via
LTX-2.5, see ltx_engineNG.py / video_jobsNG.py).

Mirrors routes/generateNG.py's job-queue shape: POST enqueues and
returns a job id immediately, the frontend polls GET .../jobs/<id> for
progress. One route family, deliberately -- no character registration,
no draft-character bookkeeping like Generate has, since a video render
isn't part of a character's sheet.
"""

import os
import subprocess
import tempfile

from flask import Blueprint, jsonify, request, send_file

import ltx_engineNG
import video_jobsNG as video_jobs

videogenNG_bp = Blueprint("videogenNG", __name__)

_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


@videogenNG_bp.route("/api/ng/videogen/status", methods=["GET"])
def videogen_status_ng():
    """Reports whether the local LTX lab (venv + model + gemma encoder)
    is reachable -- the frontend gates the Generate button on this."""
    health = ltx_engineNG.ltx_health_ng()
    return jsonify({"reachable": health["ready"], **health})


@videogenNG_bp.route("/api/ng/videogen/enhance", methods=["POST"])
def videogen_enhance_prompt_ng():
    """Gemma rewrites the given prompt into LTX's preferred verbose
    motion-description style. Synchronous (a few seconds, text only --
    no video render, no job queue) and returns the enhanced text for
    the frontend to drop into the prompt box in place of the original."""
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) else None

    try:
        enhanced = ltx_engineNG.enhance_ltx_prompt_ng(
            prompt, config=ltx_engineNG.LtxConfig(), seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({"error": "prompt enhancement timed out"}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, "prompt": enhanced})


@videogenNG_bp.route("/api/ng/videogen/generate", methods=["POST"])
def videogen_generate_ng():
    """Form fields: file (required, multipart -- the reference image),
    prompt (required), duration_s (optional float, default 3.0),
    seed (optional int, random if omitted)."""
    if "file" not in request.files:
        return jsonify({"error": "no field 'file'"}), 400
    fld = request.files["file"]
    prompt = str(request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    ext = os.path.splitext(fld.filename or "")[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = _CONTENT_TYPE_EXT.get((fld.mimetype or "").lower(), ".jpg")

    duration_s = request.form.get("duration_s", 3.0, type=float)
    seed = request.form.get("seed", type=int)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix="ringvizng_videogen_ref_", suffix=ext, delete=False) as fh:
            fld.save(fh)
            tmp_path = fh.name
        try:
            job = video_jobs.start_video_job_ng(tmp_path, prompt, duration_s, seed)
        except LookupError as e:
            return jsonify({"error": str(e)}), 404
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({
            "ok": True,
            "job_id": job.job_id,
            "poll_url": f"/api/ng/videogen/jobs/{job.job_id}",
        }), 202
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@videogenNG_bp.route("/api/ng/videogen/jobs/<job_id>", methods=["GET"])
def videogen_job_status_ng(job_id):
    try:
        status = video_jobs.job_status_ng(job_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(status)


@videogenNG_bp.route("/api/ng/videogen/jobs/<job_id>", methods=["DELETE"])
def videogen_cancel_job_ng(job_id):
    """Removes a job from the render queue -- if it hasn't started yet
    it's simply skipped, if it's already rendering its subprocess gets
    killed. `ok: false` just means there was nothing left to cancel
    (already finished, or no such job); the frontend row disappears
    from the visible queue either way."""
    cancelled = video_jobs.cancel_video_job_ng(job_id)
    return jsonify({"ok": cancelled})


@videogenNG_bp.route("/api/ng/videogen/jobs/<job_id>/video", methods=["GET"])
def videogen_serve_video_ng(job_id):
    """Inline playback -- fed to the queue row's <video src>. Not an
    attachment, so it plays in place rather than triggering a download
    dialog on every browser."""
    job = video_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.mp4_path or not os.path.isfile(job.mp4_path):
        return jsonify({"error": "video not ready"}), 404
    return send_file(job.mp4_path, mimetype="video/mp4")


@videogenNG_bp.route("/api/ng/videogen/jobs/<job_id>/download", methods=["GET"])
def videogen_download_video_ng(job_id):
    """Same file as .../video, but forced as an attachment (Content-
    Disposition) -- the queue row's explicit Download link points here
    instead, since a plain <video> element's own save option is
    unreliable (esp. on iPad Safari, which often has no working
    "download video" affordance on inline playback)."""
    job = video_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.mp4_path or not os.path.isfile(job.mp4_path):
        return jsonify({"error": "video not ready"}), 404
    return send_file(
        job.mp4_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"animate-{job_id}.mp4",
    )
