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

import job_logsNG
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
    is reachable -- the frontend gates the Generate button on this.
    Also carries the render-size/fps bounds so the frontend's inputs
    don't hardcode them separately."""
    health = ltx_engineNG.ltx_health_ng()
    return jsonify({
        "reachable": health["ready"],
        **health,
        "default_width": ltx_engineNG.LTX_WIDTH,
        "default_height": ltx_engineNG.LTX_HEIGHT,
        "default_frame_rate": ltx_engineNG.LTX_FRAME_RATE,
        "dim_step": ltx_engineNG.LTX_DIM_STEP,
        "min_dim": ltx_engineNG.LTX_MIN_DIM,
        "max_dim": ltx_engineNG.LTX_MAX_DIM,
        "min_fps": ltx_engineNG.LTX_MIN_FPS,
        "max_fps": ltx_engineNG.LTX_MAX_FPS,
    })


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


@videogenNG_bp.route("/api/ng/videogen/estimate-duration", methods=["POST"])
def videogen_estimate_duration_ng():
    """Asks Gemma how long the given shot description needs (dialogue
    pace + any lead-in/trailing action beats) and returns a seconds
    estimate for the frontend to drop into the duration slider. Same
    synchronous, no-job-queue shape as /enhance."""
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) else None

    try:
        seconds = ltx_engineNG.estimate_scene_seconds_ng(
            prompt, config=ltx_engineNG.LtxConfig(), seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({"error": "scene timing timed out"}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, "seconds": seconds})


@videogenNG_bp.route("/api/ng/videogen/discuss", methods=["POST"])
def videogen_discuss_scene_ng():
    """Multi-turn "what should the next shot be" chat against the
    enhance Gemma checkpoint. The client owns the running conversation
    and sends the full turn history each call (this is stateless,
    synchronous, no job queue); every reply proposes exactly 3 labeled
    shot options (prompt + duration) plus a short discussion note."""
    body = request.get_json(silent=True) or {}
    turns = body.get("messages")
    if not isinstance(turns, list) or not turns:
        return jsonify({"error": "messages is required"}), 400
    for t in turns:
        if not isinstance(t, dict) or t.get("role") not in ("user", "assistant") \
                or not isinstance(t.get("content"), str):
            return jsonify({"error": "each message needs role (user/assistant) and content"}), 400
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) else None

    try:
        result = ltx_engineNG.discuss_next_scene_ng(
            turns, config=ltx_engineNG.LtxConfig(), seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({"error": "scene discussion timed out"}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, **result})


@videogenNG_bp.route("/api/ng/videogen/chat", methods=["POST"])
def videogen_chat_ng():
    """Free-form, multi-turn chat against the enhance Gemma checkpoint --
    unlike /discuss, replies aren't steered toward 3 labeled shot
    options; Gemma can talk about anything. Same stateless shape as
    /discuss: the client owns the running conversation and sends the
    full turn history each call."""
    body = request.get_json(silent=True) or {}
    turns = body.get("messages")
    if not isinstance(turns, list) or not turns:
        return jsonify({"error": "messages is required"}), 400
    for t in turns:
        if not isinstance(t, dict) or t.get("role") not in ("user", "assistant") \
                or not isinstance(t.get("content"), str):
            return jsonify({"error": "each message needs role (user/assistant) and content"}), 400
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) else None

    try:
        result = ltx_engineNG.chat_with_gemma_ng(
            turns, config=ltx_engineNG.LtxConfig(), seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({"error": "chat timed out"}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, **result})


@videogenNG_bp.route("/api/ng/videogen/generate", methods=["POST"])
def videogen_generate_ng():
    """Form fields: file (multipart -- the reference image; omit it for a
    text-to-video job with no reference photo), prompt (required),
    duration_s (optional float, default 3.0), seed (optional int,
    random if omitted), width/height (optional ints, multiples of 64 --
    the two-stages-hq pipeline halves then upscales resolution
    internally, so it needs a multiple of 64, not just the VAE's own
    32 -- default the fixed-tier size), frame_rate (optional float,
    default the fixed-tier fps)."""
    prompt = str(request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    duration_s = request.form.get("duration_s", 3.0, type=float)
    seed = request.form.get("seed", type=int)
    width = request.form.get("width", type=int)
    height = request.form.get("height", type=int)
    frame_rate = request.form.get("frame_rate", type=float)

    fld = request.files.get("file")
    tmp_path = None
    try:
        if fld is not None:
            ext = os.path.splitext(fld.filename or "")[1].lower()
            if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = _CONTENT_TYPE_EXT.get((fld.mimetype or "").lower(), ".jpg")
            with tempfile.NamedTemporaryFile(
                    prefix="ringvizng_videogen_ref_", suffix=ext, delete=False) as fh:
                fld.save(fh)
                tmp_path = fh.name
        try:
            job = video_jobs.start_video_job_ng(
                tmp_path, prompt, duration_s, seed,
                width=width, height=height, frame_rate=frame_rate)
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


# ---- Durable job log (job_logsNG.py) -- separate from the queue above:
# a permanent, never-pruned history of finished renders, browsable by
# day (recent) or drilled down by year/month (archive). See
# job_logsNG.py's module docstring for the on-disk layout.

@videogenNG_bp.route("/api/ng/videogen/logs", methods=["GET"])
def videogen_logs_recent_ng():
    return jsonify({"ok": True, "entries": job_logsNG.list_recent_logs_ng()})


@videogenNG_bp.route("/api/ng/videogen/logs/archive", methods=["GET"])
def videogen_logs_archive_years_ng():
    return jsonify({"ok": True, "years": job_logsNG.list_archive_years_ng()})


@videogenNG_bp.route("/api/ng/videogen/logs/archive/<year>", methods=["GET"])
def videogen_logs_archive_months_ng(year):
    return jsonify({"ok": True, "months": job_logsNG.list_archive_months_ng(year)})


@videogenNG_bp.route("/api/ng/videogen/logs/archive/<year>/<month>", methods=["GET"])
def videogen_logs_archive_day_entries_ng(year, month):
    return jsonify({"ok": True, "entries": job_logsNG.list_archive_day_entries_ng(year, month)})


@videogenNG_bp.route("/api/ng/videogen/logs/<date>/<job_id>/ref", methods=["GET"])
def videogen_log_ref_ng(date, job_id):
    path = job_logsNG.log_ref_path_ng(date, job_id)
    if path is None:
        return jsonify({"error": "no such log entry or reference image"}), 404
    return send_file(path)


@videogenNG_bp.route("/api/ng/videogen/logs/<date>/<job_id>/video", methods=["GET"])
def videogen_log_video_ng(date, job_id):
    path = job_logsNG.log_video_path_ng(date, job_id)
    if path is None:
        return jsonify({"error": "no such log entry or video"}), 404
    return send_file(path, mimetype="video/mp4")


@videogenNG_bp.route("/api/ng/videogen/logs/<date>/<job_id>/notes", methods=["PATCH"])
def videogen_log_update_notes_ng(date, job_id):
    body = request.get_json(silent=True) or {}
    notes = str(body.get("notes") or "")
    entry = job_logsNG.update_log_notes_ng(date, job_id, notes)
    if entry is None:
        return jsonify({"error": "no such log entry"}), 404
    return jsonify({"ok": True, "entry": entry})
