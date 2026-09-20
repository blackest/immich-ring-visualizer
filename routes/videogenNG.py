"""immichRingNG -- HTTP layer for the Animate view (image-to-video via
LTX-2.5, see ltx_engineNG.py / video_jobsNG.py).

Mirrors routes/generateNG.py's job-queue shape: POST enqueues and
returns a job id immediately, the frontend polls GET .../jobs/<id> for
progress. One route family, deliberately -- no character registration,
no draft-character bookkeeping like Generate has, since a video render
isn't part of a character's sheet.
"""

import base64
import os
import subprocess

from flask import Blueprint, jsonify, request, send_file

import job_logsNG
import ltx_engineNG
import video_jobsNG as video_jobs
from ltx_loraNG import list_ltx_loras_ng, resolve_ltx_lora_path_ng
from video_analysisNG import find_cache_frame_ng

videogenNG_bp = Blueprint("videogenNG", __name__)

_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


def _resolve_ref_image_bytes():
    """Resolve the pending reference image's raw bytes (and a normalized
    extension) without ever re-fetching data this same process already
    has in memory.

    `ref_frame_id` (e.g. "<job_id>_anchor" or "<job_id>_<frame>", exactly
    what /api/ng/framefile/<frame_id> serves) is how the frontend refers
    to a live curation-session frame it pulled into the Animate view's
    ref tray -- rather than have the browser download those bytes just to
    immediately re-upload them, this resolves the SAME in-memory cache
    (find_cache_frame_ng) directly, in-process. Falls back to the
    uploaded `file` field for anything the server never had in the first
    place (a local disk pick has no frame_id).

    Returns (bytes, ext) -- ext is one of .png/.jpg/.jpeg/.webp (default
    .jpg) -- or (None, None) if neither field was sent. Raises
    LookupError if ref_frame_id was given but has since been evicted from
    the cache (e.g. the curation session was cleared)."""
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


@videogenNG_bp.route("/api/ng/videogen/loras", methods=["GET"])
def videogen_loras_ng():
    """Optional style/character LoRAs available to the Animate workflow
    -- populates the rail's LoRA dropdown. See ltx_loraNG.py."""
    return jsonify({"loras": list_ltx_loras_ng()})


@videogenNG_bp.route("/api/ng/videogen/enhance", methods=["POST"])
def videogen_enhance_prompt_ng():
    """Gemma rewrites the given prompt into LTX's preferred verbose
    motion-description style. Synchronous (a few seconds -- no video
    render, no job queue) and returns the enhanced text for the frontend
    to drop into the prompt box in place of the original.

    Form fields (multipart, matching /generate): prompt (required),
    seed (optional int), file (optional -- the reference image) OR
    ref_frame_id (optional -- a live curation-session frame already
    cached server-side, see _resolve_ref_image_bytes). When a reference
    is given, Gemma actually sees it (mlx_vlm's vision path) rather than
    guessing blind from the prompt alone -- see
    ltx_engineNG.enhance_ltx_prompt_ng's docstring. Omit both for a
    text-to-video job with no reference photo, same as /generate. Bytes
    stay in memory (base64) end to end, never written to disk -- per the
    no-disk-cache principle, a temp file here would just be a detour for
    data that starts and ends as an in-memory blob."""
    prompt = str(request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    seed = request.form.get("seed", type=int)

    try:
        img_bytes, _ext = _resolve_ref_image_bytes()
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    image_b64 = base64.b64encode(img_bytes).decode() if img_bytes is not None else None

    try:
        enhanced = ltx_engineNG.enhance_ltx_prompt_ng(
            prompt, config=ltx_engineNG.LtxConfig(), seed=seed,
            image_b64=image_b64)
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
    shot options (prompt + duration) plus a short discussion note.

    The LAST message, if role "user", may carry an optional "images"
    list (base64-encoded strings, no "data:" prefix) -- same vision path
    and "only the last turn's images matter" caveat as /chat above."""
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

    clean_turns = [{"role": t["role"], "content": t["content"]} for t in turns]
    images = None
    last = turns[-1]
    if last.get("role") == "user" and isinstance(last.get("images"), list):
        images = [im for im in last["images"] if isinstance(im, str) and im]
        images = images or None

    try:
        result = ltx_engineNG.discuss_next_scene_ng(
            clean_turns, config=ltx_engineNG.LtxConfig(), seed=seed, images=images)
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
    full turn history each call.

    The LAST message, if role "user", may carry an optional "images"
    list (base64-encoded strings, no "data:" prefix) -- Gemma then sees
    them via mlx_vlm's vision path instead of just being told about them
    in text (see ltx_engineNG.chat_with_gemma_ng's docstring for why only
    the last turn's images can ever matter). Images on any earlier
    message are ignored, not an error -- mlx_vlm's own chat template
    would silently drop them too."""
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

    clean_turns = [{"role": t["role"], "content": t["content"]} for t in turns]
    images = None
    last = turns[-1]
    if last.get("role") == "user" and isinstance(last.get("images"), list):
        images = [im for im in last["images"] if isinstance(im, str) and im]
        images = images or None

    try:
        result = ltx_engineNG.chat_with_gemma_ng(
            clean_turns, config=ltx_engineNG.LtxConfig(), seed=seed, images=images)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except subprocess.TimeoutExpired:
        return jsonify({"error": "chat timed out"}), 504
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"ok": True, **result})


@videogenNG_bp.route("/api/ng/videogen/generate", methods=["POST"])
def videogen_generate_ng():
    """Form fields: file (multipart -- the reference image) OR
    ref_frame_id (a live curation-session frame already cached
    server-side, see _resolve_ref_image_bytes) -- omit both for a
    text-to-video job with no reference photo. Also: prompt (required),
    duration_s (optional float, default 3.0), seed (optional int,
    random if omitted), width/height (optional ints, multiples of 64 --
    the two-stages-hq pipeline halves then upscales resolution
    internally, so it needs a multiple of 64, not just the VAE's own
    32 -- default the fixed-tier size), frame_rate (optional float,
    default the fixed-tier fps).

    No scratch temp file: video_jobs.start_video_job_ng() writes these
    bytes directly into the job's own permanent-for-its-lifetime
    directory in one step -- that's the one real disk write a video job
    needs (the `ltx-2-mlx generate` CLI is an external binary and needs
    a real --image path, unlike the enhance/chat vision helpers), so
    there's nothing left to stage in a temp file first."""
    prompt = str(request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    duration_s = request.form.get("duration_s", 3.0, type=float)
    seed = request.form.get("seed", type=int)
    width = request.form.get("width", type=int)
    height = request.form.get("height", type=int)
    frame_rate = request.form.get("frame_rate", type=float)

    lora_name = str(request.form.get("lora_name") or "").strip()
    lora_path = None
    if lora_name:
        lora_path = resolve_ltx_lora_path_ng(lora_name)
        if lora_path is None:
            return jsonify({"error": f"unknown LoRA {lora_name!r}"}), 400
    lora_strength = request.form.get("lora_strength", 1.0, type=float)

    try:
        img_bytes, ext = _resolve_ref_image_bytes()
    except LookupError as e:
        return jsonify({"error": str(e)}), 404

    try:
        job = video_jobs.start_video_job_ng(
            img_bytes, ext or ".jpg", prompt, duration_s, seed,
            width=width, height=height, frame_rate=frame_rate,
            lora_path=lora_path, lora_strength=lora_strength)
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
