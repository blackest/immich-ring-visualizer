"""immichRingNG -- HTTP layer for the Music view (YuE2 style+lyrics-to-
song, see music_engineNG.py / music_jobsNG.py).

Twin of routes/h3NG.py's job-queue shape: POST enqueues and returns a
job id immediately, the frontend polls GET .../jobs/<id> for progress.
/generate is JSON body (no file field, just style/lyrics text). /cover
is multipart instead, same shape as Animate's own /generate -- it's
the one operation here that takes a file (an existing track to
transcribe and render over, see music_engineNG.generate_cover_ng)."""

import os

from flask import Blueprint, jsonify, request, send_file

import music_engineNG
import music_jobsNG as music_jobs

musicNG_bp = Blueprint("musicNG", __name__)


@musicNG_bp.route("/api/ng/music/status", methods=["GET"])
def music_status_ng():
    """Reports whether the requested precision (bf16 or 8bit, default
    8bit) is reachable -- the frontend gates the Compose button on
    this."""
    precision = request.args.get("precision", music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400
    health = music_engineNG.music_health_ng(precision)
    return jsonify({
        "reachable": health["ready"],
        **health,
        "min_duration_s": music_engineNG.MUSIC_MIN_SECONDS,
        "max_duration_s": music_engineNG.MUSIC_MAX_SECONDS,
        "modes": list(music_engineNG.MUSIC_MODES),
    })


@musicNG_bp.route("/api/ng/music/generate", methods=["POST"])
def music_generate_ng():
    """JSON body: style (optional str), lyrics (optional str -- at
    least one of style/lyrics is required unless instrumental is set),
    duration_s (optional float, default 240 -- a length CEILING, not a
    requested length; the song is as long as the lyrics make it, up to
    this cap), mode (optional, "full"|"melody"|"off", default "full"),
    instrumental (optional bool), seed (optional int), cfg_scale
    (optional float), precision (optional, "bf16"|"8bit", default
    "8bit" -- see music_engineNG.MUSIC_DEFAULT_PRECISION)."""
    body = request.get_json(silent=True) or {}
    style = str(body.get("style") or "")
    lyrics = str(body.get("lyrics") or "")
    duration_s = body.get("duration_s", 240.0)
    try:
        duration_s = float(duration_s)
    except (TypeError, ValueError):
        return jsonify({"error": "duration_s must be a number"}), 400
    mode = str(body.get("mode") or "full")
    instrumental = bool(body.get("instrumental"))
    seed = body.get("seed")
    seed = int(seed) if isinstance(seed, (int, float)) else None
    cfg_scale = body.get("cfg_scale")
    cfg_scale = float(cfg_scale) if isinstance(cfg_scale, (int, float)) else None
    precision = str(body.get("precision") or music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400

    try:
        job = music_jobs.start_music_job_ng(
            style, lyrics, duration_s, seed=seed, mode=mode,
            instrumental=instrumental, cfg_scale=cfg_scale, precision=precision)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "poll_url": f"/api/ng/music/jobs/{job.job_id}",
    }), 202


@musicNG_bp.route("/api/ng/music/cover", methods=["POST"])
def music_cover_ng():
    """Multipart form: file (required -- the existing track to cover),
    style/lyrics (optional strs -- YOUR words/style over the source's
    transcribed melody; both may be empty for an instrumental re-
    render), task (optional, "full"|"melody-full"|"melody-vocal",
    default "melody-full" -- see music_engineNG.MUSIC_COVER_TASKS),
    duration_s (optional float -- a length ceiling; omitted, the
    render follows the transcribed score's own length), seed/cfg_scale/
    precision as in /generate.

    First call on a fresh install downloads two extra HF models
    (transcription + base, ~untracked size) -- see
    music_engineNG.py's own docstring for why that's unavoidable here
    specifically, unlike every other engine in this app."""
    fld = request.files.get("file")
    if fld is None or not fld.filename:
        return jsonify({"error": "file is required (the track to cover)"}), 400
    audio_bytes = fld.read()
    if not audio_bytes:
        return jsonify({"error": "uploaded file is empty"}), 400
    audio_ext = os.path.splitext(fld.filename)[1].lower()

    style = str(request.form.get("style") or "")
    lyrics = str(request.form.get("lyrics") or "")
    task = str(request.form.get("task") or music_engineNG.MUSIC_DEFAULT_COVER_TASK)
    if task not in music_engineNG.MUSIC_COVER_TASKS:
        return jsonify({"error": f"task must be one of {music_engineNG.MUSIC_COVER_TASKS} (got {task!r})"}), 400

    duration_s = request.form.get("duration_s", type=float)
    seed = request.form.get("seed", type=int)
    cfg_scale = request.form.get("cfg_scale", type=float)
    precision = str(request.form.get("precision") or music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400

    try:
        job = music_jobs.start_music_job_ng(
            style, lyrics, duration_s, seed=seed, cfg_scale=cfg_scale, precision=precision,
            audio_bytes=audio_bytes, audio_ext=audio_ext, task=task)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "poll_url": f"/api/ng/music/jobs/{job.job_id}",
    }), 202


@musicNG_bp.route("/api/ng/music/jobs/<job_id>", methods=["GET"])
def music_job_status_ng(job_id):
    try:
        status = music_jobs.job_status_ng(job_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(status)


@musicNG_bp.route("/api/ng/music/jobs/<job_id>", methods=["DELETE"])
def music_cancel_job_ng(job_id):
    """Same semantics as h3NG's cancel route -- `ok: false` just means
    there was nothing left to cancel."""
    cancelled = music_jobs.cancel_music_job_ng(job_id)
    return jsonify({"ok": cancelled})


@musicNG_bp.route("/api/ng/music/jobs/<job_id>/audio", methods=["GET"])
def music_serve_audio_ng(job_id):
    """Inline playback -- fed to the queue row's <audio src>."""
    job = music_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.audio_path or not os.path.isfile(job.audio_path):
        return jsonify({"error": "audio not ready"}), 404
    return send_file(job.audio_path, mimetype="audio/flac")


@musicNG_bp.route("/api/ng/music/jobs/<job_id>/score", methods=["GET"])
def music_serve_score_ng(job_id):
    """The actual sheet music: for a cover job, the ABC score SheetSage2
    transcribed from the source track; for plain generate, the model's own
    composed ABC (music_jobsNG.MusicJobNG.score_path either way). Plain-text
    ABC notation, not audio; 404 if it isn't ready yet or mode="off" meant
    no score was ever produced."""
    job = music_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.score_path or not os.path.isfile(job.score_path):
        return jsonify({"error": "score not ready or this job has no ABC score"}), 404
    return send_file(
        job.score_path,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"music-{job_id}.abc",
    )


@musicNG_bp.route("/api/ng/music/jobs/<job_id>/download", methods=["GET"])
def music_download_audio_ng(job_id):
    """Same file as .../audio, forced as an attachment -- same
    reasoning as h3NG/videogenNG's own download routes."""
    job = music_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if not job.audio_path or not os.path.isfile(job.audio_path):
        return jsonify({"error": "audio not ready"}), 404
    return send_file(
        job.audio_path,
        mimetype="audio/flac",
        as_attachment=True,
        download_name=f"music-{job_id}.flac",
    )
