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

import engine_installNG
import music_engineNG
import music_installNG
import music_job_logsNG
import music_jobsNG as music_jobs

musicNG_bp = Blueprint("musicNG", __name__)


@musicNG_bp.route("/api/ng/music/status", methods=["GET"])
def music_status_ng():
    """Reports whether the requested precision (bf16 or 8bit, default
    8bit) is reachable -- the frontend gates the Compose button on
    this. `installable` says whether the "Install now" button (see
    /install below) even makes sense here -- these engines are MLX
    (Apple Silicon) ports, see engine_installNG.mac_apple_silicon_ok."""
    precision = request.args.get("precision", music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400
    health = music_engineNG.music_health_ng(precision)
    installable, install_blocked_reason = engine_installNG.mac_apple_silicon_ok()
    return jsonify({
        "reachable": health["ready"],
        **health,
        "min_duration_s": music_engineNG.MUSIC_MIN_SECONDS,
        "max_duration_s": music_engineNG.MUSIC_MAX_SECONDS,
        "modes": list(music_engineNG.MUSIC_MODES),
        "installable": installable and not health["ready"],
        "install_blocked_reason": None if health["ready"] else install_blocked_reason,
    })


@musicNG_bp.route("/api/ng/music/install", methods=["POST"])
def music_install_ng():
    """Kicks off the background install job (clone+pin -> venv -> deps ->
    weights, see music_installNG.py) that gets MUSIC_REPO_DIR/
    MUSIC_MODELS_ROOT to a real install without needing Phosphene. Picking
    up a *freshly finished* install still needs a process restart --
    music_engineNG.py resolves those paths once at import time -- so the
    frontend's completed-job message says so rather than re-polling
    /status expecting it to flip live."""
    try:
        job = engine_installNG.start_install_job("music", music_installNG.build_steps())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "poll_url": f"/api/ng/music/install/{job.job_id}",
    }), 202


@musicNG_bp.route("/api/ng/music/install/<job_id>", methods=["GET"])
def music_install_status_ng(job_id):
    try:
        return jsonify(engine_installNG.install_job_status(job_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@musicNG_bp.route("/api/ng/music/generate", methods=["POST"])
def music_generate_ng():
    """JSON body: style (optional str), lyrics (optional str -- at
    least one of style/lyrics is required unless instrumental is set),
    duration_s (optional float, default 240 -- a length CEILING, not a
    requested length; the song is as long as the lyrics make it, up to
    this cap), mode (optional, "full"|"melody"|"off", default "full"),
    instrumental (optional bool), seed (optional int), cfg_scale
    (optional float), temperature (optional float, the "weirdness"/
    creativity knob -- see music_engineNG.MUSIC_MIN/MAX_TEMPERATURE),
    abc (optional str -- a caller-supplied ABC score, e.g. from Music
    Edit; a real conditioning input via `lyra generate --abc`, see
    generate_music_ng's docstring; counts as "something to work with"
    on its own, so style/lyrics may both be empty when it's given),
    precision (optional, "bf16"|"8bit", default "8bit" -- see
    music_engineNG.MUSIC_DEFAULT_PRECISION)."""
    body = request.get_json(silent=True) or {}
    style = str(body.get("style") or "")
    lyrics = str(body.get("lyrics") or "")
    abc_text = body.get("abc")
    abc_text = str(abc_text) if isinstance(abc_text, str) and abc_text.strip() else None
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
    temperature = body.get("temperature")
    temperature = float(temperature) if isinstance(temperature, (int, float)) else None
    precision = str(body.get("precision") or music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400

    try:
        job = music_jobs.start_music_job_ng(
            style, lyrics, duration_s, seed=seed, mode=mode,
            instrumental=instrumental, cfg_scale=cfg_scale, temperature=temperature,
            abc_text=abc_text, precision=precision)
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
    temperature = request.form.get("temperature", type=float)
    precision = str(request.form.get("precision") or music_engineNG.MUSIC_DEFAULT_PRECISION)
    if precision not in ("bf16", "8bit"):
        return jsonify({"error": f"precision must be 'bf16' or '8bit' (got {precision!r})"}), 400

    try:
        job = music_jobs.start_music_job_ng(
            style, lyrics, duration_s, seed=seed, cfg_scale=cfg_scale, temperature=temperature,
            precision=precision, audio_bytes=audio_bytes, audio_ext=audio_ext, task=task)
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


# ---- Durable job log (music_job_logsNG.py) -- separate from the queue
# above: a permanent, never-pruned history of finished renders (style,
# lyrics, seed -- the "recipe" that's otherwise lost once a job ages out
# of the 20-job ring buffer), browsable by day (recent) or drilled down
# by year/month (archive). Twin of videogenNG's own job-log routes --
# see music_job_logsNG.py's module docstring for the on-disk layout.

@musicNG_bp.route("/api/ng/music/logs", methods=["GET"])
def music_logs_recent_ng():
    return jsonify({"ok": True, "entries": music_job_logsNG.list_recent_logs_ng()})


@musicNG_bp.route("/api/ng/music/logs/archive", methods=["GET"])
def music_logs_archive_years_ng():
    return jsonify({"ok": True, "years": music_job_logsNG.list_archive_years_ng()})


@musicNG_bp.route("/api/ng/music/logs/archive/<year>", methods=["GET"])
def music_logs_archive_months_ng(year):
    return jsonify({"ok": True, "months": music_job_logsNG.list_archive_months_ng(year)})


@musicNG_bp.route("/api/ng/music/logs/archive/<year>/<month>", methods=["GET"])
def music_logs_archive_day_entries_ng(year, month):
    return jsonify({"ok": True, "entries": music_job_logsNG.list_archive_day_entries_ng(year, month)})


@musicNG_bp.route("/api/ng/music/logs/<date>/<job_id>/audio", methods=["GET"])
def music_log_audio_ng(date, job_id):
    path = music_job_logsNG.log_audio_path_ng(date, job_id)
    if path is None:
        return jsonify({"error": "no such log entry or audio"}), 404
    return send_file(path, mimetype="audio/flac")


@musicNG_bp.route("/api/ng/music/logs/<date>/<job_id>/score", methods=["GET"])
def music_log_score_ng(date, job_id):
    path = music_job_logsNG.log_score_path_ng(date, job_id)
    if path is None:
        return jsonify({"error": "no such log entry or score"}), 404
    return send_file(path, mimetype="text/plain")


@musicNG_bp.route("/api/ng/music/logs/<date>/<job_id>/notes", methods=["PATCH"])
def music_log_update_notes_ng(date, job_id):
    body = request.get_json(silent=True) or {}
    notes = str(body.get("notes") or "")
    entry = music_job_logsNG.update_log_notes_ng(date, job_id, notes)
    if entry is None:
        return jsonify({"error": "no such log entry"}), 404
    return jsonify({"ok": True, "entry": entry})


@musicNG_bp.route("/api/ng/music/logs/<date>/<job_id>", methods=["DELETE"])
def music_log_delete_ng(date, job_id):
    ok = music_job_logsNG.delete_log_ng(date, job_id)
    if not ok:
        return jsonify({"error": "no such log entry"}), 404
    return jsonify({"ok": True})
