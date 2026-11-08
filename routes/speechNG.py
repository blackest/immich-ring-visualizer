"""immichRingNG -- HTTP layer for the Speech view (text to speech in a preset voice, a character's saved
voice, or an uploaded clip; see speech_engineNG.py / speech_jobsNG.py).

Same job-queue shape as routes/h3NG.py and routes/musicNG.py: POST /generate enqueues and returns a job id
at once, the page polls GET .../jobs/<id>, and GET .../jobs lists everything the server holds so the page
can rebuild its queue. /generate is multipart (the clip is the one file field)."""

import os
import shutil

from flask import Blueprint, jsonify, request, send_file

import speech_engineNG as speech
import speech_jobsNG as speech_jobs

speechNG_bp = Blueprint("speechNG", __name__)


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "on", "yes")


@speechNG_bp.route("/api/ng/speech/status", methods=["GET"])
def speech_status_ng():
    return jsonify(speech.speech_health_ng())


@speechNG_bp.route("/api/ng/speech/voices", methods=["GET"])
def speech_voices_ng():
    """Everything the form offers: the nine preset speakers, the delivery styles, the languages, and the
    characters that have a saved voice sample."""
    return jsonify({
        "presets": list(speech.PRESETS),
        "styles": [{"id": s["id"], "label": s["label"]} for s in speech.STYLES],
        "languages": list(speech.LANGUAGES),
        "characters": [{"id": v["id"], "label": v["label"]} for v in speech.list_voice_samples_ng()],
        "max_chars": speech.SPEECH_MAX_CHARS,
    })


@speechNG_bp.route("/api/ng/speech/voices/<name>/sample", methods=["GET"])
def speech_voice_sample_ng(name):
    """Play a character's saved voice sample (looked up by name, never by path)."""
    path = speech.resolve_voice_sample_ng(name)
    if path is None:
        return jsonify({"error": f"no voice sample for {name!r}"}), 404
    return send_file(str(path))


@speechNG_bp.route("/api/ng/speech/generate", methods=["POST"])
def speech_generate_ng():
    form = request.form
    kind = str(form.get("kind") or "preset").strip()
    clip = request.files.get("file")
    try:
        seed_raw = str(form.get("seed") or "").strip()
        job = speech_jobs.start_speech_job_ng(
            text=str(form.get("text") or ""), kind=kind,
            voice=str(form.get("voice") or "").strip(), character=str(form.get("character") or "").strip(),
            style_id=str(form.get("style") or "natural").strip(), style_text=str(form.get("style_text") or ""),
            language=str(form.get("language") or "english").strip().lower(),
            speed=float(form.get("speed") or 1.0), seed=int(seed_raw) if seed_raw else None,
            ref_text=str(form.get("ref_text") or ""), voice_consent=_truthy(form.get("consent")),
            design_text=str(form.get("design_text") or ""),
            clip_bytes=clip.read() if clip else None,
            clip_ext=os.path.splitext(clip.filename or "")[1] if clip else "")
    except speech.SpeechRequestError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/speech/jobs/{job.job_id}"}), 202


@speechNG_bp.route("/api/ng/speech/jobs/<job_id>/save-voice", methods=["POST"])
def speech_save_voice_ng(job_id):
    """Keep a finished take as a character's saved voice (`<name>.voice.wav` beside the LoRAs), so it can be used as a
    clone reference. A voice made from a description differs on every generation; this is how one is kept."""
    job = speech_jobs.get_job_ng(job_id)
    if job is None:
        return jsonify({"error": f"no such job {job_id!r}"}), 404
    if job.finished_at is None or job.error or not job.audio_path or not os.path.isfile(job.audio_path):
        return jsonify({"error": "that take is not finished"}), 409
    name = str(request.form.get("name") or "").strip()
    if not speech._NAME_RE.match(name):
        return jsonify({"error": "the name must be letters, digits, - or _ (at most 64 characters)"}), 400
    if any(v["id"].casefold() == name.casefold() for v in speech.list_voice_samples_ng()):
        return jsonify({"error": f"a saved voice named {name} already exists; pick another name"}), 409
    dest = speech.SPEECH_VOICES_DIR / f"{name}.voice.wav"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(job.audio_path, dest)
    except OSError as e:
        return jsonify({"error": f"could not save: {e}"}), 500
    return jsonify({"ok": True, "name": name})


@speechNG_bp.route("/api/ng/speech/jobs", methods=["GET"])
def speech_jobs_list_ng():
    return jsonify({"jobs": speech_jobs.list_jobs_ng()})


@speechNG_bp.route("/api/ng/speech/jobs/<job_id>", methods=["GET"])
def speech_job_status_ng(job_id):
    try:
        return jsonify(speech_jobs.job_status_ng(job_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@speechNG_bp.route("/api/ng/speech/jobs/<job_id>", methods=["DELETE"])
def speech_cancel_job_ng(job_id):
    """`ok: false` just means there was nothing left to cancel."""
    return jsonify({"ok": speech_jobs.cancel_speech_job_ng(job_id)})


def _finished_audio(job_id):
    job = speech_jobs.get_job_ng(job_id)
    if job is None:
        return None, (jsonify({"error": f"no such job {job_id!r}"}), 404)
    if not job.audio_path or not os.path.isfile(job.audio_path):
        return None, (jsonify({"error": "no audio for this job (not finished, or it failed)"}), 404)
    return job, None


@speechNG_bp.route("/api/ng/speech/jobs/<job_id>/audio", methods=["GET"])
def speech_job_audio_ng(job_id):
    job, err = _finished_audio(job_id)
    if err:
        return err
    return send_file(job.audio_path, mimetype="audio/wav")


@speechNG_bp.route("/api/ng/speech/jobs/<job_id>/download", methods=["GET"])
def speech_job_download_ng(job_id):
    job, err = _finished_audio(job_id)
    if err:
        return err
    return send_file(job.audio_path, mimetype="audio/wav", as_attachment=True,
                     download_name=f"speech-{job_id}.wav")
