"""immichRingNG -- HTTP layer for the Talk view (a script said by a voice on a character's face, lip-synced;
see talk_jobsNG.py). One POST starts the whole chain (Speech, then an H3 audio-drive render); the page polls
GET .../jobs/<id> and GET .../jobs lists everything the server holds so the page can rebuild its queue.
The character LoRAs come from the existing GET /api/ng/h3/loras and the voices from GET /api/ng/speech/voices.
/generate is multipart: an optional voice clip (`file`) and an optional face picture (`face`)."""

import os

from flask import Blueprint, jsonify, request

import h3_engineNG
import speech_engineNG as speech
import talk_jobsNG as talk

talkNG_bp = Blueprint("talkNG", __name__)


def _truthy(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "on", "yes")


@talkNG_bp.route("/api/ng/talk/status", methods=["GET"])
def talk_status_ng():
    """Ready only when both engines are: the page greys the button with the reason otherwise."""
    sp = speech.speech_health_ng()
    h3 = h3_engineNG.h3_health_ng("h3q8")
    problems = []
    if not sp.get("ok"):
        problems.append("Speech is not ready: " + "; ".join(sp.get("missing") or ["see the Speech view"]))
    h3_missing = [label for key, label in (("python_ok", "the H3 environment"), ("runner_ok", "the H3 runner"),
                                           ("dit_ok", "the H3 model"), ("compact_ok", "the H3 VAE/text pack"),
                                           ("text_config_ok", "the H3 text config")) if not h3.get(key)]
    if h3_missing:
        problems.append("H3 is not ready: missing " + ", ".join(h3_missing))
    return jsonify({"ok": not problems, "problems": problems, "clip_s": talk.TALK_CLIP_S,
                    "qualities": list(talk.TALK_QUALITIES)})


@talkNG_bp.route("/api/ng/talk/generate", methods=["POST"])
def talk_generate_ng():
    form = request.form
    clip = request.files.get("file")
    face = request.files.get("face")
    try:
        seed_raw = str(form.get("seed") or "").strip()
        job = talk.start_talk_job_ng(
            text=str(form.get("text") or ""), kind=str(form.get("kind") or "preset").strip(),
            voice=str(form.get("voice") or "").strip(), character_voice=str(form.get("character_voice") or "").strip(),
            style_id=str(form.get("style") or "natural").strip(), style_text=str(form.get("style_text") or ""),
            language=str(form.get("language") or "english").strip().lower(),
            speed=float(form.get("speed") or 1.0), ref_text=str(form.get("ref_text") or ""),
            voice_consent=_truthy(form.get("consent")),
            clip_bytes=clip.read() if clip else None,
            clip_ext=os.path.splitext(clip.filename or "")[1] if clip else "",
            subject=str(form.get("subject") or ""), scene=str(form.get("scene") or ""),
            lora_name=str(form.get("lora") or "").strip(), lora_strength=float(form.get("lora_strength") or 1.0),
            first_frame=face.read() if face else None,
            first_frame_ext=os.path.splitext(face.filename or "")[1].lower() if face else ".jpg",
            quality=str(form.get("quality") or "draft").strip().lower(),
            seed=int(seed_raw) if seed_raw else None)
    except (speech.SpeechRequestError, talk.TalkRequestError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/talk/jobs/{job.job_id}"}), 202


@talkNG_bp.route("/api/ng/talk/jobs", methods=["GET"])
def talk_jobs_list_ng():
    return jsonify({"jobs": talk.list_jobs_ng()})


@talkNG_bp.route("/api/ng/talk/jobs/<job_id>", methods=["GET"])
def talk_job_status_ng(job_id):
    try:
        return jsonify(talk.job_status_ng(job_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@talkNG_bp.route("/api/ng/talk/jobs/<job_id>", methods=["DELETE"])
def talk_cancel_job_ng(job_id):
    """`ok: false` just means there was nothing left to cancel."""
    return jsonify({"ok": talk.cancel_talk_job_ng(job_id)})
