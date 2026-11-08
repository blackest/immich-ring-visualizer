"""immichRingNG -- HTTP layer for the Scene view (a script of lines said by different characters, rendered as one
joined clip; see scene_jobsNG.py). POST /generate takes the script, a JSON list of characters and the files they refer
to (pictures, voice clips) as uploads; the page polls GET .../jobs/<id> and GET .../jobs lists everything the server
holds so the page can rebuild its queue. The joined scene is served from .../video and .../download.

characters JSON, one object per character:
  {"name": "Knight", "subject": "a bald man with a grey beard", "lora": "name" (optional), "lora_strength": 1.0,
   "looks": [{"name": "battlements", "file": "look_0_0"}], "default_look": "battlements",
   "voice": {"kind": "preset|character|clip", "voice": "aiden", "style": "natural", "style_text": "",
             "character_voice": "", "ref_text": "", "consent": false, "language": "english", "speed": 1.0,
             "clip_file": "voice_0"}}
where each "file" / "clip_file" names a multipart upload field.
"""

import json
import os

from flask import Blueprint, jsonify, request, send_file

import finish_engineNG as finish
import h3_engineNG
import scene_jobsNG as scene
import speech_engineNG as speech

sceneNG_bp = Blueprint("sceneNG", __name__)


def _truthy(v, default=False) -> bool:
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _characters_from_request() -> list:
    raw = (request.form.get("characters") or "").strip()
    try:
        items = json.loads(raw) if raw else []
    except ValueError:
        raise scene.SceneRequestError("characters must be a JSON list") from None
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise scene.SceneRequestError("characters must be a JSON list of objects")

    def upload(field):
        f = request.files.get(str(field or ""))
        if f is None:
            raise scene.SceneRequestError(f"the file {field!r} was not uploaded")
        return f.read(), os.path.splitext(f.filename or "")[1]

    out = []
    for item in items:
        ch = {k: v for k, v in item.items() if k not in ("looks", "voice")}
        ch["looks"] = []
        for look in item.get("looks") or []:
            data, ext = upload(look.get("file"))
            ch["looks"].append({"name": look.get("name"), "bytes": data, "ext": ext})
        voice = dict(item.get("voice") or {})
        if voice.get("clip_file"):
            voice["clip_bytes"], voice["clip_ext"] = upload(voice.pop("clip_file"))
        ch["voice"] = voice
        out.append(ch)
    return out


@sceneNG_bp.route("/api/ng/scene/status", methods=["GET"])
def scene_status_ng():
    """Ready only when Speech, H3 and the finish step are; otherwise the page says why."""
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
    fin = finish.finish_health_ng()
    return jsonify({"ok": not problems, "problems": problems, "clip_s": scene.CLIP_S, "qualities": list(scene.QUALITIES),
                    "max_lines": scene.script_mod.MAX_LINES, "finish_ok": fin["ok"], "finish_problems": fin["problems"],
                    "finish_defaults": finish.DEFAULTS, "finish_limits": finish.LIMITS})


@sceneNG_bp.route("/api/ng/scene/generate", methods=["POST"])
def scene_generate_ng():
    form = request.form
    try:
        seed_raw = str(form.get("seed") or "").strip()
        finish_params = {}
        for key in finish.DEFAULTS:
            raw = str(form.get(key) or "").strip()
            if raw:
                finish_params[key] = float(raw)
        job = scene.start_scene_job_ng(
            script=str(form.get("script") or ""), characters=_characters_from_request(),
            quality=str(form.get("quality") or "draft").strip().lower(), fast=_truthy(form.get("fast"), True),
            finish_pass=_truthy(form.get("finish")), finish_params=finish_params,
            match_exposure=_truthy(form.get("match_exposure"), True), seed=int(seed_raw) if seed_raw else None)
    except ValueError as e:        # SceneRequestError, a bad number, or a refusal from the speech/finish checks
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/scene/jobs/{job.job_id}"}), 202


@sceneNG_bp.route("/api/ng/scene/jobs", methods=["GET"])
def scene_jobs_list_ng():
    return jsonify({"jobs": scene.list_jobs_ng()})


@sceneNG_bp.route("/api/ng/scene/jobs/<job_id>", methods=["GET"])
def scene_job_status_ng(job_id):
    try:
        return jsonify(scene.job_status_ng(job_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@sceneNG_bp.route("/api/ng/scene/jobs/<job_id>", methods=["DELETE"])
def scene_cancel_job_ng(job_id):
    """`ok: false` just means there was nothing left to cancel."""
    return jsonify({"ok": scene.cancel_scene_job_ng(job_id)})


def _result(job_id):
    job = scene.get_job_ng(job_id)
    if job is None:
        return None, (jsonify({"error": f"no such job {job_id!r}"}), 404)
    if not job.output_path or not os.path.isfile(job.output_path):
        return None, (jsonify({"error": "video not ready"}), 404)
    return job, None


@sceneNG_bp.route("/api/ng/scene/jobs/<job_id>/video", methods=["GET"])
def scene_video_ng(job_id):
    job, err = _result(job_id)
    return err if err else send_file(job.output_path, mimetype="video/mp4")


@sceneNG_bp.route("/api/ng/scene/jobs/<job_id>/download", methods=["GET"])
def scene_download_ng(job_id):
    job, err = _result(job_id)
    return err if err else send_file(job.output_path, mimetype="video/mp4", as_attachment=True,
                                     download_name=f"scene-{job_id}.mp4")
