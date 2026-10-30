"""NG routes for the H3 prep tools: making a kohya LoRA usable, a 2x upscale of a finished clip, and
building a RefMod from pictures. Each tool is its own module; this file is only the HTTP side.

POST /api/ng/h3/loras/<name>/convert   rewrite a kohya-format LoRA in place (original kept in _originals)
POST /api/ng/h3/refmods/create         multipart: name, files[], canvas (square|landscape), max_pictures?,
                                       description? -> a queued job that writes the RefMod; poll it at
                                       /api/ng/h3/jobs/<job_id> (task "refmod"), then reload /api/ng/h3/refmods
POST /api/ng/h3/jobs/<id>/upscale      {steps?, keep?} -> a queued 2x upscale of that render's film;
                                       poll it at /api/ng/h3/jobs/<job_id> (task "upscale"), play at .../video
"""

import os

from flask import Blueprint, jsonify, request

import h3_lora_convertNG as lora_convert
import h3_refmod_makeNG as refmod_make
import h3_upscaleNG as upscale
from h3_lora_libraryNG import H3_LORAS_DIR, list_h3_loras_ng

h3ToolsNG_bp = Blueprint("h3ToolsNG", __name__)


@h3ToolsNG_bp.route("/api/ng/h3/loras/<name>/convert", methods=["POST"])
def h3_lora_convert_ng(name):
    """Only a file this app itself listed as convertible is touched: the name is checked against the
    real library listing, never joined into a path."""
    entry = next((e for e in list_h3_loras_ng() if e["name"] == name), None)
    if entry is None:
        return jsonify({"error": f"no LoRA named {name!r} in the library"}), 404
    if entry["ok"]:
        return jsonify({"error": f"{name} is already usable"}), 400
    if not entry["convertible"]:
        return jsonify({"error": entry["reason"] or "this LoRA cannot be converted"}), 400
    try:
        done = lora_convert.convert_kohya_ng(entry["path"])
    except (ValueError, RuntimeError, OSError) as e:
        return jsonify({"error": str(e)}), 400
    fresh = next((e for e in list_h3_loras_ng() if e["name"] == name), None)
    return jsonify({"ok": True, **{k: v for k, v in done.items() if k != "backup"},
                    "backup": f"{H3_LORAS_DIR.name}/_originals/{name}.safetensors",
                    "lora": {k: v for k, v in (fresh or {}).items() if k != "path"}})


@h3ToolsNG_bp.route("/api/ng/h3/jobs/<job_id>/upscale", methods=["POST"])
def h3_upscale_ng(job_id):
    body = request.get_json(silent=True) or {}
    try:
        job = upscale.start_upscale_ng(job_id, steps=body.get("steps", 3), keep=body.get("keep", 1.0))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/h3/jobs/{job.job_id}",
                    "estimate_min": job.meta["estimate_min"]}), 202


@h3ToolsNG_bp.route("/api/ng/h3/refmods/create", methods=["POST"])
def h3_refmod_create_ng():
    pictures = [(f.read(), os.path.splitext(f.filename or "")[1]) for f in request.files.getlist("files")]
    try:
        job = refmod_make.start_refmod_job_ng(
            request.form.get("name"), pictures, canvas=(request.form.get("canvas") or "square").strip().lower(),
            max_pictures=request.form.get("max_pictures"), description=(request.form.get("description") or "").strip())
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/h3/jobs/{job.job_id}",
                    "pictures": job.meta["pictures"], "rows": job.meta["rows"]}), 202
