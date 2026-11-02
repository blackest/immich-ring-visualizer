"""NG routes for a scene of 5 s windows that each carry their own reference pictures (h3_batchNG.py).

POST   /api/ng/h3/batch        multipart -> 202 {batch_id, poll_url}. Fields: `prompts` (JSON list of
                               strings, one per window), `carry` ("true": each later window also gets the
                               last frame of the one before), `joins` (JSON, as /restitch takes),
                               `quality` or `width`+`height`, `tristep`, `flex`, `steps`,
                               `reference_size`. Files: `shared_files` (every window) and
                               `window_<n>_files` (window n, 1-based), pictures or clips.
GET    /api/ng/h3/batch/<id>   {status: running|completed|failed|cancelled, windows, film, error}; `film`
                               carries the job-log entry and a `video_url` once the scene is joined.
DELETE /api/ng/h3/batch/<id>   stop the window in progress and queue no more.
"""

import json

from flask import Blueprint, jsonify, request

import h3_batchNG as batch
import h3_engineNG as h3_engine

h3BatchNG_bp = Blueprint("h3BatchNG", __name__)


def _files(field: str) -> list:
    return [(f.read(), "." + f.filename.rsplit(".", 1)[-1].lower() if "." in (f.filename or "") else ".png")
            for f in request.files.getlist(field)]


def _json_list(field: str) -> list:
    raw = (request.form.get(field) or "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except ValueError:
        raise ValueError(f"{field} must be a JSON list") from None
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    return value


def _on(field: str) -> bool:
    return str(request.form.get(field) or "").strip().lower() in ("1", "true", "on")


@h3BatchNG_bp.route("/api/ng/h3/batch", methods=["POST"])
def h3_batch_start_ng():
    try:
        prompts = _json_list("prompts")
        if not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a list of strings")
        settings = {"model": "h3ref", "tristep": _on("tristep"), "flex": _on("flex"),
                    "reference_size": str(request.form.get("reference_size") or "match").strip().lower()}
        quality = request.form.get("quality")
        width, height = request.form.get("width", type=int), request.form.get("height", type=int)
        if quality and width is None and height is None:
            width, height = h3_engine.resolve_h3_quality_ng(quality)
        settings.update(width=width, height=height,
                        timeout_s=h3_engine.resolve_h3_quality_timeout_ng(quality, batch.WINDOW_SECONDS))
        steps = request.form.get("steps", type=int)
        if steps is not None:
            settings["steps"] = steps
        windows = [{"prompt": p, "refs": _files(f"window_{n + 1}_files")} for n, p in enumerate(prompts)]
        started = batch.start_batch_ng(windows, shared=_files("shared_files"), carry=_on("carry"),
                                       joins=_json_list("joins") or None, settings=settings)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "batch_id": started["id"], "poll_url": f"/api/ng/h3/batch/{started['id']}"}), 202


@h3BatchNG_bp.route("/api/ng/h3/batch/<batch_id>", methods=["GET"])
def h3_batch_status_ng(batch_id):
    try:
        return jsonify(batch.batch_status_ng(batch_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@h3BatchNG_bp.route("/api/ng/h3/batch/<batch_id>", methods=["DELETE"])
def h3_batch_cancel_ng(batch_id):
    if not batch.cancel_batch_ng(batch_id):
        return jsonify({"error": "no such running scene"}), 404
    return jsonify({"ok": True})
