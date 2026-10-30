"""NG routes for re-rolling one window of a chained H3 render (see h3_rerollNG.py).

GET  /api/ng/h3/jobs/<id>/windows            the windows, current joins and film versions
POST /api/ng/h3/jobs/<id>/reroll             {window, seed?, prompt?, from_previous?, joins?} -> a job to poll
POST /api/ng/h3/jobs/<id>/restitch           {joins} -> join the same clips with other joins, no render
GET  /api/ng/h3/jobs/<id>/windows/<n>/video  the clip a window currently uses
GET  /api/ng/h3/jobs/<id>/versions/<n>/video one film version (0 = the original render)

`joins` is [{seam, type: cut|dissolve|fade|audio_lead, seconds}]; seam n is the join after
window n, and a seam left out is a hard cut. The one-window job it starts is polled like any
other at /api/ng/h3/jobs/<job_id>; its status carries `reroll`, with `version` set once the new
film is joined, and the source job's own /video then plays that latest version.
"""

from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

import h3_rerollNG as reroll

h3RerollNG_bp = Blueprint("h3RerollNG", __name__)


def _fail(e):
    if isinstance(e, LookupError):
        return jsonify({"error": str(e)}), 404
    if isinstance(e, (ValueError, FileNotFoundError)):
        return jsonify({"error": str(e)}), 400
    return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@h3RerollNG_bp.route("/api/ng/h3/jobs/<job_id>/windows", methods=["GET"])
def h3_windows_ng(job_id):
    try:
        return jsonify(reroll.describe_ng(job_id))
    except Exception as e:  # noqa: BLE001 -- _fail maps each kind to its status
        return _fail(e)


@h3RerollNG_bp.route("/api/ng/h3/jobs/<job_id>/reroll", methods=["POST"])
def h3_reroll_ng(job_id):
    body = request.get_json(silent=True) or {}
    try:
        window = int(body.get("window"))
    except (TypeError, ValueError):
        return jsonify({"error": "window must be a number"}), 400
    try:
        job = reroll.start_reroll_ng(
            job_id, window, seed=body.get("seed"), prompt=body.get("prompt"),
            from_previous=bool(body.get("from_previous", True)), joins=body.get("joins"))
    except Exception as e:  # noqa: BLE001
        return _fail(e)
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/h3/jobs/{job.job_id}"}), 202


@h3RerollNG_bp.route("/api/ng/h3/jobs/<job_id>/restitch", methods=["POST"])
def h3_restitch_ng(job_id):
    body = request.get_json(silent=True) or {}
    try:
        version = reroll.restitch_ng(job_id, body.get("joins"))
        return jsonify({"ok": True, "version": version, **reroll.describe_ng(job_id)})
    except Exception as e:  # noqa: BLE001
        return _fail(e)


@h3RerollNG_bp.route("/api/ng/h3/jobs/<job_id>/windows/<int:n>/video", methods=["GET"])
def h3_window_video_ng(job_id, n):
    try:
        src = reroll.resolve_source_ng(job_id)
        man = reroll.load_manifest_ng(src)
        clip = Path(src.job_dir) / man["windows"][n - 1]["clip"]
    except (IndexError, KeyError):
        return jsonify({"error": f"no window {n}"}), 404
    except Exception as e:  # noqa: BLE001
        return _fail(e)
    return send_file(clip, mimetype="video/mp4") if clip.is_file() else (jsonify({"error": "clip missing"}), 404)


@h3RerollNG_bp.route("/api/ng/h3/jobs/<job_id>/versions/<int:n>/video", methods=["GET"])
def h3_version_video_ng(job_id, n):
    try:
        src = reroll.resolve_source_ng(job_id)
        man = reroll.load_manifest_ng(src)
        name = man["original_film"] if n == 0 else man["versions"][n - 1]["film"]
        path = Path(src.job_dir) / name
    except (IndexError, KeyError):
        return jsonify({"error": f"no version {n}"}), 404
    except Exception as e:  # noqa: BLE001
        return _fail(e)
    return send_file(path, mimetype="video/mp4") if path.is_file() else (jsonify({"error": "film missing"}), 404)
