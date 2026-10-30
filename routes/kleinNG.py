"""immichRingNG -- "Klein" view: FLUX 3 Klein pictures from text or from up to three reference pictures.

The same shape as the Hd-Multi view, with three labelled slots (subject, location, extra) like
Phosphene's, several variations per job, and a gallery of earlier jobs. A slot is filled by an upload
(`ref0`..`ref2`) or by an earlier Klein picture (`ref_from0`..`ref_from2` = "<job id>/<file>"), which
the server copies so nothing is downloaded and uploaded again. The queue is klein_jobsNG.py.
"""

import os
import shutil
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

import klein_engineNG
import klein_jobsNG as jobs
from configNG import KLEIN_DIR

kleinNG_bp = Blueprint("kleinNG", __name__)


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "prompt", "n", "width", "height", "seed", "model", "status", "error",
                                    "results", "log", "created_at", "started_at", "finished_at")} | {
        "refs": [r["slot"] for r in job["refs"]]}


def _picture_path(ref: str):
    """KLEIN_DIR/<job>/out/<file> for a "<job>/<file>" reference, or None when it is not one."""
    job_id, _, name = ref.partition("/")
    if not job_id or not name or "/" in name or ".." in ref or "\\" in ref:
        return None
    path = Path(KLEIN_DIR) / job_id / "out" / name
    return path if path.is_file() else None


@kleinNG_bp.route("/api/ng/klein/health")
def klein_health_ng():
    return jsonify(klein_engineNG.klein_health_ng() | {"slots": list(jobs.SLOT_NAMES)})


@kleinNG_bp.route("/api/ng/klein/generate", methods=["POST"])
def klein_generate_ng():
    form = request.form
    job_id, folder = jobs.new_job_dir_ng()
    try:
        refs = []
        for i, slot in enumerate(jobs.SLOT_NAMES):
            upload, earlier = request.files.get(f"ref{i}"), (form.get(f"ref_from{i}") or "").strip()
            if upload and upload.filename:
                dest = folder / f"{slot}{os.path.splitext(upload.filename)[1].lower() or '.png'}"
                upload.save(dest)
            elif earlier:
                src = _picture_path(earlier)
                if src is None:
                    raise ValueError(f"{slot}: no such earlier picture {earlier!r}")
                dest = folder / f"{slot}.png"
                shutil.copyfile(src, dest)
            else:
                continue
            refs.append({"slot": slot, "path": str(dest)})
        job = jobs.start_ng(job_id, form.get("prompt") or "", int(form.get("n", 1)),
                            int(form.get("width", 1024)), int(form.get("height", 576)),
                            (form.get("seed") or "").strip() or None,
                            form.get("model") or klein_engineNG.DEFAULT_MODEL, refs)
    except (ValueError, TypeError) as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 503
    return jsonify({"jobId": job["id"], "job": _public(job)})


@kleinNG_bp.route("/api/ng/klein/status/<job_id>")
def klein_status_ng(job_id):
    job = jobs.get_ng(job_id)
    return (jsonify(_public(job)) if job else (jsonify({"error": "unknown job"}), 404))


@kleinNG_bp.route("/api/ng/klein/jobs")
def klein_jobs_ng():
    return jsonify({"jobs": [_public(j) for j in jobs.recent_ng(int(request.args.get("limit", 30)))]})


@kleinNG_bp.route("/api/ng/klein/cancel/<job_id>", methods=["POST"])
def klein_cancel_ng(job_id):
    return jsonify({"cancelled": jobs.cancel_ng(job_id)})


@kleinNG_bp.route("/api/ng/klein/image/<job_id>/<name>")
def klein_image_ng(job_id, name):
    path = _picture_path(f"{job_id}/{name}")
    if path is None:
        return jsonify({"error": "no such picture"}), 404
    return send_file(path, mimetype="image/png")
