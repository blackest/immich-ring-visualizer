"""immichRingNG -- HTTP layer for "Make vertical" (a finished Animate clip made 9:16, 4:5 or square, the bands above
and below painted by the AI fill). The queue is outpaint_jobsNG.py, the model is outpaint_engineNG.py.

POST starts a job and returns its id at once; the page polls GET .../jobs/<id>. The source is an Animate job id, never a
path, so the client cannot point the server at an arbitrary file.
"""

from flask import Blueprint, jsonify, request, send_file

import outpaint_engineNG as engine
import outpaint_jobsNG as jobs

outpaintNG_bp = Blueprint("outpaintNG", __name__)


@outpaintNG_bp.route("/api/ng/outpaint/status", methods=["GET"])
def outpaint_status():
    return jsonify(engine.outpaint_health_ng())


@outpaintNG_bp.route("/api/ng/outpaint/start", methods=["POST"])
def outpaint_start():
    body = request.get_json(silent=True) or {}
    try:
        job = jobs.start_job_ng(str(body.get("job_id") or ""), str(body.get("ratio") or "9:16"))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(job), 202


@outpaintNG_bp.route("/api/ng/outpaint/jobs/<job_id>", methods=["GET"])
def outpaint_job(job_id):
    job = jobs.get_ng(job_id)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    return jsonify(jobs.public_ng(job))


@outpaintNG_bp.route("/api/ng/outpaint/jobs/<job_id>", methods=["DELETE"])
def outpaint_cancel(job_id):
    return jsonify({"cancelled": jobs.cancel_ng(job_id)})


def _finished_path(job_id):
    job = jobs.get_ng(job_id)
    if job is None or job["status"] != "done":
        return None
    return job["output"]


@outpaintNG_bp.route("/api/ng/outpaint/jobs/<job_id>/video", methods=["GET"])
def outpaint_video(job_id):
    path = _finished_path(job_id)
    if not path:
        return jsonify({"error": "not finished"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@outpaintNG_bp.route("/api/ng/outpaint/jobs/<job_id>/download", methods=["GET"])
def outpaint_download(job_id):
    path = _finished_path(job_id)
    if not path:
        return jsonify({"error": "not finished"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=True, download_name=f"vertical-{job_id}.mp4")
