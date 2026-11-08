"""immichRingNG -- HTTP layer for the soft-finish pass (see finish_engineNG.py / finish_jobsNG.py).

POST /api/ng/finish/jobs takes an H3 job id (the clip is looked up server-side, never a path) and the four
optional settings; the page polls GET .../jobs/<id>; the finished clip is served from .../video and
.../download. The original clip is never changed."""

import os

from flask import Blueprint, jsonify, request, send_file

import finish_engineNG as finish
import finish_jobsNG as finish_jobs

finishNG_bp = Blueprint("finishNG", __name__)


@finishNG_bp.route("/api/ng/finish/status", methods=["GET"])
def finish_status_ng():
    return jsonify(finish.finish_health_ng())


@finishNG_bp.route("/api/ng/finish/jobs", methods=["POST"])
def finish_start_ng():
    form = request.form
    params = {}
    try:
        for key in finish.DEFAULTS:
            raw = str(form.get(key) or "").strip()
            if raw:
                params[key] = float(raw)
        job = finish_jobs.start_finish_job_ng(str(form.get("h3_job_id") or "").strip(), params)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:        # FinishRequestError is a ValueError, as is a bad float
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "job_id": job.job_id, "poll_url": f"/api/ng/finish/jobs/{job.job_id}"}), 202


@finishNG_bp.route("/api/ng/finish/jobs", methods=["GET"])
def finish_list_ng():
    return jsonify({"jobs": finish_jobs.list_jobs_ng()})


@finishNG_bp.route("/api/ng/finish/jobs/<job_id>", methods=["GET"])
def finish_job_ng(job_id):
    try:
        return jsonify(finish_jobs.job_status_ng(job_id))
    except LookupError as e:
        return jsonify({"error": str(e)}), 404


@finishNG_bp.route("/api/ng/finish/jobs/<job_id>", methods=["DELETE"])
def finish_cancel_ng(job_id):
    return jsonify({"ok": finish_jobs.cancel_finish_job_ng(job_id)})


def _result_file(job_id):
    job = finish_jobs.get_job_ng(job_id)
    if job is None:
        return None, (jsonify({"error": f"no such job {job_id!r}"}), 404)
    if not job.result_path or not os.path.isfile(job.result_path):
        return None, (jsonify({"error": "video not ready"}), 404)
    return job, None


@finishNG_bp.route("/api/ng/finish/jobs/<job_id>/video", methods=["GET"])
def finish_video_ng(job_id):
    job, err = _result_file(job_id)
    return err if err else send_file(job.result_path, mimetype="video/mp4")


@finishNG_bp.route("/api/ng/finish/jobs/<job_id>/download", methods=["GET"])
def finish_download_ng(job_id):
    job, err = _result_file(job_id)
    return err if err else send_file(job.result_path, mimetype="video/mp4", as_attachment=True,
                                     download_name=f"finished-{job_id}.mp4")
