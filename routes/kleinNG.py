"""immichRingNG -- "Klein" view: FLUX 3 Klein pictures from text or from up to three reference pictures.

The same shape as the Hd-Multi view, with three labelled slots (subject, location, extra) like
Phosphene's, several variations per job, and a gallery of earlier jobs. A slot is filled by an upload
(`ref0`..`ref2`) or by an earlier Klein picture (`ref_from0`..`ref_from2` = "<job id>/<file>"), which
the server copies so nothing is downloaded and uploaded again. The queue is klein_jobsNG.py.
"""

import json
import os
import shutil
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

import klein_engineNG
import klein_lorasNG
import klein_outpaintNG
import klein_jobsNG as jobs
from configNG import KLEIN_DIR

kleinNG_bp = Blueprint("kleinNG", __name__)


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "prompt", "n", "width", "height", "seed", "model", "status", "error",
                                    "results", "log", "created_at", "started_at", "finished_at", "options", "outpaint")} | {
        "refs": [r["slot"] for r in job["refs"]]}


def _options(form) -> dict:
    """The optional settings from the form; blank fields are left out."""
    out = {}
    if (form.get("steps") or "").strip():
        out["steps"] = int(form["steps"])
    if (form.get("guidance") or "").strip():
        out["guidance"] = float(form["guidance"])
    if form.get("match_ref") == "1":
        out["match_ref"] = True
    if (form.get("loras") or "").strip():
        out["loras"] = json.loads(form["loras"])
    if form.get("low_ram") == "0":
        out["low_ram"] = False
        if (form.get("vae_tile") or "").strip():
            out["vae_tile"] = int(form["vae_tile"])
    return out


def _picture_path(ref: str):
    """KLEIN_DIR/<job>/out/<file> for a "<job>/<file>" reference, or None when it is not one."""
    job_id, _, name = ref.partition("/")
    if not job_id or not name or "/" in name or ".." in ref or "\\" in ref:
        return None
    path = Path(KLEIN_DIR) / job_id / "out" / name
    return path if path.is_file() else None


@kleinNG_bp.route("/api/ng/klein/health")
def klein_health_ng():
    loras = [{k: v for k, v in e.items() if k != "path"} for e in klein_lorasNG.list_loras_ng()]
    return jsonify(klein_engineNG.klein_health_ng() | {"slots": list(jobs.SLOT_NAMES), "loras": loras})


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
                            form.get("model") or klein_engineNG.DEFAULT_MODEL, refs, _options(form))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 503
    return jsonify({"jobId": job["id"], "job": _public(job)})


OUTPAINT_LORA = "flux-3-image-klein-outpaint"


@kleinNG_bp.route("/api/ng/klein/outpaint", methods=["POST"])
def klein_outpaint_ng():
    """Extend a picture: pad it with gray, run the Outpaint LoRA, put the original back."""
    form = request.form
    job_id, folder = jobs.new_job_dir_ng()
    try:
        upload, earlier = request.files.get("image"), (form.get("image_from") or "").strip()
        if upload and upload.filename:
            original = folder / "original.png"
            upload.save(original)
        elif earlier:
            src = _picture_path(earlier)
            if src is None:
                raise ValueError(f"no such earlier picture {earlier!r}")
            original = folder / "original.png"
            shutil.copyfile(src, original)
        else:
            raise ValueError("give a picture to extend")
        model = form.get("model") or klein_engineNG.DEFAULT_MODEL
        if model.startswith("base-"):
            raise ValueError("outpainting is tuned for the distilled model; pick a distilled one")
        if klein_lorasNG.find_lora_ng(OUTPAINT_LORA) is None:
            raise RuntimeError("the Outpaint LoRA is not installed (flux3/flux-3-image-klein-outpaint.safetensors "
                               "in the shared loras folder)")
        pads = [int(form.get(k) or 0) for k in ("left", "top", "right", "bottom")]
        canvas = folder / "padded.png"
        plan = klein_outpaintNG.prepare_ng(original, *pads, canvas, folder.parent / "source.png")
        job = jobs.start_ng(
            job_id, klein_outpaintNG.outpaint_prompt(form.get("scene") or ""), int(form.get("n", 1)),
            plan["canvas"][0], plan["canvas"][1], (form.get("seed") or "").strip() or None, model,
            [{"slot": "padded", "path": str(canvas)}],
            {"match_ref": True, "loras": [{"name": OUTPAINT_LORA, "scale": float(form.get("strength") or 1.5)}]},
            outpaint_plan={"pads": pads, "offset": list(plan["offset"]), "scale": plan["scale"],
                           "canvas": list(plan["canvas"]), "keep_original": form.get("keep_original", "1") != "0"})
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        shutil.rmtree(folder.parent, ignore_errors=True)
        return jsonify({"error": str(exc)}), 503
    return jsonify({"jobId": job["id"], "job": _public(job), "plan": plan | {"offset": list(plan["offset"])}})


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
