"""immichRingNG -- "Hd-Multi" view: one-off HiDream edit/multi-ref jobs.

NG-only file. Standalone panel rather than folded into routes/generateNG.py
-- Generate's whole model is "one character reference -> a full sheet of
posed shots"; this is "1-3 arbitrary reference images + one prompt -> one
result image" (object removal, or composing two images together, e.g. a
bad LTX render frame's pose/scene with a character reference's likeness).
Different enough inputs/outputs that bolting it onto Generate's pose grid
would tangle both, per the NG-duplication instinct elsewhere in this repo
-- see hidream_engineNG.generate_hidream_ng's own docstring for what K=1
vs K=2-3 actually do.

Shares hidream_engineNG.generate_hidream_ng (the actual subprocess call)
and character_sheetNG._SHEET_LOCK_NG (the one lock serializing every
HiDream call in the app, sheet generation included -- this is the same
shared GPU-bound resource, so a job here must never run concurrently with
one from Generate) with the existing character-sheet feature. Everything
else -- job store, output directory, routes -- is its own.
"""

import os
import threading
import uuid

from flask import Blueprint, jsonify, request, send_file

import hidream_engineNG
from character_sheetNG import _SHEET_LOCK_NG
from configNG import HDMULTI_DIR
from stateNG import _hdmulti_jobs_ng

hdmultiNG_bp = Blueprint("hdmultiNG", __name__)

MAX_REFS = 3


def _run_hdmulti_job_ng(job_id, prompt, ref_paths, width, height, steps, seed):
    job = _hdmulti_jobs_ng[job_id]
    try:
        cfg = hidream_engineNG.HiDreamConfig(steps=steps)
        output_dir = os.path.join(HDMULTI_DIR, job_id)
        with _SHEET_LOCK_NG:
            candidates = hidream_engineNG.generate_hidream_ng(
                prompt=prompt, n=1, width=width, height=height,
                output_dir=output_dir, base_seed=seed,
                refs=ref_paths, config=cfg, allow_offspec_res=True,
                on_log=lambda line: job.setdefault("log", []).append(line),
            )
        if not candidates or not candidates[0].get("png_path"):
            raise RuntimeError("HiDream returned no image")
        job["status"] = "done"
        job["pngPath"] = str(candidates[0]["png_path"])
        job["seed"] = candidates[0].get("seed")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@hdmultiNG_bp.route("/api/ng/hdmulti/generate", methods=["POST"])
def hdmulti_generate_ng():
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    ref_files = []
    for i in range(MAX_REFS):
        f = request.files.get(f"ref{i}")
        if f and f.filename:
            ref_files.append(f)
    if not ref_files:
        return jsonify({"error": "at least one reference image is required"}), 400

    width = int(request.form.get("width", 1024))
    height = int(request.form.get("height", 1024))
    steps = int(request.form.get("steps", 28))
    seed_raw = (request.form.get("seed") or "").strip()
    seed = int(seed_raw) if seed_raw else None

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(HDMULTI_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ref_paths = []
    for i, f in enumerate(ref_files):
        ext = os.path.splitext(f.filename)[1] or ".png"
        p = os.path.join(job_dir, f"ref{i}{ext}")
        f.save(p)
        ref_paths.append(p)

    _hdmulti_jobs_ng[job_id] = {
        "status": "running", "error": None, "pngPath": None,
        "prompt": prompt, "refCount": len(ref_paths),
    }

    t = threading.Thread(
        target=_run_hdmulti_job_ng,
        args=(job_id, prompt, ref_paths, width, height, steps, seed),
        daemon=True,
    )
    t.start()

    return jsonify({"jobId": job_id})


@hdmultiNG_bp.route("/api/ng/hdmulti/status/<job_id>")
def hdmulti_status_ng(job_id):
    job = _hdmulti_jobs_ng.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({
        "status": job["status"],
        "error": job.get("error"),
        "seed": job.get("seed"),
        "log": job.get("log", [])[-20:],
    })


@hdmultiNG_bp.route("/api/ng/hdmulti/result/<job_id>")
def hdmulti_result_ng(job_id):
    job = _hdmulti_jobs_ng.get(job_id)
    if not job or job.get("status") != "done" or not job.get("pngPath"):
        return jsonify({"error": "not ready"}), 404
    return send_file(job["pngPath"], mimetype="image/png")
