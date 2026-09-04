import io
import uuid
import threading
import zipfile
import os
from flask import Blueprint, request, jsonify, Response, send_file
from config import IMAGE_EXTS
from folder_analysis import run_folder_analysis
from state import _analysis_jobs

folder_bp = Blueprint('folder', __name__)

@folder_bp.route("/api/analyze-folder", methods=["POST"])
def analyze_folder():
    """Accepts either multiple image files (folder picker / multi-select)
    under the 'images' field, or a single 'zip' file containing images.
    Runs the exact same analysis pipeline as video, just over stills.

    Everything is read into memory and handed to run_folder_analysis as
    (orig_name, bytes) pairs -- nothing here is written to disk. The same
    bytes are kept in job["srcImages"] for the job's lifetime so
    /api/export-job can pull the true original later without a disk
    round-trip, the same way a video job already keeps its videoBytes in
    memory. Previously this wrote every uploaded file (or every image
    extracted from an uploaded zip) to a temp directory under FRAME_STORE
    and never cleaned it up on a successful job -- an unbounded disk leak
    for something cheap to just re-upload, unlike character-sheet
    generation (which does deliberately persist to disk, since redoing a
    30-minute render is the expensive case worth protecting against)."""
    sim_threshold = float(request.form.get("simThreshold", 0.1))
    blur_threshold = float(request.form.get("blurThreshold", 1))
    ref_index = int(request.form.get("refIndex", 1))
    cache_format = "png" if request.form.get("cacheFormat") == "png" else "jpg"

    job_id = uuid.uuid4().hex[:12]

    images = []  # list of (orig_name, bytes), in upload order

    zip_file = request.files.get("zip")
    if zip_file and zip_file.filename:
        zip_bytes = zip_file.read()
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.namelist():
                    ext = os.path.splitext(member)[1].lower()
                    if ext not in IMAGE_EXTS or member.startswith("__MACOSX"):
                        continue
                    # flatten any subfolder structure from the zip
                    safe_name = os.path.basename(member)
                    if not safe_name:
                        continue
                    images.append((safe_name, zf.read(member)))
        except zipfile.BadZipFile:
            return jsonify({"error": "uploaded file is not a valid zip"}), 400
        source_name = os.path.splitext(zip_file.filename)[0]
    else:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "provide 'images' (multiple files) or a 'zip' file"}), 400
        for f in files:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            safe_name = os.path.basename(f.filename)
            images.append((safe_name, f.read()))
        source_name = request.form.get("sourceName") or "folder_set"

    if not images:
        return jsonify({"error": "no valid images found (jpg/jpeg/png/webp/bmp)"}), 400

    images.sort(key=lambda pair: pair[0].lower())

    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "sourceType": "folder",
        "srcImages": dict(images),
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
    }

    t = threading.Thread(
        target=run_folder_analysis,
        args=(job_id, images, sim_threshold, blur_threshold, ref_index, cache_format),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(images)})
