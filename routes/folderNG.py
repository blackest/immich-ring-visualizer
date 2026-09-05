"""immichRingNG -- folder/zip image-set ingest.

NG-only file. Copied from routes/folder.py as it exists today -- per the
NG rule, this is a deliberate duplicate, not an import from the original
module. Uses folder_analysisNG.py / stateNG.py, its own NG-suffixed
twins of the originals.
"""

import io
import os
import threading
import uuid
import zipfile

from flask import Blueprint, jsonify, request

from configNG import IMAGE_EXTS
from folder_analysisNG import run_folder_analysis_ng
from stateNG import _analysis_jobs_ng

folderNG_bp = Blueprint("folderNG", __name__)


@folderNG_bp.route("/api/ng/analyze-folder", methods=["POST"])
def analyze_folder_ng():
    """Accepts either multiple image files under the 'images' field, or a
    single 'zip' file containing images. Everything is read into memory
    -- nothing here is written to disk, matching the video path's
    no-disk-writes-for-ingest principle."""
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

    _analysis_jobs_ng[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "sourceType": "folder",
        "srcImages": dict(images),
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
    }

    t = threading.Thread(
        target=run_folder_analysis_ng,
        args=(job_id, images, sim_threshold, blur_threshold, ref_index, cache_format),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(images)})
