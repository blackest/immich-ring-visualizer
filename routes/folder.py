import os
import uuid
import threading
import zipfile
import shutil
from flask import Blueprint, request, jsonify, Response, send_file
from config import FRAME_STORE, IMAGE_EXTS
from folder_analysis import run_folder_analysis
from state import _analysis_jobs

folder_bp = Blueprint('folder', __name__)

@folder_bp.route("/api/analyze-folder", methods=["POST"])
def analyze_folder():
    """Accepts either multiple image files (folder picker / multi-select)
    under the 'images' field, or a single 'zip' file containing images.
    Runs the exact same analysis pipeline as video, just over stills."""
    sim_threshold = float(request.form.get("simThreshold", 0.65))
    blur_threshold = float(request.form.get("blurThreshold", 50))
    ref_index = int(request.form.get("refIndex", 1))
    cache_format = "png" if request.form.get("cacheFormat") == "png" else "jpg"

    job_id = uuid.uuid4().hex[:12]
    src_dir = os.path.join(FRAME_STORE, f"{job_id}_srcimgs")
    os.makedirs(src_dir, exist_ok=True)

    saved_paths = []

    zip_file = request.files.get("zip")
    if zip_file and zip_file.filename:
        zip_path = os.path.join(src_dir, "_upload.zip")
        zip_file.save(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    ext = os.path.splitext(member)[1].lower()
                    if ext not in IMAGE_EXTS or member.startswith("__MACOSX"):
                        continue
                    # flatten any subfolder structure from the zip
                    safe_name = os.path.basename(member)
                    if not safe_name:
                        continue
                    out_path = os.path.join(src_dir, safe_name)
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    saved_paths.append(out_path)
        except zipfile.BadZipFile:
            shutil.rmtree(src_dir, ignore_errors=True)
            return jsonify({"error": "uploaded file is not a valid zip"}), 400
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        source_name = os.path.splitext(zip_file.filename)[0]
    else:
        images = request.files.getlist("images")
        if not images:
            shutil.rmtree(src_dir, ignore_errors=True)
            return jsonify({"error": "provide 'images' (multiple files) or a 'zip' file"}), 400
        for f in images:
            if not f.filename:
                continue
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            safe_name = os.path.basename(f.filename)
            out_path = os.path.join(src_dir, safe_name)
            f.save(out_path)
            saved_paths.append(out_path)
        source_name = request.form.get("sourceName") or "folder_set"

    if not saved_paths:
        shutil.rmtree(src_dir, ignore_errors=True)
        return jsonify({"error": "no valid images found (jpg/jpeg/png/webp/bmp)"}), 400

    saved_paths.sort(key=lambda p: os.path.basename(p).lower())

    _analysis_jobs[job_id] = {
        "status": "running", "results": [], "error": None,
        "sourceName": source_name, "sourceType": "folder",
        "srcDir": src_dir,
        "simThreshold": sim_threshold,
        "blurThreshold": blur_threshold,
        "cacheFormat": cache_format,
    }

    t = threading.Thread(
        target=run_folder_analysis,
        args=(job_id, saved_paths, sim_threshold, blur_threshold, ref_index, cache_format),
        daemon=True
    )
    t.start()

    return jsonify({"jobId": job_id, "imageCount": len(saved_paths)})

