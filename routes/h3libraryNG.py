"""immichRingNG -- "Prompt Library" view: browse the minimax_h3_1k prompt/clip dataset.

Read-only. The folder comes from the h3_library_dir address setting; all parsing is in h3_libraryNG.
GET /api/ng/h3-library/list   ?q=&look=&shots=&dialogue=&music=&offset=&limit=
GET /api/ng/h3-library/clip/<id>        full record (three fields + joined text)
GET /api/ng/h3-library/video/<id>.mp4   the clip, with Range support so a preview can scrub
"""

import os

from flask import Blueprint, jsonify, request, send_file

import h3_libraryNG
from configNG import get_h3_library_dir

h3libraryNG_bp = Blueprint("h3libraryNG", __name__)


def _folder():
    folder = get_h3_library_dir()
    return folder if os.path.isdir(folder) else None


def _tri(name):
    v = request.args.get(name, "")
    return True if v == "1" else False if v == "0" else None


@h3libraryNG_bp.route("/api/ng/h3-library/list")
def h3_library_list():
    folder = _folder()
    if not folder:
        return jsonify({"error": "Library folder not found: %s. Set it in Settings." % get_h3_library_dir()}), 404
    return jsonify(h3_libraryNG.list_clips(
        folder,
        query=request.args.get("q", ""),
        look=request.args.get("look", ""),
        shots=request.args.get("shots", 0, type=int),
        dialogue=_tri("dialogue"),
        music=_tri("music"),
        offset=max(0, request.args.get("offset", 0, type=int)),
        limit=min(200, max(1, request.args.get("limit", 60, type=int))),
    ))


@h3libraryNG_bp.route("/api/ng/h3-library/clip/<clip_id>")
def h3_library_clip(clip_id):
    folder = _folder()
    rec = h3_libraryNG.get_clip(folder, clip_id) if folder else None
    if not rec:
        return jsonify({"error": "No such clip"}), 404
    return jsonify(rec)


@h3libraryNG_bp.route("/api/ng/h3-library/video/<clip_id>.mp4")
def h3_library_video(clip_id):
    folder = _folder()
    path = h3_libraryNG.video_path(folder, clip_id) if folder else None
    if not path:
        return jsonify({"error": "No such clip"}), 404
    return send_file(path, mimetype="video/mp4", conditional=True)
