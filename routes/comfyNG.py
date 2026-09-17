"""immichRingNG -- "ComfyUI" view: run an arbitrary workflow extracted
from a PNG's embedded metadata, with a generic form for its parameters.

NG-only file. This is the "ComfyUI slot-override" idea from
RING_VIZ_MCP.md realized via the PNG-metadata angle rather than a live
node-graph editor: ComfyUI embeds the exact flattened "prompt" (API
format: {node_id: {class_type, inputs}}, where each input is either a
plain scalar or a [source_node_id, source_slot] link) into every image
it saves. Confirmed live against a real ComfyUI instance -- see the
session's own hand-flattened submission before this route existed.

Scope, deliberately (see conversation): any input whose value is a
plain scalar becomes an editable form field; anything wired via a link
stays fixed. That's correct and sufficient for straight txt2img-shaped
workflows. It does NOT special-case image-typed inputs (a LoadImage
node's `image` input is *also* just a scalar -- a filename string on
ComfyUI's own server, not something a browser can usefully paste an
image into today) -- that's real future work for ControlNet/IPAdapter-
style workflows, once one actually exists to build against, not
guessed at now.

No project/character concept, same as routes/hdmultiNG.py -- a
workflow here doesn't belong to any tab.
"""

import io
import json
import os
import time
import uuid

import requests
from flask import Blueprint, Response, jsonify, request, send_file
from PIL import Image

from configNG import COMFY_WORKFLOWS_DIR, get_comfyui_base_url
from stateNG import _comfy_extracts_ng, _comfy_jobs_ng

comfyNG_bp = Blueprint("comfyNG", __name__)

_WORKFLOWS_INDEX = os.path.join(COMFY_WORKFLOWS_DIR, "index.json")


def _is_link(value):
    """True if this input value is a wire to another node's output
    ([node_id, slot_index]) rather than a literal widget value ComfyUI's
    own UI would have shown as an editable field."""
    return (
        isinstance(value, list) and len(value) == 2
        and isinstance(value[0], str) and isinstance(value[1], int)
    )


def _value_type(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _is_image_field(class_type, input_name):
    """LoadImage's own `image` input is the img2img/img2video entry point
    -- its value is a filename on ComfyUI's *own* server, not something a
    browser can paste a photo into directly (this file's own module
    docstring flagged this as the scoped-out part). Deliberately just
    this one node type for now, the common case for a straight "load one
    photo" workflow -- a mask input or a second chained loader would need
    its own handling, not guessed at here."""
    return class_type == "LoadImage" and input_name == "image"


def _coerce_like(original, new_value):
    """Cast a form-submitted override back to the original value's type
    -- everything arrives over JSON as whatever the frontend sent
    (typically a string from an <input>), but ComfyUI's own node code
    expects the real type (an int steps count, not "8")."""
    if isinstance(original, bool):
        return bool(new_value) if not isinstance(new_value, str) else new_value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(original, int) and not isinstance(original, bool):
        return int(new_value)
    if isinstance(original, float):
        return float(new_value)
    return new_value


def _extract_node_titles(png_text):
    """Best-effort node_id -> custom title map from the PNG's separate
    'workflow' metadata chunk -- the full editor graph ComfyUI embeds
    alongside 'prompt' (same shape ComfyUI's own "Save (API Format)"
    vs. plain save produces). Only that full-graph format carries a
    node's UI title ("Load Image-insert reolace" etc); the flattened
    'prompt' format this route otherwise runs on on doesn't. This is
    purely cosmetic (tells two same-class_type LoadImage nodes apart in
    the generic form), so missing/unparsable 'workflow' metadata just
    means no titles, not an error -- extraction still works from
    'prompt' alone."""
    raw = (png_text or {}).get("workflow")
    if not raw:
        return {}
    try:
        workflow = json.loads(raw)
    except ValueError:
        return {}
    titles = {}
    for node in workflow.get("nodes") or []:
        title = node.get("title")
        node_id = node.get("id")
        if title and node_id is not None:
            titles[str(node_id)] = title
    return titles


def _extract_fields_from_png_bytes(png_bytes):
    """Parse a ComfyUI-saved PNG's embedded 'prompt' metadata into a graph
    plus the flat field list the frontend renders a form from. Shared by
    the upload-a-PNG route and the load-a-saved-workflow route below --
    same PNG-metadata format either way, just a different source.

    Returns (graph, fields). Raises ValueError with a user-facing message
    on any problem (unreadable image, no embedded workflow, bad JSON)."""
    try:
        im = Image.open(io.BytesIO(png_bytes))
        png_text = im.text if hasattr(im, "text") else {}
        raw = png_text.get("prompt")
    except Exception as e:
        raise ValueError(f"could not read that image: {e}")
    if not raw:
        raise ValueError("no ComfyUI workflow found in this PNG's metadata")

    try:
        graph = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"embedded workflow is not valid JSON: {e}")

    titles = _extract_node_titles(png_text)

    fields = []
    for node_id, node in graph.items():
        class_type = node.get("class_type", "?")
        for input_name, value in (node.get("inputs") or {}).items():
            if _is_link(value):
                continue
            fields.append({
                "nodeId": node_id,
                "classType": class_type,
                "title": titles.get(node_id),
                "inputName": input_name,
                "value": value,
                "valueType": "image" if _is_image_field(class_type, input_name) else _value_type(value),
            })
    return graph, fields


@comfyNG_bp.route("/api/ng/comfy/extract", methods=["POST"])
def comfy_extract_ng():
    f = request.files.get("png")
    if not f or not f.filename:
        return jsonify({"error": "a PNG file is required"}), 400

    try:
        graph, fields = _extract_fields_from_png_bytes(f.read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    extract_id = uuid.uuid4().hex[:12]
    _comfy_extracts_ng[extract_id] = {"graph": graph}

    return jsonify({"extractId": extract_id, "fields": fields})


@comfyNG_bp.route("/api/ng/comfy/view")
def view_comfy_image_ng():
    """Generic proxy for whatever a LoadImage-shaped field's current
    string value points at on ComfyUI's own server -- lets the frontend
    show a real preview of the image a saved/loaded workflow already
    references, not just its filename, before you replace it (or don't).
    Query params mirror ComfyUI's own /view: filename, subfolder
    (optional), type (default "input", matching a LoadImage widget)."""
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400
    subfolder = request.args.get("subfolder", "")
    type_ = request.args.get("type", "input")
    try:
        content, content_type = _fetch_comfy_view_bytes(filename, subfolder, type_)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI: {e}"}), 502
    return Response(content, mimetype=content_type)


@comfyNG_bp.route("/api/ng/comfy/server-images")
def list_comfy_server_images_ng():
    """Filenames already sitting in ComfyUI's own input/ folder -- e.g.
    stuff dropped there directly on whatever machine ComfyUI actually
    runs on ("the studio"), which the browser has no local copy of to
    paste or pick from disk. Reuses the exact combo list ComfyUI's own
    LoadImage widget populates itself from (its /object_info), rather
    than this app reaching into that machine's filesystem itself --
    if ComfyUI can see it, so can this, through the one HTTP hop it
    already talks over."""
    base_url = get_comfyui_base_url()
    try:
        resp = requests.get(f"{base_url}/object_info/LoadImage", timeout=10)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI at {base_url}: {e}"}), 502
    if resp.status_code != 200:
        return jsonify({"error": f"ComfyUI returned HTTP {resp.status_code}"}), 502

    try:
        data = resp.json()
        images = data["LoadImage"]["input"]["required"]["image"][0]
    except (ValueError, KeyError, IndexError, TypeError):
        return jsonify({"error": "unexpected response shape from ComfyUI's object_info"}), 502

    return jsonify({"images": images})


@comfyNG_bp.route("/api/ng/comfy/upload-image", methods=["POST"])
def upload_comfy_image_ng():
    """Proxies a picked photo into ComfyUI's own /upload/image (default
    type="input", exactly what a LoadImage node's `image` widget expects
    to reference) and hands back the filename value to substitute into
    that field -- img2img/img2video entry point, see _is_image_field.
    Browser -> this app -> ComfyUI, same one-hop-proxy shape as every
    other route here (this app never touches ComfyUI's actual pixels
    beyond passing bytes through)."""
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify({"error": "an image file is required"}), 400

    base_url = get_comfyui_base_url()
    try:
        resp = requests.post(
            f"{base_url}/upload/image",
            files={"image": (f.filename, f.stream, f.mimetype or "application/octet-stream")},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI at {base_url}: {e}"}), 502
    if resp.status_code != 200:
        return jsonify({"error": f"ComfyUI rejected the upload (HTTP {resp.status_code}): {resp.text[:500]}"}), 502

    data = resp.json()
    name = data.get("name")
    subfolder = data.get("subfolder") or ""
    if not name:
        return jsonify({"error": "ComfyUI's upload response had no filename"}), 502

    # Same value shape ComfyUI's own frontend widget would set on a
    # LoadImage node: bare filename, or "subfolder/filename" if it landed
    # in one.
    value = f"{subfolder}/{name}" if subfolder else name
    return jsonify({"value": value})


@comfyNG_bp.route("/api/ng/comfy/generate", methods=["POST"])
def comfy_generate_ng():
    body = request.get_json(silent=True) or {}
    extract_id = body.get("extractId")
    overrides = body.get("overrides") or {}

    extract = _comfy_extracts_ng.get(extract_id)
    if not extract:
        return jsonify({"error": "unknown or expired workflow -- extract it again"}), 404

    import copy
    graph = copy.deepcopy(extract["graph"])

    for key, new_value in overrides.items():
        try:
            node_id, input_name = key.split(".", 1)
        except ValueError:
            continue
        node = graph.get(node_id)
        if not node or input_name not in (node.get("inputs") or {}):
            continue
        original = node["inputs"][input_name]
        if _is_link(original):
            continue
        node["inputs"][input_name] = _coerce_like(original, new_value)

    base_url = get_comfyui_base_url()
    try:
        resp = requests.post(
            f"{base_url}/prompt",
            json={"prompt": graph, "client_id": uuid.uuid4().hex},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI at {base_url}: {e}"}), 502

    if resp.status_code != 200:
        return jsonify({"error": f"ComfyUI rejected the workflow (HTTP {resp.status_code}): {resp.text[:500]}"}), 502

    data = resp.json()
    if data.get("node_errors"):
        return jsonify({"error": f"ComfyUI reported node errors: {data['node_errors']}"}), 400
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        return jsonify({"error": "ComfyUI did not return a prompt_id"}), 502

    job_id = uuid.uuid4().hex[:12]
    _comfy_jobs_ng[job_id] = {"promptId": prompt_id, "status": "running"}
    return jsonify({"jobId": job_id})


@comfyNG_bp.route("/api/ng/comfy/status/<job_id>")
def comfy_status_ng(job_id):
    job = _comfy_jobs_ng.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404

    base_url = get_comfyui_base_url()
    try:
        resp = requests.get(f"{base_url}/history/{job['promptId']}", timeout=10)
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "error": f"could not reach ComfyUI: {e}"})

    history = (resp.json() or {}).get(job["promptId"])
    if not history:
        return jsonify({"status": "running"})

    status = history.get("status") or {}
    if not status.get("completed"):
        if status.get("status_str") == "error":
            return jsonify({"status": "error", "error": "ComfyUI reported the job failed -- check its own console/log"})
        return jsonify({"status": "running"})

    images = []
    for node_out in (history.get("outputs") or {}).values():
        images.extend(node_out.get("images") or [])
    if not images:
        return jsonify({"status": "error", "error": "job finished but produced no image output"})

    img = images[-1]
    job["resultFilename"] = img.get("filename")
    job["resultSubfolder"] = img.get("subfolder") or ""
    job["resultType"] = img.get("type") or "output"
    job["status"] = "done"
    return jsonify({"status": "done"})


def _fetch_comfy_view_bytes(filename, subfolder, type_):
    """GET a finished job's image bytes back from ComfyUI's own /view
    endpoint (that's where the actual file lives -- this app never saved
    a copy). Shared by the result route below and by the save-workflow
    route, which needs the same bytes to write into the library."""
    base_url = get_comfyui_base_url()
    resp = requests.get(
        f"{base_url}/view",
        params={"filename": filename, "subfolder": subfolder or "", "type": type_ or "output"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "image/png")


@comfyNG_bp.route("/api/ng/comfy/result/<job_id>")
def comfy_result_ng(job_id):
    job = _comfy_jobs_ng.get(job_id)
    if not job or job.get("status") != "done" or not job.get("resultFilename"):
        return jsonify({"error": "not ready"}), 404

    try:
        content, content_type = _fetch_comfy_view_bytes(
            job["resultFilename"], job.get("resultSubfolder", ""), job.get("resultType", "output"))
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI: {e}"}), 502
    return Response(content, mimetype=content_type)


def _load_workflows_index():
    try:
        with open(_WORKFLOWS_INDEX) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_workflows_index(index):
    os.makedirs(COMFY_WORKFLOWS_DIR, exist_ok=True)
    with open(_WORKFLOWS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


@comfyNG_bp.route("/api/ng/comfy/workflows", methods=["GET"])
def list_comfy_workflows_ng():
    """The saved-workflow picker's list -- works identically from the
    iPad or the Mac, unlike the file-picker/paste flow above, since
    nothing here needs access to whatever machine's disk the original
    PNG lived on."""
    index = _load_workflows_index()
    items = [
        {"id": wf_id, "name": entry.get("name", wf_id), "savedAt": entry.get("savedAt")}
        for wf_id, entry in index.items()
    ]
    items.sort(key=lambda x: x.get("savedAt") or 0, reverse=True)
    return jsonify({"workflows": items})


@comfyNG_bp.route("/api/ng/comfy/workflows", methods=["POST"])
def save_comfy_workflow_ng():
    """Save a finished job's result PNG into the library under a name --
    explicit action, not automatic, so this doesn't fill up with 50
    near-identical entries every time the same workflow gets reused
    (John's point). The PNG itself is the save format: ComfyUI already
    embedded the exact workflow that made it, so nothing extra to store
    beyond that file and a display name."""
    body = request.get_json(silent=True) or {}
    job_id = body.get("jobId")
    name = (body.get("name") or "").strip()
    if not job_id or not name:
        return jsonify({"error": "jobId and name are required"}), 400

    job = _comfy_jobs_ng.get(job_id)
    if not job or job.get("status") != "done" or not job.get("resultFilename"):
        return jsonify({"error": "that job has no finished result to save"}), 400

    try:
        content, _ = _fetch_comfy_view_bytes(
            job["resultFilename"], job.get("resultSubfolder", ""), job.get("resultType", "output"))
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"could not reach ComfyUI: {e}"}), 502

    wf_id = uuid.uuid4().hex[:12]
    os.makedirs(COMFY_WORKFLOWS_DIR, exist_ok=True)
    with open(os.path.join(COMFY_WORKFLOWS_DIR, f"{wf_id}.png"), "wb") as f:
        f.write(content)

    index = _load_workflows_index()
    index[wf_id] = {"name": name, "savedAt": time.time()}
    _save_workflows_index(index)

    return jsonify({"id": wf_id, "name": name})


@comfyNG_bp.route("/api/ng/comfy/workflows/<wf_id>", methods=["DELETE"])
def delete_comfy_workflow_ng(wf_id):
    index = _load_workflows_index()
    if wf_id not in index:
        return jsonify({"error": "unknown workflow"}), 404
    del index[wf_id]
    _save_workflows_index(index)
    try:
        os.remove(os.path.join(COMFY_WORKFLOWS_DIR, f"{wf_id}.png"))
    except OSError:
        pass
    return jsonify({"ok": True})


@comfyNG_bp.route("/api/ng/comfy/workflows/<wf_id>/thumb")
def comfy_workflow_thumb_ng(wf_id):
    path = os.path.join(COMFY_WORKFLOWS_DIR, f"{wf_id}.png")
    if not os.path.isfile(path):
        return jsonify({"error": "unknown workflow"}), 404
    return send_file(path, mimetype="image/png")


@comfyNG_bp.route("/api/ng/comfy/workflows/<wf_id>/load", methods=["POST"])
def load_comfy_workflow_ng(wf_id):
    """Same extraction as /api/ng/comfy/extract (and the same response
    shape, {extractId, fields}) -- just reading a saved library PNG off
    disk instead of an uploaded one, so the rest of the frontend flow
    (dynamic form, Generate) doesn't need to know which source it came
    from."""
    path = os.path.join(COMFY_WORKFLOWS_DIR, f"{wf_id}.png")
    if not os.path.isfile(path):
        return jsonify({"error": "unknown workflow"}), 404

    with open(path, "rb") as f:
        png_bytes = f.read()
    try:
        graph, fields = _extract_fields_from_png_bytes(png_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    extract_id = uuid.uuid4().hex[:12]
    _comfy_extracts_ng[extract_id] = {"graph": graph}
    return jsonify({"extractId": extract_id, "fields": fields})
