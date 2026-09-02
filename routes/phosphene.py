"""Character-sheet generation: face photo -> a shot-list of turnaround /
dataset views, rendered as a background job.

This used to proxy every request over HTTP to a separate Phosphene
process (127.0.0.1:8198). As of PHOSPHENE_DECOUPLING_PLAN.md, it doesn't
anymore -- character_sheet.py + hidream_engine.py (ported from Phosphene)
do the work locally, in-process. Phosphene keeps working exactly as it
did before; we copied logic out, we didn't remove anything from there,
and we never import mlx-vlm into this app's interpreter -- generation
still happens in a subprocess in the standalone HiDream lab venv, just
invoked directly instead of via Phosphene's HTTP layer.

It also used to block the whole HTTP request for the duration of a
render (fine for 3 views at ~9-10 min each; not fine once "extended"
15-shot / full-body / multi-angle dataset jobs at 2.5+ hours entered the
picture). Every generate/reroll route now hands off to sheet_jobs.py's
background thread and returns immediately with a job id; the frontend
polls GET /api/phosphene/sheet-jobs/<job_id> for progress instead of
sitting on one long-lived connection.

The module (and blueprint variable) keep the name "phosphene" rather
than being renamed -- the URL paths under /api/phosphene/... mostly
stay put so existing bookmarks/scripts don't break, and renaming the
file/blueprint too would just be churn for no functional benefit.
"""

import os
import tempfile

from flask import Blueprint, jsonify, request, send_file

import character_sheet
import hidream_engine
import sheet_jobs
import shot_presets
from config import IMMICH_API_KEY, IMMICH_BASE_URL

phosphene_bp = Blueprint("phosphene", __name__)

_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


def _job_started_response(job, status_code=202):
    return jsonify({
        "ok": True,
        "job_id": job.job_id,
        "trigger": job.character_id,
        "shot_keys": job.shot_keys,
        "poll_url": f"/api/phosphene/sheet-jobs/{job.job_id}",
    }), status_code


def _map_job_start_error(e: Exception):
    if isinstance(e, character_sheet.CharacterSheetBusyError):
        return jsonify({"error": str(e)}), 429
    if isinstance(e, LookupError):
        return jsonify({"error": str(e)}), 404
    if isinstance(e, FileNotFoundError):
        return jsonify({"error": str(e)}), 404
    if isinstance(e, ValueError):
        return jsonify({"error": str(e)}), 400
    return jsonify({"error": str(e)}), 500


@phosphene_bp.route("/api/phosphene/status", methods=["GET"])
def phosphene_status():
    """Reports whether the local HiDream lab (venv + model + script) is
    reachable -- replaces the old "is the Phosphene process up" ping,
    since there's no separate process to ping anymore."""
    health = hidream_engine.hidream_health()
    return jsonify({"reachable": health["ready"], **health})


@phosphene_bp.route("/api/phosphene/presets", methods=["GET"])
def list_presets():
    """Feeds the settings panel's preset/style dropdowns -- shot keys
    and pose text per preset, plus the available style names, so the
    frontend doesn't hardcode any of shot_presets.py's content."""
    return jsonify({
        "presets": {
            name: [{"key": s.key, "pose_phrase": s.pose_phrase,
                    "expression": s.expression, "background": s.background}
                   for s in shots]
            for name, shots in shot_presets.PRESETS.items()
        },
        "styles": list(shot_presets.STYLE_PRESETS),
    })


@phosphene_bp.route("/api/phosphene/characters", methods=["POST"])
def create_draft_character():
    """Register a "draft" character from a photo already on this machine
    (an Immich pick, an exported video frame, any local file) - no LoRA,
    no training.

    Body: {"trigger": str, "source_image_path": str, "name"/"pronoun"/
    "subject_noun": str, all optional except trigger + source_image_path}.
    `trigger` doubles as the character's name/id (Phosphene had separate
    trigger/name fields; Ring Visualizer's exports/<name>/ convention
    only has room for one, so they're the same value here).
    """
    body = request.get_json(silent=True) or {}
    trigger = str(body.get("trigger") or "").strip()
    source_image_path = str(body.get("source_image_path") or "").strip()
    if not trigger or not source_image_path:
        return jsonify({"error": "trigger and source_image_path are required"}), 400
    try:
        bundle = character_sheet.create_draft_character(
            trigger, source_image_path,
            pronoun=body.get("pronoun", ""),
            subject_noun=body.get("subject_noun", ""))
    except character_sheet.DraftCharacterExistsError as e:
        return jsonify({"error": str(e), "trigger": trigger}), 409
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(bundle), 200


def _generation_settings_from_body(body: dict) -> dict:
    """Shared kwargs for sheet_jobs.start_job, pulled out of a JSON
    body -- used by both the one-click asset/upload routes (which accept
    these as optional extras alongside the photo) and the explicit
    .../sheet/generate route.

    `custom_prompt`, if given, is the settings panel's free-prompt field
    -- it becomes a single one-shot ShotSpec (key "custom") instead of a
    preset, so `preset`/`views` are ignored when it's set. A raw
    ShotSpec list isn't accepted over the wire at all (no need to expose
    that shape to the client) -- this is the one place free-form shot
    content enters a job."""
    custom_prompt = str(body.get("custom_prompt") or "").strip()
    shots = [shot_presets.ShotSpec("custom", prompt_override=custom_prompt)] if custom_prompt else None
    return {
        "preset": str(body.get("preset") or "default"),
        "shots": shots,
        "views": None if shots else body.get("views"),
        "wardrobe": body.get("wardrobe", ""),
        "seed": body.get("seed", -1),
        "anchor_chain": body.get("anchor_chain", True),
        "identity_lock": body.get("identity_lock", True),
        "style": str(body.get("style") or "none"),
    }


def _register_draft_and_start_job(trigger: str, tmp_path: str, settings: dict):
    """Shared tail for every "photo -> sheet" route below, regardless of
    where the photo came from: register it as a draft character (already
    existing is fine -- a re-click on the same face reuses it rather than
    failing), then kick off a background generation job."""
    try:
        character_sheet.create_draft_character(trigger, tmp_path, pronoun="",
                                                subject_noun="")
    except character_sheet.DraftCharacterExistsError:
        pass  # fine -- reuse the existing character for this trigger
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        job = sheet_jobs.start_job(trigger, **settings)
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


@phosphene_bp.route("/api/phosphene/sheet-from-asset", methods=["POST"])
def generate_sheet_from_immich_asset():
    """The one-click path for an Immich-sourced face: an asset_id in, a
    background sheet-generation job started. Downloads the asset's
    original bytes from Immich to a temp file (nothing is kept beyond
    the request - draft characters live under exports/<name>/character/
    once registered) and hands off to the shared register+start tail.

    Body: {"asset_id": str (required), "name": str (optional - also
    becomes the trigger; falls back to the raw asset_id), plus the
    optional generation settings: preset/views/wardrobe/seed/
    anchor_chain/identity_lock/style}.
    """
    body = request.get_json(silent=True) or {}
    asset_id = str(body.get("asset_id") or "").strip()
    if not asset_id:
        return jsonify({"error": "asset_id is required"}), 400
    name = str(body.get("name") or "").strip()
    trigger = name or asset_id

    try:
        r = requests_get_immich_asset(asset_id)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not reach Immich: {e}"}), 502
    if r.status_code != 200:
        return jsonify({
            "error": f"Immich returned HTTP {r.status_code} for asset "
                     f"{asset_id}"
        }), 502

    ext = _CONTENT_TYPE_EXT.get(
        (r.headers.get("Content-Type") or "").split(";")[0].strip().lower(),
        ".jpg")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix="ringviz_char_draft_", suffix=ext, delete=False) as fh:
            fh.write(r.content)
            tmp_path = fh.name
        return _register_draft_and_start_job(
            trigger, tmp_path, _generation_settings_from_body(body))
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def requests_get_immich_asset(asset_id: str):
    import requests
    return requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
        headers={"x-api-key": IMMICH_API_KEY},
        timeout=30,
    )


@phosphene_bp.route("/api/phosphene/sheet-from-upload", methods=["POST"])
def generate_sheet_from_upload():
    """The one-click path for anything that ISN'T already an Immich
    asset: a local video frame, a folder-loaded image, or literally any
    file the user picks from disk. All three land here the same way -
    as raw image bytes in a multipart upload.

    Form fields: file (required, multipart), trigger (required - unlike
    the asset route there's no natural id to fall back to for an
    arbitrary photo, so the caller must supply one), name (optional),
    plus the same optional generation-settings fields as
    sheet-from-asset, sent as individual form fields (multipart bodies
    don't carry nested JSON) -- preset/wardrobe/seed/anchor_chain/
    identity_lock/style. `views` isn't accepted here (JSON-list-shaped,
    awkward over multipart); use the JSON .../sheet/generate route for
    that.
    """
    if "file" not in request.files:
        return jsonify({"error": "no field 'file'"}), 400
    fld = request.files["file"]
    trigger = str(request.form.get("trigger") or "").strip()
    if not trigger:
        return jsonify({"error": "trigger is required"}), 400

    ext = os.path.splitext(fld.filename or "")[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        ext = _CONTENT_TYPE_EXT.get((fld.mimetype or "").lower(), ".jpg")

    form = request.form
    custom_prompt = (form.get("custom_prompt") or "").strip()
    shots = [shot_presets.ShotSpec("custom", prompt_override=custom_prompt)] if custom_prompt else None
    settings = {
        "preset": form.get("preset") or "default",
        "shots": shots,
        "views": None,
        "wardrobe": form.get("wardrobe", ""),
        "seed": form.get("seed", -1, type=int),
        "anchor_chain": form.get("anchor_chain", "true").lower() != "false",
        "identity_lock": form.get("identity_lock", "true").lower() != "false",
        "style": form.get("style") or "none",
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix="ringviz_char_draft_", suffix=ext, delete=False) as fh:
            fld.save(fh)
            tmp_path = fh.name
        return _register_draft_and_start_job(trigger, tmp_path, settings)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@phosphene_bp.route(
    "/api/phosphene/characters/<character_id>/sheet/generate", methods=["POST"]
)
def generate_character_sheet_route(character_id):
    """Start a background sheet-generation job for an already-registered
    character, with full settings -- the direct equivalent of
    Phosphene's own /characters/<id>/sheet/generate, but async (see
    module docstring)."""
    body = request.get_json(silent=True) or {}
    settings = _generation_settings_from_body(body)
    try:
        job = sheet_jobs.start_job(character_id, **settings)
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


@phosphene_bp.route(
    "/api/phosphene/characters/<character_id>/sheet/reroll", methods=["POST"]
)
def reroll_shot_route(character_id):
    """Re-render one shot of an already-generated sheet.

    Body: {"shot_key": str (required), "seed": int (optional, random if
    omitted), "prompt": str (optional full prompt override -- omit to
    reuse the shot's existing prompt)}.
    """
    body = request.get_json(silent=True) or {}
    shot_key = str(body.get("shot_key") or "").strip()
    if not shot_key:
        return jsonify({"error": "shot_key is required"}), 400
    try:
        job = sheet_jobs.start_reroll(
            character_id, shot_key,
            seed=body.get("seed"), prompt=body.get("prompt"))
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


@phosphene_bp.route("/api/phosphene/sheet-jobs/<job_id>", methods=["GET"])
def sheet_job_status_route(job_id):
    """Poll target for every job-starting route above -- overall status,
    per-shot status inferred from disk, and a tail of the HiDream
    subprocess's own log lines (see sheet_jobs.py)."""
    try:
        status = sheet_jobs.job_status(job_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(status)


@phosphene_bp.route("/api/phosphene/characters/<character_id>/sheet", methods=["GET"])
def serve_character_sheet(character_id):
    """Serves the composited sheet.png (small shot counts only -- see
    character_sheet.MAX_SHOTS_FOR_COMPOSITE; an extended-preset job has
    no composite, use the per-shot route below instead)."""
    try:
        p = character_sheet.character_sheet_png(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if p is None:
        return jsonify({"error": f"no sheet for character {character_id!r}"}), 404
    return send_file(p, mimetype="image/png")


@phosphene_bp.route("/api/phosphene/characters/<character_id>/sheet-meta", methods=["GET"])
def serve_character_sheet_meta(character_id):
    """Full sheet.json (every shot's prompt/seed/path/refs) -- lets the
    panel restore the shot grid on page load without an active job_id
    in hand (e.g. after a refresh)."""
    try:
        character_sheet._safe_id(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(character_sheet.character_sheet_meta(character_id))


@phosphene_bp.route(
    "/api/phosphene/characters/<character_id>/shots/<shot_key>", methods=["GET"]
)
def serve_shot_thumbnail(character_id, shot_key):
    """Serves the latest rendered PNG for one shot key -- what the shot
    grid's <img> tags point at. "Latest" by mtime, so this keeps working
    across re-rolls without the frontend needing to know a specific
    filename (the engine's cand_*.png names are timestamped)."""
    try:
        cid = character_sheet._safe_id(character_id)
        key = character_sheet._safe_id(shot_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    view_dir = character_sheet._character_dir(cid) / "sheet_views" / key
    if not view_dir.is_dir():
        return jsonify({"error": f"no shot {shot_key!r} for character {character_id!r}"}), 404
    candidates = sorted(view_dir.glob("cand_*.png"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return jsonify({"error": f"shot {shot_key!r} has no rendered image yet"}), 404
    return send_file(candidates[-1], mimetype="image/png")
