"""NG twin of routes/phosphene.py -- HTTP layer for the Generate view's
character-sheet feature.

NG-only file, renamed off "phosphene" (the original kept that name for
URL-compatibility reasons that don't apply here -- there are no
existing NG bookmarks/scripts to preserve, and the code has never had
anything to do with Phosphene). Named generateNG.py to match the
Generate view itself and to stay engine-agnostic, since more than one
generation engine may exist someday (see hidream_engineNG.py's
docstring).

Every generate/reroll route hands off to sheet_jobsNG.py's FIFO queue
and returns immediately with a job id; the frontend polls
GET /api/ng/generate/sheet-jobs/<job_id> for progress. Unlike the
original, a second concurrent request never gets a busy error -- it
queues (see sheet_jobsNG.py's module docstring).

Deliberately NOT ported in this patch: the original's
.../sheet/add-to-ring route. That route relies on
folder_analysis.run_folder_analysis's always_cache=True behavior so a
low-similarity side/profile generated shot still lands in the ring
instead of being silently dropped -- folder_analysisNG.py's
run_folder_analysis_ng doesn't support always_cache yet. Add this route
back once that lands.
"""

import json
import os
import tempfile

from flask import Blueprint, jsonify, request, send_file

import character_sheetNG as character_sheet
import hidream_engineNG
import sheet_jobsNG as sheet_jobs
import shot_presetsNG as shot_presets
from configNG import IMMICH_API_KEY, IMMICH_BASE_URL

generateNG_bp = Blueprint("generateNG", __name__)

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
        "poll_url": f"/api/ng/generate/sheet-jobs/{job.job_id}",
    }), status_code


def _map_job_start_error(e: Exception):
    if isinstance(e, LookupError):
        return jsonify({"error": str(e)}), 404
    if isinstance(e, FileNotFoundError):
        return jsonify({"error": str(e)}), 404
    if isinstance(e, ValueError):
        return jsonify({"error": str(e)}), 400
    return jsonify({"error": str(e)}), 500


@generateNG_bp.route("/api/ng/generate/status", methods=["GET"])
def generate_status_ng():
    """Reports whether the local HiDream lab (venv + model + script) is
    reachable -- the frontend gates the Generate button on this."""
    health = hidream_engineNG.hidream_health_ng()
    return jsonify({"reachable": health["ready"], **health})


@generateNG_bp.route("/api/ng/generate/presets", methods=["GET"])
def list_presets_ng():
    """Feeds the settings panel's preset/style dropdowns -- shot keys
    and pose text per preset, plus the available style names, so the
    frontend doesn't hardcode any of shot_presetsNG.py's content. Also
    the render-size defaults/bounds and the trained resolutions, so the
    panel's width/height/steps controls stay in sync with the engine."""
    return jsonify({
        "presets": {
            name: [{"key": s.key, "pose_phrase": s.pose_phrase,
                    "expression": s.expression, "background": s.background}
                   for s in shots]
            for name, shots in shot_presets.PRESETS.items()
        },
        "styles": list(shot_presets.STYLE_PRESETS),
        "render": {
            "default_width": character_sheet.DEFAULT_RENDER_W,
            "default_height": character_sheet.DEFAULT_RENDER_H,
            "default_steps": character_sheet.DEFAULT_RENDER_STEPS,
            "dim_min": character_sheet.RENDER_DIM_MIN,
            "dim_max": character_sheet.RENDER_DIM_MAX,
            "steps_min": character_sheet.RENDER_STEPS_MIN,
            "steps_max": character_sheet.RENDER_STEPS_MAX,
            "trained_resolutions": [list(wh) for wh in
                                    hidream_engineNG.HIDREAM_TRAINED_RESOLUTIONS],
        },
    })


@generateNG_bp.route("/api/ng/generate/characters", methods=["POST"])
def create_draft_character_route_ng():
    """Register a "draft" character from a photo already on this machine
    (an Immich pick, an exported video frame, any local file) - no LoRA,
    no training.

    Body: {"trigger": str, "source_image_path": str, "name"/"pronoun"/
    "subject_noun": str, all optional except trigger + source_image_path}.
    """
    body = request.get_json(silent=True) or {}
    trigger = str(body.get("trigger") or "").strip()
    source_image_path = str(body.get("source_image_path") or "").strip()
    if not trigger or not source_image_path:
        return jsonify({"error": "trigger and source_image_path are required"}), 400
    try:
        bundle = character_sheet.create_draft_character_ng(
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


def _generation_settings_from_body_ng(body: dict) -> dict:
    """Shared kwargs for sheet_jobsNG.start_job_ng, pulled out of a JSON
    body -- used by both the one-click asset/upload routes and the
    explicit .../sheet/generate route.

    `custom_prompt`, if given, becomes a single one-shot ShotSpec (key
    "custom") instead of a preset, so `preset`/`views` are ignored when
    it's set."""
    custom_prompt = str(body.get("custom_prompt") or "").strip()
    shots = [shot_presets.ShotSpec("custom", prompt_override=custom_prompt)] if custom_prompt else None
    return {
        "preset": str(body.get("preset") or "default"),
        "shots": shots,
        "views": None if shots else body.get("views"),
        "wardrobe": body.get("wardrobe", ""),
        "hair_color": str(body.get("hair_color") or "").strip(),
        "seed": body.get("seed", -1),
        "anchor_chain": body.get("anchor_chain", True),
        "identity_lock": body.get("identity_lock", True),
        "style": str(body.get("style") or "none"),
        "width": body.get("width", character_sheet.DEFAULT_RENDER_W),
        "height": body.get("height", character_sheet.DEFAULT_RENDER_H),
        "steps": body.get("steps", character_sheet.DEFAULT_RENDER_STEPS),
    }


_MAX_TRIGGER_SUFFIX_ATTEMPTS = 50


def _register_draft_and_start_job_ng(trigger: str, tmp_path: str, settings: dict):
    """Shared tail for every "photo -> sheet" route below, regardless of
    where the photo came from: register it as a draft character, then
    enqueue a background generation job.

    `trigger` colliding with an existing character used to just reuse
    whatever avatar was already on disk under that name, silently
    discarding the newly-uploaded photo -- harmless for a genuine
    re-click on the same face, but a trap for the common case of a
    generic/placeholder project name (e.g. "default") outliving the
    character it was first used for. Mirrors the frontend's own
    per-reference trigger suffixing (petra, petra-2, petra-3, ...) at
    the backend instead, which is the actual source of truth across
    page reloads and separate sessions: same photo re-added under the
    same name reuses that character; a genuinely different photo gets
    the next free "<trigger>-N" id instead of clobbering someone else's
    render queue."""
    final_trigger = trigger
    for n in range(2, _MAX_TRIGGER_SUFFIX_ATTEMPTS + 2):
        try:
            character_sheet.create_draft_character_ng(final_trigger, tmp_path,
                                                       pronoun="", subject_noun="")
            break  # fresh character created under this id
        except character_sheet.DraftCharacterExistsError:
            existing = character_sheet.character_avatar_ng(final_trigger)
            if existing is not None and _same_file_bytes_ng(existing, tmp_path):
                break  # same photo re-added under the same name -- reuse it
            final_trigger = f"{trigger}-{n}"
            continue
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({
            "error": f"could not find a free character id for {trigger!r} "
                     f"after {_MAX_TRIGGER_SUFFIX_ATTEMPTS} attempts"
        }), 500

    try:
        job = sheet_jobs.start_job_ng(final_trigger, **settings)
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


def _same_file_bytes_ng(path_a, path_b, chunk_size: int = 1 << 20) -> bool:
    """Cheap identity check for 'is this the same upload as last time' --
    size first (near-free), then a streamed byte comparison so we never
    have to load either file fully into memory."""
    import os as _os

    try:
        if _os.path.getsize(path_a) != _os.path.getsize(path_b):
            return False
    except OSError:
        return False
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        while True:
            a = fa.read(chunk_size)
            b = fb.read(chunk_size)
            if a != b:
                return False
            if not a:
                return True


@generateNG_bp.route("/api/ng/generate/sheet-from-asset", methods=["POST"])
def generate_sheet_from_immich_asset_ng():
    """The one-click path for an Immich-sourced face: an asset_id in, a
    queued sheet-generation job started. Downloads the asset's original
    bytes from Immich to a temp file (nothing is kept beyond the
    request) and hands off to the shared register+start tail.

    Body: {"asset_id": str (required), "name": str (optional - also
    becomes the trigger; falls back to the raw asset_id), plus the
    optional generation settings: preset/views/wardrobe/hair_color/seed/
    anchor_chain/identity_lock/style}.
    """
    body = request.get_json(silent=True) or {}
    asset_id = str(body.get("asset_id") or "").strip()
    if not asset_id:
        return jsonify({"error": "asset_id is required"}), 400
    name = str(body.get("name") or "").strip()
    trigger = name or asset_id

    try:
        r = _requests_get_immich_asset_ng(asset_id)
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
                prefix="ringvizng_char_draft_", suffix=ext, delete=False) as fh:
            fh.write(r.content)
            tmp_path = fh.name
        return _register_draft_and_start_job_ng(
            trigger, tmp_path, _generation_settings_from_body_ng(body))
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _requests_get_immich_asset_ng(asset_id: str):
    import requests
    return requests.get(
        f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
        headers={"x-api-key": IMMICH_API_KEY},
        timeout=30,
    )


@generateNG_bp.route("/api/ng/generate/sheet-from-upload", methods=["POST"])
def generate_sheet_from_upload_ng():
    """The one-click path for anything that ISN'T already an Immich
    asset: a local video frame, a folder-loaded image, or any file
    picked from disk. All three land here the same way - as raw image
    bytes in a multipart upload.

    Form fields: file (required, multipart), trigger (required), name
    (optional), plus the same optional generation-settings fields as
    sheet-from-asset, sent as individual form fields -- preset/views/
    wardrobe/hair_color/seed/anchor_chain/identity_lock/style. `views`,
    if present, is a JSON array of shot keys (["chest_profile_left",
    ...]) -- the Generate view sends one key at a time so each pose
    queues as its own job.
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

    views = None
    views_raw = (form.get("views") or "").strip()
    if views_raw:
        try:
            views = json.loads(views_raw)
        except ValueError:
            return jsonify({"error": "views must be a JSON array of shot-key strings"}), 400
        if not isinstance(views, list) or not all(isinstance(v, str) for v in views):
            return jsonify({"error": "views must be a JSON array of shot-key strings"}), 400

    settings = {
        "preset": form.get("preset") or "default",
        "shots": shots,
        "views": None if shots else views,
        "wardrobe": form.get("wardrobe", ""),
        "hair_color": (form.get("hair_color") or "").strip(),
        "seed": form.get("seed", -1, type=int),
        "anchor_chain": form.get("anchor_chain", "true").lower() != "false",
        "identity_lock": form.get("identity_lock", "true").lower() != "false",
        "style": form.get("style") or "none",
        "width": form.get("width", character_sheet.DEFAULT_RENDER_W, type=int),
        "height": form.get("height", character_sheet.DEFAULT_RENDER_H, type=int),
        "steps": form.get("steps", character_sheet.DEFAULT_RENDER_STEPS, type=int),
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix="ringvizng_char_draft_", suffix=ext, delete=False) as fh:
            fld.save(fh)
            tmp_path = fh.name
        return _register_draft_and_start_job_ng(trigger, tmp_path, settings)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@generateNG_bp.route(
    "/api/ng/generate/characters/<character_id>/sheet/generate", methods=["POST"]
)
def generate_character_sheet_route_ng(character_id):
    """Enqueue a sheet-generation job for an already-registered
    character, with full settings."""
    body = request.get_json(silent=True) or {}
    settings = _generation_settings_from_body_ng(body)
    try:
        job = sheet_jobs.start_job_ng(character_id, **settings)
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


@generateNG_bp.route(
    "/api/ng/generate/characters/<character_id>/sheet/reroll", methods=["POST"]
)
def reroll_shot_route_ng(character_id):
    """Re-render one shot of an already-generated sheet.

    Body: {"shot_key": str (required), "seed": int (optional, random if
    omitted), "prompt": str (optional full prompt override -- omit to
    reuse the shot's existing prompt), "width"/"height"/"steps": int
    (optional -- default to the shot's previous render size)}.
    """
    body = request.get_json(silent=True) or {}
    shot_key = str(body.get("shot_key") or "").strip()
    if not shot_key:
        return jsonify({"error": "shot_key is required"}), 400
    try:
        job = sheet_jobs.start_reroll_ng(
            character_id, shot_key,
            seed=body.get("seed"), prompt=body.get("prompt"),
            width=body.get("width"), height=body.get("height"),
            steps=body.get("steps"))
    except Exception as e:  # noqa: BLE001
        return _map_job_start_error(e)
    return _job_started_response(job)


@generateNG_bp.route("/api/ng/generate/sheet-jobs/<job_id>", methods=["GET"])
def sheet_job_status_route_ng(job_id):
    """Poll target for every job-starting route above -- overall status
    (including "queued" while waiting for the GPU to free up), per-shot
    status inferred from disk, and a tail of the HiDream subprocess's
    own log lines."""
    try:
        status = sheet_jobs.job_status_ng(job_id)
    except LookupError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(status)


@generateNG_bp.route("/api/ng/generate/characters/<character_id>/sheet", methods=["GET"])
def serve_character_sheet_ng(character_id):
    """Serves the composited sheet.png (small shot counts only -- see
    character_sheetNG.MAX_SHOTS_FOR_COMPOSITE)."""
    try:
        p = character_sheet.character_sheet_png_ng(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if p is None:
        return jsonify({"error": f"no sheet for character {character_id!r}"}), 404
    return send_file(p, mimetype="image/png")


@generateNG_bp.route("/api/ng/generate/characters/<character_id>/sheet-meta", methods=["GET"])
def serve_character_sheet_meta_ng(character_id):
    """Full sheet.json (every shot's prompt/seed/path/refs) -- lets the
    panel restore the shot grid on page load without an active job_id
    in hand (e.g. after a refresh)."""
    try:
        character_sheet._safe_id_ng(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(character_sheet.character_sheet_meta_ng(character_id))


@generateNG_bp.route(
    "/api/ng/generate/characters/<character_id>/shots/<shot_key>", methods=["GET"]
)
def serve_shot_thumbnail_ng(character_id, shot_key):
    """Serves the latest rendered PNG for one shot key -- what the shot
    grid's <img> tags point at. "Latest" by mtime, so this keeps
    working across re-rolls without the frontend needing to know a
    specific filename."""
    try:
        cid = character_sheet._safe_id_ng(character_id)
        key = character_sheet._safe_id_ng(shot_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    view_dir = character_sheet._character_dir_ng(cid) / "sheet_views" / key
    if not view_dir.is_dir():
        return jsonify({"error": f"no shot {shot_key!r} for character {character_id!r}"}), 404
    candidates = sorted(view_dir.glob("cand_*.png"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return jsonify({"error": f"shot {shot_key!r} has no rendered image yet"}), 404
    return send_file(candidates[-1], mimetype="image/png")
