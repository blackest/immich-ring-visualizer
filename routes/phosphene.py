"""Integration with Phosphene's character sheet generator.

Phosphene (github.com/mrbizarro/phosphene) exposes a synchronous API,
POST /characters/<id>/sheet/generate, that takes a single reference image
already registered as a "character" in Phosphene and renders a small set
of turnaround views (front / profile_left / three_quarter by default)
using identity-preserving image editing (HiDream/Qwen-Edit), composited
into a sheet.png plus per-view PNGs and a sheet.json sidecar.

sheet/generate never touches a trained LoRA - it only reads that single
reference photo - so "already registered as a character" used to mean
"already trained," which was backwards for the actual use case: these
turnaround views are how you get enough angles to train a good LoRA in
the first place. POST /api/phosphene/characters registers a "draft" character
from a photo already on this machine (an Immich pick, a video frame,
any local file) with no LoRA required, so a face can go straight from
here to a sheet without a manual detour through Phosphene's own UI.

Phosphene has no UI for this feature yet (it's API-first, per its own
code comments as of this writing) - this blueprint is Ring Visualizer's
client for it, not a reimplementation. We never touch Phosphene's code
or generation logic; we just call its already-running local server.

Because the call is synchronous and can legitimately take a while (three
sequential renders, each depending on the previous one for identity
consistency - see Phosphene's own comments on why views are chained),
the Flask route here mirrors that: it blocks for the duration of the
request rather than pretending to poll a job. The frontend is expected
to show a real "generating..." state and use a generous timeout.
"""

import os
import tempfile

from flask import Blueprint, jsonify, request
import requests

from config import IMMICH_API_KEY, IMMICH_BASE_URL, PHOSPHENE_BASE_URL

phosphene_bp = Blueprint("phosphene", __name__)

_REQUEST_TIMEOUT_SECONDS = 600


@phosphene_bp.route("/api/phosphene/status", methods=["GET"])
def phosphene_status():
    try:
        resp = requests.get(PHOSPHENE_BASE_URL, timeout=3)
        reachable = resp.status_code < 500
    except requests.exceptions.RequestException:
        reachable = False
    return jsonify({"reachable": reachable, "base_url": PHOSPHENE_BASE_URL})


@phosphene_bp.route("/api/phosphene/characters", methods=["POST"])
def create_draft_character():
    """Register a "draft" character from a photo already on this machine
    (an Immich pick, an exported video frame, any local file) - no LoRA,
    no training. Phosphene's sheet/generate only ever needed the photo,
    not a trained LoRA, so this is what lets Ring Visualizer hand it a
    face directly instead of requiring a character to already exist.

    Body: {"trigger": str, "source_image_path": str, "name"/"pronoun"/
    "subject_noun": str, all optional except trigger + source_image_path}.
    source_image_path is a path on disk, not an upload - Phosphene reads
    it directly since both services run on the same box.
    """
    body = request.get_json(silent=True) or {}

    try:
        resp = requests.post(
            f"{PHOSPHENE_BASE_URL}/characters",
            json=body,
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Could not reach Phosphene at " + PHOSPHENE_BASE_URL
                     + " - is it running?"
        }), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Phosphene did not respond within 30s"}), 504

    try:
        payload = resp.json()
    except ValueError:
        return jsonify({
            "error": f"Phosphene returned a non-JSON response (HTTP {resp.status_code})"
        }), 502

    return jsonify(payload), resp.status_code


_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png", "image/webp": ".webp",
}


@phosphene_bp.route("/api/phosphene/sheet-from-asset", methods=["POST"])
def generate_sheet_from_immich_asset():
    """The one-click path: an Immich asset_id in, a rendered character
    sheet out. Downloads the asset's original bytes from Immich to a
    temp file (nothing is kept - draft characters live in Phosphene's
    own mlx_models/characters/ once registered), registers it as a
    draft character with Phosphene (or reuses one that already exists
    for this trigger - a re-click on the same face isn't an error),
    then calls sheet/generate and hands back a URL the browser can
    load directly from Phosphene's own server.

    Body: {"asset_id": str (required), "name": str (optional - also
    becomes the trigger; falls back to the raw asset_id, which is a
    valid trigger as-is since Phosphene's id pattern allows hyphens)}.
    """
    body = request.get_json(silent=True) or {}
    asset_id = str(body.get("asset_id") or "").strip()
    if not asset_id:
        return jsonify({"error": "asset_id is required"}), 400
    name = str(body.get("name") or "").strip()
    trigger = name or asset_id

    try:
        r = requests.get(
            f"{IMMICH_BASE_URL}/api/assets/{asset_id}/original",
            headers={"x-api-key": IMMICH_API_KEY},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
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
                prefix="phosphene_draft_", suffix=ext, delete=False) as fh:
            fh.write(r.content)
            tmp_path = fh.name

        try:
            create_resp = requests.post(
                f"{PHOSPHENE_BASE_URL}/characters",
                json={"trigger": trigger, "source_image_path": tmp_path,
                      "name": name or trigger},
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": "Could not reach Phosphene at " + PHOSPHENE_BASE_URL
                         + " - is it running?"
            }), 502
        except requests.exceptions.Timeout:
            return jsonify({"error": "Phosphene did not respond within 30s"}), 504
        # 409 = this trigger already exists (draft or trained) - fine,
        # a re-click on the same face should reuse it, not fail.
        if create_resp.status_code not in (200, 409):
            try:
                return jsonify(create_resp.json()), create_resp.status_code
            except ValueError:
                return jsonify({
                    "error": f"Phosphene returned HTTP {create_resp.status_code} "
                             "registering the character"
                }), 502

        try:
            sheet_resp = requests.post(
                f"{PHOSPHENE_BASE_URL}/characters/{trigger}/sheet/generate",
                json={}, timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": "Could not reach Phosphene at " + PHOSPHENE_BASE_URL
                         + " - is it running?"
            }), 502
        except requests.exceptions.Timeout:
            return jsonify({
                "error": "Phosphene did not respond within "
                         f"{_REQUEST_TIMEOUT_SECONDS}s"
            }), 504
        try:
            sheet_payload = sheet_resp.json()
        except ValueError:
            return jsonify({
                "error": f"Phosphene returned a non-JSON response "
                         f"(HTTP {sheet_resp.status_code})"
            }), 502
        if sheet_resp.status_code != 200:
            return jsonify(sheet_payload), sheet_resp.status_code

        return jsonify({
            "ok": True,
            "trigger": trigger,
            "sheet_url": f"{PHOSPHENE_BASE_URL}/characters/{trigger}/sheet",
            "result": sheet_payload,
        })
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@phosphene_bp.route(
    "/api/phosphene/characters/<character_id>/sheet/generate", methods=["POST"]
)
def generate_character_sheet(character_id):
    body = request.get_json(silent=True) or {}

    try:
        resp = requests.post(
            f"{PHOSPHENE_BASE_URL}/characters/{character_id}/sheet/generate",
            json=body,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Could not reach Phosphene at " + PHOSPHENE_BASE_URL
                     + " - is it running?"
        }), 502
    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Phosphene did not respond within "
                     f"{_REQUEST_TIMEOUT_SECONDS}s"
        }), 504

    try:
        payload = resp.json()
    except ValueError:
        return jsonify({
            "error": f"Phosphene returned a non-JSON response (HTTP {resp.status_code})"
        }), 502

    return jsonify(payload), resp.status_code