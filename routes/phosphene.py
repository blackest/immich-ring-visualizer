"""Integration with Phosphene's character sheet generator.

Phosphene (github.com/mrbizarro/phosphene) exposes a synchronous API,
POST /characters/<id>/sheet/generate, that takes a single reference image
already registered as a "character" in Phosphene and renders a small set
of turnaround views (front / profile_left / three_quarter by default)
using identity-preserving image editing (HiDream/Qwen-Edit), composited
into a sheet.png plus per-view PNGs and a sheet.json sidecar.

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

from flask import Blueprint, jsonify, request
import requests

from config import PHOSPHENE_BASE_URL

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