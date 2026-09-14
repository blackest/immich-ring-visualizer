"""immichRingNG -- settings-cog actions (venv maintenance, local launchers).

NG-only file, additive. Curated single endpoints rather than an open-
ended "run any command" one -- per the design note, this is meant to
grow with a reviewed list of specific actions, not become a shell.
"""

import os
import socket
import subprocess
import sys
import time

from flask import Blueprint, jsonify, request

from configNG import (
    SUNO_DIR,
    SUNO_PORT,
    get_ng_address_settings,
    get_tailscale_hostname,
    save_ng_address_settings,
)

settingsNG_bp = Blueprint("settingsNG", __name__)


@settingsNG_bp.route("/api/ng/settings/addresses", methods=["GET"])
def get_addresses_ng():
    """List every machine-specific address setting (Ollama/Hermes/Tailscale)
    with its current effective value and where that value came from --
    drives the "Addresses" section of the settings modal. Secret values
    (Hermes API key) are still sent as plain text here since this is a
    same-origin, single-user local app; the UI masks them on display."""
    return jsonify({"settings": get_ng_address_settings()})


@settingsNG_bp.route("/api/ng/settings/addresses", methods=["POST"])
def save_addresses_ng():
    """Persist edited address settings to configNG.NG_SETTINGS_FILE. Body:
    {key: value, ...} -- a blank value clears that key's override, falling
    back to its env var / hardcoded default. Takes effect immediately, no
    restart needed (configNG reads the file fresh on every call)."""
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"ok": False, "error": "body must be a JSON object"}), 400
    save_ng_address_settings(body)
    return jsonify({"ok": True, "settings": get_ng_address_settings()})


@settingsNG_bp.route("/api/ng/settings/update-ytdlp", methods=["POST"])
def update_ytdlp_ng():
    """Run `pip install --upgrade yt-dlp` in the current venv (the same
    interpreter running this Flask process) and report back pass/fail
    plus the tail of pip's output for visibility."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": False,
            "error": "Timed out after 120s waiting for pip.",
        }), 504
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-15:])

    if result.returncode != 0:
        return jsonify({
            "ok": False,
            "error": f"pip exited with code {result.returncode}",
            "output": tail,
        }), 500

    return jsonify({"ok": True, "output": tail})


def _port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@settingsNG_bp.route("/api/ng/settings/launch-suno", methods=["POST"])
def launch_suno_ng():
    """Start john's Suno Vault server (~/Music/mysunodb, its own repo and
    venv) if it isn't already running, and hand back its Tailscale URL.
    Started detached (start_new_session) so it keeps running after this
    request -- and after this Flask process -- exits."""
    url = f"http://{get_tailscale_hostname()}:{SUNO_PORT}/"

    if _port_open("127.0.0.1", SUNO_PORT):
        return jsonify({"ok": True, "already_running": True, "url": url})

    if not os.path.isdir(SUNO_DIR):
        return jsonify({
            "ok": False,
            "error": f"Suno Vault directory not found: {SUNO_DIR}",
        }), 500

    venv_python = os.path.join(SUNO_DIR, ".venv", "bin", "python3")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    try:
        subprocess.Popen(
            [python_bin, "server.py"],
            cwd=SUNO_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    for _ in range(20):
        if _port_open("127.0.0.1", SUNO_PORT):
            return jsonify({"ok": True, "already_running": False, "url": url})
        time.sleep(0.25)

    return jsonify({
        "ok": False,
        "error": "Server didn't come up within 5s -- check it manually.",
    }), 500
