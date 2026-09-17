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
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request

from configNG import (
    COMFYUI_DIR,
    SUNO_DIR,
    SUNO_PORT,
    get_comfyui_base_url,
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


@settingsNG_bp.route("/api/ng/settings/launch-comfyui", methods=["POST"])
def launch_comfyui_ng():
    """Start the local ComfyUI server (/Volumes/AI/ComfyUI, its own venv)
    if it isn't already running. Host/port come from the comfyui_base_url
    address setting rather than a hardcoded constant (unlike Suno) so this
    stays in sync if that setting is ever repointed -- only makes sense to
    call this when that URL actually names this machine, though; there's
    no check for that here.

    Deliberately doesn't open/navigate anywhere on success (contrast
    launch_suno_ng's window.open) -- ComfyUI's own graph UI isn't the
    point, the ComfyUI tab's own API calls to this server are, and
    window.open(..., "_blank") on an iPad PWA hands the external origin
    to Safari in a way that has left the installed app needing a force-
    quit to recover (John's report from the Suno launch button)."""
    base_url = get_comfyui_base_url()
    parsed = urlparse(base_url)
    port = parsed.port or 8188

    already_log = os.path.join(COMFYUI_DIR, "comfyui_launch.log")
    if _port_open("127.0.0.1", port):
        # If it's already running, this may well be an older instance
        # started before this log-file redirect existed (or started some
        # other way entirely) -- only claim a log path if one's actually
        # there to tail, rather than pointing at a file with nothing in it.
        resp = {"ok": True, "already_running": True, "url": base_url}
        if os.path.isfile(already_log):
            resp["logPath"] = already_log
        return jsonify(resp)

    if not os.path.isdir(COMFYUI_DIR):
        return jsonify({
            "ok": False,
            "error": f"ComfyUI directory not found: {COMFYUI_DIR}",
        }), 500

    venv_python = os.path.join(COMFYUI_DIR, "venv", "bin", "python3")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    # Truncated fresh on every launch -- this is "this run's log", not an
    # accumulating history, so `tail -f` always shows what the currently
    # running process is actually doing. Previously stdout/stderr went to
    # DEVNULL, so a launch's console output (custom-node errors, sampler
    # progress, anything short of ComfyUI's own web UI) was unrecoverable
    # -- John asked where to find it and there was nothing to point at.
    log_path = os.path.join(COMFYUI_DIR, "comfyui_launch.log")
    try:
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                [python_bin, "main.py", "--listen", "--port", str(port)],
                cwd=COMFYUI_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Popen dup()s the fd for the child at spawn time, so closing our
        # own handle here (the `with` block exiting) doesn't affect the
        # now-independent detached process's writes to it.
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # ComfyUI is a much heavier startup than Suno's plain server (loads
    # torch, scans custom nodes) -- give it real time before giving up.
    for _ in range(60):
        if _port_open("127.0.0.1", port):
            return jsonify({"ok": True, "already_running": False, "url": base_url, "logPath": log_path})
        time.sleep(0.5)

    return jsonify({
        "ok": False,
        "error": "Server didn't come up within 30s -- check it manually.",
    }), 500
