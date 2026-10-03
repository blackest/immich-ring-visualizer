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
import urllib.error
import urllib.request
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request

from configNG import (
    COMFYUI_DIR,
    SUNO_DIR,
    SUNO_PORT,
    get_comfyui_base_url,
    get_ng_address_settings,
    get_tailscale_hostname,
    get_tailscale_state,
    save_ng_address_settings,
    tailscale_bring_up,
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


@settingsNG_bp.route("/api/ng/settings/tailscale", methods=["GET"])
def tailscale_status_ng():
    return jsonify(get_tailscale_state())


@settingsNG_bp.route("/api/ng/settings/tailscale/up", methods=["POST"])
def tailscale_up_ng():
    """Launch/connect Tailscale on this machine (see configNG.tailscale_bring_up)."""
    return jsonify(tailscale_bring_up())


def _probe_url(url, timeout=2.0):
    """Any HTTP response -- even a 404 -- means the address is alive and
    reachable; only a connection failure (dead hostname, refused port,
    timeout) counts as unreachable. This is what actually catches a
    renamed/stale Tailscale hostname: DNS resolution itself fails, which
    surfaces here as an error rather than a silent hang somewhere a
    feature happens to use the address."""
    t0 = time.time()
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None, round((time.time() - t0) * 1000)
    except urllib.error.HTTPError:
        return True, None, round((time.time() - t0) * 1000)
    except Exception as e:
        return False, str(e), round((time.time() - t0) * 1000)


def _venv_can_import(python_path, modules):
    """Runs `python -c "import a, b, c"` in the given interpreter and
    reports ok/error -- catches a venv that exists but is missing a
    package (e.g. yue2-mlx's optional "transcription" extra never being
    synced), which a bare path-exists check on the interpreter itself
    can't see. This is exactly the class of failure that took an entire
    debugging session to trace by hand before this existed."""
    if not python_path:
        return False, "interpreter not found"
    try:
        result = subprocess.run(
            [python_path, "-c", "import " + ", ".join(modules)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, None
        return False, (result.stderr or "").strip().splitlines()[-1] if result.stderr else "import failed"
    except Exception as e:
        return False, str(e)


@settingsNG_bp.route("/api/ng/settings/models", methods=["GET"])
def get_models_ng():
    """Resolved repo/weights paths for LTX, H3 and Music, plus a
    found/not-found probe per component -- reuses each engine's own
    already-existing `_resolve_*_ng` validators (which already check
    real shape, not just existence: right files present, executable
    bit, expected sub-files) rather than duplicating that logic here.
    Read-only -- no override wiring yet, this only makes a broken/moved
    install visible in Settings instead of as a mid-job subprocess
    traceback."""
    import h3_engineNG as H3
    import ltx_engineNG as LTX
    import music_engineNG as MUSIC

    ltx_config = LTX.LtxConfig()
    h3_config = H3.H3Config()
    music_config = MUSIC.MusicConfig()

    def component(label, path, ok):
        return {"label": label, "path": path, "ok": bool(ok)}

    engines = [
        {
            "key": "ltx",
            "label": "LTX-2.5 (Animate)",
            "repo_dir": str(LTX.LTX_REPO_DIR),
            "repo_found": LTX.LTX_REPO_DIR.exists(),
            "components": [
                component("Binary", str(LTX.LTX_DEFAULT_BIN),
                           LTX._resolve_ltx_binary_ng(ltx_config)),
                component("Python", str(LTX.LTX_DEFAULT_PYTHON),
                           LTX._resolve_ltx_python_ng()),
                component("Model weights", str(LTX.LTX_DEFAULT_MODEL),
                           LTX._resolve_ltx_model_ng(ltx_config)),
                component("Gemma (generation)", str(LTX.LTX_DEFAULT_GEMMA),
                           LTX._resolve_ltx_gemma_ng(ltx_config)),
                component("Gemma (enhance)", str(LTX.LTX_DEFAULT_ENHANCE_GEMMA),
                           LTX._resolve_ltx_enhance_gemma_ng(ltx_config)),
            ],
        },
        {
            "key": "h3",
            "label": "MiniMax-H3",
            "repo_dir": str(H3.H3_REPO_DIR),
            "repo_found": H3.H3_REPO_DIR.exists(),
            "components": [
                component("Python", str(H3.H3_DEFAULT_PYTHON),
                           H3._resolve_h3_python_ng(h3_config)),
                component("Runner script", str(H3.H3_RUNNER),
                           H3._resolve_h3_runner_ng(h3_config)),
                component("Compact pack (VAE/text)", str(H3.H3_COMPACT_ROOT),
                           H3._resolve_h3_compact_root_ng(h3_config)),
                component("Text encoder config", str(H3.H3_TEXT_CONFIG),
                           H3._resolve_h3_text_config_ng(h3_config)),
                component(f"DiT weights ({h3_config.model})",
                           str(H3.H3_DIT_Q8_DIR if h3_config.model == "h3q8" else H3.H3_DIT_BF16),
                           H3._resolve_h3_dit_ng(h3_config)),
            ],
        },
        {
            "key": "music",
            "label": "YuE2 (Music)",
            "repo_dir": str(MUSIC.MUSIC_REPO_DIR),
            "repo_found": MUSIC.MUSIC_REPO_DIR.exists(),
            "components": [
                component("Binary", str(MUSIC.MUSIC_DEFAULT_BIN),
                           MUSIC._resolve_music_binary_ng(music_config)),
                component("VAE weights", str(MUSIC.MUSIC_VAE_DIR),
                           MUSIC._resolve_music_vae_dir_ng(music_config)),
                component(f"Generator weights ({music_config.precision})", str(MUSIC.MUSIC_MODEL_DIR),
                           MUSIC._resolve_music_model_dir_ng(music_config)),
            ],
        },
    ]

    # Transcription-extra deps (mido/mir-eval/pretty-midi/scipy) -- the
    # exact class of failure that started this whole feature: the venv
    # exists and the binary runs, but `lyra cover`'s transcription pass
    # dies mid-job because `uv sync` was never run with --extra
    # transcription. Path-exists checks can't see this; only an actual
    # import in that venv's interpreter can.
    music_venv_python = str(MUSIC.MUSIC_REPO_DIR / ".venv" / "bin" / "python")
    ok, err = _venv_can_import(
        music_venv_python if os.path.isfile(music_venv_python) else None,
        ["mir_eval.chord", "pretty_midi", "mido"],
    )
    engines[2]["components"].append({
        "label": "Transcription deps (mido/mir-eval/pretty-midi)",
        "path": music_venv_python,
        "ok": ok,
        "error": None if ok else err,
    })

    return jsonify({"engines": engines})


@settingsNG_bp.route("/api/ng/settings/addresses/check", methods=["POST"])
def check_addresses_ng():
    """Probe every URL-shaped address setting (those with "probe": True in
    NG_ADDRESS_SETTINGS) and report reachable/unreachable, right where
    the address is configured -- instead of as a mysterious failure in
    whatever feature uses it later. Body may include {key: value} for
    fields the settings modal has edited but not saved yet, so "Test"
    checks what's about to be saved rather than only what's on disk."""
    body = request.get_json(silent=True) or {}
    results = {}
    for setting in get_ng_address_settings():
        if not setting.get("probe"):
            continue
        key = setting["key"]
        value = (body.get(key) or "").strip() or setting["value"]
        ok, error, ms = _probe_url(value)
        results[key] = {"ok": ok, "error": error, "ms": ms}
    return jsonify({"results": results})


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
