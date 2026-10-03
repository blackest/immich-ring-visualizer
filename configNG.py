"""NG twin of config.py -- same local-dev config values, its own module
namespace so NG never imports from config.py, per the NG duplication rule
in APP_ARCHITECTURE_NOTES.md."""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

IMMICH_BASE_URL = "http://localhost:2283"

IMMICH_API_KEY = "L4mP37A5kNWHPME0024ms2SGep7KR8xP4oAB9UNGqOM"

# Suno Vault -- john's own local music-library server, lives outside this
# repo at ~/Music/mysunodb (its own git repo, its own venv). The settings
# cog can start it as a subprocess; it's reachable over Tailscale with no
# router/gateway config since it's just this machine's own tailnet name.
SUNO_DIR = os.path.expanduser("~/Music/mysunodb")
SUNO_PORT = 8001

# ComfyUI -- lives outside this repo on the external AI volume, its own
# venv. comfyui_base_url (below) already carries its host:port, since that
# server doesn't always run on this same machine; when it does (the
# settings cog's Launch button assumes so), this is where to find it.
COMFYUI_DIR = "/Volumes/AI/ComfyUI"

FRAME_STORE = tempfile.mkdtemp(prefix="ringvizng_frames_")

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exportsNG")

# Animate view (routes/videogenNG.py, video_jobsNG.py) -- LTX image-to-
# video renders. Flat, persistent (not a tempdir): these are real
# outputs the user reviews/downloads, not scratch, but they're keyed by
# job id rather than character id since a render's source reference
# isn't necessarily a saved character's avatar.
VIDEOGEN_DIR = os.path.join(EXPORT_DIR, "_videogen")

# Durable Animate job history (job_logsNG.py) -- separate from VIDEOGEN_DIR
# above, which is a ring buffer that deletes a job's files once evicted.
# This is the permanent, never-pruned record: day folders, rolled up into
# <year>/<month>/ once a month ends. See job_logsNG.py for the layout.
JOB_LOG_DIR = os.path.join(EXPORT_DIR, "_job_logsNG")

# H3 view (routes/h3NG.py, h3_jobsNG.py) -- MiniMax-H3 prompt/image-to-
# video renders, a peer of Animate's LTX queue above with its own ring
# buffer (see h3_engineNG.py for why H3/H3Q8 are one engine, not two).
H3GEN_DIR = os.path.join(EXPORT_DIR, "_h3gen")

# Durable H3 job history (h3_job_logsNG.py) -- twin of JOB_LOG_DIR above
# for the H3 ring buffer: permanent record of prompt/seed/settings, the
# finished mp4 and the full log, which survives both the 20-job eviction
# and a server restart.
H3_JOB_LOG_DIR = os.path.join(EXPORT_DIR, "_h3_job_logsNG")

# Music view (routes/musicNG.py, music_jobsNG.py) -- YuE2 style+lyrics-
# to-song renders, a peer of Animate/H3's queues above with its own ring
# buffer (see music_engineNG.py for why this is a separate venv/engine).
MUSICGEN_DIR = os.path.join(EXPORT_DIR, "_musicgen")

# Durable Music job history (music_job_logsNG.py) -- twin of JOB_LOG_DIR
# above for the music ring buffer: permanent record of style/lyrics/seed
# so a "recipe" survives past the 20-job ring-buffer eviction that would
# otherwise delete it along with the rest of MUSICGEN_DIR's job_dir.
MUSIC_JOB_LOG_DIR = os.path.join(EXPORT_DIR, "_music_job_logsNG")

# Hd-Multi view (routes/hdmultiNG.py) -- one-off HiDream edit/multi-ref
# generations (1-3 reference images + a prompt -> one result image).
# Same "flat, persistent, keyed by job id" shape as VIDEOGEN_DIR above,
# since a job's ref images aren't tied to any saved character either.
HDMULTI_DIR = os.path.join(EXPORT_DIR, "_hdmulti")

# ComfyUI view (routes/comfyNG.py) -- a small curated library of saved
# workflows, each just a successful result PNG (ComfyUI already embeds
# the exact workflow that made it in the PNG's own metadata, so the file
# *is* the save format) plus a display name in index.json. Explicit
# "Save this workflow" action only, not every run -- otherwise this fills
# with 50 near-duplicates of the same workflow (John's point). Exists so
# picking a workflow is a list click instead of a local file picker,
# which doesn't work from the iPad (no access to the Mac's disk).
COMFY_WORKFLOWS_DIR = os.path.join(EXPORT_DIR, "_comfy_workflows")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Machine-specific address settings -- Ollama/Hermes/Tailscale endpoints are
# unique to whatever box is running this app (LAN hostnames, tailnet names,
# API keys). Rather than only a hardcoded default + env var, these are also
# editable from the settings cog (Settings > Addresses -> routes/settingsNG.py)
# and persisted to NG_SETTINGS_FILE, a gitignored JSON file that never leaves
# this machine's checkout. Precedence per value: env var (if set) > saved
# settings file > hardcoded default. Read fresh on every call (not baked
# into a module-level constant) so a save from the settings UI takes effect
# immediately, no restart needed.
NG_SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ng_settings.json"
)

# Each entry: settings-file/API key, env var name, hardcoded default, label
# + description for the settings UI, and whether it's a secret (masked in
# the UI). This one list drives both value resolution below and the
# GET/POST settings API in routes/settingsNG.py -- add a new address here
# and it shows up in the UI for free.
NG_ADDRESS_SETTINGS = [
    {
        "key": "ollama_base_url",
        "env": "RINGVIZ_OLLAMA_URL",
        "default": "http://127.0.0.1:11434",
        "label": "Ollama URL",
        "description": "Local Ollama daemon powering the Chat view (same machine, so no network hop).",
        "probe": True,
    },
    {
        "key": "hermes_base_url",
        "env": "RINGVIZ_HERMES_URL",
        "default": "http://m1mini.local:8642",
        "fallback": "http://192.168.3.248:8642",
        "label": "Hermes gateway URL",
        "description": "Hermes agent gateway (OpenAI-compatible /v1) powering the Rachel view. LAN (mDNS) address -- Tailscale isn't needed for this.",
        "probe": True,
    },
    {
        "key": "hermes_api_key",
        "env": "RINGVIZ_HERMES_KEY",
        "default": "6tCD2B8MuMEFmIgxmgpXrPNu_sOGdaz5iEdIhwvcQTw",
        "label": "Hermes API key",
        "description": "Bearer token sent to the Hermes gateway.",
        "secret": True,
    },
    {
        "key": "hermes_model",
        "env": "RINGVIZ_HERMES_MODEL",
        "default": "Special Agent Rachel Hermes",
        "label": "Hermes model",
        "description": "Fixed model/agent name sent to Hermes (Rachel has no model picker).",
    },
    {
        "key": "hermes_session_id",
        "env": "RINGVIZ_HERMES_SESSION_ID",
        "default": "ringviz-rachel",
        "label": "Hermes session id",
        "description": "Keeps the server-side transcript continuous across page reloads.",
    },
    {
        "key": "hermes_session_key",
        "env": "RINGVIZ_HERMES_SESSION_KEY",
        "default": "agent:main:rachel",
        "label": "Hermes session key",
        "description": "The agent's long-term-memory scope (Honcho). Set to whatever scope the agent owner tells you to use.",
    },
    {
        "key": "tailscale_hostname",
        "env": "RINGVIZ_TAILSCALE_HOST",
        "default": "macstudio-2-1.tail74ab30.ts.net",
        "label": "Tailscale hostname",
        "description": "This machine's tailnet name, used to build the Suno Vault URL.",
        "tailscale_device": "MacStudio (2)",
    },
    {
        "key": "comfyui_base_url",
        "env": "RINGVIZ_COMFYUI_URL",
        "default": "http://192.168.3.54:8182",
        "label": "ComfyUI URL",
        "description": "ComfyUI server powering the ComfyUI view (extract a workflow from a PNG, edit its parameters, run it).",
        "probe": True,
    },
]


def _load_ng_settings():
    try:
        with open(NG_SETTINGS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_TAILSCALE_BIN_CANDIDATES = (
    "tailscale",
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
)

# `tailscale status --json` per-device HostName -> DNSName, cached briefly
# in-process. HostName ("MacStudio (2)") is the stable OS-level machine
# name; DNSName ("macstudio-2-1.tail74ab30.ts.net.") is what Tailscale
# actually hands out on the tailnet, and it gets a numeric suffix appended
# whenever a device reconnects and collides with an existing name --
# which is exactly what silently broke the Ollama URL default once. A TTL
# cache avoids a subprocess call on every settings read while still
# picking up a rename within one page load.
_TAILSCALE_CACHE_TTL = 20
_tailscale_cache = {"ts": 0.0, "devices": None}


def _tailscale_bin():
    for candidate in _TAILSCALE_BIN_CANDIDATES:
        if os.sep in candidate:
            if os.path.exists(candidate):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    return None


def _tailscale_devices():
    """{HostName: current DNSName} for this machine and every peer.
    Returns {} on any failure (tailscaled not running, CLI missing,
    timeout) rather than raising -- every caller already has a hardcoded
    default to fall back to, so a live-lookup failure should be silent,
    not a request-breaking exception."""
    now = time.time()
    cached = _tailscale_cache["devices"]
    if cached is not None and now - _tailscale_cache["ts"] < _TAILSCALE_CACHE_TTL:
        return cached
    devices = {}
    try:
        bin_path = _tailscale_bin()
        if bin_path:
            out = subprocess.run(
                [bin_path, "status", "--json"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                data = json.loads(out.stdout)
                nodes = [data.get("Self") or {}, *(data.get("Peer") or {}).values()]
                for node in nodes:
                    host_name = node.get("HostName")
                    dns_name = (node.get("DNSName") or "").rstrip(".")
                    if host_name and dns_name:
                        devices[host_name] = dns_name
    except Exception:
        devices = {}
    _tailscale_cache["ts"] = now
    _tailscale_cache["devices"] = devices
    return devices


def get_tailscale_state():
    """Live (uncached) Tailscale daemon state for the settings UI.
    Returns {"state": ..., "detail": ...}. state is Tailscale's own
    BackendState ("Running", "Stopped", "NeedsLogin", "Starting", ...) or
    "not_installed" / "unreachable" (CLI present but tailscaled/app isn't
    answering -- the usual "app not launched" case)."""
    bin_path = _tailscale_bin()
    if not bin_path:
        return {"state": "not_installed", "detail": "Tailscale CLI not found."}
    try:
        out = subprocess.run(
            [bin_path, "status", "--json"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception as e:
        return {"state": "unreachable", "detail": str(e)}
    try:
        data = json.loads(out.stdout)
    except ValueError:
        msg = (out.stderr or out.stdout or "").strip()
        return {"state": "unreachable", "detail": msg or "tailscaled not responding."}
    return {"state": data.get("BackendState") or "unknown", "detail": data.get("AuthURL") or ""}


def tailscale_bring_up():
    """Try to get Tailscale connected. If the daemon answers, `tailscale
    up`; if it doesn't (app not running), launch the macOS app first and
    wait briefly for it. Returns the post-attempt get_tailscale_state()
    plus an "ok" flag. NeedsLogin can't be fixed unattended -- the
    returned detail carries the login URL for the user to open."""
    bin_path = _tailscale_bin()
    if not bin_path:
        return {"ok": False, **get_tailscale_state()}
    state = get_tailscale_state()
    if state["state"] == "unreachable":
        subprocess.run(["open", "-a", "Tailscale"], capture_output=True, timeout=10)
        for _ in range(10):
            time.sleep(1)
            state = get_tailscale_state()
            if state["state"] != "unreachable":
                break
    if state["state"] in ("Stopped", "Starting"):
        try:
            subprocess.run([bin_path, "up"], capture_output=True, text=True, timeout=15)
        except Exception:
            pass
        for _ in range(5):
            state = get_tailscale_state()
            if state["state"] == "Running":
                break
            time.sleep(1)
    _tailscale_cache["devices"] = None  # re-resolve device names now
    return {"ok": state["state"] == "Running", **state}


def _resolve_tailscale_value(spec):
    """Swap the hardcoded default's host for that device's CURRENT tailnet
    DNS name (same scheme/port/path as the default) -- None if the spec
    doesn't name a device, or that device isn't in the live status."""
    device = spec.get("tailscale_device")
    if not device:
        return None
    dns_name = _tailscale_devices().get(device)
    if not dns_name:
        return None
    default = spec["default"]
    if "://" not in default:
        return dns_name
    parts = urlsplit(default)
    netloc = f"{dns_name}:{parts.port}" if parts.port else dns_name
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


_FALLBACK_TTL = 20
_fallback_cache = {}  # url -> (timestamp, chosen url)


def _with_fallback(spec, url):
    """If the spec has a "fallback" and `url` (the built-in default) can't
    be reached by a quick TCP connect -- e.g. mDNS .local not resolving --
    use the fallback instead. Cached briefly so settings reads stay cheap.
    Only applied to the default; a saved/env value is the user's explicit
    choice and is never swapped."""
    fallback = spec.get("fallback")
    if not fallback:
        return url
    now = time.time()
    hit = _fallback_cache.get(url)
    if hit and now - hit[0] < _FALLBACK_TTL:
        return hit[1]
    parts = urlsplit(url)
    try:
        with socket.create_connection((parts.hostname, parts.port or 80), timeout=1):
            chosen = url
    except OSError:
        chosen = fallback
    _fallback_cache[url] = (now, chosen)
    return chosen


def _ng_setting(key):
    spec = next(s for s in NG_ADDRESS_SETTINGS if s["key"] == key)
    env_val = os.environ.get(spec["env"])
    if env_val:
        return env_val
    saved = _load_ng_settings().get(key)
    if saved:
        return saved
    return _resolve_tailscale_value(spec) or _with_fallback(spec, spec["default"])


def get_ng_address_settings():
    """Effective value + source ("env"/"saved"/"tailscale"/"default") for
    each editable address, for the settings UI."""
    saved = _load_ng_settings()
    out = []
    for spec in NG_ADDRESS_SETTINGS:
        env_val = os.environ.get(spec["env"])
        live = None if env_val or saved.get(spec["key"]) else _resolve_tailscale_value(spec)
        if env_val:
            source, value = "env", env_val
        elif saved.get(spec["key"]):
            source, value = "saved", saved[spec["key"]]
        elif live:
            source, value = "tailscale", live
        else:
            source, value = "default", _with_fallback(spec, spec["default"])
        out.append({**spec, "value": value, "source": source})
    return out


def save_ng_address_settings(values):
    """values: {key: str}. Unknown keys are ignored; a blank value clears
    the saved override (falls back to env/default). Returns the saved dict."""
    known_keys = {spec["key"] for spec in NG_ADDRESS_SETTINGS}
    current = _load_ng_settings()
    for key, value in values.items():
        if key not in known_keys:
            continue
        value = (value or "").strip()
        if value:
            current[key] = value
        else:
            current.pop(key, None)
    with open(NG_SETTINGS_FILE, "w") as f:
        json.dump(current, f, indent=2, sort_keys=True)
    return current


def get_ollama_base_url():
    return _ng_setting("ollama_base_url").rstrip("/")


def get_hermes_base_url():
    return _ng_setting("hermes_base_url").rstrip("/")


def get_hermes_api_key():
    return _ng_setting("hermes_api_key")


def get_hermes_model():
    return _ng_setting("hermes_model")


def get_hermes_session_id():
    return _ng_setting("hermes_session_id")


def get_hermes_session_key():
    return _ng_setting("hermes_session_key")


def get_tailscale_hostname():
    return _ng_setting("tailscale_hostname")


def get_comfyui_base_url():
    return _ng_setting("comfyui_base_url").rstrip("/")
