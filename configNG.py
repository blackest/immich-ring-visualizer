"""NG twin of config.py -- same local-dev config values, its own module
namespace so NG never imports from config.py, per the NG duplication rule
in APP_ARCHITECTURE_NOTES.md."""

import json
import os
import tempfile

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

# Music view (routes/musicNG.py, music_jobsNG.py) -- YuE2 style+lyrics-
# to-song renders, a peer of Animate/H3's queues above with its own ring
# buffer (see music_engineNG.py for why this is a separate venv/engine).
MUSICGEN_DIR = os.path.join(EXPORT_DIR, "_musicgen")

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
        "default": "http://macstudio-2.tail74ab30.ts.net:11434",
        "label": "Ollama URL",
        "description": "Local Ollama daemon powering the Chat view.",
    },
    {
        "key": "hermes_base_url",
        "env": "RINGVIZ_HERMES_URL",
        "default": "http://m1mini-4.tail74ab30.ts.net:8642",
        "label": "Hermes gateway URL",
        "description": "Hermes agent gateway (OpenAI-compatible /v1) powering the Rachel view.",
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
        "default": "macstudio-2.tail74ab30.ts.net",
        "label": "Tailscale hostname",
        "description": "This machine's tailnet name, used to build the Suno Vault URL.",
    },
    {
        "key": "comfyui_base_url",
        "env": "RINGVIZ_COMFYUI_URL",
        "default": "http://192.168.3.54:8182",
        "label": "ComfyUI URL",
        "description": "ComfyUI server powering the ComfyUI view (extract a workflow from a PNG, edit its parameters, run it).",
    },
]


def _load_ng_settings():
    try:
        with open(NG_SETTINGS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _ng_setting(key):
    spec = next(s for s in NG_ADDRESS_SETTINGS if s["key"] == key)
    env_val = os.environ.get(spec["env"])
    if env_val:
        return env_val
    saved = _load_ng_settings().get(key)
    if saved:
        return saved
    return spec["default"]


def get_ng_address_settings():
    """Effective value + source ("env"/"saved"/"default") for each editable
    address, for the settings UI."""
    saved = _load_ng_settings()
    out = []
    for spec in NG_ADDRESS_SETTINGS:
        env_val = os.environ.get(spec["env"])
        if env_val:
            source, value = "env", env_val
        elif saved.get(spec["key"]):
            source, value = "saved", saved[spec["key"]]
        else:
            source, value = "default", spec["default"]
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
