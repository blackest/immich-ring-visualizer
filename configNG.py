"""NG twin of config.py -- same local-dev config values, its own module
namespace so NG never imports from config.py, per the NG duplication rule
in APP_ARCHITECTURE_NOTES.md."""

import os
import tempfile

IMMICH_BASE_URL = "http://localhost:2283"

# Local Ollama daemon -- powers the Chat view (routes/chatNG.py). Same
# local-dev style as IMMICH_BASE_URL above: a hardcoded default, with an
# env override for the odd non-standard setup. (Deliberately NOT reading
# OLLAMA_HOST -- that's Ollama's *server bind* address and is often just
# "0.0.0.0", useless as a client URL.)
OLLAMA_BASE_URL = os.environ.get(
    "RINGVIZ_OLLAMA_URL", "http://localhost:11434"
).rstrip("/")

# Hermes agent gateway (NousResearch hermes-agent) -- powers the Rachel
# chat view (routes/rachelNG.py). An OpenAI-compatible /v1 endpoint on a
# LAN box, Bearer-authed, single agent/model. Same local-dev style as the
# values above: hardcoded defaults with env overrides.
HERMES_BASE_URL = os.environ.get(
    "RINGVIZ_HERMES_URL", "http://192.168.3.248:8642"
).rstrip("/")
HERMES_API_KEY = os.environ.get(
    "RINGVIZ_HERMES_KEY", "6tCD2B8MuMEFmIgxmgpXrPNu_sOGdaz5iEdIhwvcQTw"
)
HERMES_MODEL = os.environ.get("RINGVIZ_HERMES_MODEL", "Special Agent Rachel Hermes")
# Session headers. Hermes loads the agent's persistent long-term memory
# (Honcho) keyed off X-Hermes-Session-Key -- without a stable key every
# call gets a fresh, empty scope. X-Hermes-Session-Id keeps the
# server-side transcript continuous across page reloads. Set the *-KEY to
# whatever scope the agent owner tells you to use.
HERMES_SESSION_ID = os.environ.get("RINGVIZ_HERMES_SESSION_ID", "ringviz-rachel")
# The agent's primary long-term-memory scope, per the agent owner. Point
# this at the persona's root identity scope (not an app-specific one) so
# the view is a portal to the same memory the agent uses everywhere else.
HERMES_SESSION_KEY = os.environ.get(
    "RINGVIZ_HERMES_SESSION_KEY", "agent:main:rachel"
)

IMMICH_API_KEY = "L4mP37A5kNWHPME0024ms2SGep7KR8xP4oAB9UNGqOM"

FRAME_STORE = tempfile.mkdtemp(prefix="ringvizng_frames_")

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exportsNG")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
