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

IMMICH_API_KEY = "L4mP37A5kNWHPME0024ms2SGep7KR8xP4oAB9UNGqOM"

FRAME_STORE = tempfile.mkdtemp(prefix="ringvizng_frames_")

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exportsNG")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
