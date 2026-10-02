"""Free resident chat/vision models before a heavy render takes the GPU.

H3 wants ~50 GB wired on a 96 GB machine; an Ollama model or the standalone
Gemma server left resident pushes it toward swap. One entry point,
`offload_resident_models_ng`, unloads both. Strictly best-effort: nothing
here ever raises, since a failed unload must never block a render. Whatever
was unloaded lazy-reloads on its next request.
"""
from typing import Callable, Optional

import requests

from configNG import get_ollama_base_url

OFFLOAD_TIMEOUT_S = 3.0


def unload_ollama_models_ng(on_log: Optional[Callable[[str], None]] = None) -> list:
    """Unload every model Ollama currently holds in memory (`/api/ps`, then
    `keep_alive: 0` per model). Returns the names unloaded; [] if Ollama is
    unreachable or idle."""
    base = get_ollama_base_url()
    unloaded = []
    try:
        resident = requests.get(f"{base}/api/ps", timeout=OFFLOAD_TIMEOUT_S).json().get("models", [])
    except (requests.exceptions.RequestException, ValueError):
        return unloaded
    for entry in resident:
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        try:
            requests.post(f"{base}/api/generate",
                          json={"model": name, "keep_alive": 0},
                          timeout=OFFLOAD_TIMEOUT_S)
            unloaded.append(name)
        except requests.exceptions.RequestException:
            pass
    if unloaded and on_log:
        on_log(f"[offload] unloaded Ollama model(s): {', '.join(unloaded)}")
    return unloaded


def offload_resident_models_ng(on_log: Optional[Callable[[str], None]] = None) -> None:
    """Unload Ollama models and the standalone Gemma server."""
    # Imported here: ltx_engineNG is heavy and already owns the Gemma helper.
    from ltx_engineNG import _unload_gemma_server_ng
    unload_ollama_models_ng(on_log=on_log)
    _unload_gemma_server_ng(on_log=on_log)
