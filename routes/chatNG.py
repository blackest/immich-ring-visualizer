"""NG-only file: HTTP layer for the Chat view (bottom-bar "Chat" task).

A thin proxy in front of the local Ollama daemon so the browser never
talks to Ollama directly (same reasoning as routes/generateNG.py's Immich
proxy -- keeps the daemon off the network surface, keeps CORS/config in
one place). Everything lives under /api/ng/chat/*.

There is deliberately no conversation state here: the transcript is held
in the browser (static/chatNG.js, one global conversation) and the full
message list is POSTed on every turn, exactly how Ollama's /api/chat
expects it. This route only forwards.

Not wired into any non-NG module. Blueprint registered in ring_viz.py.
"""

import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from configNG import OLLAMA_BASE_URL

chatNG_bp = Blueprint("chatNG", __name__)

# Ollama can sit for a long time on the first token while it loads a model
# into VRAM; give the connect a short leash but the read a generous one.
_CONNECT_TIMEOUT = 5
_READ_TIMEOUT = 300


def _ollama_get(path, timeout=_CONNECT_TIMEOUT):
    import requests

    return requests.get(f"{OLLAMA_BASE_URL}{path}", timeout=timeout)


@chatNG_bp.route("/api/ng/chat/status", methods=["GET"])
def chat_status_ng():
    """Is the local Ollama daemon reachable? Drives the Chat view's
    "Ollama isn't running" banner."""
    try:
        r = _ollama_get("/api/version")
    except Exception as e:  # noqa: BLE001
        return jsonify({
            "reachable": False,
            "base_url": OLLAMA_BASE_URL,
            "error": str(e),
        })
    if r.status_code != 200:
        return jsonify({
            "reachable": False,
            "base_url": OLLAMA_BASE_URL,
            "error": f"HTTP {r.status_code}",
        })
    body = r.json() if r.content else {}
    return jsonify({
        "reachable": True,
        "base_url": OLLAMA_BASE_URL,
        "version": body.get("version"),
    })


@chatNG_bp.route("/api/ng/chat/models", methods=["GET"])
def chat_models_ng():
    """Installed models (for the picker) plus which ones are currently
    loaded in memory, so the view can pre-select the model the running
    agent is already using instead of cold-loading another."""
    try:
        tags = _ollama_get("/api/tags")
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not reach Ollama: {e}"}), 502
    if tags.status_code != 200:
        return jsonify({
            "error": f"Ollama returned HTTP {tags.status_code} for /api/tags"
        }), 502

    models = []
    for m in (tags.json().get("models") or []):
        details = m.get("details") or {}
        models.append({
            "name": m.get("name"),
            "size": m.get("size"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "family": details.get("family"),
        })
    models.sort(key=lambda x: (x["name"] or "").lower())

    loaded = []
    try:
        ps = _ollama_get("/api/ps")
        if ps.status_code == 200:
            loaded = [m.get("name") for m in (ps.json().get("models") or [])]
    except Exception:  # noqa: BLE001
        pass  # /api/ps is a nicety, not required

    return jsonify({"models": models, "loaded": loaded})


@chatNG_bp.route("/api/ng/chat/send", methods=["POST"])
def chat_send_ng():
    """Forward one chat turn to Ollama's /api/chat and stream the NDJSON
    response straight back to the browser.

    Body: {"model": str, "messages": [{"role","content"}, ...],
           "options": {"temperature": float, ...} (optional)}
    Response: application/x-ndjson, one JSON object per line, relayed
    verbatim from Ollama (each has a "message" delta and a final line
    with "done": true).
    """
    import requests

    body = request.get_json(silent=True) or {}
    model = str(body.get("model") or "").strip()
    messages = body.get("messages")
    if not model:
        return jsonify({"error": "model is required"}), 400
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    payload = {"model": model, "messages": messages, "stream": True}
    options = body.get("options")
    if isinstance(options, dict) and options:
        payload["options"] = options

    try:
        upstream = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not reach Ollama: {e}"}), 502

    if upstream.status_code != 200:
        detail = ""
        try:
            detail = upstream.json().get("error") or ""
        except Exception:  # noqa: BLE001
            detail = (upstream.text or "")[:200]
        return jsonify({
            "error": f"Ollama returned HTTP {upstream.status_code}"
                     + (f": {detail}" if detail else "")
        }), 502

    def relay():
        try:
            for chunk in upstream.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        except Exception as e:  # noqa: BLE001
            # Surface a mid-stream break as a final NDJSON line the
            # frontend already knows how to read.
            yield (json.dumps({"error": str(e), "done": True}) + "\n").encode()
        finally:
            upstream.close()

    resp = Response(stream_with_context(relay()),
                    mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"  # don't let a proxy buffer the stream
    return resp
