"""NG-only file: HTTP layer for the Rachel view (bottom-bar "Rachel" task).

A thin proxy in front of a Hermes agent gateway (NousResearch
hermes-agent) that speaks the OpenAI /v1 protocol. Same shape as
routes/chatNG.py -- the browser never talks to the gateway directly
(keeps the API key server-side, keeps config in one place). Everything
lives under /api/ng/rachel/*.

Differences from chatNG.py:
  * one fixed model (HERMES_MODEL) -- no picker;
  * upstream is Server-Sent Events, not Ollama NDJSON, so /send translates
    the SSE stream into the exact NDJSON line shape chatNG.js's reader
    already understands ({"message": {"content": ...}} deltas, a final
    {"done": true}, errors as {"error": ..., "done": true});
  * session headers (X-Hermes-Session-Id / X-Hermes-Session-Key) are
    attached on every turn so the agent loads its persistent memory.

There is no conversation state here: the transcript lives in the browser
(static/rachelNG.js) and the full message list is POSTed every turn.
Blueprint registered in ring_viz.py.
"""

import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from configNG import (
    get_hermes_api_key,
    get_hermes_base_url,
    get_hermes_model,
    get_hermes_session_id,
    get_hermes_session_key,
)

rachelNG_bp = Blueprint("rachelNG", __name__)

# The agent can sit for minutes on the first token -- a long agent loop
# plus a big context preamble on a slow local backend -- so the read
# timeout is deliberately generous.
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 600


def _auth_headers(extra=None):
    h = {
        "Authorization": f"Bearer {get_hermes_api_key()}",
        "X-Hermes-Session-Id": get_hermes_session_id(),
        "X-Hermes-Session-Key": get_hermes_session_key(),
    }
    if extra:
        h.update(extra)
    return h


@rachelNG_bp.route("/api/ng/rachel/status", methods=["GET"])
def rachel_status_ng():
    """Is the Hermes gateway reachable? Drives the Rachel view's
    "can't reach Rachel" banner. Uses the unauthenticated /health probe."""
    import requests

    base_url = get_hermes_base_url()
    try:
        r = requests.get(f"{base_url}/health", timeout=_CONNECT_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        return jsonify({
            "reachable": False,
            "base_url": base_url,
            "error": str(e),
        })
    if r.status_code != 200:
        return jsonify({
            "reachable": False,
            "base_url": base_url,
            "error": f"HTTP {r.status_code}",
        })
    body = r.json() if r.content else {}
    return jsonify({
        "reachable": True,
        "base_url": base_url,
        "model": get_hermes_model(),
        "platform": body.get("platform"),
        "version": body.get("version"),
        "status": body.get("status"),
    })


@rachelNG_bp.route("/api/ng/rachel/send", methods=["POST"])
def rachel_send_ng():
    """Forward one chat turn to the Hermes gateway's OpenAI-compatible
    /v1/chat/completions and stream the reply back to the browser as
    NDJSON (translated from the upstream SSE).

    Body: {"messages": [{"role","content"}, ...],
           "options": {"temperature": float} (optional)}
    Response: application/x-ndjson -- {"message": {"content": "..."}} per
    delta, a final {"done": true}, and {"error": ..., "done": true} on a
    mid-stream failure.
    """
    import requests

    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    payload = {"model": get_hermes_model(), "messages": messages, "stream": True}
    options = body.get("options")
    if isinstance(options, dict):
        temp = options.get("temperature")
        if isinstance(temp, (int, float)):
            payload["temperature"] = float(temp)

    try:
        upstream = requests.post(
            f"{get_hermes_base_url()}/v1/chat/completions",
            json=payload,
            headers=_auth_headers({"Accept": "text/event-stream"}),
            stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not reach the Hermes gateway: {e}"}), 502

    if upstream.status_code != 200:
        detail = ""
        try:
            j = upstream.json()
            detail = (j.get("error") or {}).get("message") or j.get("error") or ""
        except Exception:  # noqa: BLE001
            detail = (upstream.text or "")[:200]
        upstream.close()
        return jsonify({
            "error": f"Hermes gateway returned HTTP {upstream.status_code}"
                     + (f": {detail}" if detail else "")
        }), 502

    def relay():
        try:
            for raw in upstream.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line or line.startswith(":"):
                    continue  # blank line or ": keepalive" comment
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                # OpenAI stream chunk: choices[0].delta.content
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield (json.dumps({"message": {"content": piece}}) + "\n").encode()
                elif obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    yield (json.dumps({"error": msg, "done": True}) + "\n").encode()
                    return
            yield (json.dumps({"done": True}) + "\n").encode()
        except Exception as e:  # noqa: BLE001
            yield (json.dumps({"error": str(e), "done": True}) + "\n").encode()
        finally:
            upstream.close()

    resp = Response(stream_with_context(relay()), mimetype="application/x-ndjson")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp
