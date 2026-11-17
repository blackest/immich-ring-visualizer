#!/usr/bin/env python3
"""immichring MCP server: six tools for an agent, over newline JSON-RPC on stdio.

    IMMICHRING_URL=http://192.168.3.54:5050 python3 immichring_mcp.py

Standard library only. The tool descriptions are written for a small local model: short, concrete,
each with its defaults, and the fewest possible required arguments. Failures come back as tool
results with isError (the model reads the sentence), never as protocol errors.

Register on this Mac, for example:
    claude mcp add immichring -- python3 /path/to/mcp_server/immichring_mcp.py
"""

import json
import sys

import immichring_client as api

PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER = {"name": "immichring", "version": "0.1.0"}


def _schema(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = [
    {"name": "compile_brief",
     "description": ("Turn an idea into video prompts. Give what should happen in plain words. Returns 'windows': "
                     "one prompt per 5-second part, to pass to render_video. Takes up to a minute. Does not render."),
     "inputSchema": _schema({"intent": {"type": "string", "description": "What happens in the clip."},
                             "seconds": {"type": "number", "description": "Clip length in seconds. Default 5.2."}},
                            ["intent"]),
     "run": lambda a: api.compile_brief(a["intent"], a.get("seconds", 5.2))},
    {"name": "render_video",
     "description": ("Start a Hailuo H3 video. Give 'windows' from compile_brief, or just a 'prompt' for a single "
                     "5-second clip. Defaults: draft quality, Fast. Returns a job_id like 'h3:abc'; the video is not "
                     "ready yet. Use job_status to wait, then fetch."),
     "inputSchema": _schema({"windows": {"type": "array", "items": {"type": "string"},
                                         "description": "Prompts from compile_brief, in order."},
                             "prompt": {"type": "string", "description": "A single prompt, if no windows."},
                             "quality": {"type": "string", "enum": list(api.H3_QUALITIES),
                                         "description": "Default 'draft'. 'high' is much slower."},
                             "first_frame": {"type": "string",
                                             "description": "Optional full path of a picture to start from."}}),
     "run": lambda a: api.render_video(a.get("windows"), a.get("prompt"), 5.0, a.get("quality", "draft"),
                                       a.get("first_frame"))},
    {"name": "make_image",
     "description": ("Start a Klein picture from a prompt. Optionally 1 to 3 reference pictures (full paths): the "
                     "first is the subject, then the location, then anything extra. Returns a job_id like 'klein:k1'."),
     "inputSchema": _schema({"prompt": {"type": "string"},
                             "pictures": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                                          "description": "Full paths of reference pictures."},
                             "width": {"type": "integer", "description": "Default 1024."},
                             "height": {"type": "integer", "description": "Default 576."}}, ["prompt"]),
     "run": lambda a: api.make_image(a["prompt"], a.get("pictures"), a.get("width", 1024), a.get("height", 576))},
    {"name": "job_status",
     "description": ("Where a job is: queued, rendering, done, failed or cancelled. Works for any job_id. Set "
                     "wait_seconds (up to 600) to wait for it to finish instead of asking again and again."),
     "inputSchema": _schema({"job_id": {"type": "string", "description": "For example 'h3:abc' or 'klein:k1'."},
                             "wait_seconds": {"type": "number", "description": "Default 0: answer now."}}, ["job_id"]),
     "run": lambda a: api.job_status(a["job_id"], a.get("wait_seconds", 0))},
    {"name": "fetch",
     "description": ("Save a finished job's video or first picture to a file on this machine. 'dest' is a file path, "
                     "or a folder to save into. The job must be done."),
     "inputSchema": _schema({"job_id": {"type": "string"},
                             "dest": {"type": "string", "description": "File path or folder to save to."}},
                            ["job_id", "dest"]),
     "run": lambda a: api.fetch(a["job_id"], a["dest"])},
    {"name": "recent_jobs",
     "description": "The latest jobs, videos and pictures together, newest first. Default 8.",
     "inputSchema": _schema({"limit": {"type": "integer", "description": "How many, 1 to 25."}}),
     "run": lambda a: api.recent_jobs(a.get("limit", 8))},
]
_BY_NAME = {t["name"]: t for t in TOOLS}


def _tool_list() -> list:
    return [{k: v for k, v in t.items() if k != "run"} for t in TOOLS]


def _text(obj, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}],
            "isError": error}


def call_tool(name: str, args: dict) -> dict:
    tool = _BY_NAME.get(name)
    if tool is None:
        return _text(f"Unknown tool {name}.", True)
    args = args or {}
    missing = [k for k in tool["inputSchema"]["required"] if args.get(k) in (None, "")]
    if missing:
        return _text(f"{name} needs: {', '.join(missing)}.", True)
    unknown = [k for k in args if k not in tool["inputSchema"]["properties"]]
    if unknown:
        return _text(f"{name} does not take: {', '.join(unknown)}.", True)
    try:
        return _text(tool["run"](args))
    except api.ImmichringError as e:
        return _text(str(e), True)
    except (TypeError, ValueError) as e:
        return _text(f"Bad arguments for {name}: {e}", True)


def handle(msg: dict):
    """One JSON-RPC message in, the reply (or None for a notification) out."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None
    if method == "initialize":
        asked = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
            "capabilities": {"tools": {}}, "serverInfo": SERVER}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": _tool_list()}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": mid, "result": call_tool(params.get("name", ""), params.get("arguments"))}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line))
        except Exception as e:  # noqa: BLE001 - a bad message must not kill the server
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
