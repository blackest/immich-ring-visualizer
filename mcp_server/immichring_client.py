"""HTTP client for the immichring NG app: one function per thing an agent may do.

No MCP code in here, so it is testable on its own. Standard library only, so a copy of this
folder runs on another machine (Rachel's m1mini) with whatever Python it has.

Job ids handed out carry their kind ("h3:abc123", "klein:k1") so a caller never has to say
which engine a job belongs to.

The app listens on 0.0.0.0:5050 with no auth, so only the calls below are used. Nothing here
deletes, cancels or edits a job.
"""

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

BASE = os.environ.get("IMMICHRING_URL", "http://127.0.0.1:5050").rstrip("/")

H3_QUALITIES = ("draft", "standard", "high", "native")
KLEIN_SLOTS = ("subject", "location", "extra")   # the app's first three reference slots, in order


class ImmichringError(Exception):
    """A sentence the model can read and act on."""


def _multipart(fields: dict, files: list) -> tuple:
    boundary = uuid.uuid4().hex
    out = []
    for key, value in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for key, path in files:
        name = os.path.basename(path)
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        out.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\n'
                    f'Content-Type: {kind}\r\n\r\n').encode() + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def _call(method: str, path: str, fields: dict = None, files: list = None, timeout: float = 30, raw: bool = False):
    """One request. JSON back (or raw bytes), or ImmichringError with the app's own sentence."""
    data, headers = None, {}
    if fields is not None or files:
        data, headers["Content-Type"] = _multipart(fields or {}, files or [])
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8", "replace")).get("error")
        except (ValueError, AttributeError):
            msg = None
        finally:
            e.close()
        raise ImmichringError(msg or f"The app answered {e.code} for {path}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ImmichringError(f"Cannot reach immichring at {BASE} ({getattr(e, 'reason', e)}). "
                              "Is the app running?") from None
    if raw:
        return body
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError:
        raise ImmichringError(f"The app sent something unreadable for {path}.") from None


def split_id(job_id: str) -> tuple:
    kind, sep, raw = str(job_id or "").partition(":")
    if not sep or kind not in ("h3", "klein") or not raw:
        raise ImmichringError(f"{job_id!r} is not a job id. Ids look like 'h3:abc123' or 'klein:k1'.")
    return kind, raw


def _check_pictures(paths) -> list:
    paths = [str(p) for p in (paths or [])]
    for p in paths:
        if not os.path.isfile(p):
            raise ImmichringError(f"No picture at {p}. Give the full path of a file on this machine.")
    return paths


# --- the six things --------------------------------------------------------------------------------

def _window_texts(windows) -> list:
    """The compiler returns {prompt, continues, shots} objects; the renderer wants plain strings."""
    texts = [w.get("prompt") if isinstance(w, dict) else w for w in (windows or [])]
    return [str(t) for t in texts if str(t or "").strip()]


def compile_brief(intent: str, seconds: float = 5.2) -> dict:
    """An idea in plain words -> one prompt per 5 s window, ready for render_video."""
    if not str(intent or "").strip():
        raise ImmichringError("Say what should happen first.")
    out = _call("POST", "/api/ng/enhanced-h3/brief", {"intent": intent, "seconds": seconds}, timeout=330)
    windows = _window_texts(out.get("windows"))
    if not windows:
        # the compiler may ask a question instead of answering
        raise ImmichringError(out.get("question") or "The compiler gave no prompts. Try describing the scene differently.")
    return {"windows": windows, "note": out.get("note"), "seconds": seconds}


def render_video(windows=None, prompt: str = None, seconds: float = 5.0, quality: str = "draft",
                 first_frame: str = None) -> dict:
    """Queue a Hailuo H3 clip. Defaults: the quantized model, the 'Fast' speed adapter, draft size."""
    windows = _window_texts(windows)
    if len(windows) == 1:           # the app only chains from two windows up
        prompt, windows = windows[0], []
    if not windows and not str(prompt or "").strip():
        raise ImmichringError("Give either 'windows' (from compile_brief) or a 'prompt'.")
    if quality not in H3_QUALITIES:
        raise ImmichringError(f"quality must be one of {', '.join(H3_QUALITIES)}.")
    fields = {"model": "h3q8", "quality": quality, "tristep": "1", "duration_s": seconds}
    if windows:
        fields["chain_prompts"] = json.dumps(windows)
        fields["duration_s"] = 5.0 * len(windows) if seconds == 5.0 else seconds
    else:
        fields["prompt"] = prompt
    files = [("file", p) for p in _check_pictures([first_frame] if first_frame else [])]
    out = _call("POST", "/api/ng/h3/generate", fields, files, timeout=120)
    return {"job_id": f"h3:{out['job_id']}", "status": "queued"}


def make_image(prompt: str, pictures=None, width: int = 1024, height: int = 576) -> dict:
    """Queue a Klein picture. Up to three reference pictures, used as subject, location, extra."""
    if not str(prompt or "").strip():
        raise ImmichringError("Give a prompt describing the picture.")
    paths = _check_pictures(pictures)
    if len(paths) > len(KLEIN_SLOTS):
        raise ImmichringError(f"At most {len(KLEIN_SLOTS)} pictures (subject, location, extra).")
    fields = {"prompt": prompt, "n": 1, "width": width, "height": height}
    files = [(f"ref{i}", p) for i, p in enumerate(paths)]
    out = _call("POST", "/api/ng/klein/generate", fields, files, timeout=120)
    return {"job_id": f"klein:{out['jobId']}", "status": "queued"}


def _h3_view(st: dict) -> dict:
    status = {"completed": "done"}.get(st.get("status"), st.get("status"))
    view = {"job_id": f"h3:{st['job_id']}", "kind": "video", "status": status, "prompt": st.get("prompt", "")[:140]}
    if status == "failed":
        view["error"] = st.get("error") or "The render failed."
    return view


def _klein_view(st: dict) -> dict:
    # Klein's own words -> the shared ones: waiting is queued, running is rendering, error is failed
    status = {"waiting": "queued", "running": "rendering", "error": "failed"}.get(st.get("status"), st.get("status"))
    view = {"job_id": f"klein:{st['id']}", "kind": "picture", "status": status, "prompt": (st.get("prompt") or "")[:140]}
    if status == "done":
        view["pictures"] = len(st.get("results") or [])
    if st.get("error"):
        view["error"] = st["error"]
    return view


def job_status(job_id: str, wait_s: float = 0) -> dict:
    """Where a job is. With wait_s, polls until it ends or the time runs out (max 600 s)."""
    kind, raw = split_id(job_id)
    deadline = time.monotonic() + max(0.0, min(float(wait_s or 0), 600.0))
    while True:
        if kind == "h3":
            view = _h3_view(_call("GET", f"/api/ng/h3/jobs/{urllib.parse.quote(raw)}"))
        else:
            view = _klein_view(_call("GET", f"/api/ng/klein/status/{urllib.parse.quote(raw)}"))
        if view["status"] in ("done", "failed", "cancelled") or time.monotonic() >= deadline:
            return view
        time.sleep(5)


def fetch(job_id: str, dest: str) -> dict:
    """Save a finished job's video or first picture to `dest` (a file path on this machine)."""
    kind, raw = split_id(job_id)
    view = job_status(job_id)
    if view["status"] != "done":
        raise ImmichringError(f"{job_id} is {view['status']}, not done yet." + (f" {view['error']}" if view.get("error") else ""))
    dest = os.path.expanduser(str(dest or "").strip())
    if not dest:
        raise ImmichringError("Say where to save it: 'dest' is a file path.")
    if os.path.isdir(dest):
        dest = os.path.join(dest, f"{kind}-{raw}" + (".mp4" if kind == "h3" else ".png"))
    if kind == "h3":
        body = _call("GET", f"/api/ng/h3/jobs/{urllib.parse.quote(raw)}/download", timeout=300, raw=True)
        others = 0
    else:
        results = _call("GET", f"/api/ng/klein/status/{urllib.parse.quote(raw)}").get("results") or []
        if not results:
            raise ImmichringError(f"{job_id} finished without a picture.")
        body = _call("GET", f"/api/ng/klein/image/{urllib.parse.quote(raw)}/{urllib.parse.quote(results[0]['file'])}",
                     timeout=120, raw=True)
        others = len(results) - 1
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(body)
    out = {"saved": os.path.abspath(dest), "bytes": len(body)}
    if others:
        out["note"] = f"{others} more picture(s) exist; only the first was saved."
    return out


def recent_jobs(limit: int = 8) -> dict:
    """The latest jobs of both kinds, newest first, in one short list."""
    limit = max(1, min(int(limit or 8), 25))
    found = []
    try:
        for st in _call("GET", "/api/ng/h3/jobs").get("jobs", []):
            found.append((st.get("queued_at") or 0, _h3_view(st)))
    except ImmichringError:
        pass
    for st in _call("GET", f"/api/ng/klein/jobs?limit={limit}").get("jobs", []):
        found.append((st.get("created_at") or 0, _klein_view(st)))
    found.sort(key=lambda t: t[0], reverse=True)
    return {"jobs": [v for _, v in found[:limit]]}
