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
KINDS = ("h3", "klein", "music", "speech")
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


def _call(method: str, path: str, fields: dict = None, files: list = None, timeout: float = 30, raw: bool = False,
          json_body: dict = None):
    """One request. JSON back (or raw bytes), or ImmichringError with the app's own sentence."""
    data, headers = None, {}
    if json_body is not None:
        data, headers["Content-Type"] = json.dumps(json_body).encode(), "application/json"
    elif fields is not None or files:
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
    if not sep or kind not in KINDS or not raw:
        raise ImmichringError(f"{job_id!r} is not a job id. Ids look like 'h3:abc123', 'klein:k1', 'music:abc' or 'speech:abc'.")
    return kind, raw


def _check_pictures(paths) -> list:
    paths = [str(p) for p in (paths or [])]
    for p in paths:
        if not os.path.isfile(p):
            raise ImmichringError(f"No picture at {p}. Give the full path of a file on this machine.")
    return paths


# --- the eight things --------------------------------------------------------------------------------

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


def make_song(style: str = "", lyrics: str = "", instrumental: bool = False, seconds: float = 0) -> dict:
    """Queue a song. A style and/or lyrics; the song is as long as the lyrics make it, `seconds` only caps it."""
    if not str(style or "").strip() and not str(lyrics or "").strip():
        raise ImmichringError("Give a style (for example 'slow blues, male vocal') and/or lyrics.")
    body = {"style": style or "", "lyrics": lyrics or "", "instrumental": bool(instrumental)}
    if seconds:
        body["duration_s"] = float(seconds)
    out = _call("POST", "/api/ng/music/generate", json_body=body, timeout=60)
    return {"job_id": f"music:{out['job_id']}", "status": "queued"}


def _voices() -> dict:
    return _call("GET", "/api/ng/speech/voices")


def make_speech(text: str, voice: str = "", language: str = "english") -> dict:
    """Queue spoken audio. `voice` is a preset speaker or a character with a saved voice; blank = the first preset."""
    if not str(text or "").strip():
        raise ImmichringError("Give the words to say.")
    known = _voices()
    presets = [p["id"] for p in known.get("presets", [])]
    characters = [c["id"] for c in known.get("characters", [])]
    wanted = str(voice or "").strip()
    if not wanted:
        wanted = presets[0] if presets else ""
    by_name = {n.casefold(): n for n in presets + characters}
    if wanted.casefold() not in by_name:
        raise ImmichringError(f"No voice called {wanted!r}. Choose one of: {', '.join(presets + characters)}.")
    name = by_name[wanted.casefold()]
    kind = "preset" if name in presets else "character"
    fields = {"text": text, "kind": kind, "language": str(language or "english").lower(),
              ("voice" if kind == "preset" else "character"): name}
    out = _call("POST", "/api/ng/speech/generate", fields, timeout=60)
    return {"job_id": f"speech:{out['job_id']}", "status": "queued", "voice": name}


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


def _audio_view(kind: str, st: dict) -> dict:
    """Music and speech jobs share a shape: completed means done."""
    status = {"completed": "done"}.get(st.get("status"), st.get("status"))
    text = st.get("lyrics") or st.get("style") or st.get("text") or ""
    view = {"job_id": f"{kind}:{st['job_id']}", "kind": "song" if kind == "music" else "speech", "status": status,
            "prompt": str(text)[:140]}
    if status == "done" and st.get("audio_seconds"):
        view["seconds"] = round(st["audio_seconds"], 1)
    if status == "failed":
        view["error"] = st.get("error") or "The job failed."
    return view


def _music_logs() -> list:
    """Finished songs, newest first. The app keeps these on disk, so they outlive a restart."""
    return _call("GET", "/api/ng/music/logs").get("entries", [])


def _music_log(raw: str):
    return next((e for e in _music_logs() if e.get("job_id") == raw), None)


def _view(kind: str, raw: str) -> dict:
    q = urllib.parse.quote(raw)
    if kind == "music":
        try:
            return _audio_view(kind, _call("GET", f"/api/ng/music/jobs/{q}"))
        except ImmichringError:
            entry = _music_log(raw)         # the app forgot the job (restarted), the log still has it
            if entry is None:
                raise
            return _audio_view(kind, entry)
    if kind == "h3":
        return _h3_view(_call("GET", f"/api/ng/h3/jobs/{q}"))
    if kind == "klein":
        return _klein_view(_call("GET", f"/api/ng/klein/status/{q}"))
    return _audio_view(kind, _call("GET", f"/api/ng/{kind}/jobs/{q}"))


def job_status(job_id: str, wait_s: float = 0) -> dict:
    """Where a job is. With wait_s, polls until it ends or the time runs out (max 600 s)."""
    kind, raw = split_id(job_id)
    deadline = time.monotonic() + max(0.0, min(float(wait_s or 0), 600.0))
    while True:
        view = _view(kind, raw)
        if view["status"] in ("done", "failed", "cancelled") or time.monotonic() >= deadline:
            return view
        time.sleep(5)


# kinds whose finished job is one file behind a download route: (route, extension)
DOWNLOADS = {"h3": ("/api/ng/h3/jobs/{}/download", ".mp4"),
             "music": ("/api/ng/music/jobs/{}/download", ".flac"),
             "speech": ("/api/ng/speech/jobs/{}/download", ".wav")}


def fetch(job_id: str, dest: str) -> dict:
    """Save a finished job's video, song, speech or first picture to `dest` (a file path on this machine)."""
    kind, raw = split_id(job_id)
    view = job_status(job_id)
    if view["status"] != "done":
        raise ImmichringError(f"{job_id} is {view['status']}, not done yet." + (f" {view['error']}" if view.get("error") else ""))
    dest = os.path.expanduser(str(dest or "").strip())
    if not dest:
        raise ImmichringError("Say where to save it: 'dest' is a file path.")
    if os.path.isdir(dest):
        dest = os.path.join(dest, f"{kind}-{raw}{DOWNLOADS[kind][1] if kind in DOWNLOADS else '.png'}")
    if kind in DOWNLOADS:
        try:
            body = _call("GET", DOWNLOADS[kind][0].format(urllib.parse.quote(raw)), timeout=300, raw=True)
        except ImmichringError:
            entry = _music_log(raw) if kind == "music" else None
            if not entry or not entry.get("has_audio"):
                raise
            body = _call("GET", f"/api/ng/music/logs/{urllib.parse.quote(entry['date'])}/{urllib.parse.quote(raw)}/audio",
                         timeout=300, raw=True)
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
    try:
        for st in _call("GET", "/api/ng/speech/jobs").get("jobs", []):
            found.append((st.get("queued_at") or 0, _audio_view("speech", st)))
    except ImmichringError:
        pass
    try:
        for st in _music_logs():
            found.append((st.get("finished_at") or st.get("queued_at") or 0, _audio_view("music", st)))
    except ImmichringError:
        pass
    for st in _call("GET", f"/api/ng/klein/jobs?limit={limit}").get("jobs", []):
        found.append((st.get("created_at") or 0, _klein_view(st)))
    found.sort(key=lambda t: t[0], reverse=True)
    return {"jobs": [v for _, v in found[:limit]]}
