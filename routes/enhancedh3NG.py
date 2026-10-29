"""immichRingNG -- HTTP layer for the Enhanced H3 view: a sentence plus
reference files in, an H3 brief out, split into one prompt per chained
window (see h3_engineNG.py's chain_prompts).

Forwards to OpenH3-IR's `h3ir serve` (address: configNG's h3ir_base_url
setting) and keeps nothing itself. The render is NOT started here -- the
view sends the (edited) window prompts and the keyframe to the existing
POST /api/ng/h3/generate, so the job queue, logs and speed options stay
in one place.

Files go to the compiler as uploads (PUT /v1/assets/<sha256>, then the
brief names the hash), so it never needs to read this machine's disk.
"""

import hashlib
import json
import re

import requests
from flask import Blueprint, jsonify, request

import configNG
import h3_engineNG

enhancedh3NG_bp = Blueprint("enhancedh3NG", __name__)

# A compile writes the brief with the language model: ~15-60 s warm, longer
# when the model has to load first.
_BRIEF_TIMEOUT_S = 300
_UPLOAD_TIMEOUT_S = 120
_QUICK_TIMEOUT_S = 5

_SHOT_RE = re.compile(r"\[Shot (\d+)\]")
# "At 00:07.500, the camera cuts to ..." -- a time on the whole clip's clock,
# wrong inside a window that starts partway through it.
_CLOCK_RE = re.compile(r"^\s*At \d{2}:\d{2}(?:\.\d+)?,\s*")
_CLOCK_ANY_RE = re.compile(r"\bAt (\d{2}):(\d{2}(?:\.\d+)?)\b")
_CLOCK_TIME_RE = re.compile(r"^\s*At (\d{2}):(\d{2}(?:\.\d+)?),")


class _ServiceError(Exception):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


# A brief lives in the compiler that wrote it, so a refine goes back to the same one.
_brief_base = {}


def _h3ir(method, path, timeout=_QUICK_TIMEOUT_S, base=None, **kwargs):
    base = base or configNG.get_h3ir_base_url()
    try:
        return requests.request(method, base + path, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        video = base == configNG.get_h3ir_video_base_url()
        raise _ServiceError(
            f"The H3 compiler isn't running at {base} -- start it with "
            f"`./h3ir_serveNG.sh {'qwen' if video else 'gemma'}` (or change its address in Settings).",
            503) from None
    except requests.exceptions.Timeout:
        raise _ServiceError("The H3 compiler took too long to answer.", 504) from None


def _error_text(res):
    try:
        body = res.json()
    except ValueError:
        if res.status_code >= 500:
            return "The compiler failed on this brief (see its log); try again or change the inputs."
        return f"HTTP {res.status_code}"
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, dict):  # {"detail": {"code": ..., "message": ...}}
                value = value.get("message") or value.get("error")
            if value:
                return str(value)
    return json.dumps(body)[:400]


def _window_clock(match, window_start_s):
    """A whole-clip clock inside a shot's text -> the same moment on the
    window's own clock (a later window starts again at 00:00)."""
    t = max(0.0, int(match.group(1)) * 60 + float(match.group(2)) - window_start_s)
    return f"At {int(t // 60):02d}:{t % 60:06.3f}"


def _split_windows(ir, plan, seconds):
    """One prompt per chained window, cut where the compiler's shots start.

    A shot belongs to the window that is running when it starts. A window
    that no shot starts in carries on the shot before it (`continues`) --
    the view flags those, since the same text twice is what made the first
    chained render repeat its opening. Window 0 keeps the compiler's opening
    line about the first-frame reference; later windows are conditioned by
    the runner on the previous window's last frame instead."""
    prompt = ir["prompt"]
    sections = ir["sections"]
    window_frames, windows, _total = h3_engineNG._h3_chain_plan_ng(seconds)
    if windows == 1:
        return [{"prompt": prompt, "shots": [s["n"] for s in plan.get("shots", [])],
                 "continues": False}]

    description = sections.get("integrated_multimodal_description", "")
    header = prompt.split("integrated_multimodal_description:", 1)[0].strip()
    tail = "".join(f"\n\n{name}: {text}" for name, text in sections.items()
                   if name != "integrated_multimodal_description" and text)

    parts = _SHOT_RE.split(description)  # ["", "1", text1, "2", text2, ...]
    shot_text = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    # The plan's shot times and the clock the compiler writes into the shot
    # text can disagree (plan: shot 2 at 3.4 s, text: "At 00:05.500"). The
    # text is what the video model reads, so it decides where a shot starts.
    starts = {s["n"]: float(s["from"]) for s in plan.get("shots", [])}
    for n, text in shot_text.items():
        clock = _CLOCK_TIME_RE.match(text)
        if clock:
            starts[n] = int(clock.group(1)) * 60 + float(clock.group(2))
    stride_s = (window_frames - 1) / h3_engineNG.H3_FRAME_RATE

    # A shot starting just short of an edge opens that window: the plan rounds
    # times to 0.1 s (5.1 s vs an edge at 5.125 s), and a shot that began 0.6 s
    # before the edge would otherwise be written twice, once in each window.
    edge_slack_s = 1.0
    owned = [[] for _ in range(windows)]
    for n in sorted(shot_text):
        start = starts.get(n, 0.0)
        index = max(i for i in range(windows) if i * stride_s <= start + edge_slack_s)
        owned[index].append(n)

    out, last = [], None
    for i, shots in enumerate(owned):
        continues = not shots
        if continues:
            shots = [last] if last is not None else []
        texts = []
        for k, n in enumerate(shots, 1):
            text = shot_text.get(n, "")
            if i > 0:
                text = _CLOCK_RE.sub("", text)
                text = text[:1].upper() + text[1:]
                text = _CLOCK_ANY_RE.sub(lambda m: _window_clock(m, i * stride_s), text)
            texts.append(f"[Shot {k}] {text}")
        window = "integrated_multimodal_description: " + " ".join(texts) + tail
        if i == 0 and header:
            window = header + "\n\n" + window
        out.append({"prompt": window, "shots": shots, "continues": continues})
        last = shots[-1] if shots else last
    return out


def _shape(brief, seconds):
    """What the view needs from a compiler response."""
    ir = brief.get("ir") or {}
    out = {
        "id": brief.get("id"),
        "status": brief.get("status"),
        "question": brief.get("question"),
        "default_if_unanswered": brief.get("default_if_unanswered"),
        "presentation": brief.get("presentation"),
        "plan": brief.get("plan"),
    }
    if ir:
        target = ir.get("target") or {}
        seconds = target.get("effective_seconds") or seconds
        out.update({
            "mode": ir.get("mode"),
            "prompt": ir.get("prompt"),
            "prompt_tokens": ir.get("prompt_tokens"),
            "target": target,
            "manifest": ir.get("manifest"),
            "diagnostics": ir.get("diagnostics") or [],
            "windows": _split_windows(ir, brief.get("plan") or {}, seconds),
        })
    return out


@enhancedh3NG_bp.route("/api/ng/enhanced-h3/status", methods=["GET"])
def enhanced_h3_status_ng():
    """Is the compiler up, and can it reach its language model."""
    try:
        res = _h3ir("GET", "/health")
    except _ServiceError as e:
        return jsonify({"ok": False, "llm": False, "error": str(e)})
    body = res.json() if res.ok else {}
    return jsonify({"ok": res.ok, "llm": bool(body.get("llm")), "profile": body.get("profile")})


@enhancedh3NG_bp.route("/api/ng/enhanced-h3/options", methods=["GET"])
def enhanced_h3_options_ng():
    """Roles per file type, limits, and the director profiles, straight from
    the compiler's own contract so the view never hard-codes them."""
    try:
        contract = _h3ir("GET", "/v1/contract")
        directors = _h3ir("GET", "/v1/directors")
    except _ServiceError as e:
        return jsonify({"error": str(e)}), e.status
    if not contract.ok:
        return jsonify({"error": _error_text(contract)}), 502
    body = contract.json()
    return jsonify({
        "roles": body.get("roles"),
        "limits": body.get("limits"),
        "directors": [{"id": d["id"], "name": d["name"]}
                      for d in directors.json().get("directors", [])] if directors.ok else [],
    })


def _asset_items():
    """The `assets` form field (JSON: [{field, kind, role, note, paired_field}]) as a list."""
    raw = (request.form.get("assets") or "").strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        raise _ServiceError("assets must be a JSON array", 400) from None
    if not isinstance(items, list):
        raise _ServiceError("assets must be a JSON array", 400)
    return items


def _compiler_base(items):
    """Gemma's compiler for text and pictures; the Qwen one when any asset is a video, since
    Gemma's description of a real clip crashed the compiler (see h3ir_serveNG.sh)."""
    if any(isinstance(i, dict) and i.get("kind") == "video" for i in items):
        return configNG.get_h3ir_video_base_url()
    return configNG.get_h3ir_base_url()


def _upload_assets(items, base):
    """The files `items` name -> the compiler's asset list, bytes uploaded to `base`."""

    digests = {}

    def upload(field):
        if field not in digests:
            fld = request.files.get(field)
            if fld is None:
                raise _ServiceError(f"file {field!r} was not sent", 400)
            data = fld.read()
            sha = hashlib.sha256(data).hexdigest()
            res = _h3ir("PUT", f"/v1/assets/{sha}", timeout=_UPLOAD_TIMEOUT_S, base=base, data=data,
                        headers={"Content-Type": "application/octet-stream"})
            if not res.ok:
                raise _ServiceError(_error_text(res), res.status_code)
            digests[field] = sha
        return digests[field]

    assets = []
    for item in items:
        asset = {"sha256": upload(item.get("field")), "kind": item.get("kind"),
                 "role": item.get("role")}
        if item.get("note"):
            asset["note"] = item["note"]
        if item.get("paired_field"):
            asset["paired_video_sha256"] = upload(item["paired_field"])
        assets.append(asset)
    return assets


@enhancedh3NG_bp.route("/api/ng/enhanced-h3/brief", methods=["POST"])
def enhanced_h3_brief_ng():
    """Form fields: intent (required), seconds, aspect, shots, creativity,
    director, effort, seed, silent, dialogue (JSON array of
    {text, language, speaker, voiceover}), assets (JSON array of {field, kind,
    role, note, paired_field}) plus the files those `field` names point at.
    Returns the compiler's plan, prompt and diagnostics, and `windows`: one
    prompt per chained window for POST /api/ng/h3/generate's chain_prompts."""
    form = request.form
    intent = (form.get("intent") or "").strip()
    if not intent:
        return jsonify({"error": "Describe what happens first."}), 400
    seconds = form.get("seconds", 5.167, type=float)

    payload = {"intent": intent, "seconds": seconds}
    for key in ("aspect", "creativity", "director", "effort"):
        if form.get(key):
            payload[key] = form[key]
    if form.get("shots"):
        payload["shots"] = form.get("shots", type=int)
    if form.get("seed"):
        payload["seed"] = form.get("seed", type=int)
    if str(form.get("silent") or "").lower() in ("1", "true", "on"):
        payload["silent"] = True
    if form.get("dialogue"):
        try:
            payload["dialogue"] = json.loads(form["dialogue"])
        except ValueError:
            return jsonify({"error": "dialogue must be a JSON array"}), 400

    try:
        items = _asset_items()
        base = _compiler_base(items)
        assets = _upload_assets(items, base)
        if assets:
            payload["assets"] = assets
        res = _h3ir("POST", "/v1/briefs", timeout=_BRIEF_TIMEOUT_S, base=base, json=payload)
        fallback = None
        if res.status_code == 500 and payload.get("dialogue"):
            # open-h3-ir 0.4.1 crashes on a first-frame picture plus a dialogue
            # entry ("<Subject 1> is used but never defined"). The same line
            # written into the sentence still compiles to a <d> tag.
            lines = payload.pop("dialogue")
            payload["intent"] += "".join(
                f' {(d.get("speaker") or "A character").capitalize()} says: "{d["text"]}".' for d in lines)
            fallback = "The compiler can't take a spoken line with these pictures, so the line was written into the sentence instead."
            res = _h3ir("POST", "/v1/briefs", timeout=_BRIEF_TIMEOUT_S, base=base, json=payload)
    except _ServiceError as e:
        return jsonify({"error": str(e)}), e.status
    if not res.ok:
        return jsonify({"error": _error_text(res)}), res.status_code
    out = _shape(res.json(), seconds)
    if out.get("id"):
        _brief_base[out["id"]] = base
    if fallback:
        out["note"] = fallback
    return jsonify(out), 201


@enhancedh3NG_bp.route("/api/ng/enhanced-h3/refine", methods=["POST"])
def enhanced_h3_refine_ng():
    """JSON {id, change, seconds}: rewrite an existing brief in plain words
    ("make it darker"). Returns the same shape as /brief."""
    body = request.get_json(silent=True) or {}
    brief_id = str(body.get("id") or "").strip()
    change = str(body.get("change") or "").strip()
    if not brief_id or not change:
        return jsonify({"error": "id and change are required"}), 400
    try:
        res = _h3ir("PATCH", f"/v1/briefs/{brief_id}", timeout=_BRIEF_TIMEOUT_S,
                    base=_brief_base.get(brief_id), json={"change": change})
    except _ServiceError as e:
        return jsonify({"error": str(e)}), e.status
    if not res.ok:
        return jsonify({"error": _error_text(res)}), res.status_code
    return jsonify(_shape(res.json(), body.get("seconds") or 5.167))
