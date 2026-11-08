"""immichRingNG -- "Soft finish": a per-frame colour pass over a finished clip, run by the user's ComfyUI.

The recipe is the one from the Ref2VA look-fix tutorial, measured here on a character-LoRA lip-sync clip
(2026-11-08, one frame): crushed blacks 22.5% -> 13.1%, brightness 64.6 -> 73.5, contrast about the same
(51.2 -> 53.3), skin a little softer. Two nodes from the MIT-licensed Orion4D_FXMax pack:

  1. Clarity at a small negative strength (default -0.20, radius 24): softer local contrast / skin texture.
  2. DCTL Tone Mapper in its "Cineon-ish" mode at a low strength (default 0.25) with saturation 0.9: black
     and white stay put, low shadows lift gently.

What it does NOT do: bring back shadows that were clipped during generation (the hybrid AdaLN block range
does most of that work). The tone mapper's Filmic modes are the wrong choice for normal gamma video -- at
neutral settings they darkened the same clip (mean 64.6 -> 26.5, crushed 22% -> 43%) -- so the mode is fixed
here, not offered.

This file only talks to ComfyUI: it sends the clip's path (so ComfyUI must run on THIS Mac and read the file
itself), polls the run, copies the finished clip into the caller's folder and removes ComfyUI's own copy. No
model is loaded here and nothing is cached. Needs ComfyUI-VideoHelperSuite and Orion4D_FXMax installed
(finish_health_ng says what is missing).
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import configNG

NODE_LOAD = "VHS_LoadVideoPath"
NODE_CLARITY = "Orion4D_FXMax_Clarity"
NODE_TONE = "Orion4D_FXMax_DCTLToneMapper"
NODE_COMBINE = "VHS_VideoCombine"
REQUIRED_NODES = (NODE_LOAD, NODE_CLARITY, NODE_TONE, NODE_COMBINE)
PREFIX = "ringviz_finish_"
TONE_MODE = "Cineon-ish"

DEFAULTS = {"clarity_strength": -0.20, "clarity_radius": 24.0, "tone_strength": 0.25, "saturation": 0.9}
# The ranges the form offers; wider than the node's own limits would only invite a bad grade.
LIMITS = {"clarity_strength": (-0.5, 0.0), "clarity_radius": (4.0, 64.0),
          "tone_strength": (0.0, 1.0), "saturation": (0.5, 1.2)}

_TIMEOUT_S = 900.0
_POLL_S = 1.5


class FinishRequestError(ValueError):
    pass


class FinishCancelled(Exception):
    pass


def normalize_params_ng(params: Optional[dict]) -> dict:
    """The four settings, defaults filled in and each checked against LIMITS (a bool or non-number is refused)."""
    out = dict(DEFAULTS)
    for key, value in (params or {}).items():
        if key not in DEFAULTS:
            raise FinishRequestError(f"unknown finish setting {key!r}")
        if value is None or value == "":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FinishRequestError(f"{key} must be a number")
        lo, hi = LIMITS[key]
        if not lo <= float(value) <= hi:
            raise FinishRequestError(f"{key} must be between {lo:g} and {hi:g}")
        out[key] = float(value)
    return out


def _clarity_settings(p: dict) -> dict:
    # Every other effect in the Clarity node's pipeline is zeroed: this pass softens, it never sharpens.
    return {"pipeline": "Balanced", "mode_color": "Luma", "texture_strength": 0.0, "texture_radius": 1.2,
            "clarity_strength": p["clarity_strength"], "clarity_radius": p["clarity_radius"],
            "midtone_softness": 0.35, "sharpen_amount": 0.0, "sharpen_radius": 1.4, "threshold": 0.005,
            "smart_amount": 0.0, "smart_radius": 1.8, "reduce_noise": 0.0, "fade_shadows": 0.0,
            "fade_highlights": 0.0, "highpass_amount": 0.0, "highpass_radius": 2.4,
            "highpass_blend": "SoftLight", "highpass_opacity": 0.0, "mask_blur": 0.0, "mask_opacity": 1.0,
            "invert_mask": "No"}


def _tone_settings(p: dict) -> dict:
    return {"mode": TONE_MODE, "exposure": 0.0, "contrast": 1.0, "pivot": 0.18, "highlight_rolloff": 0.65,
            "shadow_lift": 0.0, "black_floor": 0.0, "saturation": p["saturation"], "preserve_luminosity": False,
            "strength": p["tone_strength"], "clamp_output": True, "mask_blur": 0.0, "mask_opacity": 1.0,
            "invert_mask": "No"}


def build_workflow_ng(src_path: str, params: dict, fps: float, has_audio: bool, prefix: str = PREFIX) -> dict:
    """The ComfyUI API-format graph: load -> Clarity -> tone mapper -> write (with the clip's own audio)."""
    p = normalize_params_ng(params)
    combine = {"images": ["3", 0], "frame_rate": float(fps), "loop_count": 0, "filename_prefix": prefix,
               "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False,
               "pingpong": False, "save_output": True}
    if has_audio:
        combine["audio"] = ["1", 2]
    return {
        "1": {"class_type": NODE_LOAD, "inputs": {
            "video": str(src_path), "force_rate": 0, "custom_width": 0, "custom_height": 0,
            "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff"}},
        "2": {"class_type": NODE_CLARITY, "inputs": {
            "image": ["1", 0], "preset": "Natural Clarity", "raw_settings": json.dumps(_clarity_settings(p))}},
        "3": {"class_type": NODE_TONE, "inputs": {
            "image": ["2", 0], "preset": "Neutral Filmic", "raw_settings": json.dumps(_tone_settings(p))}},
        "4": {"class_type": NODE_COMBINE, "inputs": combine},
    }


# ---- talking to ComfyUI ----------------------------------------------------------------------------------

def _base() -> str:
    return configNG.get_comfyui_base_url().rstrip("/")


def _get(path: str, timeout: float = 15.0):
    with urllib.request.urlopen(_base() + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: dict, timeout: float = 30.0):
    req = urllib.request.Request(_base() + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def comfy_is_local_ng(base: Optional[str] = None) -> bool:
    """ComfyUI reads the clip from its path, so it has to be on this Mac: the host is loopback or one of this
    machine's own addresses (a socket can only bind an address that belongs to this machine)."""
    host = urlparse(base or _base()).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((socket.gethostbyname(host), 0))
        return True
    except OSError:
        return False


def finish_health_ng() -> dict:
    """Why the button is or is not offered: ComfyUI reachable, on this Mac, and the four nodes loaded."""
    problems = []
    try:
        base = _base()
    except Exception:  # noqa: BLE001
        return {"ok": False, "problems": ["ComfyUI address is not configured"]}
    if not comfy_is_local_ng(base):
        problems.append(f"ComfyUI at {base} is not on this Mac; the finish pass needs it to read the clip's file")
    missing = []
    try:
        for node in REQUIRED_NODES:
            try:
                if node not in _get(f"/object_info/{node}", timeout=8.0):
                    missing.append(node)
            except urllib.error.HTTPError:
                missing.append(node)
    except (urllib.error.URLError, OSError, ValueError):
        return {"ok": False, "problems": problems + [f"ComfyUI is not reachable at {base}"]}
    if missing:
        pack = []
        if NODE_LOAD in missing or NODE_COMBINE in missing:
            pack.append("ComfyUI-VideoHelperSuite")
        if NODE_CLARITY in missing or NODE_TONE in missing:
            pack.append("Orion4D_FXMax")
        problems.append("ComfyUI is missing " + " and ".join(pack) + " (restart it after installing)")
    return {"ok": not problems, "problems": problems, "defaults": DEFAULTS, "limits": LIMITS}


def probe_clip_ng(path: str) -> dict:
    """fps and whether there is an audio track, from ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,r_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True).stdout
        streams = json.loads(out).get("streams", [])
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        raise FinishRequestError(f"could not read the clip: {e}") from e
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FinishRequestError("the file has no video")
    num, _, den = str(video.get("r_frame_rate") or "24/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    return {"fps": fps if 1.0 <= fps <= 120.0 else 24.0, "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def finish_video_ng(src_mp4: str, out_path: str, params: Optional[dict] = None,
                    on_log: Optional[Callable[[str], None]] = None,
                    should_cancel: Optional[Callable[[], bool]] = None,
                    timeout_s: float = _TIMEOUT_S) -> dict:
    """Run the recipe on `src_mp4` and write the result to `out_path`. Raises FinishRequestError for a bad
    request, FinishCancelled when `should_cancel()` turns true, RuntimeError for anything ComfyUI reports."""
    log = on_log or (lambda _m: None)
    p = normalize_params_ng(params)
    src = Path(src_mp4)
    if not src.is_file():
        raise FinishRequestError(f"the clip is not there any more: {src}")
    health = finish_health_ng()
    if not health["ok"]:
        raise FinishRequestError("; ".join(health["problems"]))
    clip = probe_clip_ng(str(src))
    prefix = f"{PREFIX}{int(time.time())}"
    workflow = build_workflow_ng(str(src), p, clip["fps"], clip["has_audio"], prefix)
    try:
        prompt_id = _post("/prompt", {"prompt": workflow})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI refused the workflow: {e.read().decode('utf-8', 'replace')[:400]}") from e
    log(f"finish: ComfyUI run {prompt_id}")

    deadline = time.time() + timeout_s
    entry = None
    cancel_sent = False
    while time.time() < deadline:
        if should_cancel and should_cancel() and not cancel_sent:
            cancel_sent = True
            try:
                _post("/queue", {"delete": [prompt_id]})
                _post("/interrupt", {"prompt_id": prompt_id})
            except Exception:  # noqa: BLE001 -- best effort; we stop waiting either way
                pass
        if cancel_sent:
            raise FinishCancelled("cancelled")
        hist = _get(f"/history/{prompt_id}")
        if prompt_id in hist and hist[prompt_id].get("status", {}).get("completed") is not None:
            entry = hist[prompt_id]
            break
        time.sleep(_POLL_S)
    if entry is None:
        raise RuntimeError(f"ComfyUI did not finish within {int(timeout_s)} s")
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        detail = next((m[1].get("exception_message") for m in status.get("messages", [])
                       if m and m[0] == "execution_error"), "unknown error")
        raise RuntimeError(f"ComfyUI failed: {str(detail).strip()[:400]}")
    try:
        produced = Path(entry["outputs"]["4"]["gifs"][0]["fullpath"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError("ComfyUI finished but reported no output file") from e
    if not produced.is_file():
        raise RuntimeError(f"ComfyUI's output is not where it said: {produced}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, out)
    # ComfyUI's video writer leaves a silent copy and a workflow image beside the reported file. They are ours
    # (this run's unique prefix) and nothing else, so they go with it.
    for leftover in produced.parent.glob(f"{prefix}*"):
        try:
            leftover.unlink()
        except OSError:
            pass
    log("finish: done")
    return {"path": str(out), "params": p, "fps": clip["fps"], "has_audio": clip["has_audio"]}
