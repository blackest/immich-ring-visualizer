"""immichRingNG -- "Soft finish": a per-frame colour pass over a finished clip. A finished mp4 in, a finished mp4 out.

The recipe is the one from the Ref2VA look-fix tutorial, measured here on a character-LoRA lip-sync clip
(2026-11-08, one frame): crushed blacks 22.5% -> 13.1%, brightness 64.6 -> 73.5, contrast about the same
(51.2 -> 53.3), skin a little softer. Two steps per frame:

  1. Clarity at a small negative strength (default -0.20, radius 24): softer local contrast / skin texture.
  2. A Cineon-ish tone map at a low strength (default 0.25) with saturation 0.9: black and white stay put, low
     shadows lift gently.

What it does NOT do: bring back shadows that were clipped during generation (the hybrid AdaLN block range does
most of that work). The tone mapper's Filmic modes are the wrong choice for normal gamma video -- at neutral
settings they darkened the same clip (mean 64.6 -> 26.5, crushed 22% -> 43%) -- so the mode is fixed here.

The work is done by vendor/soft_finish/soft_finish.py, a numpy/OpenCV port of two Orion4D_FXMax nodes (MIT;
see its VENDORED.md): a standalone script that takes `--in/--out` and runs in this app's own Python (numpy and
OpenCV, no torch, no ComfyUI, no GPU; about 2 s for a 5 s draft clip). It runs as a subprocess so Stop can
kill it. The earlier version of this file drove the user's ComfyUI instead (commit 3f77f0c); the port matches it
to about 1.7 levels per pixel where the recipe itself moves a pixel by about 8.7.

This file only validates the request, runs the script and reports; the original clip is never touched.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

FINISH_SCRIPT = Path(__file__).resolve().parent / "vendor" / "soft_finish" / "soft_finish.py"

DEFAULTS = {"clarity_strength": -0.20, "clarity_radius": 24.0, "tone_strength": 0.25, "saturation": 0.9}
# The ranges the form offers; wider than that would only invite a bad grade.
LIMITS = {"clarity_strength": (-0.5, 0.0), "clarity_radius": (4.0, 64.0),
          "tone_strength": (0.0, 1.0), "saturation": (0.5, 1.2)}
_ARG = {"clarity_strength": "--clarity", "clarity_radius": "--radius", "tone_strength": "--tone",
        "saturation": "--saturation"}

_TIMEOUT_S = 600.0
_POLL_S = 0.5


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


def finish_health_ng() -> dict:
    """Why the button is or is not offered: ffmpeg/ffprobe on the path, the vendored script, numpy and OpenCV."""
    problems = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            problems.append(f"{tool} is not installed")
    if not FINISH_SCRIPT.is_file():
        problems.append(f"the vendored finish script is missing at {FINISH_SCRIPT}")
    missing = [m for m, pkg in (("numpy", "numpy"), ("cv2", "opencv-python")) if importlib.util.find_spec(m) is None]
    if missing:
        problems.append("this Python is missing " + ", ".join(pkg for m, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"))
                                                                if m in missing))
    return {"ok": not problems, "problems": problems, "defaults": DEFAULTS, "limits": LIMITS}


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def finish_video_ng(src_mp4: str, out_path: str, params: Optional[dict] = None,
                    on_log: Optional[Callable[[str], None]] = None,
                    should_cancel: Optional[Callable[[], bool]] = None,
                    timeout_s: float = _TIMEOUT_S) -> dict:
    """Run the recipe on `src_mp4` and write the result to `out_path`. Raises FinishRequestError for a bad
    request, FinishCancelled when `should_cancel()` turns true, RuntimeError if the script fails."""
    log = on_log or (lambda _m: None)
    p = normalize_params_ng(params)
    src = Path(src_mp4)
    if not src.is_file():
        raise FinishRequestError(f"the clip is not there any more: {src}")
    health = finish_health_ng()
    if not health["ok"]:
        raise FinishRequestError("; ".join(health["problems"]))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(FINISH_SCRIPT), "--in", str(src), "--out", str(out)]
    for key, flag in _ARG.items():
        cmd += [flag, repr(p[key])]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    tail: list = []

    def pump() -> None:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-20]
                log(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = time.time() + timeout_s
    while proc.poll() is None:
        if should_cancel and should_cancel():
            _kill(proc)
            proc.wait()
            raise FinishCancelled("cancelled")
        if time.time() > deadline:
            _kill(proc)
            proc.wait()
            raise RuntimeError(f"the finish did not complete within {int(timeout_s)} s")
        time.sleep(_POLL_S)
    reader.join(timeout=5)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError("the finish failed: " + (" | ".join(tail[-3:]) or f"exit code {proc.returncode}"))
    return {"path": str(out), "params": p}
