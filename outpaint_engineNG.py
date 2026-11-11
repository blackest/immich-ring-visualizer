"""NG: "Make it vertical" with AI fill -- one small input -> output file that runs outpaint_runnerNG.py as a
subprocess in the LTX install's own Python (a peer of klein_engineNG.py). A widescreen clip in, a tall 9:16, 4:5 or
square copy out, the bands above and below painted by Lightricks' In/Outpainting IC-LoRA.

Geometry and blending are vendor/vertical_outpaint (MIT, Phosphene v5.0.16); see the runner for the recipe.
Measured by Phosphene on an M4 Max 64 GB: about 15 minutes for a 5-second clip.
"""
from __future__ import annotations

import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "vertical_outpaint"))

import vertical_outpaint as vo      # noqa: E402  (vendored, pure Python at import)
import vertical_reframe as vr       # noqa: E402

import ltx_engineNG as ltx          # noqa: E402

RUNNER = HERE / "outpaint_runnerNG.py"
RESULT_PREFIX = "OUTPAINT_RESULT "
RATIOS = tuple(vr.RATIOS)           # ("9:16", "4:5", "1:1")
LORA_PATH = ltx.LTX_WEIGHTS_ROOT / "loras" / "ic" / vo.OUTPAINT_LORA_FILE
TIMEOUT_S = 3 * 3600


class OutpaintCancelled(RuntimeError):
    pass


def _ffprobe(path: str) -> dict:
    exe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                          "-of", "json", path], capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        raise ValueError("could not read the clip: " + out.stderr.strip()[-200:])
    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise ValueError("the file has no video")
    s = streams[0]
    num, _, den = str(s.get("r_frame_rate", "24/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 24.0
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps or 24.0,
            "frames": int(s.get("nb_read_frames") or 0)}


def outpaint_health_ng() -> dict:
    """Is everything on disk that the fill needs. Never launches the model."""
    py = ltx.LTX_DEFAULT_PYTHON
    lora_ok = LORA_PATH.is_file() and LORA_PATH.stat().st_size == vo.OUTPAINT_LORA_BYTES
    model = ltx.LTX_DEFAULT_MODEL
    gemma = ltx.LTX_DEFAULT_GEMMA
    problems = []
    if not py.is_file():
        problems.append("the LTX install is missing")
    if not Path(model).exists():
        problems.append("the LTX model is missing")
    if not Path(gemma).exists():
        problems.append("the LTX text encoder is missing")
    if not lora_ok:
        problems.append("the AI fill add-on is not downloaded (1.3 GB)")
    return {"ok": not problems, "problems": problems, "lora": str(LORA_PATH), "ratios": list(RATIOS),
            "minutes_per_5s": 15}


def plan_ng(source: str, ratio: str) -> dict:
    """Check the request and work out sizes: raises ValueError with a message for the person."""
    if ratio not in vr.RATIOS:
        raise ValueError("pick one of: " + ", ".join(RATIOS))
    if not Path(source).is_file():
        raise ValueError("the clip was not found")
    info = _ffprobe(source)
    out_w, out_h = vr.target_size(ratio, info["width"], info["height"])   # raises when already vertical
    cw, ch = vo.canvas_size(out_w, out_h)
    vo.source_rect(info["width"], info["height"], cw, ch)                 # raises when there is no band to fill
    if info["frames"] < 1:
        raise ValueError("could not count the frames of the clip")
    return {**info, "out_w": out_w, "out_h": out_h}


def outpaint_vertical_ng(source: str, output: str, ratio: str = "9:16", prompt: str = "",
                         seed: Optional[int] = None, on_log: Optional[Callable[[str], None]] = None,
                         should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Make `output` (an mp4) from `source`. Returns {output, seconds, windows}; raises OutpaintCancelled on stop."""
    health = outpaint_health_ng()
    if not health["ok"]:
        raise RuntimeError("; ".join(health["problems"]))
    info = plan_ng(source, ratio)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    work = tempfile.mkdtemp(prefix="outpaint_")
    request = {"model": str(ltx.LTX_DEFAULT_MODEL), "gemma": str(ltx.LTX_DEFAULT_GEMMA), "lora": str(LORA_PATH),
               "source": str(source), "output": str(output), "work_dir": work,
               "src_w": info["width"], "src_h": info["height"], "out_w": info["out_w"], "out_h": info["out_h"],
               "frames": info["frames"], "fps": info["fps"], "prompt": prompt,
               "seed": seed if seed is not None else random.randint(0, 2 ** 31 - 1), "crf": 18, "keep_audio": True}
    env = ltx._clean_subprocess_env_ng()
    # macOS cuts off a GPU job that runs too long ("Impacting Interactivity", ml-explore/mlx#3267). These two make MLX hand
    # the GPU shorter pieces of work: Phosphene's own fix, and what let the 4K Klein render finish.
    env.setdefault("MLX_DISABLE_COMPILE", "1")
    env.setdefault("MLX_METAL_FAST_SYNCH", "1")
    # Phosphene's "short GPU steps": smaller command buffers, bit-identical output, a little slower (10/10 first;
    # its own retry after a kill uses 1/1). Set both in the app's environment to override.
    env.setdefault("MLX_MAX_OPS_PER_BUFFER", "10")
    env.setdefault("MLX_MAX_MB_PER_BUFFER", "10")
    proc = subprocess.Popen([str(ltx.LTX_DEFAULT_PYTHON), str(RUNNER)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            env=env, cwd=str(ltx.LTX_REPO_DIR), start_new_session=True)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    threading.Thread(target=lambda: ([lines.put(l.rstrip()) for l in proc.stdout], lines.put(None)),
                     daemon=True).start()
    result, tail = None, []
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.close()
        started = time.monotonic()
        while True:
            if should_stop and should_stop():
                raise OutpaintCancelled("stopped")
            if time.monotonic() - started > TIMEOUT_S:
                raise RuntimeError("the AI fill timed out")
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX):])
            else:
                tail = (tail + [line])[-12:]
                if on_log:
                    on_log(line)
        if proc.wait(timeout=30) or result is None:
            raise RuntimeError("the AI fill failed: " + " | ".join(l for l in tail if l.strip())[-400:])
        if not Path(output).is_file() or Path(output).stat().st_size == 0:
            raise RuntimeError("the AI fill wrote no file")
        return result
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        shutil.rmtree(work, ignore_errors=True)
