"""immichRingNG -- 2x upscale of a finished H3 film as a queued job (see upscale_x2NG.py).

The script runs LTX 2.5's pixel-spatial-upscaler IC-LoRA from the clip's own latent, so the shot, face
and motion stay the clip's and only the detail is redrawn at twice the size. It needs the LTX venv's
python (it imports ltx_pipelines_mlx), so it runs as a subprocess, one at a time in the H3 queue so it
can never share the GPU with a render. The Gemma server is unloaded first, as before a render.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import h3_jobsNG as h3_jobs
import h3_rerollNG as h3_reroll
import ltx_engineNG as ltx

SCRIPT = Path(__file__).with_name("upscale_x2NG.py")
MINUTES_PER_WINDOW = 8.5   # measured on this M2 Max for a 124-frame 640x384 clip
FRAMES_PER_WINDOW = 124
MIN_STEPS, MAX_STEPS = 1, 6
MIN_KEEP, MAX_KEEP = 0.3, 1.0
ATTEMPTS = 3
WATCHDOG_MARK = "Impacting Interactivity"


def ltx_python_ng() -> str:
    """The python that has ltx_pipelines_mlx: the LTX install's own venv."""
    for name in ("python3.11", "python3", "python"):
        cand = ltx.LTX_REPO_DIR / "env" / "bin" / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"no python in the LTX venv at {ltx.LTX_REPO_DIR / 'env' / 'bin'} "
                            "(is LTX installed?)")


def frames_of_ng(clip) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(clip)],
                         capture_output=True, text=True, timeout=60).stdout.strip().split(",")[0]
    try:
        return int(out)
    except ValueError:
        raise ValueError(f"cannot read {Path(clip).name}") from None


def estimate_minutes_ng(frames: int) -> float:
    """Rough time: the cost is proportional to the frames, and one 124-frame window took about 8.5 min."""
    return round(MINUTES_PER_WINDOW * frames / FRAMES_PER_WINDOW, 1)


def _clean(value, low, high, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be {low:g} to {high:g} (got {value:g})")
    return value


def start_upscale_ng(source_job_id: str, *, steps=3, keep=1.0, seed=1234):
    """Queue a 2x upscale of a finished job's current film. Raises LookupError / ValueError."""
    src = h3_reroll.resolve_source_ng(source_job_id)
    if not src.mp4_path or not Path(src.mp4_path).is_file():
        raise ValueError("that render has no video to upscale")
    steps = int(_clean(steps, MIN_STEPS, MAX_STEPS, "steps"))
    keep = _clean(keep, MIN_KEEP, MAX_KEEP, "keep")
    clip = Path(src.mp4_path)
    frames = frames_of_ng(clip)
    python = ltx_python_ng()
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"missing {SCRIPT.name}")

    def run(job) -> str:
        out = job.job_dir / (clip.stem + "_x2.mp4")
        ltx._unload_gemma_server_ng(job.append_log)
        cmd = [python, str(SCRIPT), str(clip), "-o", str(out), "--steps", str(steps),
               "--keep", f"{keep:g}", "--seed", str(int(seed))]
        for attempt in range(1, ATTEMPTS + 1):
            job.append_log(f"[upscale] attempt {attempt}/{ATTEMPTS}: " + " ".join(cmd[1:]))
            seen = []
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            job._proc = proc
            for line in proc.stdout:
                job.append_log(line.rstrip())
                seen.append(line)
            code = proc.wait()
            if job.cancelled:
                raise RuntimeError("cancelled")
            if code == 0 and out.is_file():
                return str(out)
            # The Metal watchdog kills a long GPU command when the screen needs the GPU; it is not a
            # fault in the clip, so the same run is tried again.
            if WATCHDOG_MARK in "".join(seen) and attempt < ATTEMPTS:
                job.append_log("[upscale] the Metal watchdog stopped the GPU job; trying again")
                continue
            tail = " | ".join(job.log_tail(4))
            raise RuntimeError(f"the upscale failed (exit {code}): {tail}")

    return h3_jobs.start_task_job_ng(
        "upscale", f"2x upscale of {clip.name}", run,
        meta={"source": source_job_id, "frames": frames, "estimate_min": estimate_minutes_ng(frames),
              "steps": steps})
