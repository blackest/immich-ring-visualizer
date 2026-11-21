"""immichRingNG -- "Sharp 2x": a quick 2x sharpen of a finished H3 film on the Neural Engine (see vendor/pipersr_x2).

PiperSR is a small image model, so this keeps the picture as it is and only crisps the edges (with halos limited); a few seconds for a 5 s
clip. It is not the same as the LTX upscale (h3_upscaleNG.py), which redraws the picture and takes minutes. It runs as a
subprocess on the folder's own env, one at a time in the H3 queue like the other tasks (it uses no GPU, but the queue is
simpler that way and the clip it reads may be the one a render is still writing).

Licence: PiperSR's code is AGPL-3.0 and its model is free for personal and academic use with attribution; commercial use
needs the author's licence. The runner prints the attribution line into the job log.
"""

import os
import subprocess
from pathlib import Path

import h3_jobsNG as h3_jobs
import h3_rerollNG as h3_reroll
from h3_upscaleNG import frames_of_ng

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "vendor" / "pipersr_x2" / "pipersr_x2_run.py"
PYTHON = ROOT / "vendor" / "pipersr_x2" / ".venv" / "bin" / "python"
SECONDS_PER_FRAME = 0.15   # measured on this M2 Max: 124 frames at 640x384 in 18 s
FRAMES_PER_WINDOW = 124
ATTRIBUTION = "Powered by PiperSR from ModelPiper - https://modelpiper.com"
DEFAULT_STRENGTH = 1.0   # the halo-limited PiperSR picture; lower mixes in the plain resize (0 = plain resize)


def pipersr_python_ng() -> str:
    """The python of the PiperSR env; says how to build it if it is missing."""
    override = os.environ.get("RINGVIZ_PIPERSR_PYTHON")
    cand = Path(override) if override else PYTHON
    if not cand.is_file():
        raise FileNotFoundError(f"no PiperSR env at {cand} (run vendor/pipersr_x2/build_venv.sh)")
    return str(cand)


def clean_strength_ng(value) -> float:
    """0 (plain resize) to 1 (pure PiperSR); raises ValueError."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("strength must be a number") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"strength must be 0 to 1 (got {value:g})")
    return value


def estimate_minutes_ng(frames: int) -> float:
    return round(SECONDS_PER_FRAME * max(1, frames) / 60, 1)


def start_sharp_ng(source_job_id: str, *, strength=DEFAULT_STRENGTH):
    """Queue a 2x sharpen of a finished job's current film. Raises LookupError / ValueError / FileNotFoundError."""
    strength = clean_strength_ng(strength)
    src = h3_reroll.resolve_source_ng(source_job_id)
    if not src.mp4_path or not Path(src.mp4_path).is_file():
        raise ValueError("that render has no video to sharpen")
    clip = Path(src.mp4_path)
    frames = frames_of_ng(clip)
    python = pipersr_python_ng()
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"missing {SCRIPT.name}")

    def run(job) -> str:
        out = job.job_dir / (clip.stem + "_sr2x.mp4")
        cmd = [python, str(SCRIPT), str(clip), "-o", str(out), "--strength", f"{strength:g}"]
        job.append_log("[sharp] " + " ".join(cmd[1:]))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        job._proc = proc
        for line in proc.stdout:
            job.append_log(line.rstrip())
        code = proc.wait()
        if job.cancelled:
            raise RuntimeError("cancelled")
        if code == 0 and out.is_file():
            return str(out)
        raise RuntimeError(f"the sharpen failed (exit {code}): " + " | ".join(job.log_tail(4)))

    return h3_jobs.start_task_job_ng(
        "sharp", f"Sharp 2x of {clip.name}", run,
        meta={"source": source_job_id, "frames": frames, "estimate_min": estimate_minutes_ng(frames),
              "strength": strength})
