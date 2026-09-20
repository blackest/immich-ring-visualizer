"""NG job queue for MiniMax-H3 prompt/image-to-video renders (see
h3_engineNG.py).

Twin of video_jobsNG.py's queue design -- same reasoning throughout
(FIFO queue not a busy-error lock, own worker thread, ring-buffer
eviction), just pointed at h3_engineNG instead of ltx_engineNG. Kept
as its own file rather than folded into video_jobsNG.py: H3 is a
different subprocess/venv/model with its own field (model: "h3" |
"h3q8"), and per [[modular-engine-files]] a second engine gets a
sibling module, not a branch inside the first one's queue.
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h3_engineNG as h3
from configNG import H3GEN_DIR

_MAX_JOBS_KEPT = 20  # ring-buffer cap, same reasoning as video_jobsNG


@dataclass
class H3JobNG:
    job_id: str
    prompt: str
    duration_s: float
    seed: Optional[int]
    job_dir: Path
    model: h3.H3Model = "h3q8"
    has_image: bool = True  # False = text-to-video, no keyframe
    width: int = h3.H3_WIDTH
    height: int = h3.H3_HEIGHT
    steps: int = h3.H3_STEPS
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    mp4_path: Optional[str] = None
    log_lines: list = field(default_factory=list)
    cancelled: bool = False
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    def append_log(self, line: str) -> None:
        with self._log_lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-200:]

    def log_tail(self, n: int = 40) -> list:
        with self._log_lock:
            return list(self.log_lines[-n:])


_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _prune_old_jobs_locked() -> None:
    if len(_JOBS) <= _MAX_JOBS_KEPT:
        return
    finished = sorted(
        (j for j in _JOBS.values() if j.finished_at is not None),
        key=lambda j: j.finished_at)
    while len(_JOBS) > _MAX_JOBS_KEPT and finished:
        oldest = finished.pop(0)
        _JOBS.pop(oldest.job_id, None)
        shutil.rmtree(oldest.job_dir, ignore_errors=True)


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        if job.cancelled:
            job.finished_at = time.time()
            job.error = "cancelled before it started rendering"
            job.error_type = "H3JobCancelled"
            continue
        job.dispatched_at = time.time()
        image_path = next(job.job_dir.glob("ref.*"), None)
        try:
            if job.has_image and image_path is None:
                raise FileNotFoundError("keyframe image went missing before render")
            result = h3.generate_h3_video_ng(
                prompt=job.prompt, image_path=str(image_path) if image_path else None,
                duration_s=job.duration_s, output_dir=job.job_dir,
                seed=job.seed, config=h3.H3Config(model=job.model),
                width=job.width, height=job.height, steps=job.steps,
                on_log=job.append_log,
                on_proc_start=lambda p: setattr(job, "_proc", p))
            job.mp4_path = result["mp4_path"]
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.error = str(e)
            job.error_type = type(e).__name__
        finally:
            job.finished_at = time.time()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker_loop, name="h3-job-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


def start_h3_job_ng(image_bytes: Optional[bytes], image_ext: str, prompt: str,
                     duration_s: float, seed: Optional[int] = None,
                     model: h3.H3Model = "h3q8",
                     width: Optional[int] = None, height: Optional[int] = None,
                     steps: Optional[int] = None) -> H3JobNG:
    """Writes image_bytes directly into a fresh per-job directory under
    H3GEN_DIR -- same "this write IS the job's own copy" shape as
    video_jobsNG.start_video_job_ng (see its own docstring for the full
    reasoning). image_bytes may be None for a text-to-video job -- H3
    generates from the prompt alone, no --first-frame keyframe."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if model not in ("h3", "h3q8"):
        raise ValueError(f"model must be 'h3' or 'h3q8' (got {model!r})")
    h3._validate_h3_duration_ng(duration_s)

    width = h3.H3_WIDTH if width is None else int(width)
    height = h3.H3_HEIGHT if height is None else int(height)
    steps = h3.H3_STEPS if steps is None else int(steps)
    h3._validate_h3_dims_ng(width, height)

    has_image = image_bytes is not None

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(H3GEN_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if has_image:
        ext = image_ext.lower() if image_ext and image_ext.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".jpg"
        (job_dir / f"ref{ext}").write_bytes(image_bytes)

    job = H3JobNG(job_id=job_id, prompt=prompt, duration_s=duration_s,
                  seed=seed, job_dir=job_dir, model=model, has_image=has_image,
                  width=width, height=height, steps=steps)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_job_ng(job_id: str) -> Optional[H3JobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_h3_job_ng(job_id: str) -> bool:
    """Same semantics as video_jobsNG.cancel_video_job_ng: a queued job
    is flagged and skipped, a rendering job's subprocess is killed
    (surfaces as the usual H3JobCancelled)."""
    job = get_job_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
    if job.dispatched_at is not None and job._proc is not None:
        try:
            os.killpg(os.getpgid(job._proc.pid), signal.SIGKILL)
        except Exception:
            try:
                job._proc.kill()
            except Exception:
                pass
    return True


def job_status_ng(job_id: str) -> dict:
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")

    if job.dispatched_at is None and job.finished_at is None:
        status = "queued"
    elif job.finished_at is None:
        status = "rendering"
    elif job.error:
        status = "failed"
    else:
        status = "completed"

    return {
        "job_id": job.job_id,
        "status": status,
        "prompt": job.prompt,
        "duration_s": job.duration_s,
        "model": job.model,
        "width": job.width,
        "height": job.height,
        "steps": job.steps,
        "error": job.error,
        "error_type": job.error_type,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "log_tail": job.log_tail(40),
        "video_url": (f"/api/ng/h3/jobs/{job.job_id}/video"
                      if status == "completed" else None),
    }
