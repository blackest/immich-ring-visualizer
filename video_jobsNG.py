"""NG job queue for LTX-2.5 image-to-video renders (see ltx_engineNG.py).

Twin of sheet_jobsNG.py's queue design, trimmed to what a single
image+prompt+duration job needs: no per-shot list, no preset
resolution, no character registration. A route hands this module raw
upload bytes; start_video_job_ng() copies them into a private per-job
directory under VIDEOGEN_DIR before returning, so the route's own
tempfile can be deleted the instant this call returns -- the job owns
its reference image for its whole lifetime, not just at enqueue time.

Same reasoning as sheet_jobsNG for why this is a FIFO queue and not a
busy-error lock: LTX and HiDream are both one-subprocess-at-a-time
engines fighting over the same GPU, but two video jobs queued back to
back should each get their turn rather than the second one bouncing.
Deliberately its OWN worker thread/queue, independent of
sheet_jobsNG's -- keeps this file small and self-contained (see
[[modular-engine-files]] in memory), at the cost of not knowing about
sheet-render jobs also running concurrently on the same GPU. Fine for
now; a shared cross-engine lock can be added later if that turns out
to matter in practice.
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

import ltx_engineNG as ltx
from configNG import VIDEOGEN_DIR

_MAX_JOBS_KEPT = 20  # ring-buffer cap, same reasoning as sheet_jobsNG


@dataclass
class VideoJobNG:
    job_id: str
    prompt: str
    duration_s: float
    seed: Optional[int]
    job_dir: Path
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
            job.error_type = "VideoJobCancelled"
            continue
        job.dispatched_at = time.time()
        image_path = next(job.job_dir.glob("ref.*"), None)
        try:
            if image_path is None:
                raise FileNotFoundError("reference image went missing before render")
            result = ltx.generate_ltx_video_ng(
                prompt=job.prompt, image_path=str(image_path),
                duration_s=job.duration_s, output_dir=job.job_dir,
                seed=job.seed, config=ltx.LtxConfig(),
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
        t = threading.Thread(target=_worker_loop, name="video-job-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


def start_video_job_ng(src_image_path: str, prompt: str, duration_s: float,
                        seed: Optional[int] = None) -> VideoJobNG:
    """Copies src_image_path into a fresh per-job directory (the caller's
    own copy, e.g. a route's upload tempfile, is safe to delete right
    after this returns) and enqueues the render. Raises synchronously
    for bad input; engine/subprocess failure surfaces later through
    job_status_ng()."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if not (0.5 <= duration_s <= 8.0):
        raise ValueError("duration_s must be between 0.5 and 8.0 seconds")
    src = Path(src_image_path)
    if not src.is_file():
        raise FileNotFoundError(f"reference image not found at {src_image_path}")

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(VIDEOGEN_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() if src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".jpg"
    shutil.copyfile(src, job_dir / f"ref{ext}")

    job = VideoJobNG(job_id=job_id, prompt=prompt, duration_s=duration_s,
                      seed=seed, job_dir=job_dir)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_job_ng(job_id: str) -> Optional[VideoJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_video_job_ng(job_id: str) -> bool:
    """Removes a job from the render queue. A job still waiting its turn
    is simply flagged and the worker skips it when it comes up. A job
    already rendering gets its LTX subprocess killed (same signal path
    as the engine's own timeout watchdog, so it surfaces as the usual
    VideoJobCancelled). Returns False if the job doesn't exist or has
    already finished -- nothing left to cancel."""
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
        "error": job.error,
        "error_type": job.error_type,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "log_tail": job.log_tail(40),
        "video_url": (f"/api/ng/videogen/jobs/{job.job_id}/video"
                      if status == "completed" else None),
    }
