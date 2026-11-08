"""NG job queue for the soft-finish pass (see finish_engineNG.py).

Same FIFO shape as speech_jobsNG.py / h3_jobsNG.py: own worker thread, POST returns a job id at once, the page
polls, list_jobs_ng() lets a page rebuild its list. The source is always an H3 job's finished mp4, looked up
server-side by job id (a client never names a path); the result is written beside it as
`<name>_finished_<jobid>.mp4`, so the original is never touched. One finish at a time: ComfyUI runs the
frames through torch and two at once would only fight over memory.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import finish_engineNG as finish
import h3_rerollNG as h3_reroll

_MAX_JOBS_KEPT = 30


@dataclass
class FinishJobNG:
    job_id: str
    h3_job_id: str
    src_path: str
    out_path: str
    params: dict
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    cancelled: bool = False
    result_path: Optional[str] = None
    log_lines: list = field(default_factory=list)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, line: str) -> None:
        with self._log_lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-100:]


_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _prune_old_jobs_locked() -> None:
    if len(_JOBS) <= _MAX_JOBS_KEPT:
        return
    done = sorted((j for j in _JOBS.values() if j.finished_at is not None), key=lambda j: j.finished_at)
    while len(_JOBS) > _MAX_JOBS_KEPT and done:
        _JOBS.pop(done.pop(0).job_id, None)   # the record only; the finished mp4 stays beside its source


def run_finish_job_ng(job: FinishJobNG) -> None:
    """Run one job to completion on the calling thread (the worker's body; also what Talk calls inline)."""
    job.dispatched_at = time.time()
    try:
        if job.cancelled:
            raise finish.FinishCancelled("cancelled before it started")
        result = finish.finish_video_ng(job.src_path, job.out_path, job.params, on_log=job.append_log,
                                        should_cancel=lambda: job.cancelled)
        job.result_path = result["path"]
    except finish.FinishCancelled as e:
        job.error, job.error_type = str(e) or "cancelled", "FinishCancelled"
    except Exception as e:  # noqa: BLE001 -- job.error is the report
        job.error, job.error_type = str(e), type(e).__name__
    finally:
        job.finished_at = time.time()


def _worker_loop() -> None:
    while True:
        job = _JOBS.get(_QUEUE.get())
        if job is not None:
            run_finish_job_ng(job)


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        threading.Thread(target=_worker_loop, name="finish-job-worker-ng", daemon=True).start()
        _WORKER_STARTED = True


def make_job_ng(h3_job_id: str, params: Optional[dict] = None) -> FinishJobNG:
    """Validate and build (not queue) a job for a finished H3 clip. Used directly by Talk, which runs it inline."""
    clean = finish.normalize_params_ng(params)
    try:
        h3_job = h3_reroll.resolve_source_ng(h3_job_id)    # the live job, or its saved settings after a restart
    except ValueError as e:
        raise finish.FinishRequestError(str(e)) from e
    src = Path(h3_job.mp4_path or "")
    if not src.is_file():
        raise finish.FinishRequestError("that render has no finished clip to work on")
    job_id = uuid.uuid4().hex[:12]
    out = src.with_name(f"{src.stem}_finished_{job_id[:6]}.mp4")
    return FinishJobNG(job_id=job_id, h3_job_id=h3_job_id, src_path=str(src), out_path=str(out), params=clean)


def start_finish_job_ng(h3_job_id: str, params: Optional[dict] = None) -> FinishJobNG:
    job = make_job_ng(h3_job_id, params)
    _ensure_worker_started()
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job.job_id)
    return job


def get_job_ng(job_id: str) -> Optional[FinishJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_finish_job_ng(job_id: str) -> bool:
    job = get_job_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
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
        "job_id": job.job_id, "status": status, "h3_job_id": job.h3_job_id, "params": job.params,
        "error": job.error, "error_type": job.error_type, "cancelled": bool(job.cancelled),
        "queued_at": job.queued_at, "dispatched_at": job.dispatched_at, "finished_at": job.finished_at,
        "video_url": f"/api/ng/finish/jobs/{job.job_id}/video" if status == "completed" else None,
        "download_url": f"/api/ng/finish/jobs/{job.job_id}/download" if status == "completed" else None,
    }


def list_jobs_ng() -> list:
    with _JOBS_LOCK:
        ids = sorted(_JOBS, key=lambda jid: _JOBS[jid].queued_at or 0)
    out = []
    for jid in ids:
        try:
            out.append(job_status_ng(jid))
        except LookupError:
            continue
    return out
