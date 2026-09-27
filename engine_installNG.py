"""Generic background "install this model engine" job -- the framework
music_installNG.py / ltx_installNG.py / h3_installNG.py plug their own
per-engine step sequences into (clone+pin, venv, deps, weights).

Twin of music_jobsNG.py's job/queue/worker shape (dataclass with derived
status from queued_at/dispatched_at/finished_at, log_lines+lock,
append_log/log_tail, a ring-buffer of finished jobs) -- reused here rather
than invented fresh so an install job polls exactly the way a render job
already does. The one addition is `step`: an install is a short, ordered
pipeline (not "render this one thing"), so the poll response can say which
stage it's on for a progress readout.

No retry-per-step logic: a failing step stops the whole job with `error`
set, same as Phosphene's own installers -- re-running the job (POST the
install route again) is the retry path, and every step here (git clone,
`uv venv`, weight fetch) is written to be safe to re-run, matching what it
was ported from.
"""

from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


def run_logged(cmd: list, *, cwd: Optional[str] = None, env: Optional[dict] = None,
                on_log: Optional[Callable[[str], None]] = None) -> None:
    """Runs `cmd`, piping stdout+stderr line-by-line into on_log (same
    convention as hidream_engine.py's own subprocess handling), raising
    RuntimeError with the tail of output on a non-zero exit. No timeout --
    unlike a render, an install step (clone, venv build, weight download)
    has no fixed expected duration; the job is only ever killed by the
    process exiting."""
    if on_log:
        on_log(f"$ {' '.join(cmd)}")
    tail: list = []
    with subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if on_log:
                    on_log(line)
                tail.append(line)
                if len(tail) > 30:
                    tail.pop(0)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"{cmd[0]} exited {rc}. Last output:\n" + "\n".join(tail))

_MAX_JOBS_KEPT = 20  # same ring-buffer cap as every other *_jobsNG.py


def mac_apple_silicon_ok() -> Tuple[bool, Optional[str]]:
    """Every engine this framework installs is an MLX port -- Apple's own
    array framework, Metal-only, no Windows/Linux/Intel-Mac build exists
    upstream for any of them (see HIDREAM_SETUP.md's own "Apple Silicon
    Mac only" section, which every one of these engines shares). Fail
    fast with a plain explanation rather than letting a clone+multi-GB
    download run to completion on a platform the venv step would only
    fail on anyway.

    Returns (ok, reason) -- reason is None when ok."""
    if platform.system() != "Darwin":
        return False, (
            f"This engine is an MLX (Apple Silicon) port and cannot run on "
            f"{platform.system()}. Nothing to install here.")
    if platform.machine() != "arm64":
        return False, (
            "This engine is an MLX (Apple Silicon) port and cannot run on "
            "an Intel Mac. Nothing to install here.")
    return True, None


@dataclass
class InstallStep:
    name: str  # "clone" | "venv" | "deps" | "weights" -- shown as job.step
    run: Callable[[Callable[[str], None]], None]  # (on_log) -> None; raises on failure


@dataclass
class InstallJob:
    job_id: str
    engine: str  # "music" | "ltx" | "h3" -- for logging/display only
    steps: List[InstallStep] = field(repr=False)
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    step: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    log_lines: list = field(default_factory=list)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        job.dispatched_at = time.time()
        try:
            for install_step in job.steps:
                job.step = install_step.name
                job.append_log(f"--- {install_step.name} ---")
                install_step.run(job.append_log)
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.error = str(e)
            job.error_type = type(e).__name__
            job.append_log(f"FAILED at step {job.step!r}: {e}")
        finally:
            job.finished_at = time.time()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker_loop, name="engine-install-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


def start_install_job(engine: str, steps: List[InstallStep]) -> InstallJob:
    """Validates the platform preflight synchronously (raises before any
    thread starts, same convention as start_music_job_ng's own upfront
    validation) and enqueues. Callers build `steps` from their own
    per-engine recipe module."""
    ok, reason = mac_apple_silicon_ok()
    if not ok:
        raise ValueError(reason)

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job = InstallJob(job_id=job_id, engine=engine, steps=steps)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_install_job(job_id: str) -> Optional[InstallJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def install_job_status(job_id: str) -> dict:
    job = get_install_job(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")

    if job.dispatched_at is None and job.finished_at is None:
        status = "queued"
    elif job.finished_at is None:
        status = "installing"
    elif job.error:
        status = "failed"
    else:
        status = "completed"

    return {
        "job_id": job.job_id,
        "engine": job.engine,
        "status": status,
        "step": job.step,
        "error": job.error,
        "error_type": job.error_type,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "log_tail": job.log_tail(40),
    }
