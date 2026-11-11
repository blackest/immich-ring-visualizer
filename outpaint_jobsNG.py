"""immichRingNG -- the "Make vertical" queue: a finished Animate clip made tall (9:16, 4:5 or square) with the
bands above and below painted by the AI fill (outpaint_engineNG.py).

One job = one source clip + one shape. Jobs run one at a time on a worker thread, and never while an Animate, H3 or
Klein render holds the GPU. The finished mp4 lives in its own folder under OUTPAINT_DIR; the job records are in
memory (like the Animate queue), so a restart forgets the list but not the files.
"""

import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import h3_jobsNG
import klein_jobsNG
import outpaint_engineNG as engine
import video_jobsNG
from configNG import OUTPAINT_DIR

MAX_JOBS_KEPT = 20
_JOBS: dict = {}
_LOCK = threading.Lock()
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False


def running_ng() -> bool:
    """A Make vertical job is on the GPU right now (H3 waits for this)."""
    with _LOCK:
        return any(j["status"] == "running" for j in _JOBS.values())


h3_jobsNG.add_gpu_guard_ng(running_ng)


def _other_gpu_work() -> bool:
    return bool(h3_jobsNG.busy_ng() or video_jobsNG.busy_ng() or klein_jobsNG.running_ng())


def _prune_locked() -> None:
    done = sorted((j for j in _JOBS.values() if j["finished_at"]), key=lambda j: j["finished_at"])
    while len(_JOBS) > MAX_JOBS_KEPT and done:
        old = done.pop(0)
        _JOBS.pop(old["id"], None)
        shutil.rmtree(Path(OUTPAINT_DIR) / old["id"], ignore_errors=True)


def start_job_ng(source_job_id: str, ratio: str) -> dict:
    """Queue a Make vertical job for a finished Animate clip. Raises LookupError (no such clip, or not finished)
    or ValueError (a bad shape, or a clip that cannot be made taller)."""
    src = video_jobsNG.get_job_ng(source_job_id)
    if src is None:
        raise LookupError("that clip is no longer in the Animate list")
    if not src.mp4_path or not Path(src.mp4_path).is_file():
        raise LookupError("that clip has not finished rendering")
    health = engine.outpaint_health_ng()
    if not health["ok"]:
        raise ValueError("; ".join(health["problems"]))
    engine.plan_ng(src.mp4_path, ratio)             # raises ValueError for a bad shape or an already-tall clip
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(OUTPAINT_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {"id": job_id, "source_job_id": source_job_id, "source": src.mp4_path, "ratio": ratio,
           "prompt": src.prompt or "", "status": "queued", "error": None, "log": [], "cancelled": False,
           "output": str(job_dir / "vertical.mp4"), "queued_at": time.time(), "started_at": None,
           "finished_at": None, "seconds": None}
    with _LOCK:
        _JOBS[job_id] = job
        _prune_locked()
    _ensure_worker()
    _QUEUE.put(job_id)
    return public_ng(job)


def _log(job: dict, line: str) -> None:
    job["log"] = (job["log"] + [line])[-60:]


def _run(job: dict) -> None:
    while not job["cancelled"] and _other_gpu_work():
        time.sleep(2)
    if job["cancelled"]:
        job.update(status="cancelled", error="cancelled before it started", finished_at=time.time())
        return
    job.update(status="running", started_at=time.time())
    try:
        result = engine.outpaint_vertical_ng(job["source"], job["output"], job["ratio"], job["prompt"],
                                             on_log=lambda l: _log(job, l), should_stop=lambda: job["cancelled"])
        job.update(status="done", seconds=result.get("seconds"))
    except engine.OutpaintCancelled:
        job.update(status="cancelled", error="stopped")
    except Exception as exc:                        # noqa: BLE001 -- job["error"] is the report
        job.update(status="failed", error=str(exc))
    finally:
        job["finished_at"] = time.time()


def _worker() -> None:
    while True:
        job = _JOBS.get(_QUEUE.get())
        if job is not None:
            _run(job)


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _LOCK:
        if _WORKER_STARTED:
            return
        threading.Thread(target=_worker, name="outpaint-worker-ng", daemon=True).start()
        _WORKER_STARTED = True


def get_ng(job_id: str) -> Optional[dict]:
    with _LOCK:
        return _JOBS.get(job_id)


def cancel_ng(job_id: str) -> bool:
    """Stop a queued or running job. False when there is no such job or it has already finished."""
    job = get_ng(job_id)
    if job is None or job["finished_at"]:
        return False
    job["cancelled"] = True
    return True


def public_ng(job: dict) -> dict:
    return {"id": job["id"], "source_job_id": job["source_job_id"], "ratio": job["ratio"], "status": job["status"],
            "error": job["error"], "log_tail": job["log"][-12:], "seconds": job["seconds"],
            "queued_at": job["queued_at"], "started_at": job["started_at"], "finished_at": job["finished_at"],
            "video_url": f"/api/ng/outpaint/jobs/{job['id']}/video" if job["status"] == "done" else None}
