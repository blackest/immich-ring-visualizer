"""immichRingNG -- the Klein picture queue.

One job = one prompt, up to three labelled reference pictures (subject, location, extra) and N
variations. Jobs run one at a time on a worker thread, and never while an H3 render holds the GPU or
a HiDream call is running. Each job keeps its pictures and a `job.json` in its own folder under
KLEIN_DIR, so the gallery survives a restart. klein_engineNG.py runs the model; routes/kleinNG.py is
the HTTP side.
"""

import json
import os
import queue
import random
import threading
import time
import uuid
from pathlib import Path

import h3_jobsNG
import klein_engineNG as engine
from character_sheetNG import _SHEET_LOCK_NG
from configNG import KLEIN_DIR

SLOT_NAMES = ("subject", "location", "extra")
_JOBS: dict = {}
_ORDER: list = []
_LOCK = threading.Lock()
_QUEUE: "queue.Queue" = queue.Queue()
_WORKER = None


def _save(job: dict) -> None:
    path = Path(KLEIN_DIR) / job["id"] / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _run(job: dict) -> None:
    job_dir = Path(KLEIN_DIR) / job["id"]
    try:
        while h3_jobsNG.busy_ng() is not None:  # a video render owns the GPU: wait for it
            job["status"] = "waiting"
            if job["cancel"]:
                raise engine.KleinCancelled("stopped")
            time.sleep(2)
        with _SHEET_LOCK_NG:
            job["status"] = "running"
            job["started_at"] = time.time()
            engine.generate_klein_ng(
                job["prompt"], job["n"], job["width"], job["height"], job_dir / "out", job["seed"],
                refs=[r["path"] for r in job["refs"]], model=job["model"],
                on_log=lambda line: job.__setitem__("log", (job["log"] + [line])[-30:]),
                should_stop=lambda: job["cancel"],
                on_result=lambda r: job["results"].append(
                    {"file": Path(r["png_path"]).name, "seed": r["seed"], "width": r["width"],
                     "height": r["height"], "seconds": r["elapsed_seconds"]}))
        job["status"] = "done"
    except engine.KleinCancelled:
        job["status"] = "cancelled"
    except Exception as exc:
        job["status"], job["error"] = "error", str(exc)
    job["finished_at"] = time.time()
    _save(job)


def _worker() -> None:
    while True:
        job = _QUEUE.get()
        if job["cancel"]:
            job["status"] = "cancelled"
            _save(job)
        else:
            _run(job)


def _enqueue(prompt, n, width, height, seed, model, refs, job_id=None) -> dict:
    global _WORKER
    job = {"id": job_id or uuid.uuid4().hex[:12], "prompt": prompt.strip(), "n": n, "width": width,
           "height": height, "seed": int(seed) if seed not in (None, "") else random.randrange(2 ** 31),
           "model": model, "refs": refs, "status": "queued", "error": None, "results": [], "log": [],
           "created_at": time.time(), "cancel": False}
    (Path(KLEIN_DIR) / job["id"]).mkdir(parents=True, exist_ok=True)
    with _LOCK:
        _JOBS[job["id"]] = job
        _ORDER.append(job["id"])
        if _WORKER is None:
            _WORKER = threading.Thread(target=_worker, daemon=True)
            _WORKER.start()
    _save(job)
    _QUEUE.put(job)
    return job


def new_job_dir_ng() -> tuple:
    """(job id, folder for its reference pictures) -- made before the pictures are saved."""
    job_id = uuid.uuid4().hex[:12]
    folder = Path(KLEIN_DIR) / job_id / "refs"
    folder.mkdir(parents=True, exist_ok=True)
    return job_id, folder


def start_ng(job_id: str, prompt, n, width, height, seed, model, refs) -> dict:
    """submit_ng for a job whose folder new_job_dir_ng already made."""
    engine.validate_ng(width, height, refs, n)
    if not prompt.strip():
        raise ValueError("write a prompt: describe the whole picture you want")
    health = engine.klein_health_ng()
    if not health["ready"] or model not in health["models"]:
        raise RuntimeError("Klein is not ready: install FLUX 3 and download " + model + " in Phosphene")
    return _enqueue(prompt, n, width, height, seed, model, refs, job_id=job_id)


def get_ng(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is not None:
        return job
    path = Path(KLEIN_DIR) / job_id / "job.json"
    if "/" in job_id or not path.is_file():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    if job["status"] in ("queued", "running", "waiting"):  # the server restarted under it
        job["status"], job["error"] = "error", "the server restarted while this was running"
    return job


def recent_ng(limit: int = 30) -> list:
    """Newest first, from disk (so a restart keeps the gallery)."""
    root = Path(KLEIN_DIR)
    found = []
    for path in root.glob("*/job.json") if root.is_dir() else []:
        job = get_ng(path.parent.name)
        if job is not None:
            found.append(job)
    found.sort(key=lambda j: j["created_at"], reverse=True)
    return found[:limit]


def cancel_ng(job_id: str) -> bool:
    job = get_ng(job_id)
    if job is None or job["status"] in ("done", "error", "cancelled"):
        return False
    job["cancel"] = True
    return True
