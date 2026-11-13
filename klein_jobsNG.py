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
import shutil
import random
import threading
import time
import uuid
from pathlib import Path

import h3_image_engineNG as h3image
import h3_jobsNG
import klein_engineNG as engine
import klein_outpaintNG as outpaint
from character_sheetNG import _SHEET_LOCK_NG
from configNG import KLEIN_DIR

# Phosphene's three named slots first, then plain numbered ones; Klein takes up to ten references.
SLOT_NAMES = ("subject", "location", "extra") + tuple(f"ref{n}" for n in range(4, engine.MAX_REFS + 1))
_JOBS: dict = {}
_ORDER: list = []
_LOCK = threading.Lock()
_QUEUE: "queue.Queue" = queue.Queue()
_WORKER = None
# A "model" value beside Klein's own: the job is made by MiniMax-H3-Image instead (text to picture, or an edit of the
# reference pictures). Its shape and quality ride in job["options"]; width and height stay 0 and the results carry the real size.
H3_IMAGE_MODEL = "h3image"


def running_ng() -> bool:
    """A Klein picture is on the GPU right now (H3 waits for this)."""
    with _LOCK:
        return any(j["status"] == "running" for j in _JOBS.values())


h3_jobsNG.add_gpu_guard_ng(running_ng)


def _save(job: dict) -> None:
    path = Path(KLEIN_DIR) / job["id"] / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _landed(job: dict, job_dir: Path, r: dict) -> None:
    """A picture is finished: for an outpaint job put the original back over it, then list it."""
    plan = job.get("outpaint")
    if plan and plan.get("keep_original"):
        outpaint.paste_original_ng(r["png_path"], job_dir / "source.png", plan["offset"])
    job["results"].append({"file": Path(r["png_path"]).name, "seed": r["seed"], "width": r["width"],
                           "height": r["height"], "seconds": r["elapsed_seconds"]})


def _capped_refs_ng(refs, job_dir: Path) -> tuple:
    """The reference paths to hand to Klein. A picture longer than engine.REF_MAX_SIDE on its long side is shrunk into
    a copy under <job>/refs_capped (rotation tag applied, so a phone photo stays upright); the upload itself is kept as
    it came. A picture that is small enough, or that cannot be read here, is passed through unchanged. Returns
    (paths, notes): one note per shrunk picture, for the job card."""
    out, notes = [], []
    for r in refs:
        path = Path(r["path"])
        try:
            from PIL import Image, ImageOps
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                if max(im.size) <= engine.REF_MAX_SIDE:
                    out.append(str(path))
                    continue
                scale = engine.REF_MAX_SIDE / max(im.size)
                before = im.size
                small = im.convert("RGB").resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))),
                                                 Image.LANCZOS)
            dest = job_dir / "refs_capped" / f"{path.stem}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            small.save(dest)
            out.append(str(dest))
            notes.append(f"{r.get('slot', path.stem)} shrunk from {before[0]}x{before[1]} to {small.size[0]}x{small.size[1]}")
        except Exception:  # noqa: BLE001 -- not a picture Pillow can read: let Klein report it as before
            out.append(str(path))
    return out, notes


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
            ref_paths, job["ref_notes"] = (([], []) if job["model"] == H3_IMAGE_MODEL
                                           else _capped_refs_ng(job["refs"], job_dir))
            on_log = lambda line: job.__setitem__("log", (job["log"] + [line])[-30:])  # noqa: E731
            if job["model"] == H3_IMAGE_MODEL:
                opts = job.get("options") or {}
                h3image.generate_h3_image_ng(
                    job["prompt"], job["n"], opts["shape"], opts["tier"], job_dir / "out", job["seed"],
                    refs=[r["path"] for r in job["refs"]], on_log=on_log, should_stop=lambda: job["cancel"],
                    on_result=lambda r: _landed(job, job_dir, r))
            else:
                engine.generate_klein_ng(
                    job["prompt"], job["n"], job["width"], job["height"], job_dir / "out", job["seed"],
                    refs=ref_paths, model=job["model"], options=job.get("options"),
                    on_log=on_log, should_stop=lambda: job["cancel"],
                    on_result=lambda r: _landed(job, job_dir, r))
        job["status"] = "done"
    except (engine.KleinCancelled, h3image.H3ImageCancelled):
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


def _enqueue(prompt, n, width, height, seed, model, refs, job_id=None, options=None, outpaint_plan=None) -> dict:
    global _WORKER
    job = {"id": job_id or uuid.uuid4().hex[:12], "prompt": prompt.strip(), "n": n, "width": width,
           "height": height, "seed": int(seed) if seed not in (None, "") else random.randrange(2 ** 31),
           "model": model, "refs": refs, "options": options or {}, "outpaint": outpaint_plan, "status": "queued", "error": None, "results": [], "log": [],
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


def start_ng(job_id: str, prompt, n, width, height, seed, model, refs, options=None, outpaint_plan=None) -> dict:
    """submit_ng for a job whose folder new_job_dir_ng already made."""
    if model == H3_IMAGE_MODEL:
        opts = options or {}
        paths = [r["path"] for r in refs]
        shape, tier = opts.get("shape") or ("match" if paths else "16:9"), opts.get("tier") or "draft"
        h3image.validate_ng(prompt, n, shape, tier, paths)
        health = h3image.h3_image_health_ng()
        if not health["ok"]:
            raise RuntimeError("MiniMax-H3-Image is not ready: " + "; ".join(health["problems"]))
        return _enqueue(prompt, n, 0, 0, seed, model, refs, job_id=job_id, options={"shape": shape, "tier": tier})
    options = engine.validate_options_ng(options, model.split("-")[0], refs)
    engine.validate_ng(width, height, refs, n, options.get("match_ref", False))
    if not prompt.strip():
        raise ValueError("write a prompt: describe the whole picture you want")
    health = engine.klein_health_ng()
    if not health["ready"] or model not in health["models"]:
        raise RuntimeError("Klein is not ready: install FLUX 3 and download " + model + " in Phosphene")
    return _enqueue(prompt, n, width, height, seed, model, refs, job_id=job_id, options=options, outpaint_plan=outpaint_plan)


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


def recent_ng(limit: int = 30, query: str = "", with_total: bool = False):
    """Newest first, from disk (so a restart keeps the gallery). `query` keeps jobs whose prompt, model or
    settings text contains it (case-insensitive); with_total also returns how many matched in all."""
    root = Path(KLEIN_DIR)
    found = []
    for path in root.glob("*/job.json") if root.is_dir() else []:
        job = get_ng(path.parent.name)
        if job is not None:
            found.append(job)
    q = (query or "").strip().lower()
    if q:
        found = [j for j in found if q in json.dumps([j["prompt"], j["model"], j.get("options"), j["id"]]).lower()]
    found.sort(key=lambda j: j["created_at"], reverse=True)
    return (found[:limit], len(found)) if with_total else found[:limit]


def delete_ng(job_id: str) -> str:
    """'deleted', 'missing' or 'active' (queued/running jobs are not deleted)."""
    job = get_ng(job_id)
    if job is None or "/" in job_id or ".." in job_id:
        return "missing"
    if job["status"] in ("queued", "running", "waiting"):
        return "active"
    with _LOCK:
        _JOBS.pop(job_id, None)
        if job_id in _ORDER:
            _ORDER.remove(job_id)
    shutil.rmtree(Path(KLEIN_DIR) / job_id, ignore_errors=True)
    return "deleted"


def cancel_ng(job_id: str) -> bool:
    job = get_ng(job_id)
    if job is None or job["status"] in ("done", "error", "cancelled"):
        return False
    job["cancel"] = True
    return True
