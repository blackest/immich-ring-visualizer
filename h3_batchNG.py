"""immichRingNG -- a scene of several 5 s windows, each rendered with its own reference pictures.

Reference pictures (h3_referencesNG.py) work on one 5 s window, and a chained render carries none.
So a scene is a batch: one render per window, each with the pictures every window shares, the
pictures for that window, and (optionally) the last frame of the window before it as one more
picture. Windows run one after another because a window can need the one before it. When the last
one finishes the clips are joined (h3_stitchNG.py) and the film is filed in the H3 job log.

A window's render is an ordinary job in the H3 queue, so it shows in the queue and can be stopped.
The next window is queued from the finished job's `on_done` callback, which keeps this module free
of threads of its own. Everything outside this file is passed in, so the tests need no GPU.
"""

import shutil
import threading
import time
import uuid
from pathlib import Path

import h3_film_logNG
import h3_jobsNG
import h3_referencesNG as refs
import h3_rerollNG
import h3_stitchNG as stitch
from configNG import H3GEN_DIR

MIN_WINDOWS = 2
MAX_WINDOWS = 12
WINDOW_SECONDS = 5.0

_BATCHES: dict = {}
_LOCK = threading.Lock()


def _check_pictures(windows, shared, carry: bool) -> None:
    """Refuse a window the runner would refuse, before anything renders."""
    for n, w in enumerate(windows):
        files = [f"x{ext}" for _b, ext in list(shared) + list(w["refs"])]
        extra = 1 if carry and n > 0 else 0
        pictures = sum(refs.kind_of_ng(f) == "image" for f in files) + extra
        clips = sum(refs.kind_of_ng(f) == "video" for f in files)
        if pictures > refs.MAX_PICTURES:
            raise ValueError(f"window {n + 1} has {pictures} pictures; the most one render takes is "
                             f"{refs.MAX_PICTURES}" + (" (the carried last frame counts as one)" if extra else ""))
        if clips > refs.MAX_CLIPS:
            raise ValueError(f"window {n + 1} has {clips} clips; the most one render takes is {refs.MAX_CLIPS}")


def start_batch_ng(windows, *, shared=(), carry: bool = True, joins=None, settings=None,
                   submit=None, last_frame=None, stitcher=None, recorder=None) -> dict:
    """`windows`: [{"prompt", "refs": [(bytes, ext)]}], in order. `shared`: [(bytes, ext)] pictures for
    every window, numbered first. `carry`: add the previous window's last frame to each later window
    (numbered last). `settings`: keyword arguments for the job queue (model, width, height, tristep,
    flex, steps, timeout_s, reference_size). Queues window 1 and returns the batch."""
    submit = submit or h3_jobsNG.start_h3_job_ng
    last_frame = last_frame or h3_rerollNG.last_frame_png_ng
    stitcher = stitcher or stitch.stitch_windows_ng
    recorder = recorder or h3_film_logNG.record_film_ng
    windows = [{"prompt": str(w.get("prompt") or "").strip(), "refs": list(w.get("refs") or [])} for w in windows or []]
    if not MIN_WINDOWS <= len(windows) <= MAX_WINDOWS:
        raise ValueError(f"a scene needs {MIN_WINDOWS} to {MAX_WINDOWS} windows (got {len(windows)})")
    for n, w in enumerate(windows):
        if not w["prompt"]:
            raise ValueError(f"window {n + 1} has no prompt")
    settings = dict(settings or {})
    settings.setdefault("model", "h3ref")
    if settings["model"] != "h3ref":
        raise ValueError("references need the Ref2VA model (model='h3ref')")
    shared = list(shared)
    _check_pictures(windows, shared, carry)
    joins = stitch.normalise_joins_ng(joins, len(windows) - 1)

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = Path(H3GEN_DIR) / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch = {"id": batch_id, "dir": batch_dir, "windows": windows, "shared": shared, "carry": bool(carry),
             "joins": joins, "settings": settings, "jobs": [], "clips": [], "film": None, "error": None,
             "cancelled": False, "started_at": time.time(), "submit": submit, "last_frame": last_frame,
             "stitcher": stitcher, "recorder": recorder}
    with _LOCK:
        _BATCHES[batch_id] = batch
    try:
        _submit_window(batch, 0, None)
    except Exception:
        with _LOCK:
            _BATCHES.pop(batch_id, None)
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise
    return batch


def _submit_window(batch: dict, n: int, carried) -> None:
    w = batch["windows"][n]
    pictures = list(batch["shared"]) + list(w["refs"])
    if carried is not None:
        pictures.append((carried, ".png"))
    job = batch["submit"](
        None, ".jpg", w["prompt"], WINDOW_SECONDS, None, references=pictures,
        on_done=lambda finished, n=n: _window_done(batch, n, finished), **batch["settings"])
    batch["jobs"].append(job.job_id)


def _window_done(batch: dict, n: int, job) -> None:
    """Runs in the queue's worker after window n+1 rendered. Never raises: a failure here is the
    batch's error, and must not turn a good render into a failed one."""
    try:
        clip = batch["dir"] / f"w{n + 1}.mp4"
        shutil.copyfile(job.mp4_path, clip)
        batch["clips"].append(str(clip))
        if batch["cancelled"]:
            return
        if n + 1 < len(batch["windows"]):
            carried = batch["last_frame"](clip) if batch["carry"] else None
            _submit_window(batch, n + 1, carried)
        else:
            _finish(batch)
    except Exception as e:  # noqa: BLE001
        batch["error"] = f"{type(e).__name__}: {e}"


def _finish(batch: dict) -> None:
    out = batch["dir"] / "film.mp4"
    done = batch["stitcher"](batch["clips"], batch["joins"], out)
    s = batch["settings"]
    prompts = [w["prompt"] for w in batch["windows"]]
    notes = (f"Scene of {len(prompts)} windows, each rendered with its own references"
             + (", the last frame of one carried into the next" if batch["carry"] else "")
             + ". Parts: " + ", ".join(batch["jobs"]))
    log = "".join(f"=== window {i + 1} (job {j}) ===\n{p}\n\n" for i, (j, p) in enumerate(zip(batch["jobs"], prompts)))
    batch["film"] = batch["recorder"](
        mp4_path=out, prompt="\n\n---\n\n".join(prompts), duration_s=done["seconds"], model=s["model"],
        width=s.get("width") or 0, height=s.get("height") or 0, steps=s.get("steps") or 0, notes=notes,
        log_text=log, queued_at=batch["started_at"])


def get_batch_ng(batch_id: str):
    with _LOCK:
        return _BATCHES.get(batch_id)


def cancel_batch_ng(batch_id: str) -> bool:
    batch = get_batch_ng(batch_id)
    if batch is None or batch["film"] is not None:
        return False
    batch["cancelled"] = True
    for job_id in batch["jobs"]:
        h3_jobsNG.cancel_h3_job_ng(job_id)
    return True


def batch_status_ng(batch_id: str, job_status=None) -> dict:
    """Where the scene is: each window's state, and the film once it is filed in the job log."""
    job_status = job_status or h3_jobsNG.job_status_ng
    batch = get_batch_ng(batch_id)
    if batch is None:
        raise LookupError(f"no such batch {batch_id!r}")
    windows = []
    for n in range(len(batch["windows"])):
        if n < len(batch["jobs"]):
            try:
                state = job_status(batch["jobs"][n])["status"]
            except LookupError:
                state = "completed" if n < len(batch["clips"]) else "failed"
            windows.append({"n": n + 1, "job_id": batch["jobs"][n], "status": state})
        else:
            windows.append({"n": n + 1, "job_id": None, "status": "waiting"})
    failed = next((w for w in windows if w["status"] == "failed"), None)
    if batch["film"]:
        status = "completed"
    elif batch["cancelled"]:
        status = "cancelled"
    elif failed or batch["error"]:
        status = "failed"
    else:
        status = "running"
    film = batch["film"]
    return {"batch_id": batch["id"], "status": status, "windows": windows,
            "error": batch["error"] or (f"window {failed['n']} failed" if failed else None),
            "film": ({"job_id": film["job_id"], "date": film["date"],
                      "video_url": f"/api/ng/h3/logs/{film['date']}/{film['job_id']}/video"} if film else None)}

