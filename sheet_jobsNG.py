"""NG twin of sheet_jobs.py -- background job queue for character-sheet
generation.

NG-only file, and the one module in this feature's NG port that is a
real behavioral rewrite rather than a rename-and-relink: the original
launches a new thread per job and relies on character_sheet.py's
_SHEET_LOCK failing fast (CharacterSheetBusyError, HTTP 429) if a
second render is requested while one is already running -- correct for
a single-request/response flow with no queue concept, wrong once two
tabs/characters (Tom, Mary, ...) can both hit Generate and should both
get their sheet, just serialized like a phone queue -- "we'll get to
you when we get to you", no busy errors, no need to report a numeric
position (a dedicated queue view covers that later).

So this module owns actual FIFO ordering: start_job_ng()/
start_reroll_ng() validate synchronously (same as the original -- bad
input, missing character, missing reference image all raise before
anything is queued) and then enqueue a SheetJobNG with status "queued",
returning immediately. A single background worker thread pulls jobs off
the queue one at a time and runs them -- this, not a lock, is what
prevents two renders from actually running concurrently on this
machine's GPU. character_sheetNG.py's own lock is just a
belt-and-suspenders safety net (see its module docstring), not the
primary serialization boundary anymore.

Per-shot status is still inferred, not pushed, exactly as in the
original -- see that module's docstring for why the mtime-floor check
matters. The floor here is dispatched_at (when the worker actually
started this job), not the earlier queued_at, since nothing renders
while a job is merely waiting its turn.

Deliberately NOT persistent, same as the original: jobs live in an
in-memory dict + queue for the life of the Flask process.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import character_sheetNG as character_sheet

_MAX_JOBS_KEPT = 20  # ring-buffer cap so a long-running process doesn't
                     # accumulate finished jobs forever


@dataclass
class SheetJobNG:
    job_id: str
    character_id: str
    shot_keys: list
    params: dict
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    log_lines: list = field(default_factory=list)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Resolved prompt text per shot key -- computed up front at
    # enqueue time, same as the original (prompts don't depend on
    # render output).
    shot_prompts: dict = field(default_factory=dict)
    # Set by start_job_ng/start_reroll_ng -- what the worker thread
    # actually calls once this job is dispatched. Not part of the
    # public job_status_ng() shape.
    _target: Optional[Callable[[Callable[[str], None]], None]] = field(
        default=None, repr=False)

    def append_log(self, line: str) -> None:
        with self._log_lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 200:
                self.log_lines = self.log_lines[-200:]

    def log_tail(self, n: int = 20) -> list:
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
        if job is None or job._target is None:
            continue
        job.dispatched_at = time.time()
        try:
            job._target(job.append_log)
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
        t = threading.Thread(target=_worker_loop, name="sheet-job-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


def _enqueue(character_id: str, shot_keys: list, params: dict,
             target: Callable[[Callable[[str], None]], None]) -> SheetJobNG:
    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job = SheetJobNG(job_id=job_id, character_id=character_id,
                     shot_keys=shot_keys, params=params, _target=target)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def start_job_ng(character_id: str, *,
                  preset: str = "default",
                  shots: Optional[list] = None,
                  views: Optional[list] = None,
                  wardrobe: str = "",
                  hair_color: str = "",
                  seed: int = -1,
                  anchor_chain: bool = True,
                  identity_lock: bool = True,
                  style: str = "none",
                  width: int = character_sheet.DEFAULT_RENDER_W,
                  height: int = character_sheet.DEFAULT_RENDER_H,
                  steps: int = character_sheet.DEFAULT_RENDER_STEPS) -> SheetJobNG:
    """Validate and enqueue a full sheet-generation job. Raises
    synchronously (before anything is queued) for bad input, a missing
    character, or a missing reference image -- everything else (engine
    failure, GPU busy) surfaces later through job_status_ng(). Never
    raises for "another job is already running" -- this one just waits
    its turn in the queue."""
    cid = character_sheet._safe_id_ng(character_id)
    shot_list = character_sheet.resolve_shots_ng(preset=preset, shots=shots, views=views)
    width, height, steps = character_sheet._validate_render_params_ng(width, height, steps)
    if not character_sheet.character_exists_ng(cid):
        raise LookupError(f"character {cid!r} not found")
    if character_sheet.character_avatar_ng(cid) is None:
        raise FileNotFoundError(
            f"character {cid!r} has no reference image -- expected an "
            f"avatar under exportsNG/{cid}/character/")

    params = {"preset": preset if (shots is None and views is None) else "custom",
             "wardrobe": wardrobe, "hair_color": hair_color, "seed": seed,
             "anchor_chain": anchor_chain,
             "identity_lock": identity_lock, "style": style,
             "width": width, "height": height, "steps": steps}

    # Same prompt-building call generate_character_sheet_ng will make
    # for each shot -- pure function of (spec, wardrobe, identity_lock,
    # style), so this can't drift from what actually gets rendered.
    shot_prompts = {
        spec.key: character_sheet._shot_prompt_ng(
            spec, wardrobe, identity_lock=identity_lock, style=style,
            hair_color=hair_color)
        for spec in shot_list
    }

    def target(on_log):
        character_sheet.generate_character_sheet_ng(
            cid, preset=preset, shots=shots, views=views, wardrobe=wardrobe,
            hair_color=hair_color,
            seed=seed, anchor_chain=anchor_chain, identity_lock=identity_lock,
            style=style, width=width, height=height, steps=steps, on_log=on_log)

    job = _enqueue(cid, [s.key for s in shot_list], params, target)
    job.shot_prompts = shot_prompts
    return job


def start_reroll_ng(character_id: str, shot_key: str, *,
                     seed: Optional[int] = None,
                     prompt: Optional[str] = None,
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     steps: Optional[int] = None) -> SheetJobNG:
    """Validate and enqueue a single-shot re-roll. width/height/steps
    default to the shot's previous render size (steps: DEFAULT_RENDER_STEPS)
    when not given -- see regenerate_shot_ng."""
    cid = character_sheet._safe_id_ng(character_id)
    if not character_sheet.character_exists_ng(cid):
        raise LookupError(f"character {cid!r} not found")
    meta = character_sheet.character_sheet_meta_ng(cid)
    existing = next((v for v in meta.get("views", []) if v.get("key") == shot_key), None)
    if existing is None:
        raise LookupError(f"no shot {shot_key!r} in character {cid!r}'s current sheet")

    def target(on_log):
        character_sheet.regenerate_shot_ng(cid, shot_key, seed=seed, prompt=prompt,
                                           width=width, height=height, steps=steps,
                                           on_log=on_log)

    job = _enqueue(cid, [shot_key],
                   {"reroll": True, "shot_key": shot_key, "seed": seed,
                    "width": width, "height": height, "steps": steps}, target)
    job.shot_prompts = {shot_key: prompt if prompt is not None else existing.get("prompt", "")}
    return job


def get_job_ng(job_id: str) -> Optional[SheetJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def job_status_ng(job_id: str) -> dict:
    """Poll-friendly snapshot: overall status (now including "queued"
    for a job that hasn't been dispatched yet), per-shot status
    (inferred from disk, floored at dispatched_at rather than
    queued_at), and a log tail."""
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")

    if job.dispatched_at is None and job.finished_at is None:
        # Still sitting in the queue -- nothing has rendered yet, so
        # every shot is simply "queued" and there's no disk to check.
        shots = [{"key": key, "status": "queued", "thumbnail": None,
                 "prompt": job.shot_prompts.get(key)} for key in job.shot_keys]
        return {
            "job_id": job.job_id,
            "character_id": job.character_id,
            "status": "queued",
            "error": None,
            "error_type": None,
            "params": job.params,
            "queued_at": job.queued_at,
            "dispatched_at": None,
            "finished_at": None,
            "shots": shots,
            "log_tail": [],
            "sheet_url": None,
        }

    mtime_floor = job.dispatched_at or job.queued_at
    char_dir = character_sheet._character_dir_ng(job.character_id)
    shots = []
    still_counting = True
    done_count = 0
    for i, key in enumerate(job.shot_keys):
        view_dir = char_dir / "sheet_views" / key
        latest_png: Optional[Path] = None
        latest_mtime = 0.0
        if view_dir.is_dir():
            for p in view_dir.glob("cand_*.png"):
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if mtime >= mtime_floor and mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_png = p

        if still_counting and latest_png is not None:
            status = "done"
            done_count += 1
        else:
            still_counting = False
            if job.finished_at is None:
                status = "rendering" if i == done_count else "queued"
            elif job.error:
                status = "failed" if i == done_count else "not_started"
            else:
                status = "done"  # job succeeded; treat as done regardless
        shots.append({
            "key": key, "status": status,
            "thumbnail": str(latest_png) if latest_png else None,
            "prompt": job.shot_prompts.get(key),
        })

    if job.finished_at is None:
        overall = "running"
    elif job.error:
        overall = "failed"
    else:
        overall = "completed"

    return {
        "job_id": job.job_id,
        "character_id": job.character_id,
        "status": overall,
        "error": job.error,
        "error_type": job.error_type,
        "params": job.params,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "shots": shots,
        "log_tail": job.log_tail(20),
        "sheet_url": (f"/api/ng/generate/characters/{job.character_id}/sheet"
                      if overall == "completed" else None),
    }
