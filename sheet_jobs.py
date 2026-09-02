"""A lightweight in-process job queue for character-sheet generation.

The 3-view turnaround sheet was small enough to render inside one
blocking HTTP request (see routes/phosphene.py's original docstring).
The 15-shot "extended" dataset preset is not -- at ~10 min/shot that's
2.5+ hours, which no HTTP client should sit on. This module runs a
generation as a background thread instead and exposes poll-based status,
so the request that *starts* a job returns immediately.

Deliberately NOT persistent: jobs live in an in-memory dict for the
life of the Flask process. A restart loses in-flight job status (though
not the images already written to disk -- those are real files under
exports/<name>/character/sheet_views/ regardless of whether anything is
tracking them). Good enough for a single-user local tool; revisit if
that ever stops being true.

Per-shot status is inferred, not pushed: generation is strictly
sequential (character_sheet.generate_character_sheet renders one shot
at a time), so a shot counts as "done" once a rendered PNG shows up
under its sheet_views/<key>/ directory with an mtime at or after the
job's start time (the mtime floor matters because re-running a preset
on a character that already has a sheet would otherwise see the OLD
files and report every shot "done" instantly).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import character_sheet

_MAX_JOBS_KEPT = 20  # ring-buffer cap so a long-running process doesn't
                     # accumulate finished jobs forever


@dataclass
class SheetJob:
    job_id: str
    character_id: str
    shot_keys: list
    params: dict
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    thread: Optional[threading.Thread] = None
    log_lines: list = field(default_factory=list)
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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


def _prune_old_jobs_locked() -> None:
    if len(_JOBS) <= _MAX_JOBS_KEPT:
        return
    finished = sorted(
        (j for j in _JOBS.values() if j.finished_at is not None),
        key=lambda j: j.finished_at)
    while len(_JOBS) > _MAX_JOBS_KEPT and finished:
        oldest = finished.pop(0)
        _JOBS.pop(oldest.job_id, None)


def _launch(character_id: str, shot_keys: list, params: dict,
           target: Callable[[Callable[[str], None]], None]) -> SheetJob:
    job_id = uuid.uuid4().hex[:12]
    job = SheetJob(job_id=job_id, character_id=character_id,
                  shot_keys=shot_keys, params=params)

    def _run():
        try:
            target(job.append_log)
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.error = str(e)
            job.error_type = type(e).__name__
        finally:
            job.finished_at = time.time()

    t = threading.Thread(target=_run, name=f"sheet-job-{job_id}", daemon=True)
    job.thread = t
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    t.start()
    return job


def start_job(character_id: str, *,
             preset: str = "default",
             shots: Optional[list] = None,
             views: Optional[list] = None,
             wardrobe: str = "",
             seed: int = -1,
             anchor_chain: bool = True,
             identity_lock: bool = True,
             style: str = "none") -> SheetJob:
    """Validate and kick off a full sheet-generation job in the
    background. Raises synchronously (before any thread starts) for bad
    input, a missing character, or a missing reference image --
    everything else (engine failure, GPU busy) surfaces later through
    job_status()."""
    cid = character_sheet._safe_id(character_id)
    shot_list = character_sheet.resolve_shots(preset=preset, shots=shots, views=views)
    if not character_sheet.character_exists(cid):
        raise LookupError(f"character {cid!r} not found")
    if character_sheet.character_avatar(cid) is None:
        raise FileNotFoundError(
            f"character {cid!r} has no reference image -- expected an "
            f"avatar under exports/{cid}/character/")

    params = {"preset": preset if (shots is None and views is None) else "custom",
             "wardrobe": wardrobe, "seed": seed, "anchor_chain": anchor_chain,
             "identity_lock": identity_lock, "style": style}

    def target(on_log):
        character_sheet.generate_character_sheet(
            cid, preset=preset, shots=shots, views=views, wardrobe=wardrobe,
            seed=seed, anchor_chain=anchor_chain, identity_lock=identity_lock,
            style=style, on_log=on_log)

    return _launch(cid, [s.key for s in shot_list], params, target)


def start_reroll(character_id: str, shot_key: str, *,
                 seed: Optional[int] = None,
                 prompt: Optional[str] = None) -> SheetJob:
    """Validate and kick off a single-shot re-roll in the background."""
    cid = character_sheet._safe_id(character_id)
    if not character_sheet.character_exists(cid):
        raise LookupError(f"character {cid!r} not found")
    meta = character_sheet.character_sheet_meta(cid)
    if not any(v.get("key") == shot_key for v in meta.get("views", [])):
        raise LookupError(f"no shot {shot_key!r} in character {cid!r}'s current sheet")

    def target(on_log):
        character_sheet.regenerate_shot(cid, shot_key, seed=seed, prompt=prompt,
                                        on_log=on_log)

    return _launch(cid, [shot_key], {"reroll": True, "shot_key": shot_key, "seed": seed}, target)


def get_job(job_id: str) -> Optional[SheetJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def job_status(job_id: str) -> dict:
    """Poll-friendly snapshot: overall status, per-shot status (inferred
    from disk, not pushed -- see module docstring), and a log tail."""
    job = get_job(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")

    char_dir = character_sheet._character_dir(job.character_id)
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
                if mtime >= job.started_at and mtime > latest_mtime:
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
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "shots": shots,
        "log_tail": job.log_tail(20),
        "sheet_url": (f"/api/phosphene/characters/{job.character_id}/sheet"
                      if overall == "completed" else None),
    }
