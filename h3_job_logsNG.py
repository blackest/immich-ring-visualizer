"""Durable, never-pruned history of H3 (h3_jobsNG.py) renders.

Twin of music_job_logsNG.py / job_logsNG.py -- same day-folder-then-
year/month-archive layout, same reasoning: h3_jobsNG.py's _JOBS dict is a
ring buffer capped at 20 entries, and evicting a job deletes its whole
job_dir (the mp4, the reference image, the metrics file). Fine for "what's
rendering right now", but it also throws away the one thing a render can't
be recovered without -- the exact prompt/seed/settings that produced it,
and the log that says what actually ran (e.g. which LoRA layers applied).
It also doesn't survive a restart at all, so a power cut loses the queue.

record_job_ng() is called once per finished job (h3_jobsNG.py's worker
thread, right when a job finishes -- before it can ever be evicted) and
copies what matters into H3_JOB_LOG_DIR, laid out like the others:

    H3_JOB_LOG_DIR/
      2026-09-29/          <- current month's days: flat, top-level
        <job_id>.json      <- prompt, seed, size, steps, turbo, timings, status
        <job_id>.mp4       <- only when the job produced a video
        <job_id>.log       <- the job's full captured log
        <job_id>.metrics.json  <- runner's own per-phase timings/memory
        <job_id>.ref.<ext> <- only when the job had a keyframe image
      2026/                <- archive: past months rolled up here
        08/15/...

Only the current month's day-folders sit at the top level, so
list_recent_logs_ng() only ever lists a bounded set of folders.
_archive_past_months() moves last month's stragglers under <year>/<month>/
the first time a new month's job (or log listing) shows up; nothing is ever
deleted automatically. delete_log_ng() is the one path that deletes, and it
only runs from the Job Log UI's own delete button.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from configNG import H3_JOB_LOG_DIR

_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{2}$")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
_REF_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _current_month() -> str:
    return time.strftime("%Y-%m", time.localtime())


def _archive_past_months() -> None:
    root = Path(H3_JOB_LOG_DIR)
    if not root.is_dir():
        return
    current_month = _current_month()
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = _DAY_RE.match(entry.name)
        if not m:
            continue
        year, month, day = m.group(1), m.group(2), m.group(3)
        if f"{year}-{month}" == current_month:
            continue
        dest = root / year / month / day
        if dest.exists():
            continue  # defensive -- shouldn't happen, never overwrite history
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(dest))


def _resolve_day_dir(date_str: str) -> Optional[Path]:
    m = _DAY_RE.match(date_str or "")
    if not m:
        return None
    top = Path(H3_JOB_LOG_DIR) / date_str
    if top.is_dir():
        return top
    year, month, day = m.group(1), m.group(2), m.group(3)
    archived = Path(H3_JOB_LOG_DIR) / year / month / day
    return archived if archived.is_dir() else None


def _load_entries(day_dir: Path, date_str: str) -> list:
    entries = []
    for jf in day_dir.glob("*.json"):
        if jf.name.endswith(".metrics.json"):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["date"] = date_str
        entries.append(data)
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def record_job_ng(job) -> None:
    """Called once from h3_jobsNG.py's worker loop right after a job
    finishes (success or failure). Never raises -- callers wrap this in
    their own try/except anyway, but staying defensive here means a bad
    day never takes the render pipeline down with it."""
    try:
        if job.cancelled and job.dispatched_at is None:
            return  # never actually rendered, nothing worth showing

        _archive_past_months()

        finished_at = job.finished_at or time.time()
        date_str = time.strftime("%Y-%m-%d", time.localtime(finished_at))
        day_dir = Path(H3_JOB_LOG_DIR) / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        has_video = False
        if job.mp4_path and Path(job.mp4_path).is_file():
            shutil.copyfile(job.mp4_path, day_dir / f"{job.job_id}.mp4")
            has_video = True

        has_ref = False
        ref = next((p for p in Path(job.job_dir).glob("ref.*")
                    if p.suffix.lower() in _REF_EXTS), None)
        if ref is not None:
            shutil.copyfile(ref, day_dir / f"{job.job_id}.ref{ref.suffix.lower()}")
            has_ref = True

        metrics = next(Path(job.job_dir).glob("*.metrics.json"), None)
        has_metrics = False
        if metrics is not None:
            shutil.copyfile(metrics, day_dir / f"{job.job_id}.metrics.json")
            has_metrics = True

        with job._log_lock:
            log_text = "\n".join(job.log_lines)
        (day_dir / f"{job.job_id}.log").write_text(log_text, encoding="utf-8")

        status = "failed" if job.error else ("cancelled" if job.cancelled else "completed")
        meta = {
            "schema": "ringviz/h3_job_log@1",
            "job_id": job.job_id,
            "status": status,
            "prompt": job.prompt,
            "duration_s": job.duration_s,
            "seed": job.seed,
            "model": job.model,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "turbo": job.turbo,
            "error": job.error,
            "error_type": job.error_type,
            "queued_at": job.queued_at,
            "dispatched_at": job.dispatched_at,
            "finished_at": finished_at,
            "has_video": has_video,
            "has_ref": has_ref,
            "has_metrics": has_metrics,
            "notes": "",
        }
        json_path = day_dir / f"{job.job_id}.json"
        tmp_path = json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, json_path)
    except Exception:
        pass


def list_recent_logs_ng() -> list:
    _archive_past_months()
    root = Path(H3_JOB_LOG_DIR)
    if not root.is_dir():
        return []
    entries = []
    for entry in root.iterdir():
        if entry.is_dir() and _DAY_RE.match(entry.name):
            entries.extend(_load_entries(entry, entry.name))
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def list_archive_years_ng() -> list:
    root = Path(H3_JOB_LOG_DIR)
    if not root.is_dir():
        return []
    years = [p.name for p in root.iterdir() if p.is_dir() and _YEAR_RE.match(p.name)]
    return sorted(years, reverse=True)


def list_archive_months_ng(year: str) -> list:
    if not _YEAR_RE.match(year or ""):
        return []
    year_dir = Path(H3_JOB_LOG_DIR) / year
    if not year_dir.is_dir():
        return []
    months = [p.name for p in year_dir.iterdir() if p.is_dir() and _MONTH_RE.match(p.name)]
    return sorted(months, reverse=True)


def list_archive_day_entries_ng(year: str, month: str) -> list:
    if not _YEAR_RE.match(year or "") or not _MONTH_RE.match(month or ""):
        return []
    month_dir = Path(H3_JOB_LOG_DIR) / year / month
    if not month_dir.is_dir():
        return []
    entries = []
    for day_dir in month_dir.iterdir():
        if day_dir.is_dir() and re.match(r"^\d{2}$", day_dir.name):
            date_str = f"{year}-{month}-{day_dir.name}"
            entries.extend(_load_entries(day_dir, date_str))
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def _entry_json_path(date_str: str, job_id: str) -> Optional[Path]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    p = day_dir / f"{job_id}.json"
    return p if p.is_file() else None


def get_log_entry_ng(date_str: str, job_id: str) -> Optional[dict]:
    json_path = _entry_json_path(date_str, job_id)
    if json_path is None:
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["date"] = date_str
    return data


def update_log_notes_ng(date_str: str, job_id: str, notes: str) -> Optional[dict]:
    json_path = _entry_json_path(date_str, job_id)
    if json_path is None:
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["notes"] = notes or ""
    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, json_path)
    data["date"] = date_str
    return data


def _sidecar_path(date_str: str, job_id: str, suffix: str) -> Optional[Path]:
    json_path = _entry_json_path(date_str, job_id)
    if json_path is None:
        return None
    p = json_path.with_name(f"{job_id}{suffix}")
    return p if p.is_file() else None


def log_video_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    return _sidecar_path(date_str, job_id, ".mp4")


def log_text_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    return _sidecar_path(date_str, job_id, ".log")


def log_ref_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    for ext in _REF_EXTS:
        p = _sidecar_path(date_str, job_id, f".ref{ext}")
        if p is not None:
            return p
    return None


def delete_log_ng(date_str: str, job_id: str) -> bool:
    """Removes one entry and all its sidecar files for good -- the one
    deliberate exception to this module's "nothing is ever deleted"
    retention policy, and only reachable from the Job Log UI's own delete
    button, never automatic pruning."""
    json_path = _entry_json_path(date_str, job_id)
    if json_path is None:
        return False
    for p in json_path.parent.glob(f"{job_id}.*"):
        p.unlink(missing_ok=True)
    return True
