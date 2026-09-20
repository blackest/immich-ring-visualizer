"""Durable, never-pruned history of Animate (video_jobsNG.py) renders.

video_jobsNG.py itself is deliberately NOT persistent -- its _JOBS dict is
a ring buffer capped at 20 entries, and evicting a job deletes its whole
job_dir (reference image + rendered mp4). That's fine for "what's still
rendering right now" but useless for "what did I generate last week."

This module is the separate, permanent record: record_job_ng() is called
once per finished job (video_jobsNG.py's worker thread, right when a job
finishes -- before it can ever be evicted) and copies what matters into
JOB_LOG_DIR, laid out Lightroom-style so it never needs to delete anything
yet stays cheap to browse:

    JOB_LOG_DIR/
      2026-09-13/          <- current month's days: flat, top-level
        <job_id>.json
        <job_id>_ref.<ext>
        <job_id>.mp4
      2026-09-12/
      ...
      2026/                <- archive: past months rolled up here
        08/
          15/              <- same per-job files, just relocated
          16/

Only the current month's day-folders ever sit at the top level, so
list_recent_logs_ng() -- the common "show recent" path -- only ever lists
a bounded (~month's worth) set of folders. _archive_past_months() moves
last month's stragglers under <year>/<month>/ the moment a new month's
first job (or log listing) shows up; nothing is ever deleted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from configNG import JOB_LOG_DIR

_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{2}$")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


def _current_month() -> str:
    return time.strftime("%Y-%m", time.localtime())


def _archive_past_months() -> None:
    root = Path(JOB_LOG_DIR)
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
    top = Path(JOB_LOG_DIR) / date_str
    if top.is_dir():
        return top
    year, month, day = m.group(1), m.group(2), m.group(3)
    archived = Path(JOB_LOG_DIR) / year / month / day
    return archived if archived.is_dir() else None


def _load_entries(day_dir: Path, date_str: str) -> list:
    entries = []
    for jf in day_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["date"] = date_str
        entries.append(data)
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def record_job_ng(job, result: Optional[dict]) -> None:
    """Called once from video_jobsNG.py's worker thread right after a job
    finishes (success or failure). Never raises -- callers wrap this in
    their own try/except anyway, but staying defensive here means a bad
    day never takes the render pipeline down with it."""
    try:
        if job.cancelled and job.dispatched_at is None:
            return  # never actually rendered, nothing worth showing

        _archive_past_months()

        finished_at = job.finished_at or time.time()
        date_str = time.strftime("%Y-%m-%d", time.localtime(finished_at))
        day_dir = Path(JOB_LOG_DIR) / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        has_ref = False
        if job.has_image:
            src_ref = next(Path(job.job_dir).glob("ref.*"), None)
            if src_ref is not None:
                shutil.copyfile(src_ref, day_dir / f"{job.job_id}_ref{src_ref.suffix.lower()}")
                has_ref = True

        has_video = False
        if job.mp4_path and Path(job.mp4_path).is_file():
            shutil.copyfile(job.mp4_path, day_dir / f"{job.job_id}.mp4")
            has_video = True

        status = "failed" if job.error else ("cancelled" if job.cancelled else "completed")
        meta = {
            "schema": "ringviz/videogen_job_log@1",
            "job_id": job.job_id,
            "status": status,
            "prompt": job.prompt,
            "duration_s": job.duration_s,
            "seed": job.seed,
            "resolved_seed": (result or {}).get("seed"),
            "width": job.width,
            "height": job.height,
            "frame_rate": job.frame_rate,
            "lora_path": job.lora_path,
            "lora_strength": job.lora_strength if job.lora_path else None,
            "error": job.error,
            "error_type": job.error_type,
            "queued_at": job.queued_at,
            "dispatched_at": job.dispatched_at,
            "finished_at": finished_at,
            "has_ref": has_ref,
            "has_video": has_video,
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
    root = Path(JOB_LOG_DIR)
    if not root.is_dir():
        return []
    entries = []
    for entry in root.iterdir():
        if entry.is_dir() and _DAY_RE.match(entry.name):
            entries.extend(_load_entries(entry, entry.name))
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def list_archive_years_ng() -> list:
    root = Path(JOB_LOG_DIR)
    if not root.is_dir():
        return []
    years = [p.name for p in root.iterdir() if p.is_dir() and _YEAR_RE.match(p.name)]
    return sorted(years, reverse=True)


def list_archive_months_ng(year: str) -> list:
    if not _YEAR_RE.match(year or ""):
        return []
    year_dir = Path(JOB_LOG_DIR) / year
    if not year_dir.is_dir():
        return []
    months = [p.name for p in year_dir.iterdir() if p.is_dir() and _MONTH_RE.match(p.name)]
    return sorted(months, reverse=True)


def list_archive_day_entries_ng(year: str, month: str) -> list:
    if not _YEAR_RE.match(year or "") or not _MONTH_RE.match(month or ""):
        return []
    month_dir = Path(JOB_LOG_DIR) / year / month
    if not month_dir.is_dir():
        return []
    entries = []
    for day_dir in month_dir.iterdir():
        if day_dir.is_dir() and re.match(r"^\d{2}$", day_dir.name):
            date_str = f"{year}-{month}-{day_dir.name}"
            entries.extend(_load_entries(day_dir, date_str))
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def get_log_entry_ng(date_str: str, job_id: str) -> Optional[dict]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    json_path = day_dir / f"{job_id}.json"
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["date"] = date_str
    return data


def update_log_notes_ng(date_str: str, job_id: str, notes: str) -> Optional[dict]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    json_path = day_dir / f"{job_id}.json"
    if not json_path.is_file():
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


def log_ref_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    return next(day_dir.glob(f"{job_id}_ref.*"), None)


def log_video_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    p = day_dir / f"{job_id}.mp4"
    return p if p.is_file() else None
