"""Durable, never-pruned history of Music (music_jobsNG.py) renders.

Twin of job_logsNG.py (Animate's own durable job history) -- same day-
folder-then-year/month-archive layout, same reasoning: music_jobsNG.py's
_JOBS dict is a ring buffer capped at 20 entries, and evicting a job
deletes its whole job_dir (audio.flac + score.abc + request.json). That's
fine for "what's still rendering right now" but means the one thing a
song generation can't be recovered without -- the exact style/lyrics/seed
that produced it -- is gone for good once a job ages out.

record_job_ng() is called once per finished job (music_jobsNG.py's worker
thread, right when a job finishes -- before it can ever be evicted) and
copies what matters into MUSIC_JOB_LOG_DIR, laid out the same way as
job_logsNG.py:

    MUSIC_JOB_LOG_DIR/
      2026-09-13/          <- current month's days: flat, top-level
        <job_id>.json
        <job_id>.flac
        <job_id>.abc       <- only when the job produced a score
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
first job (or log listing) shows up; nothing is ever deleted automatically.
delete_log_ng() is the one path that does delete something, and it only
ever runs from a user clicking the Job Log UI's own delete button.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

from configNG import MUSIC_JOB_LOG_DIR

_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_YEAR_RE = re.compile(r"^\d{4}$")
_MONTH_RE = re.compile(r"^\d{2}$")
_JOB_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


def _current_month() -> str:
    return time.strftime("%Y-%m", time.localtime())


def _archive_past_months() -> None:
    root = Path(MUSIC_JOB_LOG_DIR)
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
    top = Path(MUSIC_JOB_LOG_DIR) / date_str
    if top.is_dir():
        return top
    year, month, day = m.group(1), m.group(2), m.group(3)
    archived = Path(MUSIC_JOB_LOG_DIR) / year / month / day
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
    """Called once from music_jobsNG.py's worker loop right after a job
    finishes (success or failure). Never raises -- callers wrap this in
    their own try/except anyway, but staying defensive here means a bad
    day never takes the render pipeline down with it."""
    try:
        if job.cancelled and job.dispatched_at is None:
            return  # never actually rendered, nothing worth showing

        _archive_past_months()

        finished_at = job.finished_at or time.time()
        date_str = time.strftime("%Y-%m-%d", time.localtime(finished_at))
        day_dir = Path(MUSIC_JOB_LOG_DIR) / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        has_audio = False
        if job.audio_path and Path(job.audio_path).is_file():
            shutil.copyfile(job.audio_path, day_dir / f"{job.job_id}.flac")
            has_audio = True

        has_score = False
        if job.score_path and Path(job.score_path).is_file():
            shutil.copyfile(job.score_path, day_dir / f"{job.job_id}.abc")
            has_score = True

        is_cover = job.source_audio_path is not None
        status = "failed" if job.error else ("cancelled" if job.cancelled else "completed")
        meta = {
            "schema": "ringviz/music_job_log@1",
            "job_id": job.job_id,
            "status": status,
            "style": job.style,
            "lyrics": job.lyrics,
            "duration_s": job.duration_s,
            "seed": job.seed,
            "resolved_seed": (result or {}).get("seed"),
            "mode": job.mode,
            "instrumental": job.instrumental,
            "cfg_scale": job.cfg_scale,
            "temperature": job.temperature,
            "precision": job.precision,
            "is_cover": is_cover,
            "task": job.task if is_cover else None,
            "audio_seconds": job.audio_seconds,
            "error": job.error,
            "error_type": job.error_type,
            "queued_at": job.queued_at,
            "dispatched_at": job.dispatched_at,
            "finished_at": finished_at,
            "has_audio": has_audio,
            "has_score": has_score,
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
    root = Path(MUSIC_JOB_LOG_DIR)
    if not root.is_dir():
        return []
    entries = []
    for entry in root.iterdir():
        if entry.is_dir() and _DAY_RE.match(entry.name):
            entries.extend(_load_entries(entry, entry.name))
    entries.sort(key=lambda e: e.get("finished_at") or 0, reverse=True)
    return entries


def list_archive_years_ng() -> list:
    root = Path(MUSIC_JOB_LOG_DIR)
    if not root.is_dir():
        return []
    years = [p.name for p in root.iterdir() if p.is_dir() and _YEAR_RE.match(p.name)]
    return sorted(years, reverse=True)


def list_archive_months_ng(year: str) -> list:
    if not _YEAR_RE.match(year or ""):
        return []
    year_dir = Path(MUSIC_JOB_LOG_DIR) / year
    if not year_dir.is_dir():
        return []
    months = [p.name for p in year_dir.iterdir() if p.is_dir() and _MONTH_RE.match(p.name)]
    return sorted(months, reverse=True)


def list_archive_day_entries_ng(year: str, month: str) -> list:
    if not _YEAR_RE.match(year or "") or not _MONTH_RE.match(month or ""):
        return []
    month_dir = Path(MUSIC_JOB_LOG_DIR) / year / month
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


def log_audio_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    p = day_dir / f"{job_id}.flac"
    return p if p.is_file() else None


def log_score_path_ng(date_str: str, job_id: str) -> Optional[Path]:
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return None
    p = day_dir / f"{job_id}.abc"
    return p if p.is_file() else None


def delete_log_ng(date_str: str, job_id: str) -> bool:
    """Removes one entry and its sidecar files (audio + score) for good --
    the one deliberate exception to this module's "nothing is ever
    deleted" retention policy above, and only ever reachable from the Job
    Log UI's own delete button (a user clearing out e.g. a failed job),
    never automatic pruning."""
    if not _JOB_ID_RE.match(job_id or ""):
        return False
    day_dir = _resolve_day_dir(date_str)
    if day_dir is None:
        return False
    json_path = day_dir / f"{job_id}.json"
    if not json_path.is_file():
        return False
    for p in day_dir.glob(f"{job_id}.*"):
        p.unlink(missing_ok=True)
    return True
