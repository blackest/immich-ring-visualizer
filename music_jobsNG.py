"""NG job queue for YuE2 style+lyrics-to-song renders (see
music_engineNG.py).

Twin of h3_jobsNG.py/video_jobsNG.py's queue design -- same FIFO-queue-
not-a-busy-lock reasoning, own worker thread, ring-buffer eviction.
Simpler than either in one way for a plain generate job: no reference/
keyframe image at all, no job_dir disk write before the render even
starts -- style and lyrics are plain text carried on the job object
itself, so job_dir only comes into existence to hold the request.json/
audio.flac the engine writes.

A cover job (source_audio_path set) is the one case that DOES need an
eager job_dir/disk write, same reason video_jobsNG needs one for its
reference image: the source track has to be a real file on disk before
generate_cover_ng's subprocess can read it (see start_music_job_ng).
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import music_engineNG as music
from configNG import MUSICGEN_DIR

_MAX_JOBS_KEPT = 20  # ring-buffer cap, same reasoning as h3_jobsNG/video_jobsNG


@dataclass
class MusicJobNG:
    job_id: str
    style: str
    lyrics: str
    duration_s: Optional[float]
    seed: Optional[int]
    job_dir: Path
    mode: music.MusicMode = "full"
    instrumental: bool = False
    cfg_scale: Optional[float] = None
    precision: music.MusicPrecision = music.MUSIC_DEFAULT_PRECISION
    # Cover job fields -- source_audio_path set means "this is a cover,
    # not a plain generate" (see start_music_job_ng/_worker_loop). task
    # is cover-only (music.MUSIC_COVER_TASKS); mode/instrumental above
    # are plain-generate-only and ignored for a cover.
    source_audio_path: Optional[str] = None
    task: str = music.MUSIC_DEFAULT_COVER_TASK
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    audio_path: Optional[str] = None
    audio_seconds: Optional[float] = None
    score_path: Optional[str] = None  # cover only -- the transcribed ABC sheet music
    log_lines: list = field(default_factory=list)
    cancelled: bool = False
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)

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
        shutil.rmtree(oldest.job_dir, ignore_errors=True)


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        if job.cancelled:
            job.finished_at = time.time()
            job.error = "cancelled before it started rendering"
            job.error_type = "MusicJobCancelled"
            continue
        job.dispatched_at = time.time()
        try:
            if job.source_audio_path is not None:
                result = music.generate_cover_ng(
                    audio_path=job.source_audio_path, output_dir=job.job_dir, seed=job.seed,
                    config=music.MusicConfig(precision=job.precision),
                    task=job.task, style=job.style, lyrics=job.lyrics,
                    duration_s=job.duration_s, cfg_scale=job.cfg_scale,
                    on_log=job.append_log,
                    on_proc_start=lambda p: setattr(job, "_proc", p))
                job.score_path = result.get("score_path")
            else:
                result = music.generate_music_ng(
                    style=job.style, lyrics=job.lyrics, duration_s=job.duration_s,
                    output_dir=job.job_dir, seed=job.seed,
                    config=music.MusicConfig(precision=job.precision),
                    mode=job.mode, instrumental=job.instrumental, cfg_scale=job.cfg_scale,
                    on_log=job.append_log,
                    on_proc_start=lambda p: setattr(job, "_proc", p))
            job.audio_path = result["audio_path"]
            job.audio_seconds = result.get("audio_seconds")
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
        t = threading.Thread(target=_worker_loop, name="music-job-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


_AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac")


def start_music_job_ng(style: str, lyrics: str, duration_s: Optional[float] = None,
                        seed: Optional[int] = None,
                        mode: music.MusicMode = "full",
                        instrumental: bool = False,
                        cfg_scale: Optional[float] = None,
                        precision: music.MusicPrecision = music.MUSIC_DEFAULT_PRECISION,
                        audio_bytes: Optional[bytes] = None, audio_ext: str = "",
                        task: str = music.MUSIC_DEFAULT_COVER_TASK) -> MusicJobNG:
    """Validates and enqueues -- the actual style/lyrics text lives on
    the job object itself (see MusicJobNG), not written to disk until
    the worker calls generate_music_ng/generate_cover_ng, which writes
    its own request.json into job_dir as real provenance (see either
    function's docstring).

    audio_bytes present (a cover job) is the one thing that DOES touch
    disk here, eagerly -- same "this write IS the job's own copy for
    its whole lifetime" reasoning as video_jobsNG.start_video_job_ng's
    reference image, since generate_cover_ng's subprocess needs a real
    --audio path to read. For a cover, duration_s is optional (None =
    follow the transcribed score's own length, see generate_cover_ng);
    for a plain generate it's required by the caller (the route applies
    its own default before calling this)."""
    style = (style or "").strip()
    lyrics = (lyrics or "").replace("\r\n", "\n").strip("\n")
    is_cover = audio_bytes is not None
    if is_cover:
        if task not in music.MUSIC_COVER_TASKS:
            raise ValueError(f"task must be one of {music.MUSIC_COVER_TASKS} (got {task!r})")
        if duration_s is not None:
            music._validate_music_duration_ng(duration_s)
    else:
        if mode not in music.MUSIC_MODES:
            raise ValueError(f"mode must be one of {music.MUSIC_MODES} (got {mode!r})")
        if not instrumental and not style and not lyrics:
            raise ValueError("Give it something to work with -- style, lyrics, or both.")
        if duration_s is None:
            raise ValueError("duration_s is required for a plain generate job")
        music._validate_music_duration_ng(duration_s)

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(MUSICGEN_DIR) / job_id

    source_audio_path = None
    if is_cover:
        job_dir.mkdir(parents=True, exist_ok=True)
        ext = audio_ext.lower() if audio_ext and audio_ext.lower() in _AUDIO_EXTS else ".mp3"
        source_path = job_dir / f"source{ext}"
        source_path.write_bytes(audio_bytes)
        source_audio_path = str(source_path)

    job = MusicJobNG(job_id=job_id, style=style, lyrics=lyrics, duration_s=duration_s,
                      seed=seed, job_dir=job_dir, mode=mode, instrumental=instrumental,
                      cfg_scale=cfg_scale, precision=precision,
                      source_audio_path=source_audio_path, task=task)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_job_ng(job_id: str) -> Optional[MusicJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_music_job_ng(job_id: str) -> bool:
    """Same semantics as h3_jobsNG/video_jobsNG's own cancel: a queued
    job is flagged and skipped, a rendering job's subprocess is killed
    (surfaces as the usual MusicJobCancelled)."""
    job = get_job_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
    if job.dispatched_at is not None and job._proc is not None:
        try:
            os.killpg(os.getpgid(job._proc.pid), signal.SIGKILL)
        except Exception:
            try:
                job._proc.kill()
            except Exception:
                pass
    return True


def job_status_ng(job_id: str) -> dict:
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")

    if job.dispatched_at is None and job.finished_at is None:
        status = "queued"
    elif job.finished_at is None:
        status = "rendering"
    elif job.error:
        status = "failed"
    else:
        status = "completed"

    is_cover = job.source_audio_path is not None
    return {
        "job_id": job.job_id,
        "status": status,
        "style": job.style,
        "lyrics": job.lyrics,
        "duration_s": job.duration_s,
        "mode": job.mode,
        "instrumental": job.instrumental,
        "precision": job.precision,
        "is_cover": is_cover,
        "task": job.task if is_cover else None,
        "audio_seconds": job.audio_seconds,
        "error": job.error,
        "error_type": job.error_type,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "log_tail": job.log_tail(40),
        "audio_url": (f"/api/ng/music/jobs/{job.job_id}/audio"
                      if status == "completed" else None),
        "score_url": (f"/api/ng/music/jobs/{job.job_id}/score"
                      if status == "completed" and job.score_path else None),
    }
