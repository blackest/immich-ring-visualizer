"""NG job queue for Speech renders (see speech_engineNG.py).

Same FIFO-queue-not-a-busy-lock design as h3_jobsNG.py/music_jobsNG.py: own worker thread, POST returns a
job id at once and the browser polls. Two deliberate differences:

* list_jobs_ng() exists from the start, so a page that lost its own list can rebuild it from the server
  (the H3 view needed that added later).
* The in-memory list is a 20-job ring buffer, but trimming it NEVER deletes a finished job's folder: a
  speech wav is small, cheap to keep and annoying to lose. Only the in-memory record goes.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import speech_engineNG as speech
from configNG import SPEECHGEN_DIR

_MAX_JOBS_KEPT = 20
_CLIP_EXTS = speech.VOICE_EXTS


@dataclass
class SpeechJobNG:
    job_id: str
    text: str
    kind: str                      # "preset" | "character" | "clip"
    voice: str                     # preset id, or "" for a clone
    voice_label: str
    style: str                     # the resolved delivery instruction (presets)
    language: str
    speed: float
    seed: Optional[int]
    job_dir: Path
    ref_audio_path: Optional[str] = None
    ref_text: str = ""
    design_text: str = ""          # kind "design": the voice, in words
    voice_consent: bool = False
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    audio_path: Optional[str] = None
    audio_seconds: Optional[float] = None
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
    finished = sorted((j for j in _JOBS.values() if j.finished_at is not None), key=lambda j: j.finished_at)
    while len(_JOBS) > _MAX_JOBS_KEPT and finished:
        _JOBS.pop(finished.pop(0).job_id, None)   # the record only; the folder and its wav stay on disk


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        if job.cancelled:
            job.finished_at = time.time()
            job.error = "cancelled before it started rendering"
            job.error_type = "SpeechJobCancelled"
            continue
        job.dispatched_at = time.time()
        try:
            result = speech.generate_speech_ng(
                text=job.text, output_dir=job.job_dir, kind=job.kind, voice=job.voice, style=job.style,
                ref_audio_path=job.ref_audio_path, ref_text=job.ref_text, language=job.language,
                speed=job.speed, seed=job.seed, voice_consent=job.voice_consent,
                voice_label=job.voice_label, design_text=job.design_text, on_log=job.append_log,
                on_proc_start=lambda p: setattr(job, "_proc", p))
            job.audio_path = result["path"]
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
        threading.Thread(target=_worker_loop, name="speech-job-worker-ng", daemon=True).start()
        _WORKER_STARTED = True


def start_speech_job_ng(text: str, kind: str, voice: str = "", character: str = "",
                        style_id: str = "natural", style_text: str = "", language: str = "english",
                        speed: float = 1.0, seed: Optional[int] = None, ref_text: str = "",
                        voice_consent: bool = False, clip_bytes: Optional[bytes] = None,
                        clip_ext: str = "", design_text: str = "") -> SpeechJobNG:
    """Validates and enqueues. A character's voice is looked up by NAME in the shared loras folder (never a
    client path); a clip is written into the job folder eagerly because the runner needs a real file."""
    text = (text or "").replace("\r\n", "\n").strip()
    ref_audio_src: Optional[Path] = None
    voice_label = ""
    if kind == "character":
        ref_audio_src = speech.resolve_voice_sample_ng(character)
        if ref_audio_src is None:
            raise speech.SpeechRequestError("That character has no saved voice sample.")
        voice_label = character
    elif kind == "clip":
        if not clip_bytes:
            raise speech.SpeechRequestError("Add a voice clip first.")
        voice_label = "uploaded clip"
    elif kind == "preset":
        voice_label = next((p["label"] for p in speech.PRESETS if p["id"] == voice), voice)
    elif kind == "design":
        voice_label = "described voice"

    speech.validate_request_ng(kind, text, voice, language, float(speed),
                               has_ref_audio=bool(ref_audio_src or clip_bytes), voice_consent=bool(voice_consent),
                               design_text=design_text)

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(SPEECHGEN_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_audio_path: Optional[str] = None
    if kind == "character":
        ref_audio_path = str(ref_audio_src)
    elif kind == "clip":
        ext = clip_ext.lower() if clip_ext and clip_ext.lower() in _CLIP_EXTS else ".wav"
        clip_path = job_dir / f"reference{ext}"
        clip_path.write_bytes(clip_bytes)
        ref_audio_path = str(clip_path)

    job = SpeechJobNG(
        job_id=job_id, text=text, kind=kind, voice=voice if kind == "preset" else "", voice_label=voice_label,
        style=speech.instruction_for(style_id, style_text) if kind == "preset" else "",
        language=language, speed=float(speed), seed=seed, job_dir=job_dir, ref_audio_path=ref_audio_path,
        ref_text=(ref_text or "").strip(), voice_consent=bool(voice_consent),
        design_text=" ".join((design_text or "").split()) if kind == "design" else "")
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_job_ng(job_id: str) -> Optional[SpeechJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_speech_job_ng(job_id: str) -> bool:
    """A queued job is flagged and skipped, a rendering job's subprocess is killed."""
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
    return {
        "job_id": job.job_id, "status": status, "text": job.text, "kind": job.kind,
        "voice": job.voice_label, "style": job.style, "design": job.design_text, "language": job.language,
        "speed": job.speed,
        "seed": job.seed, "audio_seconds": job.audio_seconds, "error": job.error,
        "error_type": job.error_type, "cancelled": bool(job.cancelled), "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at, "finished_at": job.finished_at, "log_tail": job.log_tail(40),
        "audio_url": f"/api/ng/speech/jobs/{job.job_id}/audio" if status == "completed" else None,
    }


def list_jobs_ng() -> list:
    """Every job the server still holds, oldest first (no log tail): the page rebuilds its list from this."""
    with _JOBS_LOCK:
        ids = sorted(_JOBS, key=lambda jid: _JOBS[jid].queued_at or 0)
    out = []
    for jid in ids:
        try:
            st = job_status_ng(jid)
        except LookupError:
            continue
        st.pop("log_tail", None)
        out.append(st)
    return out
