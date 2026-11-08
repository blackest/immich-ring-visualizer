"""NG "Talk" jobs: a script said by a voice, on a character's face, lip-synced -- one click, two stages.

  1. speaking  -- a Speech job (speech_jobsNG: a preset voice, a character's saved voice, or a clip) makes the WAV.
  2. rendering -- an H3 job (h3_jobsNG) generates the picture to that WAV (`--audio-drive`, h3_audio_driveNG),
                  with the character's LoRA and, optionally, a first-frame picture.

Talk owns no GPU work and no files of its own: each stage is an ordinary job in its own queue, so cancel, the
Job Log and the memory handling are the existing ones. This file only chains them, keeps the stage and the
progress, and tidies the request into the two calls. One Talk job at a time (FIFO), so two submissions do not
interleave their stages.

  3. finishing -- optional: the soft-finish pass (finish_jobsNG, ComfyUI) over the clip. A failed finish never loses
                  the render: the job still completes with the original clip and reports finish_error.

The clip is a fixed 5 s window (TALK_CLIP_S): that is the length measured to work (a speech WAV drove a clear,
articulated mouth that stopped when the voice stopped). Shorter and chained lengths are untested, so a script
whose speech runs longer than the window is refused after the speech stage with a plain message, not cut off.
Two speeds: full (16 steps, about 20 minutes for a 5 s draft) and Fast (the 3-forward TaoMate adapter, about 6
minutes). Measured 2026-11-08 on the same picture, prompt, seed and speech WAV: the mouth still follows the voice and
stops when it stops, a little softer than the full run. Other speed adapters (Turbo, Flex) are untested with audio drive.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import finish_engineNG as finish
import finish_jobsNG as finish_jobs
import h3_engineNG as h3
import h3_jobsNG as h3_jobs
import h3_lora_libraryNG as h3_lora_lib
import h3_refmod_libraryNG as h3_refmod_lib
import speech_engineNG as speech
import speech_jobsNG as speech_jobs

TALK_CLIP_S = 5.0
TALK_QUALITIES = ("draft", "standard")
TALK_STEPS = 16
TALK_MAX_SUBJECT = 200
TALK_MAX_SCENE = 400
_MAX_JOBS_KEPT = 20
_POLL_S = 1.0
_STEP_RE = re.compile(r"step (\d+)/(\d+)")
_STAGE_RE = re.compile(r"^== (\w+) ==")


class TalkRequestError(ValueError):
    pass


class TalkJobCancelled(Exception):
    pass


def build_prompt_ng(subject: str, scene: str = "") -> str:
    """The H3 prompt: a close-up of `subject` talking straight to camera. Neutral wording ("they") so it fits
    any subject; the lips line is what the audio drive needs the text to agree with."""
    subject = (subject or "").strip()
    scene = (scene or "").strip()
    who = subject or "a person"
    text = (f"Live-action, close-up portrait, {who} looks straight into the camera and speaks directly to the "
            f"viewer, lips moving clearly with each word while talking, calm warm light, shallow depth of "
            f"field, steady camera.")
    if scene:
        text += f" {scene}"
    return text


@dataclass
class TalkJobNG:
    job_id: str
    text: str
    kind: str                       # voice: "preset" | "character" | "clip"
    voice: str
    character_voice: str
    voice_label: str
    style_id: str
    style_text: str
    language: str
    speed: float
    ref_text: str
    voice_consent: bool
    subject: str
    scene: str
    prompt: str
    lora_name: str
    lora_path: str
    lora_strength: float
    quality: str
    fast: bool
    width: int
    height: int
    seed: Optional[int]
    timeout_s: Optional[float]
    first_frame: Optional[bytes] = field(default=None, repr=False)
    first_frame_ext: str = ".jpg"
    clip_bytes: Optional[bytes] = field(default=None, repr=False)
    clip_ext: str = ""
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    stage: str = "queued"           # queued | speaking | rendering | done | failed | cancelled
    error: Optional[str] = None
    error_type: Optional[str] = None
    cancelled: bool = False
    speech_job_id: Optional[str] = None
    h3_job_id: Optional[str] = None
    mp4_path: Optional[str] = None
    finish: bool = False
    finish_params: dict = field(default_factory=dict)
    finish_job_id: Optional[str] = None
    finish_error: Optional[str] = None
    refmod_name: str = ""
    refmod_entry: dict = field(default_factory=dict, repr=False)


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
        _JOBS.pop(finished.pop(0).job_id, None)   # the record only; the speech wav and the mp4 stay with their jobs


def _wait_for(job: TalkJobNG, done, cancel_stage) -> None:
    """Block until `done()`; if the Talk job is cancelled meanwhile, cancel the running stage once and keep
    waiting for it to wind down, then raise."""
    cancel_sent = False
    while not done():
        if job.cancelled and not cancel_sent:
            cancel_stage()
            cancel_sent = True
        time.sleep(_POLL_S)
    if job.cancelled:
        raise TalkJobCancelled("cancelled")


def _run_job(job: TalkJobNG) -> None:
    job.stage = "speaking"
    sj = speech_jobs.start_speech_job_ng(
        job.text, job.kind, voice=job.voice, character=job.character_voice, style_id=job.style_id,
        style_text=job.style_text, language=job.language, speed=job.speed, seed=job.seed, ref_text=job.ref_text,
        voice_consent=job.voice_consent, clip_bytes=job.clip_bytes, clip_ext=job.clip_ext)
    job.speech_job_id = sj.job_id
    job.clip_bytes = None                      # the speech job holds its own copy in its folder
    _wait_for(job, lambda: sj.finished_at is not None, lambda: speech_jobs.cancel_speech_job_ng(sj.job_id))
    if sj.error:
        raise RuntimeError(f"speech: {sj.error}")
    secs = sj.audio_seconds or 0.0
    if secs > TALK_CLIP_S + 0.05:
        raise TalkRequestError(
            f"the speech runs {secs:.1f} s but a talking clip is {TALK_CLIP_S:.0f} s -- shorten the script "
            f"(about 12 words) or raise the speed")
    wav = Path(sj.audio_path).read_bytes()

    job.stage = "rendering"
    hj = h3_jobs.start_h3_job_ng(
        job.first_frame, job.first_frame_ext, job.prompt, TALK_CLIP_S, seed=job.seed,
        model="h3ref" if job.refmod_name else "h3q8", refmods=[job.refmod_entry] if job.refmod_name else [],
        width=job.width, height=job.height, steps=TALK_STEPS, tristep=job.fast,
        loras=[{"name": job.lora_name, "path": job.lora_path, "strength": job.lora_strength}] if job.lora_name else [],
        timeout_s=job.timeout_s, audio_bytes=wav, audio_ext=".wav")
    job.h3_job_id = hj.job_id
    job.first_frame = None
    _wait_for(job, lambda: hj.finished_at is not None, lambda: h3_jobs.cancel_h3_job_ng(hj.job_id))
    if hj.error:
        raise RuntimeError(hj.error)
    job.mp4_path = hj.mp4_path

    if job.finish:
        job.stage = "finishing"
        try:
            fj = finish_jobs.start_finish_job_ng(hj.job_id, job.finish_params)
            job.finish_job_id = fj.job_id
            _wait_for(job, lambda: fj.finished_at is not None, lambda: finish_jobs.cancel_finish_job_ng(fj.job_id))
            if fj.error:
                job.finish_error = fj.error
        except TalkJobCancelled:
            raise
        except Exception as e:  # noqa: BLE001 -- the render is safe; say why the finish did not happen
            job.finish_error = str(e)


def _worker_loop() -> None:
    while True:
        job_id = _QUEUE.get()
        job = _JOBS.get(job_id)
        if job is None:
            continue
        if job.cancelled:
            job.stage, job.error, job.error_type = "cancelled", "cancelled before it started", "TalkJobCancelled"
            job.finished_at = time.time()
            continue
        job.dispatched_at = time.time()
        try:
            _run_job(job)
            job.stage = "done"
        except TalkJobCancelled:
            job.stage, job.error, job.error_type = "cancelled", "cancelled", "TalkJobCancelled"
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.stage, job.error, job.error_type = "failed", str(e), type(e).__name__
        finally:
            job.finished_at = time.time()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        threading.Thread(target=_worker_loop, name="talk-job-worker-ng", daemon=True).start()
        _WORKER_STARTED = True


def start_talk_job_ng(text: str, kind: str, voice: str = "", character_voice: str = "", style_id: str = "natural",
                      style_text: str = "", language: str = "english", speed: float = 1.0, ref_text: str = "",
                      voice_consent: bool = False, clip_bytes: Optional[bytes] = None, clip_ext: str = "",
                      subject: str = "", scene: str = "", lora_name: str = "", lora_strength: float = 1.0,
                      first_frame: Optional[bytes] = None, first_frame_ext: str = ".jpg", quality: str = "draft",
                      seed: Optional[int] = None, fast: bool = False, finish_pass: bool = False,
                      finish_params: Optional[dict] = None, refmod_name: str = "") -> TalkJobNG:
    """Validates everything that can be checked now (so a bad request fails the POST, not 20 minutes in) and
    enqueues. A LoRA and a saved voice are looked up by NAME in the shared library, never a client path."""
    text = (text or "").replace("\r\n", "\n").strip()
    subject = (subject or "").strip()
    scene = (scene or "").strip()
    if len(subject) > TALK_MAX_SUBJECT or len(scene) > TALK_MAX_SCENE:
        raise TalkRequestError("the description is too long")
    if quality not in TALK_QUALITIES:
        raise TalkRequestError(f"quality must be one of {', '.join(TALK_QUALITIES)}")
    if isinstance(lora_strength, bool) or not 0.0 <= float(lora_strength) <= 2.0:
        raise TalkRequestError("LoRA strength must be between 0 and 2")

    # The voice, exactly as the Speech view checks it.
    has_ref = False
    voice_label = ""
    if kind == "character":
        if speech.resolve_voice_sample_ng(character_voice) is None:
            raise speech.SpeechRequestError("That character has no saved voice sample.")
        has_ref, voice_label = True, character_voice
    elif kind == "clip":
        if not clip_bytes:
            raise speech.SpeechRequestError("Add a voice clip first.")
        has_ref, voice_label = True, "uploaded clip"
    elif kind == "preset":
        voice_label = next((p["label"] for p in speech.PRESETS if p["id"] == voice), voice)
    speech.validate_request_ng(kind, text, voice, language, float(speed), has_ref_audio=has_ref,
                               voice_consent=bool(voice_consent))

    lora_path = ""
    if lora_name:
        lora_path = h3_lora_lib.resolve_h3_lora_path_ng(lora_name) or ""
        if not lora_path:
            raise TalkRequestError(f"unknown or unusable LoRA {lora_name!r}")
    refmod_entry: dict = {}
    if refmod_name:
        if first_frame or lora_name:
            raise TalkRequestError("a RefMod is the whole character here: leave the face picture and the LoRA empty")
        entry = h3_refmod_lib.resolve_h3_refmod_ng(refmod_name)
        if entry is None:
            raise TalkRequestError(f"unknown or unusable RefMod {refmod_name!r}")
        refmod_entry = {"name": refmod_name, "path": entry["path"], "strength": 1.0,
                        "images": entry["images"], "videos": entry["videos"]}
    elif not (lora_name or first_frame):
        raise TalkRequestError("pick a character (a RefMod or a LoRA) or add a face picture -- otherwise there is no face to speak")

    if fast and h3._resolve_h3_tristep_ng() is None:
        raise TalkRequestError("Fast needs the 3-step adapter (TaoMate) in the H3 turbo-lora folder, and it is not there")
    try:
        finish_clean = finish.normalize_params_ng(finish_params) if finish_pass else {}
    except finish.FinishRequestError as e:
        raise TalkRequestError(str(e)) from e
    width, height = h3.resolve_h3_quality_ng(quality)
    timeout_s = h3.resolve_h3_quality_timeout_ng(quality, TALK_CLIP_S)

    _ensure_worker_started()
    job = TalkJobNG(
        job_id=uuid.uuid4().hex[:12], text=text, kind=kind, voice=voice if kind == "preset" else "",
        character_voice=character_voice if kind == "character" else "", voice_label=voice_label,
        style_id=style_id, style_text=style_text, language=language, speed=float(speed),
        ref_text=(ref_text or "").strip(), voice_consent=bool(voice_consent), subject=subject, scene=scene,
        prompt=build_prompt_ng(subject, scene), lora_name=lora_name or "", lora_path=lora_path,
        lora_strength=float(lora_strength), quality=quality, fast=bool(fast), width=width, height=height, seed=seed,
        timeout_s=timeout_s, first_frame=first_frame or None, first_frame_ext=first_frame_ext or ".jpg",
        clip_bytes=clip_bytes if kind == "clip" else None, clip_ext=clip_ext,
        finish=bool(finish_pass), finish_params=finish_clean, refmod_name=refmod_name or "", refmod_entry=refmod_entry)
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job.job_id)
    return job


def get_job_ng(job_id: str) -> Optional[TalkJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_talk_job_ng(job_id: str) -> bool:
    """A queued job is flagged and skipped; a running one cancels whichever stage it is in."""
    job = get_job_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
    return True


def _render_progress(h3_job_id: Optional[str]) -> dict:
    """The H3 stage's own progress, read from its log tail: the runner's current stage and denoise step."""
    if not h3_job_id:
        return {}
    try:
        tail = h3_jobs.job_status_ng(h3_job_id).get("log_tail") or []
    except LookupError:
        return {}
    out: dict = {}
    for line in tail:
        m = _STAGE_RE.match(line.strip())
        if m:
            out["phase"] = m.group(1)
        m = _STEP_RE.search(line)
        if m:
            out["step"], out["steps"] = int(m.group(1)), int(m.group(2))
    return out


def start_finish_ng(job_id: str, params: Optional[dict] = None):
    """Soft-finish a Talk clip that is already done (the row's button). One finish at a time per Talk job."""
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")
    if job.stage != "done" or not job.h3_job_id:
        raise TalkRequestError("only a finished clip can be finished")
    if _finish_status(job).get("status") in ("queued", "rendering"):
        raise TalkRequestError("a finish is already running on this clip")
    fj = finish_jobs.start_finish_job_ng(job.h3_job_id, params)
    job.finish_job_id, job.finish_error = fj.job_id, None
    return fj


def _finish_status(job: TalkJobNG) -> dict:
    if not job.finish_job_id:
        return {}
    try:
        return finish_jobs.job_status_ng(job.finish_job_id)
    except LookupError:
        return {}


def job_status_ng(job_id: str) -> dict:
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")
    status = {"queued": "queued", "done": "completed", "failed": "failed", "cancelled": "failed"}.get(job.stage, "rendering")
    h3_status = None
    if job.h3_job_id:
        try:
            h3_status = h3_jobs.job_status_ng(job.h3_job_id)["status"]
        except LookupError:
            pass
    fin = _finish_status(job)
    finished = fin.get("status") == "completed"
    original = f"/api/ng/h3/jobs/{job.h3_job_id}/video" if job.stage in ("finishing", "done") and job.h3_job_id else None
    return {
        "job_id": job.job_id, "status": status, "stage": job.stage, "text": job.text, "voice": job.voice_label,
        "kind": job.kind, "subject": job.subject, "lora": job.lora_name, "refmod": job.refmod_name, "quality": job.quality, "fast": job.fast,
        "width": job.width, "height": job.height, "seed": job.seed, "prompt": job.prompt,
        "error": job.error, "error_type": job.error_type, "cancelled": bool(job.cancelled),
        "queued_at": job.queued_at, "dispatched_at": job.dispatched_at, "finished_at": job.finished_at,
        "speech_job_id": job.speech_job_id, "h3_job_id": job.h3_job_id, "h3_status": h3_status,
        "progress": _render_progress(job.h3_job_id) if job.stage == "rendering" else {},
        "audio_url": f"/api/ng/speech/jobs/{job.speech_job_id}/audio" if job.speech_job_id and job.stage in ("rendering", "finishing", "done") else None,
        "finish": job.finish, "finish_status": fin.get("status"), "finish_error": job.finish_error or fin.get("error"),
        "finished": finished,
        "original_video_url": original if job.stage == "done" else None,
        "video_url": (fin.get("video_url") if finished else original) if job.stage == "done" and job.h3_job_id else None,
        "download_url": (fin.get("download_url") if finished else f"/api/ng/h3/jobs/{job.h3_job_id}/download")
                        if job.stage == "done" and job.h3_job_id else None,
    }


def list_jobs_ng() -> list:
    """Every Talk job the server still holds, oldest first: the page rebuilds its list from this."""
    with _JOBS_LOCK:
        ids = sorted(_JOBS, key=lambda jid: _JOBS[jid].queued_at or 0)
    out = []
    for jid in ids:
        try:
            out.append(job_status_ng(jid))
        except LookupError:
            continue
    return out
