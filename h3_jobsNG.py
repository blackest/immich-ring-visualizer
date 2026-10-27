"""NG job queue for MiniMax-H3 prompt/image-to-video renders (see
h3_engineNG.py).

Twin of video_jobsNG.py's queue design -- same reasoning throughout
(FIFO queue not a busy-error lock, own worker thread, ring-buffer
eviction), just pointed at h3_engineNG instead of ltx_engineNG. Kept
as its own file rather than folded into video_jobsNG.py: H3 is a
different subprocess/venv/model with its own field (model: "h3" |
"h3q8"), and per [[modular-engine-files]] a second engine gets a
sibling module, not a branch inside the first one's queue.
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

import h3_engineNG as h3
import h3_ref2vaNG as h3_ref2va
import h3_job_logsNG
from configNG import H3GEN_DIR

_MAX_JOBS_KEPT = 20  # ring-buffer cap, same reasoning as video_jobsNG


@dataclass
class H3JobNG:
    job_id: str
    prompt: str
    duration_s: float
    seed: Optional[int]
    job_dir: Path
    model: h3.H3Model = "h3q8"
    has_image: bool = True  # False = text-to-video, no keyframe
    width: int = h3.H3_WIDTH
    height: int = h3.H3_HEIGHT
    steps: int = h3.H3_STEPS
    turbo: bool = False  # 4-step distilled LoRA -- forces steps to H3_TURBO_STEPS
    v13: bool = False  # lightx2v v1.3 faster-Best -- forces steps to H3_V13_STEPS
    tristep: bool = False  # TaoMate 3-step "Fast" -- forces steps to H3_TRISTEP_STEPS; allowed with RefMods
    loras: list = field(default_factory=list)  # [{"name", "path", "strength"}] user LoRAs
    refmods: list = field(default_factory=list)  # [{"name", "path", "strength", "images", "videos"}] RefMods
    timeout_s: Optional[float] = None  # None = H3Config's own default (3600s)
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    mp4_path: Optional[str] = None
    log_lines: list = field(default_factory=list)
    cancelled: bool = False
    _log_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    def append_log(self, line: str) -> None:
        with self._log_lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 2000:  # durable log copies this whole list
                self.log_lines = self.log_lines[-2000:]

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
            job.error_type = "H3JobCancelled"
            continue
        job.dispatched_at = time.time()
        image_path = next(job.job_dir.glob("ref.*"), None)
        try:
            if job.has_image and image_path is None:
                raise FileNotFoundError("keyframe image went missing before render")
            cfg_kwargs = {"model": job.model}
            if job.timeout_s is not None:
                cfg_kwargs["timeout_s"] = job.timeout_s
            result = h3.generate_h3_video_ng(
                prompt=job.prompt, image_path=str(image_path) if image_path else None,
                duration_s=job.duration_s, output_dir=job.job_dir,
                seed=job.seed, config=h3.H3Config(**cfg_kwargs),
                width=job.width, height=job.height, steps=job.steps,
                turbo=job.turbo, v13=job.v13, tristep=job.tristep,
                loras=[(l["path"], l["strength"]) for l in job.loras],
                refmods=job.refmods,
                on_log=job.append_log,
                on_proc_start=lambda p: setattr(job, "_proc", p))
            job.mp4_path = result["mp4_path"]
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.error = str(e)
            job.error_type = type(e).__name__
        finally:
            job.finished_at = time.time()
            try:
                h3_job_logsNG.record_job_ng(job)
            except Exception:
                pass  # the durable log must never take the render pipeline down


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker_loop, name="h3-job-worker-ng", daemon=True)
        t.start()
        _WORKER_STARTED = True


def start_h3_job_ng(image_bytes: Optional[bytes], image_ext: str, prompt: str,
                     duration_s: float, seed: Optional[int] = None,
                     model: h3.H3Model = "h3q8",
                     width: Optional[int] = None, height: Optional[int] = None,
                     steps: Optional[int] = None, turbo: bool = False,
                     v13: bool = False, loras: Optional[list] = None,
                     refmods: Optional[list] = None,
                     timeout_s: Optional[float] = None,
                     tristep: bool = False) -> H3JobNG:
    """Writes image_bytes directly into a fresh per-job directory under
    H3GEN_DIR -- same "this write IS the job's own copy" shape as
    video_jobsNG.start_video_job_ng (see its own docstring for the full
    reasoning). `loras` is an optional [{"name", "path", "strength"}] list
    (paths resolved from h3_lora_libraryNG by the route). image_bytes may be None for a text-to-video job -- H3
    generates from the prompt alone, no --first-frame keyframe."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    if model not in ("h3", "h3q8", "h3ref"):
        raise ValueError(f"model must be 'h3', 'h3q8' or 'h3ref' (got {model!r})")
    if turbo and model != "h3":
        # Same guard generate_h3_video_ng enforces -- checked here too so
        # this fails the POST synchronously (400) instead of wasting a
        # queued job slot on something the worker will reject anyway.
        raise ValueError(
            "turbo requires model='h3' (bf16) -- the turbo LoRA silently "
            "fails to apply against the quantized 'h3q8' DiT")
    if turbo and v13:
        raise ValueError("turbo and v13 are separate speed adapters -- pick one")
    if v13 and model != "h3":
        raise ValueError("v13 requires model='h3' (bf16) -- same limitation as turbo")
    if tristep and (turbo or v13):
        raise ValueError("Fast (3-step) is its own speed adapter -- not with turbo or v13")
    if tristep and h3._resolve_h3_tristep_ng() is None:
        raise FileNotFoundError(
            "Fast (TaoMate 3-step) adapter not found in the H3 turbo-lora folder")
    h3._validate_h3_duration_ng(duration_s)
    # RefMods are checked here too, so a combination the runner cannot do (a speed
    # adapter, a keyframe, a chained length, the wrong model) fails the POST
    # synchronously instead of after the DiT has loaded.
    refmods = h3_ref2va.validate_refmods_ng(
        refmods, model=model, image_path="keyframe" if image_bytes is not None else None,
        turbo=turbo, v13=v13, chain_windows=h3._h3_chain_plan_ng(duration_s)[1],
        runner=h3._resolve_h3_runner_ng(h3.H3Config()) or h3.H3_RUNNER, tristep=tristep)

    width = h3.H3_WIDTH if width is None else int(width)
    height = h3.H3_HEIGHT if height is None else int(height)
    # Same override generate_h3_video_ng itself applies -- computed here
    # too so job.steps (and the status API) reports the step count that
    # will actually run, not whatever default was requested before turbo
    # was known.
    steps = (h3.H3_TURBO_STEPS if turbo else h3.H3_V13_STEPS if v13
             else h3.H3_TRISTEP_STEPS if tristep
             else (h3.H3_STEPS if steps is None else int(steps)))
    h3._validate_h3_dims_ng(width, height)
    if v13 and (width, height) != (h3.H3_WIDTH, h3.H3_HEIGHT):
        raise ValueError(
            f"v13 is only validated at the 'high' {h3.H3_WIDTH}x{h3.H3_HEIGHT} canvas")
    if v13 and not h3.H3_V13_LORA_PATH.is_file():
        raise FileNotFoundError(
            f"lightx2v v1.3 adapter not found at {h3.H3_V13_LORA_PATH}")

    loras = loras or []
    cleaned = h3._validate_h3_loras_ng([(l["path"], l["strength"]) for l in loras])
    loras = [{"name": l["name"], "path": p, "strength": st}
             for l, (p, st) in zip(loras, cleaned)]
    if loras:
        limit = h3.h3_lora_limit_ng(h3.H3_RUNNER, speed_adapter=bool(turbo or v13 or tristep))
        if len(loras) > limit:
            raise ValueError(
                f"this H3 runner takes at most {limit} LoRA(s) with this setup "
                f"(got {len(loras)})")

    has_image = image_bytes is not None

    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(H3GEN_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if has_image:
        ext = image_ext.lower() if image_ext and image_ext.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".jpg"
        (job_dir / f"ref{ext}").write_bytes(image_bytes)

    job = H3JobNG(job_id=job_id, prompt=prompt, duration_s=duration_s,
                  seed=seed, job_dir=job_dir, model=model, has_image=has_image,
                  width=width, height=height, steps=steps, turbo=turbo, v13=v13,
                  tristep=tristep, loras=loras, refmods=refmods, timeout_s=timeout_s)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_old_jobs_locked()
    _QUEUE.put(job_id)
    return job


def get_job_ng(job_id: str) -> Optional[H3JobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_h3_job_ng(job_id: str) -> bool:
    """Same semantics as video_jobsNG.cancel_video_job_ng: a queued job
    is flagged and skipped, a rendering job's subprocess is killed
    (surfaces as the usual H3JobCancelled)."""
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
        "job_id": job.job_id,
        "status": status,
        "prompt": job.prompt,
        "duration_s": job.duration_s,
        "model": job.model,
        "width": job.width,
        "height": job.height,
        "steps": job.steps,
        "turbo": job.turbo,
        "v13": job.v13,
        "tristep": job.tristep,
        "loras": [{"name": l["name"], "strength": l["strength"]} for l in job.loras],
        "refmods": [{"name": r["name"], "strength": r["strength"]} for r in job.refmods],
        "error": job.error,
        "error_type": job.error_type,
        "queued_at": job.queued_at,
        "dispatched_at": job.dispatched_at,
        "finished_at": job.finished_at,
        "log_tail": job.log_tail(40),
        "video_url": (f"/api/ng/h3/jobs/{job.job_id}/video"
                      if status == "completed" else None),
    }
