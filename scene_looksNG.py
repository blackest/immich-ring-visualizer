"""immichRingNG -- make a character's pictures with Klein, from ONE picture of them.

Two things, both a Klein job (or eight) with the character's picture as the `subject` reference:

  * a LOOK: the same person somewhere else. The Scene view uses a look as the first frame of a clip, so a character
    can be on the battlements in one line and in the great hall in the next. Measured 2026-11-08: from a bathroom
    portrait Klein kept the face and pinned-up hair and put her on castle battlements (about 170 s a picture).

  * an IDENTITY KIT: the eight-picture dataset recipe from the RefMod tutorial the user shared -- four face close-ups
    at different angles and expressions and four body shots (waist-up, full-body front, three-quarter, from behind),
    every prompt ending with the same "same person, identical face, hairstyle, skin tone and outfit" line -- so the
    identity stays put while the angle changes (different LOOKS are the opposite). Optionally the pictures are turned
    straight into a RefMod with the app's own RefMod maker. Eight pictures is also the number found to work earlier.

No model runs here: this builds prompts and queues Klein jobs (klein_jobsNG) and, for a kit, one RefMod job
(h3_refmod_makeNG). A kit is one background thread that submits the eight Klein jobs together and waits for them.
"""

from __future__ import annotations

import io
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h3_refmod_makeNG as refmod_make
import klein_engineNG as klein_engine
import klein_jobsNG as klein_jobs
from configNG import KLEIN_DIR

IDENTITY = "Same person as image one, identical face, hairstyle, skin tone, and outfit."
# (label, what the picture shows). The face four vary the angle and the expression; the body four vary the framing.
KIT_SHOTS = (
    ("face, turned left", "Close-up portrait, face turned slightly to the left, calm neutral expression, soft light."),
    ("face, three-quarter right", "Close-up portrait, three-quarter view toward the right, a gentle smile."),
    ("face, looking up", "Close-up portrait, chin slightly raised, eyes looking up and to the side, lips slightly parted as if about to speak."),
    ("face, front, laughing", "Close-up portrait, facing the camera, laughing naturally with teeth showing."),
    ("body, waist up", "Waist-up shot facing the camera, relaxed posture, hands visible."),
    ("body, full front", "Full-body shot, standing, front view, whole figure in frame."),
    ("body, full three-quarter", "Full-body shot, standing, three-quarter view, whole figure in frame."),
    ("body, from behind", "Full-body shot seen from behind, whole figure in frame, head turned slightly."),
)
KIT_SIZE = 1024                  # square pictures: the RefMod maker's square canvas is 512x512
LOOK_SIZE = (1024, 576)          # 16:9, like the clips' first frames
MAX_SETTING = 160
MAX_LOOKS = 4
_POLL_S = 2.0
_MAX_KITS_KEPT = 6


class LooksRequestError(ValueError):
    pass


def _check_picture(data: bytes) -> None:
    from PIL import Image
    if not data:
        raise LooksRequestError("add the character's picture first")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
            w, h = im.size
    except Exception:  # noqa: BLE001 -- Pillow raises many kinds for a bad file
        raise LooksRequestError("that file is not a readable picture") from None
    if min(w, h) < 128:
        raise LooksRequestError(f"the picture is too small ({w}x{h}); use one at least 128 px on the short side")


def look_prompt_ng(setting: str, who: str = "", keep_outfit: bool = False) -> str:
    """The Klein prompt for the same person in a new place. `who` (optional) is a short description ("an older man
    with a bald head") that helps Klein when the picture is small; `keep_outfit` pins the clothes instead of letting
    the setting suggest them."""
    setting = " ".join((setting or "").split())
    if not setting:
        raise LooksRequestError("say where: describe the setting, e.g. 'in a torch-lit great hall'")
    if len(setting) > MAX_SETTING:
        raise LooksRequestError(f"the setting is too long (at most {MAX_SETTING} characters)")
    subject = f"The same {who.strip()}" if who and who.strip() else "The same person"
    keep = " Keep the same clothes." if keep_outfit else ""      # otherwise the setting may suggest the clothes
    return (f"{subject}, same face and same hair, now {setting}. Medium shot, looking toward the camera, "
            f"cinematic natural light.{keep}")


def kit_prompts_ng() -> list:
    """The eight (label, prompt) pairs, each ending with the identity line."""
    return [(label, f"{shot} {IDENTITY}") for label, shot in KIT_SHOTS]


def _submit_klein(picture: bytes, ext: str, prompt: str, n: int, width: int, height: int, seed) -> str:
    job_id, folder = klein_jobs.new_job_dir_ng()
    dest = folder / f"subject{(ext or '.png').lower() if (ext or '').lower() in ('.png', '.jpg', '.jpeg', '.webp') else '.png'}"
    dest.write_bytes(picture)
    job = klein_jobs.start_ng(job_id, prompt, n, width, height, seed, klein_engine.DEFAULT_MODEL,
                              [{"slot": "subject", "path": str(dest)}], {})
    return job["id"]


def start_look_ng(picture: bytes, ext: str, setting: str, who: str = "", n: int = 2, seed: Optional[int] = None,
                  keep_outfit: bool = False) -> str:
    """Queue a look. Returns the Klein job id; the page polls Klein's own status/image routes for the pictures."""
    _check_picture(picture)
    if not 1 <= int(n) <= MAX_LOOKS:
        raise LooksRequestError(f"ask for between 1 and {MAX_LOOKS} variations")
    prompt = look_prompt_ng(setting, who, keep_outfit)
    return _submit_klein(picture, ext, prompt, int(n), LOOK_SIZE[0], LOOK_SIZE[1], seed)


# ---- the identity kit -------------------------------------------------------------------------------------------

@dataclass
class KitJobNG:
    job_id: str
    picture: bytes
    ext: str
    name: str                                   # the RefMod to make from the pictures ("" = pictures only)
    seed: Optional[int]
    klein_ids: list = field(default_factory=list)
    shots: list = field(default_factory=list)   # [{label, status, klein_job, file}]
    stage: str = "queued"                       # queued | generating | refmod | done | failed | cancelled
    refmod_job_id: Optional[str] = None
    error: Optional[str] = None
    cancelled: bool = False
    queued_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


_KITS: dict = {}
_KITS_LOCK = threading.Lock()


def start_kit_ng(picture: bytes, ext: str, name: str = "", seed: Optional[int] = None) -> KitJobNG:
    """Validate (a bad request fails now, not after twenty minutes of pictures) and start the background run."""
    _check_picture(picture)
    name = (name or "").strip()
    if name:
        if not refmod_make.NAME_RE.match(name):
            raise LooksRequestError("the RefMod name must be letters, digits, - or _ (at most 63 characters)")
        if (refmod_make.refmods_dir_ng() / f"{name}.safetensors").exists():
            raise LooksRequestError(f"a RefMod named {name} already exists; pick another name")
    health = klein_engine.klein_health_ng()
    if not health["ready"]:
        raise LooksRequestError("Klein is not ready: install FLUX 3 and download " + klein_engine.DEFAULT_MODEL + " in Phosphene")
    job = KitJobNG(job_id=uuid.uuid4().hex[:12], picture=picture, ext=ext or ".png", name=name, seed=seed,
                   shots=[{"label": label, "status": "waiting", "klein_job": None, "file": None} for label, _ in KIT_SHOTS])
    with _KITS_LOCK:
        _KITS[job.job_id] = job
        old = sorted((j for j in _KITS.values() if j.finished_at is not None), key=lambda j: j.finished_at)
        while len(_KITS) > _MAX_KITS_KEPT and old:
            _KITS.pop(old.pop(0).job_id, None)
    threading.Thread(target=_run_kit, args=(job,), name=f"kit-{job.job_id}", daemon=True).start()
    return job


def _sleep() -> None:
    time.sleep(_POLL_S)


def _run_kit(job: KitJobNG) -> None:
    try:
        job.stage = "generating"
        for shot, (label, prompt) in zip(job.shots, kit_prompts_ng()):
            seed = None if job.seed is None else job.seed + len(job.klein_ids)
            shot["klein_job"] = _submit_klein(job.picture, job.ext, prompt, 1, KIT_SIZE, KIT_SIZE, seed)
            job.klein_ids.append(shot["klein_job"])
        pictures = [(job.picture, job.ext if job.ext.lower() in (".png", ".jpg", ".jpeg", ".webp") else ".png")]
        for shot in job.shots:
            shot["status"] = "rendering"
            while True:
                if job.cancelled:                      # stop now: tell Klein to drop every picture still to come
                    for klein_id in job.klein_ids:
                        klein_jobs.cancel_ng(klein_id)
                    shot["status"] = "failed"
                    raise _Cancelled()
                kj = klein_jobs.get_ng(shot["klein_job"])
                if kj is None or kj["status"] in ("error", "cancelled"):
                    shot["status"] = "failed"
                    raise RuntimeError(f"{shot['label']}: " + ((kj or {}).get("error") or "the picture job disappeared"))
                if kj["status"] == "done":
                    break
                _sleep()
            file = kj["results"][0]["file"]
            shot["file"], shot["status"] = file, "done"
            pictures.append(((Path(KLEIN_DIR) / shot["klein_job"] / "out" / file).read_bytes(), ".png"))
        if job.cancelled:
            raise _Cancelled()
        if job.name:
            job.stage = "refmod"
            rj = refmod_make.start_refmod_job_ng(job.name, pictures, canvas="square",
                                                 description=f"identity kit {job.name} ({len(pictures)} pictures)")
            job.refmod_job_id = rj.job_id
            while rj.finished_at is None:
                if job.cancelled:
                    raise _Cancelled()
                _sleep()
            if rj.error:
                raise RuntimeError(f"building the RefMod: {rj.error}")
        job.stage = "done"
    except _Cancelled:
        job.stage, job.error = "cancelled", "cancelled"
    except Exception as e:  # noqa: BLE001 -- job.error is the report
        job.stage, job.error = "failed", str(e)
    finally:
        job.finished_at = time.time()


class _Cancelled(Exception):
    pass


def get_kit_ng(job_id: str) -> Optional[KitJobNG]:
    with _KITS_LOCK:
        return _KITS.get(job_id)


def cancel_kit_ng(job_id: str) -> bool:
    job = get_kit_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
    return True


def kit_status_ng(job_id: str) -> dict:
    job = get_kit_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")
    done = sum(1 for s in job.shots if s["status"] == "done")
    return {
        "job_id": job.job_id, "stage": job.stage, "name": job.name, "done": done, "total": len(job.shots),
        "error": job.error, "refmod_job_id": job.refmod_job_id,
        "shots": [{"label": s["label"], "status": s["status"],
                   "url": f"/api/ng/klein/image/{s['klein_job']}/{s['file']}" if s["file"] else None} for s in job.shots],
        "finished": job.finished_at is not None,
    }
