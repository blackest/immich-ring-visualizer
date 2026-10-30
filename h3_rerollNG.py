"""immichRingNG -- re-roll one window of a chained H3 render and join it back in.

A chained render keeps each 5 s window as its own clip (`h3_<id>_wN.mp4`). Re-rolling window n
renders just that window again (a new seed, optionally a changed prompt, optionally starting on
the previous window's last frame) as an ordinary one-window job, then joins the clips again with
the chosen join at each seam (h3_stitchNG.py). The windows after n are not re-rendered: the seam
is a deliberate cut, dissolve or fade, not a continuation. Every change is a new film version
(`film_vN.mp4`) beside the original, listed in `windows.json` in the source job's folder.

This file reads and writes that folder and starts the one-window job; h3_jobsNG.py runs it and
calls `finish_reroll_ng` when it ends, and routes/h3_rerollNG.py is the HTTP side.
"""

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h3_engineNG as h3
import h3_jobsNG as h3_jobs
import h3_stitchNG as stitch

MANIFEST_NAME = "windows.json"
_CLIP_RE = re.compile(r"^h3_\d+_w(\d+)\.mp4$")
_FILM_RE = re.compile(r"^h3_\d+\.mp4$")
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(job_dir) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(job_dir), threading.Lock())


def window_seconds_ng() -> float:
    """How long one window is: the same 124 frames a chained render uses."""
    return h3._h3_chain_plan_ng(h3.H3_CHAIN_WINDOW_S)[0] / h3.H3_FRAME_RATE


def resolve_source_ng(job_id: str):
    """The finished job to re-roll from: the live one, or (after a restart) the settings the job
    saved beside its clips. LookupError when neither exists."""
    job = h3_jobs.get_job_ng(job_id)
    if job is not None:
        if job.finished_at is None or job.error:
            raise ValueError("that render has not finished successfully")
        return job
    job_dir = Path(h3_jobs.H3GEN_DIR) / str(job_id)
    saved = job_dir / h3_jobs.SETTINGS_NAME
    if not job_id or "/" in str(job_id) or not saved.is_file():
        raise LookupError(f"no such job {job_id!r} (jobs made before windows could be re-rolled "
                          "are gone after a restart: render it again)")
    data = json.loads(saved.read_text(encoding="utf-8"))
    film = next((p for p in job_dir.iterdir() if _FILM_RE.match(p.name)), None)
    manifest = job_dir / MANIFEST_NAME
    if manifest.is_file():
        versions = json.loads(manifest.read_text(encoding="utf-8")).get("versions") or []
        if versions:
            film = job_dir / versions[-1]["film"]
    return SimpleNamespace(job_id=job_id, job_dir=job_dir, mp4_path=str(film) if film else None,
                           finished_at=1, error=None, **data)


def _clips_on_disk(job_dir: Path) -> dict:
    found = {}
    for path in job_dir.iterdir():
        m = _CLIP_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path.name
    return found


def load_manifest_ng(src) -> dict:
    """windows.json for the source job, built from the runner's own files the first time."""
    job_dir = Path(src.job_dir)
    path = job_dir / MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    clips = _clips_on_disk(job_dir)
    if len(clips) < 2 or sorted(clips) != list(range(1, len(clips) + 1)):
        raise ValueError("this render is a single window, so there is nothing to re-roll apart from "
                         "rendering it again")
    metrics_file = next(job_dir.glob("*.metrics.json"), None)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file else {}
    per_window = {int(w["window"]): w for w in metrics.get("windows") or []}
    raw = sum(stitch.probe_frames_ng(job_dir / clips[n]) for n in clips) - (len(clips) - 1)
    delivered = int((metrics.get("settings") or {}).get("chain_delivered_frames")
                    or metrics.get("delivered_frames") or raw)
    man = {"windows": [{"index": n, "clip": clips[n], "seed": (per_window.get(n) or {}).get("seed"),
                        "prompt": (per_window.get(n) or {}).get("prompt", ""), "rerolled": False}
                       for n in sorted(clips)],
           "joins": [], "trim_tail": max(0, raw - delivered),
           "original_film": Path(src.mp4_path).name if src.mp4_path else "", "versions": []}
    save_manifest_ng(job_dir, man)
    return man


def save_manifest_ng(job_dir, man: dict) -> None:
    path = Path(job_dir) / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def describe_ng(job_id: str) -> dict:
    """What the windows panel shows: each window, the current join at each seam, every version."""
    src = resolve_source_ng(job_id)
    man = load_manifest_ng(src)
    joins = stitch.normalise_joins_ng(man["joins"], len(man["windows"]) - 1)
    return {"job_id": job_id, "windows": man["windows"], "joins": joins,
            "window_seconds": round(window_seconds_ng(), 3),
            "versions": [{"n": 0, "label": "original"}] + [
                {"n": v["n"], "label": v["change"], "joins": v["joins"]} for v in man["versions"]],
            "join_types": list(stitch.JOIN_TYPES),
            "min_seconds": stitch.MIN_SECONDS, "max_seconds": stitch.MAX_SECONDS}


def last_frame_png_ng(clip) -> bytes:
    """The final frame of a clip as PNG bytes."""
    last = stitch.probe_frames_ng(clip) - 1
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"select=eq(n\\,{last})",
                          "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
                         capture_output=True, timeout=120)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError("could not read the last frame of " + Path(clip).name)
    return out.stdout


def _stitch_version(src, man: dict, joins, change: str, on_log=None) -> dict:
    """Join the current clips with `joins` into the next film version and make it the job's video."""
    job_dir = Path(src.job_dir)
    version = len(man["versions"]) + 1
    film = f"film_v{version}.mp4"
    done = stitch.stitch_windows_ng([job_dir / w["clip"] for w in man["windows"]], joins,
                                    job_dir / film, trim_tail=man["trim_tail"], on_log=on_log)
    man["joins"] = done["joins"]
    man["versions"].append({"n": version, "film": film, "change": change, "joins": done["joins"],
                            "at": time.time(), "frames": done["frames"]})
    save_manifest_ng(job_dir, man)
    src.mp4_path = str(job_dir / film)
    return man["versions"][-1]


def restitch_ng(job_id: str, joins) -> dict:
    """Join the same clips again with different joins (no rendering)."""
    src = resolve_source_ng(job_id)
    with _lock_for(src.job_dir):
        man = load_manifest_ng(src)
        stitch.normalise_joins_ng(joins, len(man["windows"]) - 1)  # refuse bad joins before any work
        return _stitch_version(src, man, joins, "joins changed")


def start_reroll_ng(job_id: str, window: int, *, seed=None, prompt=None, from_previous: bool = True,
                    joins=None):
    """Queue the one-window render for `window` (1-based) and return its job. When it ends the new
    clip replaces the window and a new film version is joined with `joins` (the current joins
    where none are given)."""
    src = resolve_source_ng(job_id)
    man = load_manifest_ng(src)
    count = len(man["windows"])
    if not 1 <= window <= count:
        raise ValueError(f"window must be 1 to {count} (got {window})")
    joins = stitch.normalise_joins_ng(joins if joins is not None else man["joins"], count - 1)
    entry = man["windows"][window - 1]
    text = (entry["prompt"] if prompt is None else str(prompt)).strip()
    if not text:
        raise ValueError("the window has no prompt to render")
    if seed in (None, ""):
        seed = random.randrange(2 ** 31)
    first, first_ext = None, ".png"
    job_dir = Path(src.job_dir)
    if window > 1 and from_previous:
        first = last_frame_png_ng(job_dir / man["windows"][window - 2]["clip"])
    elif window == 1 and getattr(src, "has_image", False):
        ref = next(job_dir.glob("ref.*"), None)
        if ref is not None:
            first, first_ext = ref.read_bytes(), ref.suffix
    return h3_jobs.start_h3_job_ng(
        first, first_ext, text, window_seconds_ng(), int(seed), model=src.model, width=src.width,
        height=src.height, steps=src.steps, turbo=src.turbo, v13=getattr(src, "v13", False),
        tristep=src.tristep, flex=getattr(src, "flex", False), loras=[dict(l) for l in src.loras],
        reroll={"source": job_id, "window": window, "joins": joins, "version": None},
        on_done=finish_reroll_ng)


def finish_reroll_ng(job) -> None:
    """Called by the worker when a re-roll's one-window render has finished without error."""
    info = job.reroll
    src = resolve_source_ng(info["source"])
    with _lock_for(src.job_dir):
        man = load_manifest_ng(src)
        n = info["window"]
        version = len(man["versions"]) + 1
        clip = f"reroll_w{n}_v{version}.mp4"
        shutil.copyfile(job.mp4_path, Path(src.job_dir) / clip)
        man["windows"][n - 1].update(clip=clip, seed=job.seed, prompt=job.prompt, rerolled=True)
        made = _stitch_version(src, man, info["joins"], f"window {n} rerolled (seed {job.seed})",
                               on_log=job.append_log)
    info["version"] = made["n"]
