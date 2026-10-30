"""immichRingNG -- build a RefMod from pictures, as a queued task job.

A RefMod is a saved, VAE-encoded set of pictures that the Ref2VA model reads as extra reference rows
(h3_ref2vaNG.py uses them; h3_refmod_libraryNG.py lists them). The library's files are about twenty
pictures of one person on a 512x512 canvas. This validates the pictures and the name, then runs the
fork's own scripts/make_refmod.py (the same encoder a --reference takes at render time) and drops the
file in the refmods folder, where the RefMod rows pick it up. It runs in the H3 queue so it never
shares the GPU with a render.
"""

import io
import re
import subprocess
from pathlib import Path

import h3_engineNG as h3
import h3_jobsNG as h3_jobs
import h3_referencesNG as refs
from h3_refmod_libraryNG import refmods_dir_ng

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
CANVASES = {"square": (512, 512), "landscape": (768, 448)}
MIN_PICTURES = 1
MAX_PICTURES = 24          # 24 x 256 rows = 6,144, the biggest file in the library
MIN_SIDE = 128
LATENT_RATIO = 16


def script_path_ng() -> Path:
    return Path(h3.H3_RUNNER).with_name("make_refmod.py")


def rows_per_picture_ng(canvas: str) -> int:
    width, height = CANVASES[canvas]
    return (height // LATENT_RATIO // 2) * (width // LATENT_RATIO // 2)


def evenly_spaced_ng(count: int, keep: int) -> list:
    """Indices of `keep` of `count` pictures, evenly spaced, first and last included (22 -> 8 is
    0,3,6,...,21, the way the library's 8-picture files were cut)."""
    if keep >= count:
        return list(range(count))
    if keep == 1:
        return [0]
    return sorted({round(i * (count - 1) / (keep - 1)) for i in range(keep)})


def validate_ng(name, pictures, canvas, max_pictures=None) -> tuple:
    """-> (clean name, kept [(bytes, ext)], rows). Raises ValueError with one plain sentence."""
    from PIL import Image
    name = str(name or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("the name must be letters, digits, - or _ (at most 63 characters, starting with a letter or digit)")
    if canvas not in CANVASES:
        raise ValueError(f"canvas must be one of {', '.join(CANVASES)} (got {canvas!r})")
    target = refmods_dir_ng() / f"{name}.safetensors"
    if target.exists():
        raise ValueError(f"a RefMod named {name} already exists; pick another name")
    pictures = list(pictures or [])
    if len(pictures) < MIN_PICTURES:
        raise ValueError("add at least one picture")
    for data, ext in pictures:
        if ext.lower() not in refs.PICTURE_EXTS:
            raise ValueError(f"RefMod pictures must be {', '.join(refs.PICTURE_EXTS)} (got {ext or 'no extension'})")
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
                width, height = img.size
        except Exception:  # noqa: BLE001 -- Pillow raises many kinds for a bad file
            raise ValueError("one of the files is not a readable picture") from None
        if min(width, height) < MIN_SIDE:
            raise ValueError(f"pictures must be at least {MIN_SIDE} px on the short side (got {width}x{height})")
    limit = MAX_PICTURES
    if max_pictures not in (None, ""):
        try:
            limit = min(limit, max(1, int(max_pictures)))
        except (TypeError, ValueError):
            raise ValueError("the picture limit must be a whole number") from None
    kept = [pictures[i] for i in evenly_spaced_ng(len(pictures), limit)]
    return name, kept, len(kept) * rows_per_picture_ng(canvas)


def start_refmod_job_ng(name, pictures, canvas="square", max_pictures=None, description=""):
    """Queue the build. `pictures` is [(bytes, ext)] in the order they were chosen. Raises ValueError
    / FileNotFoundError at once; returns the job."""
    name, kept, rows = validate_ng(name, pictures, canvas, max_pictures)
    script = script_path_ng()
    python = h3._resolve_h3_python_ng(h3.H3Config())
    if not script.is_file():
        raise FileNotFoundError(f"this H3 install has no {script.name} (the fork build with RefMod making)")
    if not python:
        raise FileNotFoundError("the H3 python was not found")
    compact = h3._resolve_h3_compact_root_ng(h3.H3Config())
    if not compact:
        raise FileNotFoundError("the H3 video VAE was not found")
    width, height = CANVASES[canvas]
    out_dir = refmods_dir_ng()

    def run(job):
        pics = job.job_dir / "pictures"
        pics.mkdir(exist_ok=True)
        for n, (data, ext) in enumerate(kept):
            (pics / f"{n:03d}{ext.lower()}").write_bytes(data)
        cmd = [python, str(script), str(pics), "--name", name, "--compact-root", str(compact),
               "--out", str(out_dir), "--width", str(width), "--height", str(height),
               "--description", description or f"RefMod dataset {name} ({len(kept)} pictures)"]
        job.append_log("[refmod] " + " ".join(cmd[1:]))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True, env=h3._clean_subprocess_env_ng(),
                                cwd=str(script.parent.parent))
        job._proc = proc
        for line in proc.stdout:
            job.append_log(line.rstrip())
        code = proc.wait()
        if job.cancelled:
            raise RuntimeError("cancelled")
        if code != 0 or not (out_dir / f"{name}.safetensors").is_file():
            raise RuntimeError(f"making the RefMod failed (exit {code}): " + " | ".join(job.log_tail(4)))
        return None  # no video: the result is the file in the refmods folder

    return h3_jobs.start_task_job_ng(
        "refmod", f"RefMod {name} from {len(kept)} pictures", run,
        meta={"name": name, "pictures": len(kept), "given": len(pictures), "rows": rows, "canvas": canvas})
