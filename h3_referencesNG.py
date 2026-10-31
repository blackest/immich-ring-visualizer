"""immichRingNG -- `--ref2va-reference` for the H3 runner: pictures and clips handed to a Ref2VA render
as references (swap a character into a clip, carry a face through a new scene).

Unlike a RefMod (h3_ref2vaNG.py: a saved latent from the library) a reference is a plain picture
or mp4 that the runner encodes itself at render time, so there is nothing to build first. The
file's kind comes from its extension, and the order within each kind is its number: the first
picture is <Picture 1>, the first clip <Video 1>, which is what the compiler's brief names.
This file only checks what the runner would refuse and builds its arguments; h3_engineNG.py
calls it and h3_jobsNG.py stores the uploaded files.
"""

from pathlib import Path
from typing import Optional

import h3_loraNG as h3_lora

REFERENCE_FLAG = "--ref2va-reference"
REFERENCE_SIZE_FLAG = "--ref2va-reference-size"
PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CLIP_EXTS = (".mp4", ".mov", ".m4v", ".webm")
SIZES = ("match", "max")
MAX_PICTURES = 9
MAX_CLIPS = 3


def runner_supports_reference_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, REFERENCE_FLAG)


def kind_of_ng(path_or_name) -> Optional[str]:
    ext = Path(str(path_or_name)).suffix.lower()
    return "image" if ext in PICTURE_EXTS else "video" if ext in CLIP_EXTS else None


def clean_ext_ng(filename: str) -> str:
    """The extension a stored upload keeps, or ValueError for a file that is neither a picture
    nor a clip."""
    ext = Path(filename or "").suffix.lower()
    if ext not in PICTURE_EXTS + CLIP_EXTS:
        raise ValueError(f"a reference must be a picture ({', '.join(PICTURE_EXTS)}) or a clip "
                         f"({', '.join(CLIP_EXTS)}); got {filename!r}")
    return ext


def validate_references_ng(references, size: str = "match", *, model: str, image_path,
                           last_frame_path=None, keyframes=None, extend_from=None, refmods=None,
                           turbo: bool = False, v13: bool = False, chain_windows: int = 1,
                           runner, check_files: bool = True) -> list:
    """`references`: file paths -> the same list as strings. Raises ValueError with one plain
    sentence on anything the runner cannot do. An empty list is always fine. The job queue checks
    before the files are written (check_files=False); the engine checks again with them on disk."""
    references = [str(p) for p in references or []]
    if not references:
        return []
    if size not in SIZES:
        raise ValueError(f"reference size must be one of {', '.join(SIZES)} (got {size!r})")
    if model != "h3ref":
        raise ValueError("references need the Ref2VA model (model='h3ref'); "
                         f"{model!r} was never trained to read reference rows")
    if not runner_supports_reference_ng(runner):
        raise ValueError("this H3 runner has no --ref2va-reference support -- it needs the minimax-h3-mlx "
                         "build with the reference flag (branch reference-flag)")
    if image_path is not None or last_frame_path is not None or keyframes:
        raise ValueError("references do not combine with a first, last or keyframe image "
                         "(one conditioning source at a time)")
    if extend_from is not None:
        raise ValueError("references do not combine with extending a clip")
    if refmods:
        raise ValueError("references do not combine with a RefMod: use one or the other")
    if turbo or v13:
        raise ValueError("references do not combine with Turbo or LightX2V v1.3: those adapters "
                         "were not made for the Ref2VA weights")
    if chain_windows > 1:
        raise ValueError("references render a single 5 s window; shorten the clip (chained windows "
                         "carry no references yet)")
    pictures = clips = 0
    for path in references:
        kind = kind_of_ng(path)
        if kind is None:
            raise ValueError(f"a reference must be a picture or a clip; got {Path(path).name!r}")
        if check_files and not Path(path).is_file():
            raise FileNotFoundError(f"reference not found at {path}")
        pictures += kind == "image"
        clips += kind == "video"
    for label, have, limit in (("picture", pictures, MAX_PICTURES), ("clip", clips, MAX_CLIPS)):
        if have > limit:
            raise ValueError(f"at most {limit} {label} references per render, got {have}")
    return references


def reference_argv_ng(references, size: str = "match") -> list:
    """The runner arguments: one `--ref2va-reference PATH` per file, in order, then the size."""
    argv = []
    for path in references or []:
        argv += [REFERENCE_FLAG, str(path)]
    if argv and size != "match":
        argv += [REFERENCE_SIZE_FLAG, size]
    return argv
