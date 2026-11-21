"""immichRingNG -- `--ref2va-reference` for the H3 runner (and, in "faces" mode, upstream's `--reference`): pictures and clips handed to a Ref2VA render
as references (swap a character into a clip, carry a face through a new scene).

Unlike a RefMod (h3_ref2vaNG.py: a saved latent from the library) a reference is a plain picture
or mp4 that the runner encodes itself at render time, so there is nothing to build first. The
file's kind comes from its extension, and the order within each kind is its number: the first
picture is <Picture 1>, the first clip <Video 1>, which is what the compiler's brief names.
This file only checks what the runner would refuse and builds its arguments; h3_engineNG.py
calls it and h3_jobsNG.py stores the uploaded files.

Two modes, chosen by `reference_mode`:

* "ref2va" (the default): the Ref2VA model reads the pictures and clips in ONE 5 s window; nothing else is
  combined with them (`--ref2va-reference`).
* "faces": the ordinary first-frame model (FL2VA, or its Q8) with upstream's `--reference`, which packs each picture
  as a reference block in EVERY window of a chained clip, so a person's look is not left to the text in window 2 and
  later (the runner's own help: "Every window gets it"). Pictures only, up to three; a first-frame picture may come
  with them; Phosphene's "Face to keep" uses the same flag on a single picture. The pictures are named in each
  window's prompt as <Picture n>, after the first frame if there is one (the runner moves the number for the later
  windows, which also hold the carried frame). An extension of a finished clip is allowed too: every
  extension window starts from the carried last frame, so the pictures are <Picture 2> there as well.
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
MODE_REF2VA, MODE_FACES = "ref2va", "faces"
MODES = (MODE_REF2VA, MODE_FACES)
FACES_FLAG = "--reference"
FACES_SIZE_FLAG = "--reference-size"
FACES_MODELS = ("h3q8", "h3")      # the first-frame model: never the Ref2VA one
FACES_MAX_PICTURES = 3
FACES_SIZE = {"match": 1024, "max": 2048}   # short edge; Phosphene's Face to keep uses 1024 (2048 is several times slower)
FACES_SENTENCE = "The person from <Picture {n}> is in this shot, with the same face, hair and facial hair."


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


def runner_supports_faces_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, FACES_FLAG)


def validate_faces_ng(references, *, model: str, last_frame_path=None, keyframes=None, extend_from=None,
                      refmods=None, turbo: bool = False, v13: bool = False, runner, check_files: bool = True) -> list:
    """The "faces" mode rules: pictures only, the first-frame model, a chain is fine."""
    if model not in FACES_MODELS:
        raise ValueError("pictures kept in every window need the first-frame model (H3 or H3 Q8); "
                         f"{model!r} reads references its own way (one 5 s window)")
    if not runner_supports_faces_ng(runner):
        raise ValueError("this H3 runner has no --reference -- update the minimax-h3-mlx runner")
    if last_frame_path is not None or keyframes:
        raise ValueError("pictures kept in every window combine with a first frame, not with a last or timed frame")
    if refmods:
        raise ValueError("pictures kept in every window do not combine with a RefMod: use one or the other")
    if turbo or v13:
        raise ValueError("pictures kept in every window do not combine with Turbo or LightX2V v1.3")
    for path in references:
        if kind_of_ng(path) != "image":
            raise ValueError(f"only pictures can be kept in every window; got {Path(path).name!r}")
        if check_files and not Path(path).is_file():
            raise FileNotFoundError(f"reference not found at {path}")
    if len(references) > FACES_MAX_PICTURES:
        raise ValueError(f"at most {FACES_MAX_PICTURES} pictures can be kept in every window, got {len(references)}")
    return references


def validate_references_ng(references, size: str = "match", *, model: str, image_path,
                           last_frame_path=None, keyframes=None, extend_from=None, refmods=None,
                           turbo: bool = False, v13: bool = False, chain_windows: int = 1,
                           runner, check_files: bool = True, mode: str = MODE_REF2VA) -> list:
    """`references`: file paths -> the same list as strings. Raises ValueError with one plain
    sentence on anything the runner cannot do. An empty list is always fine. The job queue checks
    before the files are written (check_files=False); the engine checks again with them on disk."""
    references = [str(p) for p in references or []]
    if mode not in MODES:
        raise ValueError(f"reference mode must be one of {', '.join(MODES)} (got {mode!r})")
    if not references:
        return []
    if size not in SIZES:
        raise ValueError(f"reference size must be one of {', '.join(SIZES)} (got {size!r})")
    if mode == MODE_FACES:
        return validate_faces_ng(references, model=model, last_frame_path=last_frame_path, keyframes=keyframes,
                                 extend_from=extend_from, refmods=refmods, turbo=turbo, v13=v13, runner=runner,
                                 check_files=check_files)
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


def reference_argv_ng(references, size: str = "match", mode: str = MODE_REF2VA) -> list:
    """The runner arguments: one `--ref2va-reference PATH` per file, in order, then the size. In "faces" mode
    one `--reference PATH` per picture and the short edge it is encoded at (always given: the runner's own
    default is the slow 2048)."""
    if mode == MODE_FACES:
        argv = []
        for path in references or []:
            argv += [FACES_FLAG, str(path)]
        return argv + [FACES_SIZE_FLAG, str(FACES_SIZE[size])] if argv else []
    argv = []
    for path in references or []:
        argv += [REFERENCE_FLAG, str(path)]
    if argv and size != "match":
        argv += [REFERENCE_SIZE_FLAG, size]
    return argv


_SECTION_ENDS = ("\n\noverall_soundscape:", "\n\nnon_diegetic_music:", "\n\nsummary:", "\n\nretention_analysis:")


def append_to_description_ng(prompt: str, sentences) -> str:
    """The sentences added at the end of the prompt's integrated_multimodal_description (before the sound
    sections); at the very end when the prompt has no such section."""
    extra = " ".join(s.strip() for s in sentences if s and s.strip())
    if not extra:
        return prompt
    head = "integrated_multimodal_description:"
    start = prompt.find(head)
    if start < 0:
        return prompt.rstrip() + " " + extra
    ends = [i for i in (prompt.find(m, start) for m in _SECTION_ENDS) if i >= 0]
    cut = min(ends) if ends else len(prompt)
    return prompt[:cut].rstrip() + " " + extra + prompt[cut:]


def name_faces_in_prompt_ng(prompt: str, count: int, first_picture: int) -> str:
    """Name the kept pictures in a window's prompt: `first_picture` is 2 when a first-frame picture comes before
    them, else 1. A picture the prompt already names (<Picture n>) is left alone."""
    lower = prompt.lower()
    missing = [FACES_SENTENCE.format(n=first_picture + k) for k in range(count)
               if f"<picture {first_picture + k}>" not in lower]
    return append_to_description_ng(prompt, missing)
