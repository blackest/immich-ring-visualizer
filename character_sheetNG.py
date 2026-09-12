"""NG twin of character_sheet.py -- character-sheet generation: one face
photo -> a shot list of turnaround / dataset views.

NG-only file: same generation logic as character_sheet.py as it exists
today (prompt building, anchor-chaining, per-shot seeding, composite
strip), duplicated per the NG rule in APP_ARCHITECTURE_NOTES.md so
nothing under *NG.py ever imports the original module. Wired to
configNG.py / hidream_engineNG.py / shot_presetsNG.py instead of
config.py / hidream_engine.py / shot_presets.py.

One deliberate behavioral change from the original, not just a rename:
the original fails fast (CharacterSheetBusyError, HTTP 429) if a second
render is requested while one is already running -- "queueing behind
another one is worse than an honest busy error" was the right call
for a single request/response flow with no queue UI. NG's Generate view
adds a real FIFO job queue (sheet_jobsNG.py) specifically so a second
tab/character's job waits its turn instead of bouncing -- so
_SHEET_LOCK_NG here is a plain blocking wait (belt-and-suspenders
against any caller that bypasses the queue), not a fail-fast guard.
There is deliberately no CharacterSheetBusyError in this module.

Storage follows Ring Visualizer's own export convention, under
configNG.EXPORT_DIR (exportsNG/, kept separate from the original app's
exports/ tree):

    exportsNG/<n>/
      character/
        avatar.<ext>            <- the reference photo
        bundle.json              <- {schema, id, name, pronoun, subject_noun}
        sheet.png                <- composited strip (small shot counts only)
        sheet.json                <- per-shot prompt/seed/path metadata
        sheet_views/<shot key>/*.png
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from configNG import EXPORT_DIR
import hidream_engineNG
import shot_presetsNG as shot_presets
from shot_presetsNG import ShotSpec

_ID_RE = re.compile(r"^[A-Za-z0-9 _-]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES_PER_IMAGE = 32 * 1024 * 1024

# Above this many shots, a single horizontal composite strip stops being
# useful -- see generate_character_sheet_ng.
MAX_SHOTS_FOR_COMPOSITE = 6

# Render-size / step bounds for the Generate view's per-job width/height/
# steps controls. Width/height are patch-aligned to 32 downstream (see
# hidream_engineNG._patch_align_ng); anything outside the trained
# resolutions renders off-spec (faster, faint 32px patch grid possible).
RENDER_DIM_MIN, RENDER_DIM_MAX = 256, 4096
RENDER_STEPS_MIN, RENDER_STEPS_MAX = 1, 100
DEFAULT_RENDER_W = DEFAULT_RENDER_H = 2048
DEFAULT_RENDER_STEPS = 28


def _validate_render_params_ng(width, height, steps):
    """(width, height, steps) -> validated ints. Raises ValueError (which
    the routes map to HTTP 400) on non-integers or out-of-range values."""
    try:
        w, h, s = int(width), int(height), int(steps)
    except (TypeError, ValueError):
        raise ValueError("width, height and steps must be integers")
    if not (RENDER_DIM_MIN <= w <= RENDER_DIM_MAX
            and RENDER_DIM_MIN <= h <= RENDER_DIM_MAX):
        raise ValueError(
            f"width/height must be between {RENDER_DIM_MIN} and {RENDER_DIM_MAX}")
    if not (RENDER_STEPS_MIN <= s <= RENDER_STEPS_MAX):
        raise ValueError(
            f"steps must be between {RENDER_STEPS_MIN} and {RENDER_STEPS_MAX}")
    return w, h, s


class DraftCharacterExistsError(Exception):
    """Raised by create_draft_character_ng() when `name` is already taken."""


# Plain blocking lock -- see module docstring for why this isn't a
# fail-fast guard here the way it is in the original.
_SHEET_LOCK_NG = threading.Lock()


def _safe_id_ng(value: str) -> str:
    v = (value or "").strip()
    if not v or not _ID_RE.match(v):
        raise ValueError("invalid character name -- use letters, digits, "
                         "spaces, underscore or hyphen")
    return v


def _character_dir_ng(name: str) -> Path:
    return Path(EXPORT_DIR) / name / "character"


def _bundle_path_ng(name: str) -> Path:
    return _character_dir_ng(name) / "bundle.json"


def character_bundle_ng(name: str) -> dict:
    p = _bundle_path_ng(name)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def character_avatar_ng(name: str) -> Optional[Path]:
    char_dir = _character_dir_ng(name)
    if not char_dir.is_dir():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = char_dir / f"avatar{ext}"
        if p.is_file():
            return p
    return None


def character_exists_ng(name: str) -> bool:
    return _bundle_path_ng(name).is_file()


def character_sheet_png_ng(name: str) -> Optional[Path]:
    """The CURRENT sheet's composite, if it has one -- reads sheet.json's
    own sheet_png field rather than just checking sheet.png's existence,
    since a job whose shot count exceeds MAX_SHOTS_FOR_COMPOSITE
    intentionally produces no composite."""
    meta = character_sheet_meta_ng(name)
    sheet_png = meta.get("sheet_png")
    if not sheet_png:
        return None
    p = Path(sheet_png)
    return p if p.is_file() else None


def sheet_shot_image_paths_ng(name: str) -> list:
    """Every shot's *current* rendered image path -- the latest
    cand_*.png by mtime in each sheet_views/<key>/ dir. Reads the
    on-disk layout rather than sheet.json's views list so it reflects
    rerolls and in-progress shots immediately rather than only after a
    whole job finishes."""
    views_dir = _character_dir_ng(_safe_id_ng(name)) / "sheet_views"
    if not views_dir.is_dir():
        return []
    paths = []
    for shot_dir in sorted(p for p in views_dir.iterdir() if p.is_dir()):
        candidates = sorted(shot_dir.glob("cand_*.png"), key=lambda p: p.stat().st_mtime)
        if candidates:
            paths.append(str(candidates[-1]))
    return paths


# ---- character-picker grid: the saved "project" doc (characterIONG.js's
# .character.json -- video/task/settings/source pointers, NOT the
# generation bundle above) mirrored server-side at exportsNG/<n>/
# character.json, one level up from character/ -- sibling to it, not
# nested inside, since a project doc can exist for a character that's
# never been through Generate at all (video/folder/Immich analysis only).
def _project_doc_path_ng(name: str) -> Path:
    return Path(EXPORT_DIR) / _safe_id_ng(name) / "character.json"


def save_character_project_doc_ng(name: str, doc: dict) -> None:
    p = _project_doc_path_ng(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def load_character_project_doc_ng(name: str) -> Optional[dict]:
    p = _project_doc_path_ng(name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_characters_ng() -> list:
    """Every exportsNG/<n>/ dir that's "known" -- either it's been
    registered as a generation character (character/bundle.json) or a
    project was explicitly Saved onto it (character.json) -- for the
    Generate view's landing-state picker grid. Merges both sources per
    id since a character can have either, both, or (having only ever
    been used for video/folder/Immich analysis, then Saved) just the
    latter."""
    root = Path(EXPORT_DIR)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            cid = _safe_id_ng(d.name)
        except ValueError:
            continue  # skip anything that couldn't have been created by this app
        bundle = character_bundle_ng(cid) if character_exists_ng(cid) else {}
        doc = load_character_project_doc_ng(cid)
        if not bundle and not doc:
            continue
        avatar = character_avatar_ng(cid)
        shot_count = len(sheet_shot_image_paths_ng(cid)) if bundle else 0
        out.append({
            "id": cid,
            # Prefer the project doc's user-facing name (e.g. "Mary") over
            # the bundle's, which is usually just the id/trigger itself.
            "name": (doc and doc.get("identity", {}).get("name")) or bundle.get("name") or cid,
            "hasAvatar": avatar is not None,
            "hasProjectDoc": doc is not None,
            "task": (doc or {}).get("task"),
            "shotCount": shot_count,
            "savedAt": (doc or {}).get("savedAt"),
        })
    out.sort(key=lambda c: (c.get("savedAt") or ""), reverse=True)
    return out


def character_sheet_meta_ng(name: str) -> dict:
    p = _character_dir_ng(_safe_id_ng(name)) / "sheet.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def create_draft_character_ng(name: str, source_image_path: str, *,
                               pronoun: str = "", subject_noun: str = "") -> dict:
    """Register a character from a single reference photo already on
    disk. No LoRA, no training.

    Raises ValueError on a bad name/extension, FileNotFoundError if
    source_image_path doesn't exist, DraftCharacterExistsError if the
    name is already registered.
    """
    cid = _safe_id_ng(name)
    if character_exists_ng(cid):
        raise DraftCharacterExistsError(f"character {cid!r} already exists")

    src = Path(source_image_path)
    if not src.is_file():
        raise FileNotFoundError(f"source_image_path not found: {src}")
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"unsupported image type {ext!r}; want one of {sorted(IMAGE_EXTS)}")
    size = src.stat().st_size
    if size <= 0 or size > MAX_BYTES_PER_IMAGE:
        raise ValueError(f"source image is {size} bytes; must be > 0 and <= "
                         f"{MAX_BYTES_PER_IMAGE} (32 MB)")

    char_dir = _character_dir_ng(cid)
    char_dir.mkdir(parents=True, exist_ok=True)
    avatar_path = char_dir / f"avatar{ext}"
    shutil.copyfile(src, avatar_path)
    bundle = {
        "schema": "ringviz/character_bundle@1",
        "id": cid,
        "name": cid,
        "pronoun": (pronoun or "").strip() or "they",
        "subject_noun": (subject_noun or "").strip() or "person",
    }
    _bundle_path_ng(cid).write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


def _view_prompt_ng(view_phrase: str, wardrobe: str = "", hair_color: str = "") -> str:
    """Identity-lock clause set for a plain default-preset shot. Kept as
    its own function -- rather than folded into _shot_prompt_ng -- so
    the original 3-view sheet's output stays provably identical to
    what's already been verified end-to-end on real hardware;
    _shot_prompt_ng delegates to this for any shot that doesn't use the
    newer per-shot fields.

    `hair_color`, if given, names the color explicitly ("blonde", "dark
    red", ...) instead of the default relative "same hair color as the
    reference image" clause."""
    hair_clause = f"{hair_color} hair" if hair_color else "same hair color"
    prompt = (
        "Keep this person exactly as they are in the reference image -- same "
        f"face, same skin tone and complexion, {hair_clause}, same exact "
        "hairstyle (do not restyle, tie back, loosen, or otherwise change "
        "how the hair is worn -- keep the same length and the same way it "
        "falls), same build, wearing exactly the same clothes as in "
        "the reference image. Change only the camera and pose: "
        f"{view_phrase}, the person centered in the frame and filling most "
        "of it. Neutral seamless studio background, soft even lighting that "
        "matches the reference image's skin tone, photorealistic."
    )
    if wardrobe:
        prompt += f" They are wearing {wardrobe}."
    return prompt


def _shot_prompt_ng(spec: ShotSpec, wardrobe: str = "", *,
                     identity_lock: bool = True, style: str = "none",
                     hair_color: str = "") -> str:
    """General shot-spec prompt builder. For a plain default-preset shot
    (no background/expression/override) with identity_lock on, this
    produces byte-identical output to _view_prompt_ng -- see that
    function's docstring for why that matters.

    `hair_color`: see _view_prompt_ng's docstring -- same optional
    explicit color name, job-level only (unlike wardrobe there's no
    per-shot override; hair color shouldn't vary shot to shot within one
    sheet)."""
    pose = spec.prompt_override or spec.pose_phrase
    shot_wardrobe = spec.wardrobe or wardrobe

    if not identity_lock:
        # Power-user / free-prompt escape hatch: no identity clauses at
        # all, just the pose text (or full override) as-is.
        prompt = pose
        style_clause = shot_presets.STYLE_PRESETS.get(style, "")
        if spec.background:
            prompt += f" {spec.background}."
        if style_clause:
            prompt += f" {style_clause}."
        if shot_wardrobe:
            prompt += f" They are wearing {shot_wardrobe}."
        return prompt

    if not spec.background and not spec.expression and not spec.prompt_override \
            and style in (None, "none"):
        return _view_prompt_ng(pose, shot_wardrobe, hair_color)

    expression_clause = f", {spec.expression}" if spec.expression else ""
    background_clause = spec.background or (
        "Neutral seamless studio background, soft even lighting that "
        "matches the reference image's skin tone")
    style_clause = shot_presets.STYLE_PRESETS.get(style, "")
    hair_clause = f"{hair_color} hair" if hair_color else "same hair color"
    prompt = (
        "Keep this person exactly as they are in the reference image -- same "
        f"face, same skin tone and complexion, {hair_clause}, same exact "
        "hairstyle (do not restyle, tie back, loosen, or otherwise change "
        "how the hair is worn -- keep the same length and the same way it "
        "falls), same build, wearing exactly the same clothes as in "
        "the reference image. Change only the camera and pose: "
        f"{pose}{expression_clause}, the person centered in the frame and "
        f"filling most of it. {background_clause}, photorealistic."
    )
    if style_clause:
        prompt += f" {style_clause}."
    if shot_wardrobe:
        prompt += f" They are wearing {shot_wardrobe}."
    return prompt


def _compose_sheet_row_ng(image_paths, out_path: Path) -> Path:
    """1 row x N columns strip, cell height 1024, aspect preserved, thin
    light gutters. Only called for small shot counts -- see
    MAX_SHOTS_FOR_COMPOSITE."""
    from PIL import Image
    paths = [p for p in (image_paths or []) if p and Path(p).exists()]
    if not paths:
        raise RuntimeError("character sheet composite got no view images")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(imgs) == 1:
        imgs[0].save(out_path, format="PNG")
        return out_path
    BG = (235, 233, 229)
    g, ch = 12, 1024
    cells = []
    for im in imgs:
        w = max(1, round(im.width * ch / im.height))
        cells.append(im.resize((w, ch), Image.LANCZOS))
    total_w = sum(c.width for c in cells) + g * (len(cells) + 1)
    canvas = Image.new("RGB", (total_w, ch + 2 * g), BG)
    x = g
    for c in cells:
        canvas.paste(c, (x, g))
        x += c.width + g
    canvas.save(out_path, format="PNG")
    return out_path


def resolve_shots_ng(*, preset: str = "default",
                      shots: Optional[list] = None,
                      views: Optional[list] = None) -> list:
    """Turn (preset | shots | views) into a concrete, deduped list of
    ShotSpec. Exposed standalone so the job queue can resolve the shot
    list up front -- before any rendering starts -- to report initial
    per-shot status.

    Precedence: explicit `shots` wins, then `views` (a list of shot
    keys, validated against every registered preset's catalogue so the
    Generate view can cherry-pick individual poses out of the 15-shot
    "extended" preset one at a time), else `preset` by name.
    """
    if shots is not None:
        if not isinstance(shots, list) or not all(isinstance(s, ShotSpec) for s in shots):
            raise ValueError("shots must be a list of ShotSpec")
        seen = set()
        deduped = []
        for s in shots:
            if s.key in seen:
                continue
            seen.add(s.key)
            deduped.append(s)
        if not deduped:
            raise ValueError("shots resolved to an empty list")
        return deduped

    if views is not None:
        if not isinstance(views, list) or not all(isinstance(v, str) for v in views):
            raise ValueError("views must be a list of view-name strings")
        by_key = {s.key: s
                  for preset_shots in shot_presets.PRESETS.values()
                  for s in preset_shots}
        keys = list(dict.fromkeys(v.strip() for v in views if v.strip()))
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise ValueError(f"unknown views: {', '.join(unknown)} -- available: "
                             f"{', '.join(by_key)}")
        if not keys:
            raise ValueError("views resolved to an empty list")
        return [by_key[k] for k in keys]

    return list(shot_presets.resolve_preset_ng(preset))


def _shot_by_key_ng(key: str) -> ShotSpec:
    for preset_shots in shot_presets.PRESETS.values():
        for s in preset_shots:
            if s.key == key:
                return s
    raise ValueError(f"unknown view {key!r}")


def _safe_shot_key_ng(text: str) -> str:
    """Slugifies free-text pose names/phrases into a filesystem-safe shot
    key for a brand-new custom pose -- prefixed so it can never collide
    with a real preset key even if someone types e.g. "front", truncated
    so a caller that passes the whole pose phrase (rather than a short
    name) as custom_key doesn't produce an unwieldy sheet_views/ dir
    name."""
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")[:40].strip("_")
    return "custom_" + (slug or "pose")


def build_custom_shot_ng(*, views: Optional[list] = None, custom_prompt: str = "",
                          custom_pose: str = "", custom_key: str = "",
                          prompt_override: str = "", scene: str = "") -> Optional[ShotSpec]:
    """Builds the single ShotSpec behind the Generate view's per-shot
    escape hatches, or None if none of them apply (caller should just
    resolve `views`/`preset` as normal in that case). Shared by the
    generate/reroll routes AND the prompt-preview route (see
    preview_shot_prompt_ng) so a previewed prompt always matches what
    actually renders.

    Precedence, most to least specific:
      custom_prompt  -- pure free-text prompt, bypasses the identity-lock
                        template entirely (the original, pre-existing
                        escape hatch).
      custom_pose    -- a brand-new pose under a new key ("add custom
                        pose"), still goes through the normal template
                        (identity-lock clauses, wardrobe, hair color) so
                        it behaves like any preset shot, just with
                        caller-supplied pose text. `prompt_override`, if
                        also given, replaces the pose text (for fine-
                        tuning a previewed prompt before queuing it).
      views[0] + (prompt_override or scene) -- edits one existing preset
                        shot's pose text and/or scene ("editable prompt" /
                        "per-shot scene dropdown") before rendering it;
                        only applies when exactly one view is selected --
                        multi-view batches use the presets unmodified.

    `scene`, if given, is a key into shot_presets.SCENES (mirrors how
    `style` is a key into shot_presets.STYLE_PRESETS elsewhere) -- an
    unknown/empty key just means no scene override.
    """
    custom_prompt = (custom_prompt or "").strip()
    if custom_prompt:
        return ShotSpec("custom", prompt_override=custom_prompt)

    custom_pose = (custom_pose or "").strip()
    prompt_override = (prompt_override or "").strip()
    scene_phrase = shot_presets.SCENES.get((scene or "").strip(), "")

    if custom_pose:
        key = _safe_shot_key_ng(custom_key or custom_pose)
        return ShotSpec(key, pose_phrase=prompt_override or custom_pose,
                         background=scene_phrase)

    if views and len(views) == 1 and (prompt_override or scene_phrase):
        base = _shot_by_key_ng(views[0])
        return ShotSpec(base.key, pose_phrase=prompt_override or base.pose_phrase,
                         expression=base.expression,
                         background=scene_phrase or base.background,
                         wardrobe=base.wardrobe, use_anchor=base.use_anchor)

    return None


def preview_shot_prompt_ng(*, preset: str = "default", views: Optional[list] = None,
                            wardrobe: str = "", hair_color: str = "",
                            identity_lock: bool = True, style: str = "none",
                            custom_prompt: str = "", custom_pose: str = "",
                            custom_key: str = "", prompt_override: str = "",
                            scene: str = "") -> dict:
    """Resolves the exact prompt a queued shot WOULD render with, given
    the same inputs generate_character_sheet_ng would see, without
    touching the engine or disk -- lets the Generate view show/edit a
    shot's prompt before actually submitting it. Returns {key, prompt}
    for the first (only, in practice -- the route only ever previews
    one shot at a time) resolved shot.

    Raises ValueError for the same reasons resolve_shots_ng would (bad
    preset/views/shot) -- the route maps this to HTTP 400.
    """
    custom = build_custom_shot_ng(
        views=views, custom_prompt=custom_prompt, custom_pose=custom_pose,
        custom_key=custom_key, prompt_override=prompt_override, scene=scene)
    shot_list = resolve_shots_ng(preset=preset, shots=[custom] if custom else None,
                                  views=None if custom else views)
    spec = shot_list[0]
    prompt = _shot_prompt_ng(spec, wardrobe, identity_lock=identity_lock,
                              style=style, hair_color=hair_color)
    return {"key": spec.key, "prompt": prompt}


def generate_character_sheet_ng(name: str, *,
                                 preset: str = "default",
                                 shots: Optional[list] = None,
                                 views: Optional[list] = None,
                                 wardrobe: str = "",
                                 hair_color: str = "",
                                 seed: int = -1,
                                 anchor_chain: bool = True,
                                 identity_lock: bool = True,
                                 style: str = "none",
                                 width: int = DEFAULT_RENDER_W,
                                 height: int = DEFAULT_RENDER_H,
                                 steps: int = DEFAULT_RENDER_STEPS,
                                 on_log=None) -> dict:
    """Render a shot-list character sheet from one reference photo.

    Same seed handling as the original (one resolved seed shared across
    the job, each shot offset by its position -- resolved_seed + i, not
    resolved_seed alone, so shots genuinely diverge in pose), same
    anchor_chain default (every shot after the first also gets the
    first rendered shot as a second reference, to stop hair-color
    drift), same atomic sheet.png write for small shot counts. NOT the
    same GPU-busy-fails-fast philosophy -- see module docstring; this
    acquires _SHEET_LOCK_NG with a plain blocking wait instead of
    raising when busy, since sheet_jobsNG.py's queue is the real
    serialization boundary and callers of this function directly should
    simply wait their turn too.

    Raises:
      ValueError               -- bad name / preset / shots / views / seed
                                                            (route -> 400)
      LookupError               -- no such character          (route -> 404)
      FileNotFoundError         -- no reference image on disk (route -> 404)
      RuntimeError               -- engine failure           (route -> 500)
    """
    cid = _safe_id_ng(name)
    shot_list = resolve_shots_ng(preset=preset, shots=shots, views=views)
    width, height, steps = _validate_render_params_ng(width, height, steps)

    wardrobe = str(wardrobe or "").strip()
    hair_color = str(hair_color or "").strip()
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise ValueError("seed must be an integer")
    resolved_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)

    if not character_exists_ng(cid):
        raise LookupError(f"character {cid!r} not found")
    ref = character_avatar_ng(cid)
    if ref is None:
        raise FileNotFoundError(
            f"character {cid!r} has no reference image -- expected an "
            f"avatar under {EXPORT_DIR}/{cid}/character/")

    char_dir = _character_dir_ng(cid)
    char_dir.mkdir(parents=True, exist_ok=True)

    with _SHEET_LOCK_NG:
        t0 = time.time()
        cfg = hidream_engineNG.HiDreamConfig(steps=steps)
        view_records = []
        anchor_png = None
        for i, spec in enumerate(shot_list):
            prompt = _shot_prompt_ng(spec, wardrobe, identity_lock=identity_lock,
                                      style=style, hair_color=hair_color)
            view_dir = char_dir / "sheet_views" / spec.key
            view_dir.mkdir(parents=True, exist_ok=True)
            view_refs = [str(ref)] + (
                [anchor_png] if (anchor_png and anchor_chain and spec.use_anchor)
                else [])
            if on_log:
                on_log(f"[sheet] {cid}: shot {i + 1}/{len(shot_list)} ({spec.key})")
            # Each shot gets its own seed, derived from the job's
            # resolved seed plus its position in shot_list -- an
            # identical seed across shots suppresses genuine pose
            # divergence (HiDream's Dev recipe is CFG-free, refs are
            # conditioned as clean/unnoised tokens rather than classic
            # img2img noising). Still fully deterministic/reproducible
            # from one job-level `seed` input.
            candidates = hidream_engineNG.generate_hidream_ng(
                prompt=prompt, n=1, width=width, height=height,
                output_dir=view_dir,
                base_seed=resolved_seed + i,
                refs=view_refs,
                config=cfg,
                allow_offspec_res=True,
                on_log=on_log,
            )
            if not candidates or not candidates[0].get("png_path"):
                raise RuntimeError(f"engine returned no image for shot {spec.key!r}")
            c = candidates[0]
            if anchor_png is None:
                anchor_png = c.get("png_path")
            view_records.append({
                "key": spec.key,
                "prompt": prompt,
                "seed": c.get("seed"),
                "png_path": c.get("png_path"),
                "engine": c.get("engine"),
                "width": c.get("width"),
                "height": c.get("height"),
                "refs": view_refs,
            })

    sheet_path = None
    if len(view_records) <= MAX_SHOTS_FOR_COMPOSITE:
        sheet_path = char_dir / "sheet.png"
        tmp = char_dir / f".sheet.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.png"
        try:
            _compose_sheet_row_ng([r["png_path"] for r in view_records], tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, sheet_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    else:
        # No composite for this shot count -- clear a stale one left
        # over from an earlier, smaller-preset run on this character.
        try:
            (char_dir / "sheet.png").unlink()
        except FileNotFoundError:
            pass

    elapsed = round(time.time() - t0, 2)
    sheet_meta = {
        "schema": "ringviz/character_sheet@2",
        "character_id": cid,
        "engine": "hidream",
        "reference": str(ref),
        "preset": preset if (shots is None and views is None) else "custom",
        "wardrobe": wardrobe,
        "hair_color": hair_color,
        "identity_lock": identity_lock,
        "style": style,
        "seed": seed,
        "resolved_seed": resolved_seed,
        "views": view_records,
        "sheet_png": str(sheet_path) if sheet_path else None,
        "created_at": time.time(),
        "elapsed_sec": elapsed,
    }
    sheet_json_path = char_dir / "sheet.json"
    tmp_json = sheet_json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(sheet_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_json, sheet_json_path)

    return {
        "ok": True,
        "trigger": cid,
        "sheet_path": str(sheet_path) if sheet_path else None,
        "result": sheet_meta,
    }


def regenerate_shot_ng(name: str, shot_key: str, *,
                        seed: Optional[int] = None,
                        prompt: Optional[str] = None,
                        width: Optional[int] = None,
                        height: Optional[int] = None,
                        steps: Optional[int] = None,
                        on_log=None) -> dict:
    """Re-render a single shot from an already-generated sheet and
    recomposite (when the shot count is small enough to composite at
    all -- see MAX_SHOTS_FOR_COMPOSITE).

    Reuses that shot's existing prompt and reference images from
    sheet.json unless overridden.

    Raises:
      LookupError        -- no such character, or no such shot in its
                            current sheet                    (route -> 404)
      FileNotFoundError   -- no sheet.json yet to re-roll a shot in
                                                              (route -> 404)
      RuntimeError         -- engine failure                (route -> 500)
    """
    cid = _safe_id_ng(name)
    if not character_exists_ng(cid):
        raise LookupError(f"character {cid!r} not found")
    char_dir = _character_dir_ng(cid)
    sheet_json_path = char_dir / "sheet.json"
    if not sheet_json_path.is_file():
        raise FileNotFoundError(
            f"character {cid!r} has no existing sheet to re-roll a shot in "
            f"-- generate a sheet first")
    meta = json.loads(sheet_json_path.read_text(encoding="utf-8"))
    views = meta.get("views", [])
    idx = next((i for i, v in enumerate(views) if v.get("key") == shot_key), None)
    if idx is None:
        raise LookupError(f"no shot {shot_key!r} in character {cid!r}'s current sheet")

    # Fall back to the shot's previous render size when the caller didn't
    # ask for a specific one; steps aren't persisted per shot, so default.
    prev = views[idx]
    use_w, use_h, use_steps = _validate_render_params_ng(
        width if width is not None else (prev.get("width") or DEFAULT_RENDER_W),
        height if height is not None else (prev.get("height") or DEFAULT_RENDER_H),
        steps if steps is not None else DEFAULT_RENDER_STEPS)

    with _SHEET_LOCK_NG:
        t0 = time.time()
        cfg = hidream_engineNG.HiDreamConfig(steps=use_steps)
        use_prompt = prompt if prompt is not None else views[idx]["prompt"]
        use_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        refs = views[idx].get("refs") or []
        view_dir = char_dir / "sheet_views" / shot_key
        view_dir.mkdir(parents=True, exist_ok=True)
        if on_log:
            on_log(f"[sheet] {cid}: re-rolling shot {shot_key!r}")
        candidates = hidream_engineNG.generate_hidream_ng(
            prompt=use_prompt, n=1, width=use_w, height=use_h,
            output_dir=view_dir, base_seed=use_seed, refs=refs,
            config=cfg, allow_offspec_res=True, on_log=on_log)
        if not candidates or not candidates[0].get("png_path"):
            raise RuntimeError(f"engine returned no image for shot {shot_key!r}")
        c = candidates[0]
        views[idx] = {
            **views[idx],
            "prompt": use_prompt,
            "seed": c.get("seed"),
            "png_path": c.get("png_path"),
            "engine": c.get("engine"),
            "width": c.get("width"),
            "height": c.get("height"),
        }

    sheet_path = None
    if len(views) <= MAX_SHOTS_FOR_COMPOSITE:
        sheet_path = char_dir / "sheet.png"
        tmp = char_dir / f".sheet.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.png"
        try:
            _compose_sheet_row_ng([v["png_path"] for v in views], tmp)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, sheet_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    meta["views"] = views
    meta["sheet_png"] = str(sheet_path) if sheet_path else None
    meta["updated_at"] = time.time()
    meta["last_reroll"] = {"shot_key": shot_key, "elapsed_sec": round(time.time() - t0, 2)}
    tmp_json = sheet_json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_json, sheet_json_path)

    return {
        "ok": True,
        "trigger": cid,
        "shot_key": shot_key,
        "sheet_path": str(sheet_path) if sheet_path else None,
        "result": meta,
    }
