"""Character-sheet generation: one face photo -> a shot list of turnaround
/ dataset views, with no Phosphene involved.

Ported from Phosphene's mlx_ltx_panel.py (the Characters-tab bookkeeping
+ generate_character_sheet orchestrator) per PHOSPHENE_DECOUPLING_PLAN.md,
then grown from a fixed 3-view sheet into a shot-list system (see
shot_presets.py) once "a couple of reference angles" turned into "a real
15+ shot training dataset with full-body framing and varied backgrounds".

Storage follows Ring Visualizer's own export convention instead of
mirroring Phosphene's LoRA-training-oriented mlx_models/characters/ tree:

    exports/<name>/
      fr00002_sim0.39.png      <- existing curated training crops, untouched
      character/
        avatar.<ext>            <- the reference photo
        bundle.json              <- {schema, id, name, pronoun, subject_noun}
        sheet.png                <- composited strip (small shot counts only
                                     -- see generate_character_sheet)
        sheet.json                <- per-shot prompt/seed/path metadata
        sheet_views/<shot key>/*.png

No LoRA-training lifecycle exists here (that's what Phosphene's
list_characters(include_drafts=...) machinery was for) -- a "character"
in this module is just: does exports/<name>/character/bundle.json exist.
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

from config import EXPORT_DIR
import hidream_engine
import shot_presets
from shot_presets import ShotSpec

_ID_RE = re.compile(r"^[A-Za-z0-9 _-]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES_PER_IMAGE = 32 * 1024 * 1024

# Kept for backward compatibility (older callers / the legacy `views=`
# kwarg reference this) -- derived from shot_presets.DEFAULT_PRESET
# rather than hand-duplicated, so the two can't drift apart.
CHARACTER_SHEET_VIEWS = {s.key: s.pose_phrase for s in shot_presets.DEFAULT_PRESET}

# Above this many shots, a single horizontal composite strip stops being
# useful (15 shots at 1024px cell height is a >10000px-wide image nobody
# can look at) -- generate_character_sheet skips the strip and leaves
# sheet_png=None, relying on the per-shot files under sheet_views/ for
# review instead. The 3-view turnaround sheet stays under this easily.
MAX_SHOTS_FOR_COMPOSITE = 6


class DraftCharacterExistsError(Exception):
    """Raised by create_draft_character() when `name` is already taken."""


class CharacterSheetBusyError(RuntimeError):
    """Another sheet generation is already running in this process. Fails
    fast rather than queueing -- same contract as Phosphene's: a sheet is
    3+ full renders, silently queueing behind another one is worse than
    an honest busy error. (This guards Ring Visualizer's own process
    only; Phosphene/Ring-Viz cross-app GPU contention is handled by
    convention per the decoupling plan -- don't run both at once.)"""


_SHEET_LOCK = threading.Lock()


def _safe_id(value: str) -> str:
    v = (value or "").strip()
    if not v or not _ID_RE.match(v):
        raise ValueError("invalid character name -- use letters, digits, "
                         "spaces, underscore or hyphen")
    return v


def _character_dir(name: str) -> Path:
    return Path(EXPORT_DIR) / name / "character"


def _bundle_path(name: str) -> Path:
    return _character_dir(name) / "bundle.json"


def character_bundle(name: str) -> dict:
    p = _bundle_path(name)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def character_avatar(name: str) -> Optional[Path]:
    char_dir = _character_dir(name)
    if not char_dir.is_dir():
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = char_dir / f"avatar{ext}"
        if p.is_file():
            return p
    return None


def character_exists(name: str) -> bool:
    return _bundle_path(name).is_file()


def character_sheet_png(name: str) -> Optional[Path]:
    """The CURRENT sheet's composite, if it has one. Reads sheet.json's
    own sheet_png field rather than just checking sheet.png's existence
    on disk -- a job whose shot count exceeds MAX_SHOTS_FOR_COMPOSITE
    intentionally produces no composite, and without this check a stale
    sheet.png left over from an earlier, smaller-preset run on the same
    character would otherwise still get served as if it were current."""
    meta = character_sheet_meta(name)
    sheet_png = meta.get("sheet_png")
    if not sheet_png:
        return None
    p = Path(sheet_png)
    return p if p.is_file() else None


def character_sheet_meta(name: str) -> dict:
    p = _character_dir(_safe_id(name)) / "sheet.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def create_draft_character(name: str, source_image_path: str, *,
                            pronoun: str = "", subject_noun: str = "") -> dict:
    """Register a character from a single reference photo already on
    disk. No LoRA, no training -- see module docstring.

    Raises ValueError on a bad name/extension, FileNotFoundError if
    source_image_path doesn't exist, DraftCharacterExistsError if the
    name is already registered.
    """
    cid = _safe_id(name)
    if character_exists(cid):
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

    char_dir = _character_dir(cid)
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
    _bundle_path(cid).write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


def _view_prompt(view_phrase: str, wardrobe: str = "", hair_color: str = "") -> str:
    """Verbatim from Phosphene's _character_sheet_view_prompt (the
    session-tuned version) -- every clause below exists because a real
    render defect motivated it; see PHOSPHENE_DECOUPLING_PLAN.md's
    "things learned this session" for the failure each one fixes
    (skin-tone warming, hair-color drift on side angles, hairstyle
    loosening even with correct anchor-chaining). Kept as its own
    function -- rather than folded into _shot_prompt -- so the original
    3-view sheet's output stays provably identical to what's already
    been verified end-to-end on real hardware; _shot_prompt delegates to
    this for any shot that doesn't use the newer per-shot fields.

    `hair_color`, if given, names the color explicitly ("blonde", "dark
    red", ...) instead of the default relative "same hair color as the
    reference image" clause. Added 2026-09-03 alongside use_anchor=False
    on the extended preset's angle shots (see ShotSpec's docstring) --
    hair color is a small enough vocabulary that a concrete name should
    be a more reliable instruction than "same as reference" once a shot
    no longer also gets the anchor image as a second, color-carrying
    reference."""
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


def _shot_prompt(spec: ShotSpec, wardrobe: str = "", *,
                 identity_lock: bool = True, style: str = "none",
                 hair_color: str = "") -> str:
    """General shot-spec prompt builder. For a plain default-preset shot
    (no background/expression/override) with identity_lock on, this
    produces byte-identical output to _view_prompt -- see that
    function's docstring for why that matters.

    `hair_color`: see _view_prompt's docstring -- same optional explicit
    color name, job-level only (unlike wardrobe there's no per-shot
    override; hair color shouldn't vary shot to shot within one sheet)."""
    pose = spec.prompt_override or spec.pose_phrase
    shot_wardrobe = spec.wardrobe or wardrobe

    if not identity_lock:
        # Power-user / free-prompt escape hatch: no identity clauses at
        # all, just the pose text (or full override) as-is. Chosen over
        # John (2026-09-02): identity-lock stays the default everywhere
        # else, this is opt-in per job. hair_color is an identity clause
        # too, so it's correctly skipped here along with the rest.
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
        return _view_prompt(pose, shot_wardrobe, hair_color)

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


def _compose_sheet_row(image_paths, out_path: Path) -> Path:
    """1 row x N columns strip, cell height 1024, aspect preserved, thin
    light gutters. Verbatim layout from Phosphene's
    _compose_character_sheet_row. Only called for small shot counts --
    see MAX_SHOTS_FOR_COMPOSITE."""
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


def resolve_shots(*, preset: str = "default",
                  shots: Optional[list] = None,
                  views: Optional[list] = None) -> list:
    """Turn (preset | shots | views) into a concrete, deduped list of
    ShotSpec. Exposed standalone so a job queue can resolve the shot
    list up front -- before any rendering starts -- to report initial
    per-shot status.

    Precedence: explicit `shots` wins, then legacy `views` (validated
    against the default 3-view catalogue only, for old callers), else
    `preset` by name.
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
        by_key = {s.key: s for s in shot_presets.DEFAULT_PRESET}
        keys = list(dict.fromkeys(v.strip() for v in views if v.strip()))
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            raise ValueError(f"unknown views: {', '.join(unknown)} -- available: "
                             f"{', '.join(by_key)}")
        if not keys:
            raise ValueError("views resolved to an empty list")
        return [by_key[k] for k in keys]

    return list(shot_presets.resolve_preset(preset))


def generate_character_sheet(name: str, *,
                             preset: str = "default",
                             shots: Optional[list] = None,
                             views: Optional[list] = None,
                             wardrobe: str = "",
                             hair_color: str = "",
                             seed: int = -1,
                             anchor_chain: bool = True,
                             identity_lock: bool = True,
                             style: str = "none",
                             on_log=None) -> dict:
    """Render a shot-list character sheet from one reference photo.

    Ported from Phosphene's generate_character_sheet -- same seed
    handling (one resolved seed shared across every shot, not seed+i),
    same anchor_chain default (every shot after the first also gets the
    first rendered shot as a second reference, to stop hair-color drift
    -- see _view_prompt's docstring), same GPU-busy-fails-fast
    philosophy, same atomic sheet.png write for small shot counts.
    Grown from the original fixed 3 views into a preset/shot-list system
    (shot_presets.py) plus wardrobe/seed/identity-lock/style knobs for
    the settings panel. Simplified vs. Phosphene: no engine_override
    (hidream is the only engine wired up) and no cross-app GPU lock (see
    CharacterSheetBusyError's docstring).

    Shot list resolution: `shots` (explicit ShotSpec list) > `views`
    (legacy, default-preset keys only) > `preset` (by name, default
    "default" = the original 3-view turnaround, unchanged output).

    Raises:
      ValueError               -- bad name / preset / shots / views / seed
                                                            (route -> 400)
      LookupError               -- no such character          (route -> 404)
      FileNotFoundError         -- no reference image on disk (route -> 404)
      CharacterSheetBusyError -- a sheet render is already in flight in
                                  this process               (route -> 429)
      RuntimeError               -- engine failure           (route -> 500)
    """
    cid = _safe_id(name)
    shot_list = resolve_shots(preset=preset, shots=shots, views=views)

    wardrobe = str(wardrobe or "").strip()
    hair_color = str(hair_color or "").strip()
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise ValueError("seed must be an integer")
    resolved_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)

    if not character_exists(cid):
        raise LookupError(f"character {cid!r} not found")
    ref = character_avatar(cid)
    if ref is None:
        raise FileNotFoundError(
            f"character {cid!r} has no reference image -- expected an "
            f"avatar under exports/{cid}/character/")

    char_dir = _character_dir(cid)
    char_dir.mkdir(parents=True, exist_ok=True)

    if not _SHEET_LOCK.acquire(blocking=False):
        raise CharacterSheetBusyError(
            "a character-sheet render is already in progress -- try again "
            "once it finishes.")
    t0 = time.time()
    cfg = hidream_engine.HiDreamConfig()
    try:
        view_records = []
        anchor_png = None
        for i, spec in enumerate(shot_list):
            prompt = _shot_prompt(spec, wardrobe, identity_lock=identity_lock,
                                  style=style, hair_color=hair_color)
            view_dir = char_dir / "sheet_views" / spec.key
            view_dir.mkdir(parents=True, exist_ok=True)
            view_refs = [str(ref)] + (
                [anchor_png] if (anchor_png and anchor_chain and spec.use_anchor)
                else [])
            if on_log:
                on_log(f"[sheet] {cid}: shot {i + 1}/{len(shot_list)} ({spec.key})")
            candidates = hidream_engine.generate_hidream(
                prompt=prompt, n=1, width=1024, height=1024,
                output_dir=view_dir,
                base_seed=resolved_seed,
                refs=view_refs,
                config=cfg,
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
    finally:
        _SHEET_LOCK.release()

    sheet_path = None
    if len(view_records) <= MAX_SHOTS_FOR_COMPOSITE:
        sheet_path = char_dir / "sheet.png"
        tmp = char_dir / f".sheet.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.png"
        try:
            _compose_sheet_row([r["png_path"] for r in view_records], tmp)
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
        # over from an earlier, smaller-preset run on this character so
        # nothing on disk contradicts sheet.json's sheet_png=None
        # (character_sheet_png() also checks sheet.json rather than raw
        # file existence, so this is belt-and-suspenders tidiness, not
        # the only guard).
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


def regenerate_shot(name: str, shot_key: str, *,
                    seed: Optional[int] = None,
                    prompt: Optional[str] = None,
                    on_log=None) -> dict:
    """Re-render a single shot from an already-generated sheet and
    recomposite (when the shot count is small enough to composite at
    all -- see MAX_SHOTS_FOR_COMPOSITE).

    Reuses that shot's existing prompt and reference images from
    sheet.json unless overridden -- so a plain re-roll is "same shot,
    new random seed", not a full re-resolve of the anchor-chain logic
    (which would need re-running every other shot too).

    Raises:
      LookupError        -- no such character, or no such shot in its
                            current sheet                    (route -> 404)
      FileNotFoundError   -- no sheet.json yet to re-roll a shot in
                                                              (route -> 404)
      CharacterSheetBusyError -- a render is already in flight
                                                              (route -> 429)
      RuntimeError         -- engine failure                (route -> 500)
    """
    cid = _safe_id(name)
    if not character_exists(cid):
        raise LookupError(f"character {cid!r} not found")
    char_dir = _character_dir(cid)
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

    if not _SHEET_LOCK.acquire(blocking=False):
        raise CharacterSheetBusyError(
            "a character-sheet render is already in progress -- try again "
            "once it finishes.")
    t0 = time.time()
    cfg = hidream_engine.HiDreamConfig()
    try:
        use_prompt = prompt if prompt is not None else views[idx]["prompt"]
        use_seed = int(seed) if seed is not None else random.randint(0, 2**31 - 1)
        refs = views[idx].get("refs") or []
        view_dir = char_dir / "sheet_views" / shot_key
        view_dir.mkdir(parents=True, exist_ok=True)
        if on_log:
            on_log(f"[sheet] {cid}: re-rolling shot {shot_key!r}")
        candidates = hidream_engine.generate_hidream(
            prompt=use_prompt, n=1, width=1024, height=1024,
            output_dir=view_dir, base_seed=use_seed, refs=refs,
            config=cfg, on_log=on_log)
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
    finally:
        _SHEET_LOCK.release()

    sheet_path = None
    if len(views) <= MAX_SHOTS_FOR_COMPOSITE:
        sheet_path = char_dir / "sheet.png"
        tmp = char_dir / f".sheet.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.png"
        try:
            _compose_sheet_row([v["png_path"] for v in views], tmp)
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
