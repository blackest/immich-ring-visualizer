"""NG: Ref2VA reference conditioning (RefMods) for the H3 engine.

Small, single-purpose helpers h3_engineNG.py calls; nothing here runs a model.

What this guards, and why it is stricter than the runner:

* `--refmod` exists only on a runner that carries the Ref2VA patch
  (minimax-h3-mlx PR #2), and only does anything on a Ref2VA DiT. The official
  Ref2VA transformer is a different fine-tune from FL2VA (different shard
  hashes and tensor bytes), so it is its own model, "h3ref", with its own Q8 --
  FL2VA weights were never trained to read reference rows.
* The runner packs references into ONE window. Chained windows, a first-frame
  keyframe and the speed adapters (Turbo, LightX2V v1.3) are refused here, up
  front, because the runner would only discover it after loading 20 GiB of
  weights -- or, for the speed adapters, never (untested on Ref2VA).
* The reference limits (9 images, 3 videos, 12 in all) are the model's; the
  runner enforces them too, but only after the DiT is loaded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import h3_loraNG as h3_lora

REF2VA_DIT_ENV = "LTX_H3_REF2VA_DIT"
REFMOD_FLAG = "--refmod"
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
# (The model also caps a request at 12 references, but that is 9 + 3 -- with visual
# references only, the per-kind limits above already enforce it.)


def ref2va_dit_candidates_ng(models_root: Path) -> list:
    """Where a locally built Ref2VA Q8 may live: the env override first, then the
    shared model store beside the FL2VA weights, then a plain home folder."""
    cands = []
    env = os.environ.get(REF2VA_DIT_ENV)
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path(models_root) / "ref2va-q8")
    cands.append(Path.home() / "models" / "h3-ref2va" / "q8")
    return cands


def _is_q8_dir(p: Path) -> bool:
    return (p / "config.json").is_file() and (p / "quant_config.json").is_file() \
        and any(p.glob("model-*.safetensors"))


def resolve_ref2va_dit_ng(override: str, models_root: Path) -> Optional[str]:
    cands = [Path(override).expanduser()] if override else ref2va_dit_candidates_ng(models_root)
    for p in cands:
        if _is_q8_dir(p):
            return str(p)
    return None


def runner_supports_refmod_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, REFMOD_FLAG)


def validate_refmods_ng(refmods, *, model: str, image_path, turbo: bool, v13: bool,
                        chain_windows: int, runner) -> list:
    """`refmods`: [{"path", "strength", "images", "videos"}] -> the same list,
    cleaned. Raises ValueError with one plain sentence on anything the runner
    cannot do. An empty list is always fine (a plain render on any model)."""
    refmods = list(refmods or [])
    if not refmods:
        return []
    if model != "h3ref":
        raise ValueError("RefMods need the Ref2VA model (model='h3ref'); "
                         f"{model!r} was never trained to read reference rows")
    if not runner_supports_refmod_ng(runner):
        raise ValueError("this H3 runner has no --refmod support -- it needs the minimax-h3-mlx "
                         "build with the Ref2VA reference patch")
    if image_path is not None:
        raise ValueError("RefMods do not combine with a first-frame keyframe (one conditioning source at a time)")
    if turbo or v13:
        raise ValueError("RefMods do not combine with Turbo or LightX2V v1.3: those adapters "
                         "were not made for the Ref2VA weights and are untested with references")
    if chain_windows > 1:
        raise ValueError("RefMods render a single 5 s window; shorten the clip (chained windows "
                         "carry no references yet)")
    cleaned, images, videos = [], 0, 0
    for item in refmods:
        path = Path(str(item.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"RefMod not found at {path}")
        strength = item.get("strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"RefMod {path.stem!r} needs a strength between 0 and 1")
        images += int(item.get("images", 0))
        videos += int(item.get("videos", 0))
        if strength > 0.0:  # strength 0 is dropped by the runner, so it costs nothing
            cleaned.append({**item, "path": str(path), "strength": float(strength)})
    for label, have, limit in (("image", images, MAX_REF_IMAGES), ("video", videos, MAX_REF_VIDEOS)):
        if have > limit:
            raise ValueError(f"at most {limit} {label} references per render, got {have}")
    if not cleaned:
        raise ValueError("every RefMod is at strength 0; nothing to condition on")
    return cleaned


def refmod_argv_ng(refmods) -> list:
    """The runner arguments: one `--refmod PATH:STRENGTH` per cleaned entry."""
    argv = []
    for item in refmods or []:
        argv += [REFMOD_FLAG, f"{item['path']}:{item['strength']:g}"]
    return argv
