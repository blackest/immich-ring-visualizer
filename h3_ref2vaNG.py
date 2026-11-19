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

import json
import os
import re
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
                        chain_windows: int, runner, tristep: bool = False) -> list:
    """`refmods`: [{"path", "strength", "images", "videos"}] -> the same list,
    cleaned. Raises ValueError with one plain sentence on anything the runner
    cannot do. An empty list is always fine (a plain render on any model).

    `tristep` (Fast, the TaoMate 3-step adapter) is allowed: it was tested with a
    RefMod on the Ref2VA Q8 (640x384, 5 s, the likeness held). The LightX2V
    adapters (turbo, v1.3) still are not."""
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


# --- Hybrid ("clean look"): FL2VA weights, Ref2VA AdaLN on the last blocks -----------------------
# Ref2VA's reference conditioning comes with a harsher, more contrasty look (Minimax has said the
# two checkpoints went through different post-training). The two models differ almost only in the
# per-block AdaLN projections, so the hybrid runs the FL2VA Q8 everywhere and takes Ref2VA's AdaLN
# for a block range only (community testing: 45-49). The runner does it at the modulation-table
# level (minimax_h3_mlx/adaln_overlay.py); this side only validates and builds the argv.
ADALN_OVERLAY_FLAG = "--adaln-overlay"
ADALN_BLOCKS_FLAG = "--adaln-blocks"
DEFAULT_ADALN_BLOCKS = "45-49"
_BLOCKS_RE = re.compile(r"^\d+(-\d*)?(,\d+(-\d*)?)*$")


def runner_supports_adaln_overlay_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, ADALN_OVERLAY_FLAG)


def validate_adaln_blocks_ng(spec, *, model: str, flex: bool, turbo: bool, v13: bool, runner) -> str:
    """`spec` ("" = no hybrid) -> the cleaned block range, or "". Raises ValueError with one plain sentence."""
    spec = str(spec or "").replace(" ", "")
    if not spec:
        return ""
    if model != "h3ref":
        raise ValueError("the hybrid look takes Ref2VA's AdaLN, so it is a Ref2VA render (model='h3ref')")
    if flex or turbo or v13:
        raise ValueError("the hybrid look does not combine with Flex, Turbo or LightX2V v1.3; "
                         "Fast (the 3-step adapter) and the plain 16-step run are fine")
    if spec.lower() == "none":
        spec = "none"
    elif not _BLOCKS_RE.match(spec):
        raise ValueError(f"adaln blocks {spec!r} is not a block range like 45-49")
    if not runner_supports_adaln_overlay_ng(runner):
        raise ValueError("this H3 runner has no --adaln-overlay -- it needs the vendored runner")
    return spec


# Where the hybrid takes Ref2VA's AdaLN from. The overlay reads only 14 tensors per block range, so the
# runner accepts either the full Ref2VA Q8 (47.5 GB) or the small pack scripts/extract_adaln_overlay_pack.py
# cuts from it (2.7 GB for blocks 45-49); the tables come out identical. The switch is an env var like the
# other RINGVIZ_H3_* ones: auto (the full model if it is on disk, else the pack), full, or pack.
ADALN_SOURCE_ENV = "RINGVIZ_H3_ADALN_OVERLAY"
ADALN_PACK_ENV = "RINGVIZ_H3_ADALN_PACK"
ADALN_PACK_DIRNAME = "adaln-overlay-45-49"
ADALN_SOURCES = ("auto", "full", "pack")
_TIME_EMBEDDER = tuple(f"time_embedder.{part}.{kind}" for part in ("proj_in", "proj_out") for kind in ("weight", "bias"))


def adaln_source_mode_ng(raw=None) -> str:
    """The switch: "auto" (default), "full" or "pack". `raw` defaults to the env var."""
    mode = str(os.environ.get(ADALN_SOURCE_ENV, "") if raw is None else raw).strip().lower() or "auto"
    if mode not in ADALN_SOURCES:
        raise ValueError(f"{ADALN_SOURCE_ENV} is {mode!r}; it must be one of {', '.join(ADALN_SOURCES)}")
    return mode


def adaln_pack_candidates_ng(models_root: Path) -> list:
    cands = []
    env = os.environ.get(ADALN_PACK_ENV)
    if env:
        cands.append(Path(env).expanduser())
    cands.append(Path(models_root) / ADALN_PACK_DIRNAME)
    cands.append(Path.home() / "models" / "h3-ref2va" / ADALN_PACK_DIRNAME)
    return cands


def _is_pack_dir(p: Path) -> bool:
    return (p / "config.json").is_file() and (p / "model.safetensors").is_file()


def resolve_adaln_pack_ng(models_root: Path) -> Optional[str]:
    for p in adaln_pack_candidates_ng(models_root):
        if _is_pack_dir(p):
            return str(p)
    return None


def _blocks_of(spec: str, num_layers: int) -> list:
    """"45-49", "45-", "3" or "0,2,45-49" -> block numbers; "none" -> []."""
    if spec == "none":
        return []
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        lo_n = int(lo)
        hi_n = (int(hi) if hi else num_layers - 1) if "-" in part else lo_n
        out.update(range(lo_n, hi_n + 1))
    return sorted(out)


def pack_missing_blocks_ng(pack_dir, blocks_spec: str) -> list:
    """Block numbers `blocks_spec` needs that the pack does not hold ([] = fine), plus -1 when its
    time embedder is missing. Reads only the safetensors header."""
    pack = Path(pack_dir)
    with open(pack / "model.safetensors", "rb") as f:
        head = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    have = set(head)
    layers = int(json.loads((pack / "config.json").read_text()).get("num_layers", 50))
    blocks = _blocks_of(blocks_spec, layers)
    missing = [b for b in blocks if not all(f"blocks.{b}.adaln_proj.linear.{k}" in have for k in ("weight", "bias"))]
    if blocks and not all(n in have for n in _TIME_EMBEDDER):
        missing.append(-1)
    return missing


def pick_overlay_source_ng(mode: str, full_dit, pack, blocks_spec: str) -> str:
    """The folder to hand the runner as `--adaln-overlay`, per the switch. Raises FileNotFoundError
    (the wanted source is not on disk) or ValueError (the pack lacks a block) with one plain sentence."""
    if mode == "full" or (mode == "auto" and full_dit):
        if not full_dit:
            raise FileNotFoundError("the hybrid look is set to use the full Ref2VA model "
                                    f"({ADALN_SOURCE_ENV}=full), which is not on this machine")
        return str(full_dit)
    if not pack:
        raise FileNotFoundError(
            "the hybrid look needs the full Ref2VA model or the small AdaLN pack "
            f"({ADALN_PACK_DIRNAME}, made by scripts/extract_adaln_overlay_pack.py); neither was found")
    missing = pack_missing_blocks_ng(pack, blocks_spec)
    if missing:
        what = "its time embedder" if missing == [-1] else "block " + ", ".join(str(b) for b in missing if b >= 0)
        raise ValueError(f"the small AdaLN pack does not hold {what}; rebuild it with "
                         "scripts/extract_adaln_overlay_pack.py --blocks, or set "
                         f"{ADALN_SOURCE_ENV}=full to use the full Ref2VA model")
    return str(pack)


def adaln_overlay_argv_ng(overlay_dir, blocks: str) -> list:
    """`--adaln-overlay REF2VA_DIR --adaln-blocks RANGE`, or [] when there is no hybrid."""
    if not blocks:
        return []
    return [ADALN_OVERLAY_FLAG, str(overlay_dir), ADALN_BLOCKS_FLAG, blocks]
