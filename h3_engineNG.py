"""NG engine for MiniMax-H3 (Hailuo H3) prompt/image-to-video generation.

Talks to the standalone minimax-h3-mlx install as a subprocess (its own
venv, never imported in-process) -- same shape as ltx_engineNG.py: one
input (prompt + optional keyframe image + duration) -> one output (an
mp4 path + result dict).

H3 and H3Q8 are the SAME model at two DiT precisions -- bf16
("deepbeep-pruned") vs quantized ("h3-dit-q8") -- selected by which
--dit path this passes to generate_staged.py, not two separate engines
or two separate subprocess calls. The VAEs/text-encoder (the "compact"
pack) and the upstream text-encoder config are shared by both.

Defaults to H3's "high" tier -- 1024x576, true 16:9, 16 steps -- the
same canvas ltx_engineNG.py defaults to (LTX_WIDTH/LTX_HEIGHT), and
phosphene's own recommended delivery canvas (see mlx_ltx_panel.py's
_h3_qualities()["high"]). H3_QUALITIES below mirrors that same tier
table (draft/standard/high/native) so callers can pick a size instead.
Measured at "high", 8 forwards/1024x576: 18.8 min wall, no memory-wall
difference vs the smaller "standard" canvas (the DiT weights dominate,
not activations) -- so H3_TIMEOUT_S below budgets roughly double that
for the 16-step default. Not (yet) re-measured at "native", the one
tier above "high" -- if that turns out to need longer than the shared
timeout, raise it per-call via RINGVIZ_H3_TIMEOUT_S rather than baking
a bigger default in for every tier.

Chained multi-window clips (--chain-windows) ARE in scope, as of the
minimax-h3-mlx pull that landed 69ec486 ("Chain 5-second windows into
clips the dense path cannot reach") -- a duration beyond one window
(H3_CHAIN_WINDOW_S) is rendered as N windows, each conditioned on the
previous window's last decoded frame (the runner's own --first-frame
path, reused), then stitched into one mp4 by the runner itself. See
_h3_chain_plan_ng.

The turbo LoRA (larryvrh/MiniMax-H3-Turbo-Lora, Apache-2.0, a 4-step
distillation adapter -- see H3_TURBO_LORA_PATH) IS also wired, as an
opt-in `turbo=True` on generate_h3_video_ng: forces --steps 4 and adds
--lora/--lora-adaln, cutting each window's denoise from ~15 steps to
4 for roughly a 3-4x speedup, at the real quality cost the runner's
own README is explicit about (4 steps without the adapter is
"unresolved", not broken -- the adapter is what makes 4 usable, not a
free identical-quality shortcut). Both required files were already
present on disk under H3_MODELS_ROOT/turbo-lora/ -- nothing to fetch.

--chain-prompts (a different prompt per window) IS wired as the optional
`chain_prompts` list, one string per window. Without it every window gets
the same prompt (the runner's own fallback), so a multi-shot prompt plays
its opening again in each window.

Deliberately still NOT wired: live preview, draft/TAE decode, step-cache,
prompt-cache reuse. User
style/character LoRAs ARE wired: `loras=[(path, strength)]` becomes a
repeated --lora PATH:SCALE after any speed adapter (names are resolved to
paths by h3_lora_libraryNG.py; each file is first laid out in the base's
qkv row order by h3_loraNG.lora_for_runner_ng).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import h3_loraNG as h3_lora
import h3_ref2vaNG as h3_ref2va

# "h3ref" is the official Ref2VA transformer (own Q8, see h3_ref2vaNG.py): the only
# model that reads RefMod reference rows. Same compact VAE/text-encoder pack.
H3Model = Literal["h3", "h3q8", "h3ref"]


class H3JobCancelled(RuntimeError):
    """Raised when the H3 subprocess is killed by its watchdog timeout.
    No cancel button exists yet, so the only source here is the timeout."""


def _resolve_h3_repo_dir_ng() -> Path:
    """LTX_H3_ROOT env var (same name phosphene's own mlx_ltx_panel.py
    reads, so one override works for both apps), then conventional
    install locations, falling back to the first even if it doesn't
    exist so callers get a clean "not found" error pointing at the
    conventional location."""
    env_p = os.environ.get("LTX_H3_ROOT")
    if env_p:
        return Path(env_p).expanduser()
    home = Path.home()
    # Phosphene's own vendored checkout comes first: it sits on the pinned
    # engine commit with a venv held to its locked package set, and sharing
    # it avoids a second ~copy drifting out of date.
    for cand in (Path("/Volumes/AI/Pinkio/api/phosphene.git/minimax-h3-mlx"),
                 home / "minimax-h3-mlx",
                 home / "AI" / "minimax-h3-mlx",
                 Path("/Volumes/AI/minimax-h3-mlx")):
        if cand.exists():
            return cand
    return home / "minimax-h3-mlx"


def _resolve_h3_models_root_ng() -> Path:
    """LTX_H3_MODELS env var (same name phosphene reads), then the
    conventional shared mlx_models/hailuo-h3/models tree phosphene
    itself resolves onto (same drive ltx_engineNG.py's
    LTX_WEIGHTS_ROOT lives under -- one shared model store, per the
    two apps' agreed split)."""
    env_p = os.environ.get("LTX_H3_MODELS")
    if env_p:
        return Path(env_p).expanduser()
    cand = Path("/Volumes/AI/Pinkio/drive/drives/peers/d1785024377531"
                "/mlx_models/hailuo-h3/models")
    if cand.exists():
        return cand
    return Path.home() / "mlx_models" / "hailuo-h3" / "models"


H3_REPO_DIR = _resolve_h3_repo_dir_ng()
H3_MODELS_ROOT = _resolve_h3_models_root_ng()
H3_DEFAULT_PYTHON = H3_REPO_DIR / ".venv" / "bin" / "python"
H3_RUNNER = H3_REPO_DIR / "scripts" / "generate_staged.py"

H3_DIT_BF16 = H3_MODELS_ROOT / "deepbeep-pruned-bf16" / "MiniMax-H3-FL2VA-pruned_bf16.safetensors"
H3_DIT_Q8_DIR = H3_MODELS_ROOT / "h3-dit-q8"
H3_COMPACT_ROOT = H3_MODELS_ROOT / "ddalcu-q8"
H3_COMPACT_FILES = ("text_encoder.safetensors", "video_vae.safetensors", "audio_vae.safetensors")
H3_TEXT_CONFIG = H3_MODELS_ROOT / "upstream-meta" / "FL2VA" / "text_encoder" / "config.json"

# larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0), the 4-step distillation
# adapter -- see generate_staged.py's README "Turbo LoRA: 4-step
# sampling" section for the two loader traps (qkv row permutation,
# adaLN absorption) this file relies on the runner having already
# fixed. --lora-adaln needs the second file (the upstream
# time_embedder the pruned checkpoint dropped), not the DiT itself.
H3_TURBO_LORA_PATH = H3_MODELS_ROOT / "turbo-lora" / "minimax_h3_turbo_4step_ema_ckpt500.safetensors"
H3_TURBO_ADALN_PATH = H3_MODELS_ROOT / "turbo-lora" / "upstream_time_embedder.safetensors"
# Phosphene's current default (v4.19.x): LightX2V's v1.0 4-step adapter
# repacked into the runner layout. It carries no adaLN pairs, so it needs
# no --lora-adaln companion and is passed as PATH:SCALE. Preferred over
# the retired ckpt500 pair above, which stays as the fallback.
H3_TURBO_V1_LORA_PATH = H3_MODELS_ROOT / "turbo-lora" / "lightx2v_v1.0_768p_ourlayout.safetensors"
H3_TURBO_STEPS = 4
# lightx2v v1.3 ("Best, faster", Phosphene 4.19.11): LightX2V's 768p adapter,
# built locally by Phosphene's scripts/fetch_h3_v13.py already in the DiT's
# per-head qkv order (so, unlike the turbo adapters, NO bake step). 7 sigma
# points = 6 forwards at scale 1.0, ~870 s vs ~1950 s for the full 16-step
# sampler on a 5 s 1024x576 clip. Only validated at the "high" canvas.
H3_V13_LORA_PATH = H3_MODELS_ROOT / "turbo-lora" / "lightx2v_v1.3_768p_phosphene.safetensors"
H3_V13_STEPS = 7
H3_V13_VIDEO_SHIFT = "6"  # LightX2V's training shift; only sent if the runner has --video-shift
# TaoMate-H3's 3-step adapter ("Fast" in Phosphene). It was distilled onto FOUR points of the
# 50-point shifted sigma grid, so it runs THREE forwards: `--steps 4 --sigma-subset
# 50:0,16,33,49` on the runner, not `--steps 4` alone (the adapter never saw a re-spaced grid).
# Candidates mirror Phosphene's own allowlist: a full-rank copy if one is there, else the
# 182 MB rank-19 file; the size floor rejects a truncated download. Measured on the Ref2VA Q8
# with an 8-image RefMod: 640x384, 5 s, 369 s on an M2 Max, and the likeness held. It goes in as
# `--lora PATH:1.0` untouched -- Phosphene's Fast tier does the same, no qkv bake.
H3_TRISTEP_STEPS = 4
H3_TRISTEP_SIGMA_SUBSET = "50:0,16,33,49"
H3_TRISTEP_CANDIDATES = (
    (H3_MODELS_ROOT / "turbo-lora" / "taomate_h3_3step_ourlayout.safetensors", 2 * 1024 ** 3),
    (H3_MODELS_ROOT / "turbo-lora" / "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors",
     150 * 1024 ** 2),
)
# The runner's own summary line for each adapter, e.g.
# "LoRA name.safetensors scale 1 mode runtime: 208 applied, 0 skipped, ...". The speed adapter
# is first on the command line, so its line is the first match.
_H3_LORA_APPLIED_RE = re.compile(r"LoRA\s+\S+.*?:\s*(\d+)\s+applied")
# `--lazy-adaln` (minimax-h3-mlx): builds the AdaLN cache block by block so the ~24 GiB of
# projections are never resident together -- the Q8 DiT load peak drops from 44 to 20 GiB and the
# output is byte-identical. Used automatically when the runner has it; RINGVIZ_H3_LAZY_ADALN=0 opts out.
H3_LAZY_ADALN_FLAG = "--lazy-adaln"
# `--adaln-cache-dir` (minimax-h3-mlx branch adaln-cache): the AdaLN tables are stored once per
# schedule and reloaded, and a hit never reads the ~24 GiB of AdaLN weights at all (the AdaLN stage
# drops from 4.5 s to 0.1 s; the output is byte-identical). Used automatically when the runner has
# it; the folder defaults to ~/.cache/ringviz/h3-adaln, RINGVIZ_H3_ADALN_CACHE points it elsewhere
# and RINGVIZ_H3_ADALN_CACHE=0 turns it off.
H3_ADALN_CACHE_FLAG = "--adaln-cache-dir"

H3_FRAME_RATE = 24.0
# packing.py: 17n+5 frame grid, 32px canvas grid -- duplicated here (not
# imported) since this file talks to the H3 venv only as a subprocess,
# same reasoning ltx_engineNG.py gives for its own duplicated constants.
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_DIM_STEP = 32
H3_MIN_DIM = 256
H3_MAX_DIM = 1536

# phosphene's "high" tier -- true 16:9, the recommended delivery canvas,
# same dims LTX_WIDTH/LTX_HEIGHT already default to in ltx_engineNG.py.
H3_WIDTH = 1024
H3_HEIGHT = 576
H3_STEPS = 16

# Named size presets ("quality" in the UI) -- phosphene's own tier names
# and dims (mlx_ltx_panel.py's _h3_qualities()), all true 16:9. "high" is
# just H3_WIDTH/H3_HEIGHT under another name, so the existing default is
# unchanged; "native" is above 720p (callers exporting to a 720p cap
# should downscale on export, not pick native for delivery).
H3_QUALITIES = {
    "draft": (640, 384),
    "standard": (768, 448),
    "high": (H3_WIDTH, H3_HEIGHT),
    "native": (1344, 768),
}

# Per-tier watchdog budget -- self-attention over the packed video tokens
# scales worse than linearly with pixel count, so "native" (above "high"
# on both dims) isn't safely covered by the same 3600s the smaller tiers
# are measured against; give it more room rather than let a legitimate
# native render race H3JobCancelled. Unmeasured guess, not a benchmark --
# tune once a real native run's wall time is known. RINGVIZ_H3_TIMEOUT_S
# (env) still overrides any of these when set, for manual tuning.
H3_QUALITY_TIMEOUTS = {
    "draft": 3600.0,
    "standard": 3600.0,
    "high": 3600.0,
    "native": 7200.0,
}

H3_MIN_DURATION_S = 3.0
# 15.0 was phosphene's own validated ceiling (3 chained windows -- nobody,
# including phosphene, has data past that). Raised to 30.0 (6 windows) to
# test past it, 2026-09-29 -- the runner itself has no hard cap
# (chain_windows math in _h3_chain_plan_ng is unbounded), this is purely
# an "our own choice, untested past 3 windows" ceiling. Revisit if a
# 6-window chain's seams/drift look worse than the validated 3-window
# ones did.
H3_MAX_DURATION_S = 30.0

# The dense (--chain-windows 1) path's proven-reliable ceiling -- phosphene
# chains anything longer into this many seconds per window (see
# minimax-h3-mlx's 69ec486, "Chain 5-second windows into clips the dense
# path cannot reach"). A duration that fits in one window renders exactly
# as it always has; only a longer one pays the chain's extra windows.
H3_CHAIN_WINDOW_S = 5.0


def _validate_h3_dims_ng(width: int, height: int) -> None:
    for name, v in (("width", width), ("height", height)):
        if v % H3_DIM_STEP != 0:
            raise ValueError(f"{name} must be a multiple of {H3_DIM_STEP} (got {v})")
        if not (H3_MIN_DIM <= v <= H3_MAX_DIM):
            raise ValueError(f"{name} must be between {H3_MIN_DIM} and {H3_MAX_DIM} (got {v})")


def resolve_h3_quality_ng(quality: str) -> tuple:
    """Maps a named tier (see H3_QUALITIES) to (width, height). Raises
    ValueError for an unknown name -- same "clean 400, not a KeyError"
    shape as _validate_h3_dims_ng."""
    dims = H3_QUALITIES.get(quality)
    if dims is None:
        raise ValueError(
            f"quality must be one of {', '.join(H3_QUALITIES)} (got {quality!r})")
    return dims


def resolve_h3_quality_timeout_ng(quality: Optional[str],
                                   duration_s: Optional[float] = None) -> Optional[float]:
    """Maps a named tier to its H3_QUALITY_TIMEOUTS budget. None (no
    named tier -- an explicit width/height call) or an unrecognized
    name both return None, meaning "let H3Config.timeout_s's own
    default apply" rather than raise -- unlike resolve_h3_quality_ng,
    an unknown quality here isn't fatal to the render.

    duration_s scales the budget by how many chained windows that
    duration needs (_h3_chain_plan_ng) -- a 3-window chain takes
    roughly 3x one window's wall time, so the flat per-tier budget
    alone would let the watchdog kill a legitimate long chain."""
    base = H3_QUALITY_TIMEOUTS.get(quality) if quality else None
    if base is None or not duration_s:
        return base
    _, chain_windows, _ = _h3_chain_plan_ng(duration_s)
    return base * chain_windows


def _validate_h3_duration_ng(duration_s: float) -> None:
    if not (H3_MIN_DURATION_S <= duration_s <= H3_MAX_DURATION_S):
        raise ValueError(
            f"duration_s must be between {H3_MIN_DURATION_S} and "
            f"{H3_MAX_DURATION_S} (got {duration_s})")


def _align_h3_frames_ng(duration_s: float, frame_rate: float = H3_FRAME_RATE) -> int:
    """Snap seconds -> the 17n+5 frame count the video VAE encodes --
    same math as packing.align_num_frames, starting from round(seconds *
    frame_rate) the way MiniMaxH3Pipeline.__call__ does."""
    n = max(1, round(duration_s * frame_rate))
    while n % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        n += 1
    return n


def _h3_chain_plan_ng(duration_s: float) -> tuple[int, int, int]:
    """(window_frames, chain_windows, total_frames) for duration_s.

    chain_windows is 1 -- today's untouched dense path, same output as
    before this existed -- whenever the whole clip fits in one
    H3_CHAIN_WINDOW_S window. Otherwise this is the same stride math
    generate_staged.py's own --chain-windows uses (frames + (n-1) *
    (frames-1) delivered, trimmed to total_frames via
    --chain-total-frames), computed here up front so the timeout budget
    and the subprocess command agree on the same plan."""
    total_frames = _align_h3_frames_ng(duration_s)
    window_frames = _align_h3_frames_ng(H3_CHAIN_WINDOW_S)
    if total_frames <= window_frames:
        return total_frames, 1, total_frames
    stride = window_frames - 1
    chain_windows = 1 + math.ceil((total_frames - window_frames) / stride)
    return window_frames, chain_windows, total_frames


def _clean_subprocess_env_ng() -> dict:
    """os.environ.copy() with macOS Malloc* debug vars stripped, so the
    H3 subprocess doesn't spam stderr with debug-allocator noise (same
    as ltx_engineNG.py's own helper)."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("Malloc"):
            del env[key]
    return env


@dataclass
class H3Config:
    """How to run minimax-h3-mlx's generate_staged.py for one render."""
    model: H3Model = "h3q8"  # phosphene's own AUTO default since v4.8.0
    python_path: str = ""       # default: H3_DEFAULT_PYTHON
    runner_path: str = ""       # default: H3_RUNNER
    compact_root: str = ""      # default: H3_COMPACT_ROOT
    text_config: str = ""       # default: H3_TEXT_CONFIG
    dit_bf16_path: str = ""     # default: H3_DIT_BF16
    dit_q8_dir: str = ""        # default: H3_DIT_Q8_DIR
    dit_ref2va_dir: str = ""    # default: h3_ref2vaNG.ref2va_dit_candidates_ng()
    timeout_s: float = 3600.0   # absolute per-call ceiling; override via RINGVIZ_H3_TIMEOUT_S
    stall_s: float = 1200.0     # no-output watchdog; override via RINGVIZ_H3_STALL_S


def _resolve_h3_python_ng(config: H3Config) -> Optional[str]:
    p = Path(config.python_path) if config.python_path else H3_DEFAULT_PYTHON
    return str(p) if p.is_file() else None


def _resolve_h3_runner_ng(config: H3Config) -> Optional[str]:
    p = Path(config.runner_path) if config.runner_path else H3_RUNNER
    return str(p) if p.is_file() else None


def _resolve_h3_compact_root_ng(config: H3Config) -> Optional[str]:
    p = Path(config.compact_root) if config.compact_root else H3_COMPACT_ROOT
    return str(p) if all((p / f).is_file() for f in H3_COMPACT_FILES) else None


def _resolve_h3_text_config_ng(config: H3Config) -> Optional[str]:
    p = Path(config.text_config) if config.text_config else H3_TEXT_CONFIG
    return str(p) if p.is_file() else None


def _resolve_h3_dit_ng(config: H3Config) -> Optional[str]:
    """Which DiT weights to pass, per config.model: bf16 is a single
    .safetensors file, q8 is a sharded directory (config.json +
    quant_config.json + model-*.safetensors) -- generate_staged.py's
    --dit accepts either. "h3ref" is the Ref2VA Q8 directory."""
    if config.model == "h3ref":
        return h3_ref2va.resolve_ref2va_dit_ng(config.dit_ref2va_dir, H3_MODELS_ROOT)
    if config.model == "h3q8":
        p = Path(config.dit_q8_dir) if config.dit_q8_dir else H3_DIT_Q8_DIR
        ok = (p / "config.json").is_file() and (p / "quant_config.json").is_file() \
            and any(p.glob("model-*.safetensors"))
        return str(p) if ok else None
    p = Path(config.dit_bf16_path) if config.dit_bf16_path else H3_DIT_BF16
    return str(p) if p.is_file() else None


def _resolve_h3_turbo_ng() -> tuple:
    """(lora_path, adaln_path) as strs, or (None, None) if no complete
    adapter is on disk. Prefers the v1.0 LightX2V adapter (adaln_path is
    None -- it has no adaLN pairs); otherwise the retired ckpt500 pair,
    which is all-or-nothing: --lora-adaln without --lora (or vice versa)
    isn't a state generate_staged.py's turbo mode is documented for."""
    if H3_TURBO_V1_LORA_PATH.is_file():
        return str(H3_TURBO_V1_LORA_PATH), None
    lora, adaln = H3_TURBO_LORA_PATH, H3_TURBO_ADALN_PATH
    if lora.is_file() and adaln.is_file():
        return str(lora), str(adaln)
    return None, None


def _h3_turbo_argv_ng(turbo_lora: str, turbo_adaln: Optional[str]) -> list:
    """The runner argv for the resolved adapter: v1.0 (no adaLN) uses the
    PATH:SCALE spelling the pinned runner documents; the legacy ckpt500
    pair keeps its original --lora + --lora-adaln form."""
    if turbo_adaln is None:
        return ["--lora", f"{turbo_lora}:1.0"]
    return ["--lora", turbo_lora, "--lora-adaln", turbo_adaln]


def _resolve_h3_tristep_ng() -> Optional[str]:
    """The TaoMate 3-step adapter on disk (first candidate that is a real file of at least its
    floor size), or None."""
    for path, floor in H3_TRISTEP_CANDIDATES:
        try:
            if path.is_file() and path.stat().st_size >= floor:
                return str(path)
        except OSError:
            continue
    return None


def _h3_tristep_argv_ng(tristep_lora: str) -> list:
    """The runner argv for Fast: the TaoMate ladder and the adapter at 1.0. `--steps` is set
    separately (H3_TRISTEP_STEPS) like every speed adapter's."""
    return ["--sigma-subset", H3_TRISTEP_SIGMA_SUBSET, "--lora", f"{tristep_lora}:1.0"]


def _h3_lazy_adaln_argv_ng(runner, legacy_adaln: bool) -> list:
    """`--lazy-adaln` when the runner has it. Not for the retired ckpt500 turbo, which folds adaLN
    pairs into the cache (`--lora-adaln`) -- that combination is untested with the lazy load."""
    if legacy_adaln:
        return []
    if os.environ.get("RINGVIZ_H3_LAZY_ADALN", "1").strip().lower() in ("0", "false", "off", "no"):
        return []
    return [H3_LAZY_ADALN_FLAG] if h3_lora.runner_has_flag_ng(runner, H3_LAZY_ADALN_FLAG) else []


def _h3_adaln_cache_dir_ng() -> Optional[Path]:
    """Where the AdaLN tables are stored, or None when the cache is switched off."""
    raw = os.environ.get("RINGVIZ_H3_ADALN_CACHE", "").strip()
    if raw.lower() in ("0", "false", "off", "no"):
        return None
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "ringviz" / "h3-adaln"


def _h3_adaln_cache_argv_ng(runner, legacy_adaln: bool) -> list:
    """`--adaln-cache-dir DIR` when the runner has it. Not for the retired ckpt500 turbo, whose adaLN
    pairs are folded into the tables (`--lora-adaln`) -- that combination was not tested."""
    folder = _h3_adaln_cache_dir_ng()
    if legacy_adaln or folder is None or not h3_lora.runner_has_flag_ng(runner, H3_ADALN_CACHE_FLAG):
        return []
    return [H3_ADALN_CACHE_FLAG, str(folder)]


# The phrase the stackable runner's `--lora` help carries -- probing the help
# text, not the flag, is what tells a repeatable `--lora` from the older
# single-slot one (both spell the flag identically).
H3_LORA_STACK_MARKER = "Repeat the flag to stack adapters"
H3_LORA_STACK_LIMIT = 4
H3_LORA_STRENGTH_MIN, H3_LORA_STRENGTH_MAX = 0.0, 2.0


def h3_lora_limit_ng(runner, speed_adapter: bool) -> int:
    """How many user LoRAs a render may carry on this runner: up to
    H3_LORA_STACK_LIMIT on a stacking runner; on the single-slot one, one --
    and none if a speed adapter (turbo / v1.3) already spent the slot."""
    if h3_lora.runner_has_flag_ng(runner, H3_LORA_STACK_MARKER):
        return H3_LORA_STACK_LIMIT
    return 0 if speed_adapter else 1


def _validate_h3_loras_ng(loras) -> list:
    """[(path, strength)] cleaned up, or ValueError. Paths must already be
    resolved from the library (h3_lora_libraryNG) -- this only checks shape."""
    out = []
    for item in loras or []:
        try:
            path, strength = item
            strength = float(strength)
        except (TypeError, ValueError):
            raise ValueError("each LoRA needs a path and a numeric strength") from None
        if not H3_LORA_STRENGTH_MIN <= strength <= H3_LORA_STRENGTH_MAX:
            raise ValueError(
                f"LoRA strength must be between {H3_LORA_STRENGTH_MIN:g} and "
                f"{H3_LORA_STRENGTH_MAX:g} (got {strength:g})")
        out.append((str(path), strength))
    return out


# Metal device contention -- same reasoning as ltx_engineNG.py's own
# _LTX_SUBPROCESS_LOCK, kept separate since this is a different venv
# and a different subprocess.
_H3_SUBPROCESS_LOCK = threading.Lock()


def h3_health_ng(model: H3Model = "h3q8") -> dict:
    """Cheap up-front check the routes layer can surface to the UI/status
    endpoint without launching a subprocess."""
    cfg = H3Config(model=model)
    python = _resolve_h3_python_ng(cfg)
    runner = _resolve_h3_runner_ng(cfg)
    compact_root = _resolve_h3_compact_root_ng(cfg)
    text_config = _resolve_h3_text_config_ng(cfg)
    dit = _resolve_h3_dit_ng(cfg)
    turbo_lora, turbo_adaln = _resolve_h3_turbo_ng()
    return {
        "repo_dir": str(H3_REPO_DIR),
        "models_root": str(H3_MODELS_ROOT),
        "model": model,
        "python_ok": python is not None,
        "python_path": python or str(H3_DEFAULT_PYTHON),
        "runner_ok": runner is not None,
        "runner_path": runner or str(H3_RUNNER),
        "dit_ok": dit is not None,
        "dit_path": dit or (str(H3_DIT_Q8_DIR) if model == "h3q8"
                            else str(h3_ref2va.ref2va_dit_candidates_ng(H3_MODELS_ROOT)[0]) if model == "h3ref"
                            else str(H3_DIT_BF16)),
        # RefMods need BOTH a Ref2VA DiT on disk and a runner that takes --refmod; the
        # UI shows its rows only when `refmod_ready`, with the missing half spelled out.
        "ref2va_ok": h3_ref2va.resolve_ref2va_dit_ng("", H3_MODELS_ROOT) is not None,
        "refmod_supported": bool(runner) and h3_ref2va.runner_supports_refmod_ng(runner),
        "refmod_ready": h3_ref2va.resolve_ref2va_dit_ng("", H3_MODELS_ROOT) is not None
                        and bool(runner) and h3_ref2va.runner_supports_refmod_ng(runner),
        "compact_ok": compact_root is not None,
        "compact_path": compact_root or str(H3_COMPACT_ROOT),
        "text_config_ok": text_config is not None,
        "text_config_path": text_config or str(H3_TEXT_CONFIG),
        "turbo_ok": turbo_lora is not None,
        # Fast = the TaoMate 3-step adapter: needs the file AND a runner with --sigma-subset.
        "tristep_ok": _resolve_h3_tristep_ng() is not None,
        "tristep_supported": bool(runner) and h3_lora.runner_has_flag_ng(runner, "--sigma-subset"),
        # Informational: this runner will use the lazy AdaLN load on its own.
        "lazy_adaln": bool(runner) and bool(_h3_lazy_adaln_argv_ng(runner, legacy_adaln=False)),
        # Informational: this runner will store and reuse the AdaLN tables on its own.
        "adaln_cache": bool(runner) and bool(_h3_adaln_cache_argv_ng(runner, legacy_adaln=False)),
        "v13_ok": H3_V13_LORA_PATH.is_file(),
        "ready": all((python, runner, dit, compact_root, text_config)),
    }


def generate_h3_video_ng(prompt: str, image_path: Optional[str], duration_s: float,
                          output_dir: Path, seed: Optional[int],
                          config: H3Config,
                          width: Optional[int] = None, height: Optional[int] = None,
                          steps: Optional[int] = None,
                          turbo: bool = False, v13: bool = False,
                          loras: Optional[list] = None,
                          refmods: Optional[list] = None,
                          on_log: Optional[Callable[[str], None]] = None,
                          on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None,
                          tristep: bool = False,
                          chain_prompts: Optional[list] = None,
                          extend_from: Optional[str] = None,
                          last_frame_path: Optional[str] = None) -> dict:
    """One subprocess call: prompt + duration in, one mp4 out. image_path
    is optional -- when given, it's H3's FL2VA first-frame keyframe
    conditioning (--first-frame); when None, it's a plain text-to-video
    render. config.model picks bf16 ("h3") or quantized ("h3q8") DiT
    weights -- same compact VAE/text-encoder pack either way. width/
    height/steps default to H3_WIDTH/H3_HEIGHT/H3_STEPS (the "high" true
    16:9 canvas) when omitted.

    turbo=True applies the larryvrh 4-step distillation LoRA and FORCES
    steps to H3_TURBO_STEPS regardless of the steps argument -- turbo
    is a coherent package (this exact adapter measured at this exact
    step count), not a step count you pick independently of it. Raises
    FileNotFoundError if the adapter files aren't on disk (check
    h3_health_ng()['turbo_ok'] first to avoid that).

    v13=True renders the faster "Best" path: LightX2V's v1.3 adapter at
    7 steps (6 forwards), forcing steps to H3_V13_STEPS. Same bf16-only
    reason as turbo, and only validated on the "high" canvas; mutually
    exclusive with turbo (one speed adapter per run).

    loras is an optional [(path, strength)] list of user style/character
    LoRAs (paths already resolved from h3_lora_libraryNG), each passed as a
    repeated --lora PATH:SCALE after any speed adapter. Each file is first
    laid out in the base's qkv row order (h3_loraNG.lora_for_runner_ng).

    refmods is an optional [{"path", "strength", "images", "videos"}] list of
    saved RefMod references (h3_refmod_libraryNG), each passed as a repeated
    --refmod PATH:STRENGTH. Needs model="h3ref" and a runner that takes
    --refmod; single window only, no keyframe, and no LightX2V adapter (turbo /
    v13) -- see h3_ref2vaNG.validate_refmods_ng, which refuses the rest before
    launch.

    tristep=True is "Fast": the TaoMate 3-step adapter on its own sigma ladder
    (FORCES steps to H3_TRISTEP_STEPS, adds --sigma-subset). Unlike turbo it
    works on the quantized DiTs -- the runner's runtime LoRA wrapping attaches
    all 208 modules there -- and it is the one speed adapter that is allowed
    with a RefMod. Needs a runner with --sigma-subset. If the runner reports
    that the adapter attached 0 modules the render is stopped rather than
    finished undistilled. One speed adapter per run.

    last_frame_path is an optional still the clip ends on (--last-frame), alone
    or with image_path or extend_from; not with RefMods.

    extend_from is an existing clip to continue (--extend-from): duration_s is
    then the NEW time added after it, the output is the source followed by the
    continuation, and the runner takes the hand-over from the source's own tail,
    so no keyframe image and no RefMod. Each new window adds window_frames - 1
    frames; chain_prompts, if given, names one prompt per new window."""
    loras = _validate_h3_loras_ng(loras)
    width = H3_WIDTH if width is None else int(width)
    height = H3_HEIGHT if height is None else int(height)
    if turbo and v13:
        raise ValueError("turbo and v13 are separate speed adapters -- pick one")
    if tristep and (turbo or v13):
        raise ValueError("Fast (3-step) is its own speed adapter -- not with turbo or v13")
    steps = (H3_TURBO_STEPS if turbo else H3_V13_STEPS if v13 else H3_TRISTEP_STEPS if tristep
             else (H3_STEPS if steps is None else int(steps)))
    _validate_h3_dims_ng(width, height)
    _validate_h3_duration_ng(duration_s)
    if v13:
        if config.model != "h3":
            raise ValueError(
                "v13 requires model='h3' (bf16) -- same quantized-loader "
                "limitation as turbo")
        if (width, height) != (H3_WIDTH, H3_HEIGHT):
            raise ValueError(
                f"v13 is only validated at the 'high' {H3_WIDTH}x{H3_HEIGHT} "
                f"canvas (got {width}x{height})")
        if not H3_V13_LORA_PATH.is_file():
            raise FileNotFoundError(
                f"lightx2v v1.3 adapter not found at {H3_V13_LORA_PATH} -- "
                f"build it from Phosphene's Speed row (Get it, 1.4 GB)")

    turbo_lora = turbo_adaln = None
    if turbo:
        if config.model != "h3":
            # Measured 2026-09-29: against h3q8 (quantized) the loader's
            # shape check compares the LoRA's A matrix to the QUANTIZED
            # DiT's packed tensor shape, not its logical dense shape --
            # every non-adaLN module (208/259) silently mismatches and
            # gets skipped (0 applied), so the adapter contributes
            # nothing beyond the small adaLN absorption and --steps 4
            # runs essentially undistilled. Against h3 (bf16) the same
            # weights apply correctly (208 applied, only the documented
            # 51 adaLN modules skipped). Not a training/architecture
            # mismatch -- specific to the quantized loader path -- so
            # refuse rather than silently degrade instead of erroring.
            raise ValueError(
                "turbo requires model='h3' (bf16) -- against 'h3q8' "
                "(quantized) the turbo LoRA's shape check silently fails "
                "on every non-adaLN module and the adapter does nothing")
        turbo_lora, turbo_adaln = _resolve_h3_turbo_ng()
        if not turbo_lora:
            raise FileNotFoundError(
                f"H3 turbo LoRA not found -- expected {H3_TURBO_V1_LORA_PATH}, "
                f"or both {H3_TURBO_LORA_PATH} and {H3_TURBO_ADALN_PATH}")

    tristep_lora = None
    if tristep:
        tristep_lora = _resolve_h3_tristep_ng()
        if not tristep_lora:
            raise FileNotFoundError(
                f"Fast (TaoMate 3-step) adapter not found -- expected "
                f"{H3_TRISTEP_CANDIDATES[-1][0]} (or the full-rank copy beside it); "
                f"Phosphene's Speed row downloads it")

    python = _resolve_h3_python_ng(config)
    if not python:
        raise FileNotFoundError(
            f"minimax-h3-mlx venv python not found at "
            f"{config.python_path or H3_DEFAULT_PYTHON}")
    runner = _resolve_h3_runner_ng(config)
    if not runner:
        raise FileNotFoundError(
            f"generate_staged.py not found at {config.runner_path or H3_RUNNER}")
    if tristep and not h3_lora.runner_has_flag_ng(runner, "--sigma-subset"):
        raise ValueError("this H3 runner has no --sigma-subset, which Fast (3-step) needs -- "
                         "update the minimax-h3-mlx runner")
    dit = _resolve_h3_dit_ng(config)
    if not dit:
        expected = (H3_DIT_Q8_DIR if config.model == "h3q8"
                    else h3_ref2va.ref2va_dit_candidates_ng(H3_MODELS_ROOT)[0] if config.model == "h3ref"
                    else H3_DIT_BF16)
        raise FileNotFoundError(f"H3 {config.model} DiT weights not found at {expected}")
    compact_root = _resolve_h3_compact_root_ng(config)
    if not compact_root:
        raise FileNotFoundError(
            f"H3 compact VAE/text-encoder pack not found at "
            f"{config.compact_root or H3_COMPACT_ROOT}")
    text_config = _resolve_h3_text_config_ng(config)
    if not text_config:
        raise FileNotFoundError(
            f"H3 upstream text-encoder config not found at "
            f"{config.text_config or H3_TEXT_CONFIG}")
    if image_path is not None and not Path(image_path).is_file():
        raise FileNotFoundError(f"H3 keyframe image not found at {image_path}")
    if last_frame_path is not None:
        if not Path(last_frame_path).is_file():
            raise FileNotFoundError(f"H3 last-frame image not found at {last_frame_path}")
        if refmods:
            raise ValueError("a RefMod render takes no first or last frame")
        if not h3_lora.runner_has_flag_ng(runner, "--last-frame"):
            raise ValueError("this H3 runner has no --last-frame -- update the minimax-h3-mlx runner")
    if extend_from is not None:
        if not Path(extend_from).is_file():
            raise FileNotFoundError(f"H3 extend source not found at {extend_from}")
        if image_path is not None or refmods:
            raise ValueError("extend continues the source clip's own last frame -- "
                             "no keyframe image or RefMod with it")
        if not h3_lora.runner_has_flag_ng(runner, "--extend-from"):
            raise ValueError("this H3 runner has no --extend-from -- update the minimax-h3-mlx runner")
    if loras:
        limit = h3_lora_limit_ng(runner, speed_adapter=bool(turbo or v13 or tristep))
        if len(loras) > limit:
            raise ValueError(
                f"this H3 runner takes at most {limit} LoRA(s) alongside "
                f"{'a speed adapter' if turbo or v13 or tristep else 'a render'} "
                f"(got {len(loras)}) -- update the H3 runner to stack more")
        for lora_path, _s in loras:
            if not Path(lora_path).is_file():
                raise FileNotFoundError(f"H3 LoRA not found at {lora_path}")

    window_frames, chain_windows, frames = _h3_chain_plan_ng(duration_s)
    if extend_from is not None:
        window_frames = _align_h3_frames_ng(H3_CHAIN_WINDOW_S)
        frames = round(duration_s * H3_FRAME_RATE)
        chain_windows = math.ceil(frames / (window_frames - 1))
    refmods = h3_ref2va.validate_refmods_ng(
        refmods, model=config.model, image_path=image_path, turbo=turbo, v13=v13,
        chain_windows=chain_windows, runner=runner, tristep=tristep)
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_mp4 = output_dir / f"h3_{ts}.mp4"
    metrics_path = output_dir / f"h3_{ts}.metrics.json"

    if chain_prompts:
        # --chain-prompts and the positional prompt are mutually exclusive in
        # the runner, and it wants exactly one prompt per window.
        if chain_windows == 1 and extend_from is None:
            raise ValueError("chain_prompts needs a clip longer than one window")
        if len(chain_prompts) != chain_windows:
            raise ValueError(f"this clip is {chain_windows} windows but {len(chain_prompts)} "
                             f"chain prompts were given")
        chain_prompts_path = output_dir / f"h3_{ts}.chain_prompts.json"
        chain_prompts_path.write_text(json.dumps(chain_prompts))

    cmd = [
        python, runner,
        *([] if chain_prompts else [prompt]),
        "--dit", dit,
        "--compact-root", compact_root,
        "--text-config", text_config,
        "--frames", str(window_frames),
        "--height", str(height),
        "--width", str(width),
        "--steps", str(steps),
        "--seed", str(seed),
        "-o", str(out_mp4),
        "--metrics", str(metrics_path),
    ]
    if extend_from is not None:
        cmd += ["--extend-from", extend_from, "--extend-frames", str(frames),
                "--chain-windows", str(chain_windows), "--chain-keep-windows"]
        if chain_prompts:
            cmd += ["--chain-prompts", str(chain_prompts_path)]
    elif chain_windows > 1:
        # --chain-keep-windows: also write each window's own mp4 beside
        # the stitched output. Costs almost nothing (small per-window
        # files) and means a crash/kill partway through a long chain --
        # see the 2026-09-29 broken-pipe incident that lost a full
        # ~33-minute render with nothing to show for it -- still leaves
        # every window that finished before the failure on disk.
        cmd += ["--chain-windows", str(chain_windows), "--chain-total-frames", str(frames),
                "--chain-keep-windows"]
        if chain_prompts:
            cmd += ["--chain-prompts", str(chain_prompts_path)]
    if image_path is not None:
        cmd += ["--first-frame", image_path]
    if last_frame_path is not None:
        cmd += ["--last-frame", last_frame_path]
    if turbo:
        # The pinned runner only re-orders a speed adapter's qkv rows under
        # --lora-qkv-permute (off by default, global to the run); without
        # this bake the adapter writes onto the wrong rows. A runner that
        # predates the flag re-orders every --lora itself, so it gets the
        # source file as-is -- re-ordering twice would undo it.
        if h3_lora.runner_has_flag_ng(runner, "--lora-qkv-permute"):
            turbo_lora = str(h3_lora.bake_runner_layout_ng(turbo_lora, on_log))
        cmd += _h3_turbo_argv_ng(turbo_lora, turbo_adaln)
    if v13:
        cmd += ["--lora", f"{H3_V13_LORA_PATH}:1.0"]
        if h3_lora.runner_has_flag_ng(runner, "--video-shift"):
            cmd += ["--video-shift", H3_V13_VIDEO_SHIFT]
    if tristep:
        cmd += _h3_tristep_argv_ng(tristep_lora)
    for lora_path, strength in loras:
        # A speed adapter's own file is already in the runner's order (turbo
        # is baked above, v1.3 ships that way); a user LoRA is not, so it goes
        # through the layout check -- before this, its q/k/v update landed on
        # the wrong heads.
        runner_file, _qkv = h3_lora.lora_for_runner_ng(lora_path, runner, on_log)
        cmd += ["--lora", f"{runner_file}:{strength:g}"]
    cmd += h3_ref2va.refmod_argv_ng(refmods)
    lazy_argv = _h3_lazy_adaln_argv_ng(runner, legacy_adaln=turbo_adaln is not None)
    cmd += lazy_argv
    adaln_cache_argv = _h3_adaln_cache_argv_ng(runner, legacy_adaln=turbo_adaln is not None)
    cmd += adaln_cache_argv

    if on_log:
        turbo_note = (" [turbo 4-step]" if turbo
                      else " [lightx2v v1.3]" if v13
                      else " [fast 3-step]" if tristep else "")
        if lazy_argv:
            turbo_note += " [lazy adaln]"
        if adaln_cache_argv:
            turbo_note += " [adaln cache]"
        if loras:
            turbo_note += " [" + ", ".join(f"{Path(p).stem}@{s:g}" for p, s in loras) + "]"
        if refmods:
            turbo_note += " [refmods " + ", ".join(
                f"{Path(r['path']).stem}@{r['strength']:g}" for r in refmods) + "]"
        if extend_from is not None:
            on_log(f"[h3] launching {config.model} extend: {chain_windows} new window(s) "
                   f"(+{frames} frames, ~{frames / H3_FRAME_RATE:.1f}s) after "
                   f"{Path(extend_from).name} at {width}x{height}, "
                   f"{steps} steps{turbo_note}, seed={seed}")
        elif chain_windows > 1:
            on_log(f"[h3] launching {config.model} {chain_windows} chained windows "
                   f"({window_frames}f each) -> {frames} frames "
                   f"(~{frames / H3_FRAME_RATE:.1f}s) at {width}x{height}, "
                   f"{steps} steps{turbo_note}, seed={seed}")
        else:
            on_log(f"[h3] launching {config.model} {frames} frames "
                   f"(~{frames / H3_FRAME_RATE:.1f}s) at {width}x{height}, "
                   f"{steps} steps{turbo_note}, seed={seed}")

    with _H3_SUBPROCESS_LOCK:
        from model_offloadNG import offload_resident_models_ng
        offload_resident_models_ng(on_log=on_log)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(H3_REPO_DIR),
            start_new_session=True,
        )
        if on_proc_start:
            on_proc_start(proc)

        timeout_s = float(os.environ.get("RINGVIZ_H3_TIMEOUT_S", config.timeout_s))
        stall_s = float(os.environ.get("RINGVIZ_H3_STALL_S", config.stall_s))
        timed_out = {"v": False}
        stalled = {"v": False}
        adapter_seen = {"v": False}
        adapter_failed = {"v": False}
        last_activity = {"t": time.time()}

        def _kill_proc():
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        def _kill_on_timeout():
            timed_out["v"] = True
            _kill_proc()

        # Two watchdogs, not one: `watchdog` is the absolute ceiling for
        # the whole call (catches a process that's genuinely still
        # working but just slow/large); `stall_watch` kills it as soon
        # as it goes quiet for stall_s, well before that ceiling, which
        # is what actually catches a hang -- see the H3 gen that stalled
        # mid-render (real log progress, then dead silence) but only
        # got reaped an hour later by the single timeout that used to
        # be here.
        watchdog = threading.Timer(timeout_s, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        stop_stall_watch = threading.Event()

        def _stall_watch():
            while not stop_stall_watch.wait(30):
                if time.time() - last_activity["t"] > stall_s:
                    stalled["v"] = True
                    _kill_proc()
                    return

        stall_thread = threading.Thread(target=_stall_watch, daemon=True)
        stall_thread.start()

        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    last_activity["t"] = time.time()
                    line = line.rstrip()
                    if not line:
                        continue
                    if on_log:
                        on_log(f"[h3] {line}")
                    if tristep and not adapter_seen["v"]:
                        # The first adapter summary is the speed adapter's. 0 modules means the
                        # render would run undistilled at 4 steps, so stop it now, not in 6 minutes.
                        m = _H3_LORA_APPLIED_RE.search(line)
                        if m:
                            adapter_seen["v"] = True
                            if int(m.group(1)) == 0:
                                adapter_failed["v"] = True
                                _kill_proc()
            rc = proc.wait()
        finally:
            watchdog.cancel()
            stop_stall_watch.set()

    if adapter_failed["v"]:
        raise RuntimeError(
            "the Fast (3-step) adapter attached to 0 modules on this DiT, so the render was "
            "stopped instead of running undistilled -- check the runner's LoRA summary line")
    if stalled["v"]:
        raise H3JobCancelled(
            f"H3 gen produced no output for {stall_s:.0f}s and was "
            f"killed (likely a hung render). Raise RINGVIZ_H3_STALL_S "
            f"if this machine legitimately goes quiet between log lines "
            f"for longer than that.")
    if timed_out["v"]:
        raise H3JobCancelled(
            f"H3 gen exceeded its {timeout_s:.0f}s deadline and was "
            f"killed (likely a hung render). Raise RINGVIZ_H3_TIMEOUT_S "
            f"if this machine legitimately needs longer.")
    if rc != 0:
        if rc in (-15, 143, -9, 137):
            raise H3JobCancelled(f"H3 gen cancelled (rc={rc})")
        raise RuntimeError(f"H3 gen failed with rc={rc}")

    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError(f"H3 gen finished but no/empty mp4 at {out_mp4}")

    return {
        "mp4_path": str(out_mp4),
        "seed": seed,
        "engine": f"minimax-{config.model}",
        "width": width,
        "height": height,
        "frame_rate": H3_FRAME_RATE,
        "frames": frames,
        "duration_s": frames / H3_FRAME_RATE,
        "refmods": [{"name": Path(r["path"]).stem, "strength": r["strength"]} for r in refmods],
    }
