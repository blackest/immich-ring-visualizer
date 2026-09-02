"""HiDream-O1 subprocess engine for Ring Visualizer's character-sheet feature.

Ported from Phosphene's agent/image_engine.py (_generate_hidream and its
resolution helpers) -- see PHOSPHENE_DECOUPLING_PLAN.md. Ring Visualizer
talks to the same standalone HiDream lab install Phosphene calls
(scripts/hidream_o1/generate_hidream_o1_mlx.py, a subprocess, never
imported in-process), so this file owns nothing about the model itself --
just how to find it and how to build/run the command line.

Kept deliberately smaller than Phosphene's version: Phosphene dispatches
across four backends (mock/bfl/mflux/hidream) with a shared
ImageEngineConfig; Ring Visualizer only ever needs hidream for this
feature, so this module hardcodes that path instead of porting the whole
dispatch layer.

Step count: 28, not the 6 Phosphene currently uses. Phosphene's
hidream_steps=6 default was chosen assuming --fb-cache made 6 steps
comparable to a ~20-step render, but the actual generate_hidream_o1_mlx.py
script (as vendored today, with its own "[compat]" comments) does NOT
implement First-Block Cache -- it prints a warning and just runs the full
step count it's given. So Phosphene's character sheets are, right now,
genuinely 6-step renders, below the lab's own documented Dev-recipe floor
of 28 and below what Ring Visualizer's own routes/phosphene.py comment
assumed ("each its own full 28-step diffusion loop"). Decided with John
2026-09-02: port the honest 28-step default; revisit if it's too slow.
"""

from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


class ImageJobCancelled(RuntimeError):
    """Raised when the HiDream subprocess is killed by its watchdog
    timeout. Ring Visualizer has no cancel button yet, so the only
    source here is the timeout -- kept as a distinct exception in case a
    cancel path gets added later (matches Phosphene's own distinction
    between a real engine failure and a killed job)."""


def _resolve_hidream_lab_dir() -> Path:
    """Same search order as Phosphene's image_engine.py: HIDREAM_LAB_DIR
    env var, then ~/HIDREAM-O1-MLX-LAB-active, then
    ~/AI/HIDREAM-O1-MLX-LAB-active. Falls back to the first of those even
    if it doesn't exist so callers get a clean "not found" error pointing
    at the conventional location."""
    env_p = os.environ.get("HIDREAM_LAB_DIR")
    if env_p:
        return Path(env_p).expanduser()
    home = Path.home()
    for cand in (home / "HIDREAM-O1-MLX-LAB-active",
                 home / "AI" / "HIDREAM-O1-MLX-LAB-active",
                 home / "AI" / "projects" / "HIDREAM-O1-MLX-LAB-active",
                 # Confirmed 2026-09-02: on this machine the lab lives on
                 # an external volume, not under $HOME -- Phosphene's own
                 # repo (README/STATE.md) documents HIDREAM_LAB_DIR as the
                 # way to point at a non-default location but doesn't set
                 # it anywhere in its own checked-in launch config, so
                 # whatever makes Phosphene find it today is set outside
                 # the repo (shell profile / Pinokio env). Adding this as
                 # a fallback candidate means Ring Visualizer finds the
                 # lab out of the box on this machine without requiring
                 # that same external configuration to be duplicated.
                 Path("/Volumes/AI/HIDREAM-O1-MLX-LAB-active")):
        if cand.exists():
            return cand
    return home / "HIDREAM-O1-MLX-LAB-active"


HIDREAM_LAB_DIR = _resolve_hidream_lab_dir()
HIDREAM_DEFAULT_PY = HIDREAM_LAB_DIR / ".venv" / "bin" / "python"
# BF16 is what's actually on disk today (mlx_models/hidream-o1-dev-bf16).
# Phosphene moved off Q8 the same day it shipped this feature -- its
# checked-in CLAUDE.md hard rule still says "Q8 only" but STATE.md and
# the live ImageEngineConfig default both moved to BF16 same-session.
# Point at what's really there rather than the stale doc.
HIDREAM_DEFAULT_MODEL = HIDREAM_LAB_DIR / "mlx_models" / "hidream-o1-dev-bf16"
HIDREAM_GENERATE_SCRIPT = HIDREAM_LAB_DIR / "scripts" / "hidream_o1" / "generate_hidream_o1_mlx.py"
HIDREAM_PATCH_SIZE = 32

# From generate_hidream_o1_mlx.py / upstream pipeline.py utils.py -- the
# resolutions HiDream-O1 was actually trained on. Off-spec dims produce a
# visible 32px patch grid because the model never saw those mrope codes.
HIDREAM_TRAINED_RESOLUTIONS = [
    (2048, 2048),
    (2304, 1728), (1728, 2304),
    (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496),
    (3104, 1312), (1312, 3104),
    (2304, 1792), (1792, 2304),
]


def _snap_to_trained_resolution(width: int, height: int) -> tuple[int, int]:
    img_ratio = width / height
    best, min_diff = (2048, 2048), float("inf")
    for w, h in HIDREAM_TRAINED_RESOLUTIONS:
        diff = abs(w / h - img_ratio)
        if diff < min_diff:
            min_diff, best = diff, (w, h)
    return best


def _patch_align(value: int, patch: int = HIDREAM_PATCH_SIZE) -> int:
    return max(patch, (value // patch) * patch)


def _clean_subprocess_env() -> dict:
    """os.environ.copy() with macOS Malloc* debug vars stripped -- same
    fix as Phosphene's, otherwise every HiDream subprocess spams stderr
    with "MallocStackLogging: can't turn off..." noise."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("Malloc"):
            del env[key]
    return env


@dataclass
class HiDreamConfig:
    """How to run the HiDream lab. Deliberately smaller than Phosphene's
    ImageEngineConfig -- Ring Visualizer only ever runs the Dev T2I/edit
    recipe for character sheets, so there's no per-backend dispatch to
    configure around. fb-cache / guidance-scale are intentionally absent:
    both are accepted-but-ignored compat flags in the real script today
    (see module docstring), so wiring them through here would just be
    dead configuration."""
    python_path: str = ""          # default: HIDREAM_LAB_DIR/.venv/bin/python
    model_path: str = ""           # default: HIDREAM_LAB_DIR/mlx_models/hidream-o1-dev-bf16
    steps: int = 28                # Dev's distillation floor -- see module docstring
    noise_scale: float = 7.5       # FlashFlowMatch tuned default; lowering collapses the image
    noise_clip_std: float = 2.5
    editing_scheduler: str = "flow_match"   # only value the script implements today
    timeout_s: float = 1800.0      # per-subprocess-call watchdog; override via RINGVIZ_HIDREAM_TIMEOUT_S


def _resolve_hidream_python(config: HiDreamConfig) -> Optional[str]:
    p = Path(config.python_path) if config.python_path else HIDREAM_DEFAULT_PY
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _resolve_hidream_model(config: HiDreamConfig) -> Optional[str]:
    p = Path(config.model_path) if config.model_path else HIDREAM_DEFAULT_MODEL
    return str(p) if (p / "model.safetensors").exists() and (p / "extras" / "custom_heads.safetensors").exists() else None


def hidream_health() -> dict:
    """Cheap up-front check the routes layer can surface to the UI/status
    endpoint without launching a subprocess -- mirrors Phosphene's own
    "honest about both the venv and the model dir" health check."""
    cfg = HiDreamConfig()
    py = _resolve_hidream_python(cfg)
    model = _resolve_hidream_model(cfg)
    script_ok = HIDREAM_GENERATE_SCRIPT.is_file()
    return {
        "lab_dir": str(HIDREAM_LAB_DIR),
        "python_ok": py is not None,
        "python_path": py or str(cfg.python_path or HIDREAM_DEFAULT_PY),
        "model_ok": model is not None,
        "model_path": model or str(cfg.model_path or HIDREAM_DEFAULT_MODEL),
        "script_ok": script_ok,
        "script_path": str(HIDREAM_GENERATE_SCRIPT),
        "ready": py is not None and model is not None and script_ok,
    }


def generate_hidream(prompt: str, n: int, width: int, height: int,
                      output_dir: Path, base_seed: Optional[int],
                      config: HiDreamConfig,
                      refs: Optional[list] = None,
                      on_log: Optional[Callable[[str], None]] = None) -> list:
    """Subprocess pattern ported from Phosphene's _generate_hidream: one
    process, n candidates in one call (the generator script accepts
    multiple --output/--seed values so the model loads once per batch).

    refs non-empty runs HiDream's native edit/multi-ref path -- K=1 is an
    instruction edit, K=2-3 composes multiple references. Character-sheet
    generation always calls this with n=1 (one full-res image per view;
    a batched n>1 call would give every candidate the same prompt).
    """
    py = _resolve_hidream_python(config)
    if not py:
        raise FileNotFoundError(
            f"HiDream venv python not found at "
            f"{config.python_path or HIDREAM_DEFAULT_PY}")
    model = _resolve_hidream_model(config)
    if not model:
        raise FileNotFoundError(
            f"HiDream model not found at "
            f"{config.model_path or HIDREAM_DEFAULT_MODEL}")
    script = str(HIDREAM_GENERATE_SCRIPT)
    if not Path(script).is_file():
        raise FileNotFoundError(f"HiDream generator script missing at {script}")

    snap_w, snap_h = _snap_to_trained_resolution(width, height)
    if (snap_w, snap_h) != (width, height) and on_log:
        on_log(f"[hidream] snapping {width}x{height} -> {snap_w}x{snap_h} (trained dim)")
    aligned_w = _patch_align(snap_w)
    aligned_h = _patch_align(snap_h)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = []
    pngs = []
    ts = int(time.time() * 1000)
    for i in range(n):
        seed = (base_seed + i) if base_seed is not None else random.randint(0, 2**31 - 1)
        seeds.append(seed)
        pngs.append(output_dir / f"cand_{i:02d}_hidream_{ts}.png")

    cmd = [
        py, script,
        "--model-path", model,
        "--model-type", "dev",
        "--prompt", prompt,
        "--width", str(aligned_w),
        "--height", str(aligned_h),
        "--output", *map(str, pngs),
        "--seed", *map(str, seeds),
        "--num-inference-steps", str(config.steps),
        "--noise-scale-start", str(config.noise_scale),
        "--noise-scale-end", str(config.noise_scale),
        "--noise-clip-std", str(config.noise_clip_std),
    ]
    if refs:
        cmd.extend(["--ref-images", *map(str, refs)])
        cmd.extend(["--editing-scheduler", config.editing_scheduler])

    if on_log:
        on_log(f"[hidream] launching {n} candidate(s) in one process, seeds={seeds}"
               + (f" with {len(refs)} ref(s)" if refs else ""))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_clean_subprocess_env(),
        start_new_session=True,
    )

    timeout_s = float(os.environ.get("RINGVIZ_HIDREAM_TIMEOUT_S", config.timeout_s))
    timed_out = {"v": False}

    def _kill_on_timeout():
        timed_out["v"] = True
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    watchdog = threading.Timer(timeout_s, _kill_on_timeout)
    watchdog.daemon = True
    watchdog.start()
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if on_log:
                    on_log(f"[hidream] {line}")
        rc = proc.wait()
    finally:
        watchdog.cancel()

    if timed_out["v"]:
        raise ImageJobCancelled(
            f"HiDream gen exceeded its {timeout_s:.0f}s deadline and was "
            f"killed (likely a hung render). Raise RINGVIZ_HIDREAM_TIMEOUT_S "
            f"if this machine legitimately needs longer.")
    if rc != 0:
        if rc in (-15, 143, -9, 137):
            raise ImageJobCancelled(f"HiDream gen cancelled (rc={rc})")
        raise RuntimeError(f"HiDream gen failed with rc={rc}")

    results = []
    for seed, png in zip(seeds, pngs):
        if not png.exists():
            raise RuntimeError(f"HiDream gen finished but no PNG at {png}")
        results.append({
            "png_path": str(png),
            "seed": seed,
            "engine": "hidream-o1-dev-bf16",
            "width": aligned_w,
            "height": aligned_h,
        })
    return results
