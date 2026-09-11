"""NG twin of hidream_engine.py -- HiDream-O1 subprocess engine for the
Generate view's character-sheet feature.

NG-only file: identical logic to hidream_engine.py as it exists today
(same subprocess invocation, same resolution snapping, same watchdog/
truncated-PNG handling), duplicated per the NG rule in
APP_ARCHITECTURE_NOTES.md so nothing under *NG.py ever imports the
original module. Named hidream_engineNG.py rather than a generic
engineNG.py deliberately -- if a second generation backend (e.g.
ComfyUI) gets ported later it gets its own file alongside this one
rather than both sharing one dispatch-y module.

Talks to the same standalone HiDream lab install the original engine
does (scripts/hidream_o1/generate_hidream_o1_mlx.py, a subprocess,
never imported in-process) -- this file owns nothing about the model
itself, just how to find it and how to build/run the command line.
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

from PIL import Image


class ImageJobCancelled(RuntimeError):
    """Raised when the HiDream subprocess is killed by its watchdog
    timeout. No cancel button exists yet, so the only source here is
    the timeout -- kept as a distinct exception in case a cancel path
    gets added later."""


def _resolve_hidream_lab_dir_ng() -> Path:
    """Same search order as the original engine: HIDREAM_LAB_DIR env
    var, then a handful of conventional install locations, falling back
    to the first of those even if it doesn't exist so callers get a
    clean "not found" error pointing at the conventional location."""
    env_p = os.environ.get("HIDREAM_LAB_DIR")
    if env_p:
        return Path(env_p).expanduser()
    home = Path.home()
    for cand in (home / "HIDREAM-O1-MLX-LAB-active",
                 home / "AI" / "HIDREAM-O1-MLX-LAB-active",
                 home / "AI" / "projects" / "HIDREAM-O1-MLX-LAB-active",
                 Path("/Volumes/AI/HIDREAM-O1-MLX-LAB-active")):
        if cand.exists():
            return cand
    return home / "HIDREAM-O1-MLX-LAB-active"


HIDREAM_LAB_DIR = _resolve_hidream_lab_dir_ng()
HIDREAM_DEFAULT_PY = HIDREAM_LAB_DIR / ".venv" / "bin" / "python"
# BF16 is what's actually on disk today (mlx_models/hidream-o1-dev-bf16).
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


def _snap_to_trained_resolution_ng(width: int, height: int) -> tuple[int, int]:
    img_ratio = width / height
    best, min_diff = (2048, 2048), float("inf")
    for w, h in HIDREAM_TRAINED_RESOLUTIONS:
        diff = abs(w / h - img_ratio)
        if diff < min_diff:
            min_diff, best = diff, (w, h)
    return best


def _patch_align_ng(value: int, patch: int = HIDREAM_PATCH_SIZE) -> int:
    return max(patch, (value // patch) * patch)


def _clean_subprocess_env_ng() -> dict:
    """os.environ.copy() with macOS Malloc* debug vars stripped, so a
    HiDream subprocess doesn't spam stderr with "MallocStackLogging:
    can't turn off..." noise."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("Malloc"):
            del env[key]
    return env


@dataclass
class HiDreamConfig:
    """How to run the HiDream lab for a character-sheet shot. Only ever
    runs the Dev T2I/edit recipe -- no per-backend dispatch to configure
    around."""
    python_path: str = ""          # default: HIDREAM_LAB_DIR/.venv/bin/python
    model_path: str = ""           # default: HIDREAM_LAB_DIR/mlx_models/hidream-o1-dev-bf16
    steps: int = 28                # Dev's distillation floor
    noise_scale: float = 7.5       # FlashFlowMatch tuned default; lowering collapses the image
    noise_clip_std: float = 2.5
    editing_scheduler: str = "flow_match"   # only value the script implements today
    timeout_s: float = 1800.0      # per-subprocess-call watchdog; override via RINGVIZ_HIDREAM_TIMEOUT_S


def _resolve_hidream_python_ng(config: HiDreamConfig) -> Optional[str]:
    p = Path(config.python_path) if config.python_path else HIDREAM_DEFAULT_PY
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _resolve_hidream_model_ng(config: HiDreamConfig) -> Optional[str]:
    p = Path(config.model_path) if config.model_path else HIDREAM_DEFAULT_MODEL
    return str(p) if (p / "model.safetensors").exists() and (p / "extras" / "custom_heads.safetensors").exists() else None


def hidream_health_ng() -> dict:
    """Cheap up-front check the routes layer can surface to the UI/status
    endpoint without launching a subprocess."""
    cfg = HiDreamConfig()
    py = _resolve_hidream_python_ng(cfg)
    model = _resolve_hidream_model_ng(cfg)
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


def generate_hidream_ng(prompt: str, n: int, width: int, height: int,
                         output_dir: Path, base_seed: Optional[int],
                         config: HiDreamConfig,
                         refs: Optional[list] = None,
                         allow_offspec_res: bool = False,
                         on_log: Optional[Callable[[str], None]] = None) -> list:
    """One subprocess call, n candidates in one call (the generator
    script accepts multiple --output/--seed values so the model loads
    once per batch).

    refs non-empty runs HiDream's native edit/multi-ref path -- K=1 is an
    instruction edit, K=2-3 composes multiple references. Character-sheet
    generation always calls this with n=1 (one full-res image per shot;
    a batched n>1 call would give every candidate the same prompt).

    allow_offspec_res=True renders the requested width/height as given
    (only 32px patch-aligned), skipping the snap to a trained resolution.
    Off-spec dims render much faster but can show a faint 32px patch grid
    -- fine for draft/framing passes, not final output.
    """
    py = _resolve_hidream_python_ng(config)
    if not py:
        raise FileNotFoundError(
            f"HiDream venv python not found at "
            f"{config.python_path or HIDREAM_DEFAULT_PY}")
    model = _resolve_hidream_model_ng(config)
    if not model:
        raise FileNotFoundError(
            f"HiDream model not found at "
            f"{config.model_path or HIDREAM_DEFAULT_MODEL}")
    script = str(HIDREAM_GENERATE_SCRIPT)
    if not Path(script).is_file():
        raise FileNotFoundError(f"HiDream generator script missing at {script}")

    if allow_offspec_res:
        aligned_w = _patch_align_ng(width)
        aligned_h = _patch_align_ng(height)
        if (aligned_w, aligned_h) != (width, height) and on_log:
            on_log(f"[hidream] patch-aligning {width}x{height} -> "
                   f"{aligned_w}x{aligned_h} (off-spec: trained-resolution snap skipped)")
    else:
        snap_w, snap_h = _snap_to_trained_resolution_ng(width, height)
        if (snap_w, snap_h) != (width, height) and on_log:
            on_log(f"[hidream] snapping {width}x{height} -> {snap_w}x{snap_h} (trained dim)")
        aligned_w = _patch_align_ng(snap_w)
        aligned_h = _patch_align_ng(snap_h)

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
    if allow_offspec_res:
        # The script has its own independent trained-resolution snap that
        # runs by default regardless of what we computed above -- without
        # this flag it silently re-snaps aligned_w/aligned_h back up,
        # which both defeats allow_offspec_res entirely and makes the
        # actual-size check below raise a false "partial write" (the
        # image is fine, it's just not the size we told the caller to
        # expect).
        cmd.append("--no-snap-resolution")
    if refs:
        cmd.extend(["--ref-images", *map(str, refs)])
        cmd.extend(["--editing-scheduler", config.editing_scheduler])

    if on_log:
        on_log(f"[hidream] launching {n} candidate(s) at {aligned_w}x{aligned_h}, "
               f"{config.steps} steps, seeds={seeds}"
               + (f" with {len(refs)} ref(s)" if refs else ""))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_clean_subprocess_env_ng(),
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
        # A process interrupted mid-write (an OOM kill, the Metal
        # watchdog, a killed/orphaned subprocess) can leave a truncated
        # file behind: valid header, a handful of real scanlines, then
        # nothing -- which decodes as mostly-black rather than failing
        # outright. Force a full decode (not just Image.open(), which is
        # lazy) so a truncated file raises here instead of silently
        # getting composited into the sheet.
        try:
            with Image.open(png) as im:
                im.load()
                actual_size = im.size
        except Exception as exc:
            raise RuntimeError(
                f"HiDream gen wrote an incomplete/corrupt PNG at {png} "
                f"({exc}) -- likely the process was interrupted mid-write "
                f"(OOM kill, a crash, or a killed subprocess). Reroll "
                f"this shot.") from exc
        if actual_size != (aligned_w, aligned_h):
            raise RuntimeError(
                f"HiDream gen wrote a {actual_size[0]}x{actual_size[1]} "
                f"image but expected {aligned_w}x{aligned_h} at {png} -- "
                f"likely a partial write. Reroll this shot.")
        results.append({
            "png_path": str(png),
            "seed": seed,
            "engine": "hidream-o1-dev-bf16",
            "width": aligned_w,
            "height": aligned_h,
        })
    return results
