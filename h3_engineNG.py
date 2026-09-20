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

Renders at H3's "high" tier -- 1024x576, true 16:9, 16 steps -- the
same canvas ltx_engineNG.py defaults to (LTX_WIDTH/LTX_HEIGHT), and
phosphene's own recommended delivery canvas (see mlx_ltx_panel.py's
_h3_qualities()["high"]). Measured there at 8 forwards/1024x576: 18.8
min wall, no memory-wall difference vs the smaller "standard" canvas
(the DiT weights dominate, not activations) -- so H3_TIMEOUT_S below
budgets roughly double that for the 16-step default.

Deliberately out of scope here (keep this file small -- add a sibling
module later if any of these turn out to be needed, don't grow this
one): chained multi-window clips (--chain-windows), live preview,
draft/TAE decode, step-cache, turbo-LoRA auto-apply, prompt-cache
reuse. Optional LoRA pass-through (generate_staged.py's own repeatable
--lora PATH[:SCALE]) is in scope in the same thin way ltx_engineNG.py
does it, once something resolves a UI name to a path -- not wired yet.
"""

from __future__ import annotations

import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

H3Model = Literal["h3", "h3q8"]


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
    for cand in (home / "minimax-h3-mlx",
                 home / "AI" / "minimax-h3-mlx",
                 Path("/Volumes/AI/minimax-h3-mlx"),
                 Path("/Volumes/AI/Pinkio/api/phosphene.git/minimax-h3-mlx")):
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

H3_MIN_DURATION_S = 3.0
H3_MAX_DURATION_S = 15.0


def _validate_h3_dims_ng(width: int, height: int) -> None:
    for name, v in (("width", width), ("height", height)):
        if v % H3_DIM_STEP != 0:
            raise ValueError(f"{name} must be a multiple of {H3_DIM_STEP} (got {v})")
        if not (H3_MIN_DIM <= v <= H3_MAX_DIM):
            raise ValueError(f"{name} must be between {H3_MIN_DIM} and {H3_MAX_DIM} (got {v})")


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
    timeout_s: float = 3600.0   # per-call watchdog; override via RINGVIZ_H3_TIMEOUT_S


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
    --dit accepts either."""
    if config.model == "h3q8":
        p = Path(config.dit_q8_dir) if config.dit_q8_dir else H3_DIT_Q8_DIR
        ok = (p / "config.json").is_file() and (p / "quant_config.json").is_file() \
            and any(p.glob("model-*.safetensors"))
        return str(p) if ok else None
    p = Path(config.dit_bf16_path) if config.dit_bf16_path else H3_DIT_BF16
    return str(p) if p.is_file() else None


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
    return {
        "repo_dir": str(H3_REPO_DIR),
        "models_root": str(H3_MODELS_ROOT),
        "model": model,
        "python_ok": python is not None,
        "python_path": python or str(H3_DEFAULT_PYTHON),
        "runner_ok": runner is not None,
        "runner_path": runner or str(H3_RUNNER),
        "dit_ok": dit is not None,
        "dit_path": dit or (str(H3_DIT_Q8_DIR) if model == "h3q8" else str(H3_DIT_BF16)),
        "compact_ok": compact_root is not None,
        "compact_path": compact_root or str(H3_COMPACT_ROOT),
        "text_config_ok": text_config is not None,
        "text_config_path": text_config or str(H3_TEXT_CONFIG),
        "ready": all((python, runner, dit, compact_root, text_config)),
    }


def generate_h3_video_ng(prompt: str, image_path: Optional[str], duration_s: float,
                          output_dir: Path, seed: Optional[int],
                          config: H3Config,
                          width: Optional[int] = None, height: Optional[int] = None,
                          steps: Optional[int] = None,
                          on_log: Optional[Callable[[str], None]] = None,
                          on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: prompt + duration in, one mp4 out. image_path
    is optional -- when given, it's H3's FL2VA first-frame keyframe
    conditioning (--first-frame); when None, it's a plain text-to-video
    render. config.model picks bf16 ("h3") or quantized ("h3q8") DiT
    weights -- same compact VAE/text-encoder pack either way. width/
    height/steps default to H3_WIDTH/H3_HEIGHT/H3_STEPS (the "high" true
    16:9 canvas) when omitted."""
    width = H3_WIDTH if width is None else int(width)
    height = H3_HEIGHT if height is None else int(height)
    steps = H3_STEPS if steps is None else int(steps)
    _validate_h3_dims_ng(width, height)
    _validate_h3_duration_ng(duration_s)

    python = _resolve_h3_python_ng(config)
    if not python:
        raise FileNotFoundError(
            f"minimax-h3-mlx venv python not found at "
            f"{config.python_path or H3_DEFAULT_PYTHON}")
    runner = _resolve_h3_runner_ng(config)
    if not runner:
        raise FileNotFoundError(
            f"generate_staged.py not found at {config.runner_path or H3_RUNNER}")
    dit = _resolve_h3_dit_ng(config)
    if not dit:
        expected = H3_DIT_Q8_DIR if config.model == "h3q8" else H3_DIT_BF16
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

    frames = _align_h3_frames_ng(duration_s)
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_mp4 = output_dir / f"h3_{ts}.mp4"
    metrics_path = output_dir / f"h3_{ts}.metrics.json"

    cmd = [
        python, runner,
        prompt,
        "--dit", dit,
        "--compact-root", compact_root,
        "--text-config", text_config,
        "--frames", str(frames),
        "--height", str(height),
        "--width", str(width),
        "--steps", str(steps),
        "--seed", str(seed),
        "-o", str(out_mp4),
        "--metrics", str(metrics_path),
    ]
    if image_path is not None:
        cmd += ["--first-frame", image_path]

    if on_log:
        on_log(f"[h3] launching {config.model} {frames} frames "
               f"(~{frames / H3_FRAME_RATE:.1f}s) at {width}x{height}, "
               f"{steps} steps, seed={seed}")

    with _H3_SUBPROCESS_LOCK:
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
                        on_log(f"[h3] {line}")
            rc = proc.wait()
        finally:
            watchdog.cancel()

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
    }
