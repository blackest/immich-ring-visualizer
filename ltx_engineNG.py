"""NG engine for LTX-2.5 image-to-video generation.

Talks to the standalone ltx-2-mlx install as a subprocess (its own venv,
never imported in-process) -- this file owns nothing about the model
itself, just how to find it, build the CLI invocation, and validate the
output. One input (reference image + prompt + duration) -> one output
(an mp4 path + result dict), matching hidream_engineNG.py's shape.

Deliberately out of scope here (keep this file small -- add a sibling
module later if any of these turn out to be needed, don't grow this
one): quality-tier selection (always the "high" tier: two-stages-hq,
1024x576, 10+3 steps), LoRA, and phosphene's persistent warm-helper
process (fine for occasional single renders to eat the model-load
cost per call instead).
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


class VideoJobCancelled(RuntimeError):
    """Raised when the LTX subprocess is killed by its watchdog timeout.
    No cancel button exists yet, so the only source here is the timeout."""


def _resolve_ltx_repo_dir_ng() -> Path:
    """LTX_MLX_DIR env var, then conventional install locations, falling
    back to the first even if it doesn't exist so callers get a clean
    "not found" error pointing at the conventional location."""
    env_p = os.environ.get("LTX_MLX_DIR")
    if env_p:
        return Path(env_p).expanduser()
    home = Path.home()
    for cand in (home / "ltx-2-mlx",
                 home / "AI" / "ltx-2-mlx",
                 Path("/Volumes/AI/Pinkio/api/phosphene.git/ltx-2-mlx")):
        if cand.exists():
            return cand
    return home / "ltx-2-mlx"


def _resolve_ltx_weights_root_ng() -> Path:
    """LTX_MLX_MODELS_DIR env var, then the conventional shared mlx_models
    drive phosphene itself resolves onto."""
    env_p = os.environ.get("LTX_MLX_MODELS_DIR")
    if env_p:
        return Path(env_p).expanduser()
    cand = Path("/Volumes/AI/Pinkio/drive/drives/peers/d1785024377531/mlx_models")
    if cand.exists():
        return cand
    return Path.home() / "mlx_models"


LTX_REPO_DIR = _resolve_ltx_repo_dir_ng()
LTX_WEIGHTS_ROOT = _resolve_ltx_weights_root_ng()
LTX_DEFAULT_BIN = LTX_REPO_DIR / "env" / "bin" / "ltx-2-mlx"
LTX_DEFAULT_MODEL = LTX_WEIGHTS_ROOT / "ltx-2.5-mlx-q8"
# Paired with the DiT tower it was fine-tuned alongside for LTX-2.5 --
# LTX-2.3's Gemma-3 encoder is NOT interchangeable with this (loads, runs,
# and returns plausible garbage with no shape ever disagreeing).
LTX_DEFAULT_GEMMA = LTX_WEIGHTS_ROOT / "gemma4-12b-ltx25-q4"
# A separate, plain chat-tuned Gemma-3 checkpoint for prompt enhancement
# (ltx-2-mlx's `enhance` command and `generate --enhance-prompt`, both of
# which go through mlx-lm's chat/generate path). NOT interchangeable with
# LTX_DEFAULT_GEMMA above: that one is "gemma4_unified", an architecture
# mlx-lm's generic loader doesn't know how to load at all (confirmed
# live -- ValueError: Model type gemma4_unified not supported).
LTX_DEFAULT_ENHANCE_GEMMA = LTX_WEIGHTS_ROOT / "gemma-3-12b-it-4bit"
LTX_FRAME_RATE = 24.0

# "high" tier (2nd of phosphene's 4: draft/balanced/high/high_720p) --
# two-stages-hq pipeline at 1024x576, 10+3 steps. Fixed; no tier plumbing.
LTX_WIDTH = 1024
LTX_HEIGHT = 576
LTX_STAGE1_STEPS = 10
LTX_STAGE2_STEPS = 3
# --two-stages-hq's stage-2 refine step fuses a distilled LoRA that lives
# inside the model dir; the CLI's own default filename is LTX-2.3's, not
# 2.5's -- passing the wrong one raises FileNotFoundError (confirmed live).
LTX_DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450.safetensors"


def _duration_to_frames_ng(seconds: float) -> int:
    """LTX's temporal compression requires frames % 8 == 1 at 24fps."""
    blocks = max(1, round(seconds * LTX_FRAME_RATE / 8))
    return blocks * 8 + 1


def _clean_subprocess_env_ng() -> dict:
    """os.environ.copy() with macOS Malloc* debug vars stripped, so the
    LTX subprocess doesn't spam stderr with debug-allocator noise."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("Malloc"):
            del env[key]
    return env


@dataclass
class LtxConfig:
    """How to run ltx-2-mlx for an image-to-video render."""
    binary_path: str = ""   # default: LTX_DEFAULT_BIN
    model_path: str = ""    # default: LTX_DEFAULT_MODEL
    gemma_path: str = ""    # default: LTX_DEFAULT_GEMMA
    enhance_gemma_path: str = ""  # default: LTX_DEFAULT_ENHANCE_GEMMA
    timeout_s: float = 1800.0   # per-call watchdog; override via RINGVIZ_LTX_TIMEOUT_S


def _resolve_ltx_binary_ng(config: LtxConfig) -> Optional[str]:
    p = Path(config.binary_path) if config.binary_path else LTX_DEFAULT_BIN
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _resolve_ltx_model_ng(config: LtxConfig) -> Optional[str]:
    p = Path(config.model_path) if config.model_path else LTX_DEFAULT_MODEL
    return str(p) if p.exists() else None


def _resolve_ltx_gemma_ng(config: LtxConfig) -> Optional[str]:
    p = Path(config.gemma_path) if config.gemma_path else LTX_DEFAULT_GEMMA
    return str(p) if p.exists() else None


def _resolve_ltx_enhance_gemma_ng(config: LtxConfig) -> Optional[str]:
    p = Path(config.enhance_gemma_path) if config.enhance_gemma_path else LTX_DEFAULT_ENHANCE_GEMMA
    return str(p) if p.exists() else None


# Shared between generate_ltx_video_ng and enhance_ltx_prompt_ng so an
# enhance call (Gemma only, ~seconds) can never run on the GPU at the
# same moment as a queued render (both use the same Metal device via
# the same ltx-2-mlx subprocess).
_LTX_SUBPROCESS_LOCK = threading.Lock()

LTX_ENHANCE_TIMEOUT_S = 180.0


def ltx_health_ng() -> dict:
    """Cheap up-front check the routes layer can surface to the UI/status
    endpoint without launching a subprocess."""
    cfg = LtxConfig()
    binary = _resolve_ltx_binary_ng(cfg)
    model = _resolve_ltx_model_ng(cfg)
    gemma = _resolve_ltx_gemma_ng(cfg)
    enhance_gemma = _resolve_ltx_enhance_gemma_ng(cfg)
    return {
        "repo_dir": str(LTX_REPO_DIR),
        "binary_ok": binary is not None,
        "binary_path": binary or str(cfg.binary_path or LTX_DEFAULT_BIN),
        "model_ok": model is not None,
        "model_path": model or str(cfg.model_path or LTX_DEFAULT_MODEL),
        "gemma_ok": gemma is not None,
        "gemma_path": gemma or str(cfg.gemma_path or LTX_DEFAULT_GEMMA),
        # Not required for "ready" -- only gates the Enhance button, not
        # rendering itself.
        "enhance_gemma_ok": enhance_gemma is not None,
        "enhance_gemma_path": enhance_gemma or str(cfg.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA),
        "ready": binary is not None and model is not None and gemma is not None,
    }


def generate_ltx_video_ng(prompt: str, image_path: str, duration_s: float,
                           output_dir: Path, seed: Optional[int],
                           config: LtxConfig,
                           on_log: Optional[Callable[[str], None]] = None,
                           on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: reference image + prompt + duration in, one
    mp4 out."""
    binary = _resolve_ltx_binary_ng(config)
    if not binary:
        raise FileNotFoundError(
            f"ltx-2-mlx binary not found at "
            f"{config.binary_path or LTX_DEFAULT_BIN}")
    model = _resolve_ltx_model_ng(config)
    if not model:
        raise FileNotFoundError(
            f"LTX-2.5 model not found at "
            f"{config.model_path or LTX_DEFAULT_MODEL}")
    gemma = _resolve_ltx_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"LTX-2.5's paired Gemma text encoder not found at "
            f"{config.gemma_path or LTX_DEFAULT_GEMMA}")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"LTX reference image not found at {image_path}")

    frames = _duration_to_frames_ng(duration_s)
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_mp4 = output_dir / f"ltx_{ts}.mp4"

    cmd = [
        binary, "generate",
        "--prompt", prompt,
        "--image", image_path,
        "--output", str(out_mp4),
        "--model", model,
        "--gemma", gemma,
        "--seed", str(seed),
        "--height", str(LTX_HEIGHT),
        "--width", str(LTX_WIDTH),
        "--frames", str(frames),
        "--frame-rate", str(LTX_FRAME_RATE),
        "--two-stages-hq",
        "--stage1-steps", str(LTX_STAGE1_STEPS),
        "--stage2-steps", str(LTX_STAGE2_STEPS),
        "--distilled-lora", LTX_DISTILLED_LORA,
    ]

    if on_log:
        on_log(f"[ltx] launching {frames} frames (~{(frames - 1) / LTX_FRAME_RATE:.1f}s) "
               f"at {LTX_WIDTH}x{LTX_HEIGHT}, seed={seed}")

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            start_new_session=True,
        )
        if on_proc_start:
            on_proc_start(proc)

        timeout_s = float(os.environ.get("RINGVIZ_LTX_TIMEOUT_S", config.timeout_s))
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
                        on_log(f"[ltx] {line}")
            rc = proc.wait()
        finally:
            watchdog.cancel()

    if timed_out["v"]:
        raise VideoJobCancelled(
            f"LTX gen exceeded its {timeout_s:.0f}s deadline and was "
            f"killed (likely a hung render). Raise RINGVIZ_LTX_TIMEOUT_S "
            f"if this machine legitimately needs longer.")
    if rc != 0:
        if rc in (-15, 143, -9, 137):
            raise VideoJobCancelled(f"LTX gen cancelled (rc={rc})")
        raise RuntimeError(f"LTX gen failed with rc={rc}")

    if not out_mp4.exists() or out_mp4.stat().st_size == 0:
        raise RuntimeError(f"LTX gen finished but no/empty mp4 at {out_mp4}")

    return {
        "mp4_path": str(out_mp4),
        "seed": seed,
        "engine": "ltx-2.5-mlx-q8",
        "width": LTX_WIDTH,
        "height": LTX_HEIGHT,
        "frames": frames,
        "duration_s": (frames - 1) / LTX_FRAME_RATE,
    }


def enhance_ltx_prompt_ng(prompt: str, config: LtxConfig,
                           seed: Optional[int] = None) -> str:
    """Runs `ltx-2-mlx enhance` -- Gemma rewrites a short prompt into
    LTX's preferred verbose motion-description style. Text only, no
    video render; always --mode i2v since the Animate view is always
    image-conditioned. Shares _LTX_SUBPROCESS_LOCK with
    generate_ltx_video_ng so this never contends with a queued render
    for the same GPU."""
    binary = _resolve_ltx_binary_ng(config)
    if not binary:
        raise FileNotFoundError(
            f"ltx-2-mlx binary not found at "
            f"{config.binary_path or LTX_DEFAULT_BIN}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for prompt enhancement not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [
        binary, "enhance",
        "--prompt", prompt,
        "--mode", "i2v",
        "--gemma", gemma,
        "--seed", str(seed),
    ]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"prompt enhancement failed (rc={proc.returncode}): "
            f"{proc.stdout[-500:]}")

    marker = "\nEnhanced: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(
            f"prompt enhancement produced no 'Enhanced:' output: "
            f"{proc.stdout[-500:]}")
    enhanced = proc.stdout[idx + len(marker):].strip()
    if not enhanced:
        raise RuntimeError("prompt enhancement returned empty text")
    return enhanced
