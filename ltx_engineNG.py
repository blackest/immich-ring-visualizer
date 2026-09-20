"""NG engine for LTX-2.5 image-to-video generation.

Talks to the standalone ltx-2-mlx install as a subprocess (its own venv,
never imported in-process) -- this file owns nothing about the model
itself, just how to find it, build the CLI invocation, and validate the
output. One input (reference image + prompt + duration) -> one output
(an mp4 path + result dict), matching hidream_engineNG.py's shape.

Deliberately out of scope here (keep this file small -- add a sibling
module later if any of these turn out to be needed, don't grow this
one): quality-tier selection (always the "high" tier: two-stages-hq,
1024x576, 10+3 steps), and phosphene's persistent warm-helper process
(fine for occasional single renders to eat the model-load cost per
call instead).

Optional style/character LoRAs (on top of the fixed --distilled-lora
refine step above) ARE in scope, but only as a thin pass-through of
an already-resolved path + strength -- see ltx_loraNG.py, a sibling
module, for discovering what's on disk and resolving a UI-supplied
name to a real path.
"""

from __future__ import annotations

import json
import os
import random
import re
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
LTX_DEFAULT_PYTHON = LTX_REPO_DIR / "env" / "bin" / "python"
LTX_SCENE_TIMING_HELPER = Path(__file__).resolve().parent / "ltx_scene_timing_helperNG.py"
LTX_SCENE_DISCUSS_HELPER = Path(__file__).resolve().parent / "ltx_scene_discuss_helperNG.py"
LTX_SCENE_CHAT_HELPER = Path(__file__).resolve().parent / "ltx_scene_chat_helperNG.py"
LTX_VISION_ENHANCE_HELPER = Path(__file__).resolve().parent / "ltx_vision_enhance_helperNG.py"
LTX_VISION_CHAT_HELPER = Path(__file__).resolve().parent / "ltx_vision_chat_helperNG.py"
LTX_VISION_SCENE_DISCUSS_HELPER = Path(__file__).resolve().parent / "ltx_vision_scene_discuss_helperNG.py"
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

# Optional standalone mlx_vlm.server (see gemma_server.sh) holding the
# enhance Gemma checkpoint resident instead of reloading it from disk on
# every chat turn. Purely an optimization: chat_with_gemma_ng tries this
# first and falls back to the existing per-call subprocess helpers
# unchanged if nothing's listening here.
LTX_GEMMA_SERVER_URL = os.environ.get("RINGVIZ_GEMMA_SERVER_URL", "http://127.0.0.1:8811")
LTX_GEMMA_SERVER_CONNECT_TIMEOUT_S = 3.0


def _unload_gemma_server_ng(on_log: Optional[Callable[[str], None]] = None) -> None:
    """Best-effort: free the standalone Gemma server's resident model
    before a render takes the GPU, so the two never fight over memory on
    a machine where they wouldn't both fit (see gemma_server.sh's own
    docstring). Silently does nothing if the server isn't running -- it
    not being up is the common case, not an error -- and never raises,
    since a failed unload attempt must never block a render. She lazy-
    reloads on her own next chat request, same as any cold start."""
    import requests
    try:
        requests.post(f"{LTX_GEMMA_SERVER_URL}/unload",
                       timeout=LTX_GEMMA_SERVER_CONNECT_TIMEOUT_S)
        if on_log:
            on_log("[gemma] unloaded from the standalone server before render")
    except requests.exceptions.RequestException:
        pass

# "high" tier (2nd of phosphene's 4: draft/balanced/high/high_720p) --
# two-stages-hq pipeline, 10+3 steps. Fixed; no tier plumbing. Width/height/
# frame-rate are the caller's choice (validated below); these are just the
# defaults when none is given.
LTX_WIDTH = 1024
LTX_HEIGHT = 576
LTX_STAGE1_STEPS = 10
LTX_STAGE2_STEPS = 3

# LTX's video VAE downsamples space by 32x, so the model itself only needs
# dims divisible by 32 (confirmed against the vendored ltx-2-mlx's own
# orchestration code: "Encoder spatial height (must be divisible by 32)").
# BUT this engine always renders with --two-stages-hq, which halves both
# dims for stage 1 (`half_h, half_w = height // 2, width // 2` in
# ti2vid_two_stages.py) then upscales x2 back for stage 2 -- so the real
# constraint here is divisible by 64, not 32. Learned this the hard way:
# a 32-only-validated non-64 size crashed the LTX subprocess (confirmed via
# ~/Library/Logs/DiagnosticReports -- python3.11 SIGABRT, stack rooted in
# mlx::core::gpu::check_error(MTL::CommandBuffer*) -> std::terminate ->
# abort(), always right after LoRA fusion / before stage-2 denoising,
# i.e. exactly at the half-res -> full-res handoff). Bounds are a guess at
# what's sane on this machine, not a hard model limit -- widen if that
# turns out to be wrong in practice.
LTX_DIM_STEP = 64
LTX_MIN_DIM = 256
LTX_MAX_DIM = 1536
LTX_MIN_FPS = 8.0
LTX_MAX_FPS = 30.0


def _validate_ltx_dims_ng(width: int, height: int) -> None:
    for name, v in (("width", width), ("height", height)):
        if v % LTX_DIM_STEP != 0:
            raise ValueError(f"{name} must be a multiple of {LTX_DIM_STEP} (got {v})")
        if not (LTX_MIN_DIM <= v <= LTX_MAX_DIM):
            raise ValueError(f"{name} must be between {LTX_MIN_DIM} and {LTX_MAX_DIM} (got {v})")


def _validate_ltx_fps_ng(frame_rate: float) -> None:
    if not (LTX_MIN_FPS <= frame_rate <= LTX_MAX_FPS):
        raise ValueError(f"frame_rate must be between {LTX_MIN_FPS} and {LTX_MAX_FPS} (got {frame_rate})")
# --two-stages-hq's stage-2 refine step fuses a distilled LoRA that lives
# inside the model dir; the CLI's own default filename is LTX-2.3's, not
# 2.5's -- passing the wrong one raises FileNotFoundError (confirmed live).
LTX_DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450.safetensors"


def _duration_to_frames_ng(seconds: float, frame_rate: float = LTX_FRAME_RATE) -> int:
    """LTX's temporal compression requires frames % 8 == 1, at whatever
    frame rate the render targets."""
    blocks = max(1, round(seconds * frame_rate / 8))
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


def _resolve_ltx_python_ng() -> Optional[str]:
    """The same venv's own interpreter -- used to run
    LTX_SCENE_TIMING_HELPER directly against ltx_core_mlx, since the
    `ltx-2-mlx enhance` CLI subcommand has no raw-question mode."""
    return str(LTX_DEFAULT_PYTHON) if LTX_DEFAULT_PYTHON.is_file() else None


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


def generate_ltx_video_ng(prompt: str, image_path: Optional[str], duration_s: float,
                           output_dir: Path, seed: Optional[int],
                           config: LtxConfig,
                           width: Optional[int] = None, height: Optional[int] = None,
                           frame_rate: Optional[float] = None,
                           lora_path: Optional[str] = None, lora_strength: float = 1.0,
                           on_log: Optional[Callable[[str], None]] = None,
                           on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: prompt + duration in, one mp4 out. image_path
    is optional -- when given, it's an image-to-video render (LTX
    animates that photo); when None, it's a plain text-to-video render.
    Same CLI/model/quality settings either way -- `ltx-2-mlx generate`
    already treats --image as optional, so this is just whether the
    flag is present, not a different pipeline. width/height/frame_rate
    default to LTX_WIDTH/LTX_HEIGHT/LTX_FRAME_RATE when omitted; when
    given, they're validated (dims must be multiples of LTX_DIM_STEP,
    see its docstring). lora_path is optional and separate from the
    fixed --distilled-lora stage-2 refine step above -- it's the CLI's
    own repeatable --lora PATH STRENGTH flag (see ltx_loraNG.py for how
    a name from the UI resolves to a real path on disk)."""
    width = LTX_WIDTH if width is None else int(width)
    height = LTX_HEIGHT if height is None else int(height)
    frame_rate = LTX_FRAME_RATE if frame_rate is None else float(frame_rate)
    _validate_ltx_dims_ng(width, height)
    _validate_ltx_fps_ng(frame_rate)

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
    if image_path is not None and not Path(image_path).is_file():
        raise FileNotFoundError(f"LTX reference image not found at {image_path}")
    if lora_path is not None and not Path(lora_path).is_file():
        raise FileNotFoundError(f"LTX LoRA checkpoint not found at {lora_path}")

    frames = _duration_to_frames_ng(duration_s, frame_rate)
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out_mp4 = output_dir / f"ltx_{ts}.mp4"

    cmd = [
        binary, "generate",
        "--prompt", prompt,
        "--output", str(out_mp4),
        "--model", model,
        "--gemma", gemma,
        "--seed", str(seed),
        "--height", str(height),
        "--width", str(width),
        "--frames", str(frames),
        "--frame-rate", str(frame_rate),
        "--two-stages-hq",
        "--stage1-steps", str(LTX_STAGE1_STEPS),
        "--stage2-steps", str(LTX_STAGE2_STEPS),
        "--distilled-lora", LTX_DISTILLED_LORA,
    ]
    if image_path is not None:
        cmd += ["--image", image_path]
    if lora_path is not None:
        cmd += ["--lora", lora_path, str(lora_strength)]

    if on_log:
        on_log(f"[ltx] launching {frames} frames (~{(frames - 1) / frame_rate:.1f}s) "
               f"at {width}x{height}, {frame_rate}fps, seed={seed}")

    with _LTX_SUBPROCESS_LOCK:
        _unload_gemma_server_ng(on_log=on_log)
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
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "frames": frames,
        "duration_s": (frames - 1) / frame_rate,
    }


def enhance_ltx_prompt_ng(prompt: str, config: LtxConfig,
                           seed: Optional[int] = None,
                           image_b64: Optional[str] = None) -> str:
    """Runs `ltx-2-mlx enhance` -- Gemma rewrites a short prompt into
    LTX's preferred verbose motion-description style. Text only, no
    video render; always --mode i2v since the Animate view is always
    image-conditioned. Shares _LTX_SUBPROCESS_LOCK with
    generate_ltx_video_ng so this never contends with a queued render
    for the same GPU.

    image_b64, when given (base64, no "data:" prefix), routes through
    _enhance_ltx_prompt_vision_ng instead of the CLI below. The
    `ltx-2-mlx enhance` CLI's underlying GemmaLanguageModel loads Gemma
    through mlx_lm, which is text-only -- its own enhance_i2v docstring
    admits it "does not pass the image to Gemma" despite the i2v system
    prompt telling it to analyze one. The vision path loads the exact
    same Gemma checkpoint through mlx_vlm, which does have a vision
    tower, and gives it the real pixels -- kept as base64 the whole way
    through rather than written to a temp file, per the no-disk-cache
    principle (only real exports belong on disk)."""
    if image_b64 is not None:
        return _enhance_ltx_prompt_vision_ng(prompt, image_b64, config, seed=seed)

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


def _enhance_ltx_prompt_vision_ng(prompt: str, image_b64: str, config: LtxConfig,
                                   seed: Optional[int] = None) -> str:
    """Vision-aware sibling of enhance_ltx_prompt_ng's CLI path -- runs
    LTX_VISION_ENHANCE_HELPER via the venv's own python (same subprocess
    boundary as the scene-timing/discuss/chat helpers) so Gemma can
    actually see the reference image instead of only being told it's an
    i2v render. image_b64 travels over the subprocess's stdin and is
    never written to disk -- mlx_vlm's own image loader accepts a base64
    data URI directly, so a temp file would buy nothing here. Same
    _LTX_SUBPROCESS_LOCK, same 'Enhanced:' marker protocol as the
    text-only path above."""
    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_VISION_ENHANCE_HELPER.is_file():
        raise FileNotFoundError(
            f"vision-enhance helper script not found at {LTX_VISION_ENHANCE_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for prompt enhancement not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [
        python, str(LTX_VISION_ENHANCE_HELPER),
        "--gemma", gemma,
        "--prompt", prompt,
        "--seed", str(seed),
    ]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            input=image_b64,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"vision prompt enhancement failed (rc={proc.returncode}): "
            f"{proc.stdout[-500:]}")

    marker = "\nEnhanced: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(
            f"vision prompt enhancement produced no 'Enhanced:' output: "
            f"{proc.stdout[-500:]}")
    enhanced = proc.stdout[idx + len(marker):].strip()
    if not enhanced:
        raise RuntimeError("vision prompt enhancement returned empty text")
    return enhanced


def estimate_scene_seconds_ng(prompt: str, config: LtxConfig,
                               seed: Optional[int] = None) -> float:
    """Asks the enhance Gemma checkpoint how long this shot needs, in
    seconds -- runs LTX_SCENE_TIMING_HELPER via the venv's own python
    (not the `ltx-2-mlx` binary: its `enhance` subcommand only exposes
    Gemma's fixed prompt-rewrite templates, with no way to ask a
    different question). Shares _LTX_SUBPROCESS_LOCK so this never
    contends with a queued render or an enhance call for the same GPU."""
    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_SCENE_TIMING_HELPER.is_file():
        raise FileNotFoundError(
            f"scene-timing helper script not found at {LTX_SCENE_TIMING_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for scene timing not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [
        python, str(LTX_SCENE_TIMING_HELPER),
        "--gemma", gemma,
        "--prompt", prompt,
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
            f"scene timing failed (rc={proc.returncode}): "
            f"{proc.stdout[-500:]}")

    marker = "\nSeconds: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(
            f"scene timing produced no 'Seconds:' output: "
            f"{proc.stdout[-500:]}")
    tail = proc.stdout[idx + len(marker):].strip()
    m = re.search(r"[\d.]+", tail)
    if not m:
        raise RuntimeError(f"scene timing returned no parseable number: {tail!r}")
    seconds = float(m.group(0))
    return max(0.5, min(30.0, seconds))


def discuss_next_scene_ng(turns: list, config: LtxConfig,
                           seed: Optional[int] = None,
                           images: Optional[list] = None) -> dict:
    """Multi-turn "storyboard the next shot" chat against the enhance
    Gemma checkpoint via LTX_SCENE_DISCUSS_HELPER (same venv-python
    subprocess pattern as estimate_scene_seconds_ng, for the same reason:
    no public multi-turn chat method exists on this library, only
    single-shot prompt-rewrite templates). `turns` is the caller's full
    conversation so far -- a list of {"role": "user"|"assistant",
    "content": str} dicts, excluding the system prompt (the helper script
    always prepends its own). Returns {"raw": <the exact text Gemma
    produced -- feed this back as the next "assistant" turn>,
    "discussion": str, "options": [{"label", "prompt", "seconds"}, ...]}.

    images, when given, is a list of base64-encoded image strings (no
    "data:" prefix) belonging to the CURRENT turn only -- routes through
    _discuss_next_scene_vision_ng instead, same "only the last turn's
    images are ever visible" caveat as chat_with_gemma_ng."""
    if images:
        return _discuss_next_scene_vision_ng(turns, images, config, seed=seed)

    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_SCENE_DISCUSS_HELPER.is_file():
        raise FileNotFoundError(
            f"scene-discuss helper script not found at {LTX_SCENE_DISCUSS_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for scene discussion not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [python, str(LTX_SCENE_DISCUSS_HELPER), "--gemma", gemma, "--seed", str(seed)]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            input=json.dumps(turns),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"scene discussion failed (rc={proc.returncode}): "
            f"{proc.stdout[-500:]}")

    marker = "\nReply: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(
            f"scene discussion produced no 'Reply:' output: "
            f"{proc.stdout[-500:]}")
    raw = proc.stdout[idx + len(marker):].strip()

    parsed = _parse_scene_discuss_json_ng(raw)
    return {"raw": raw, **parsed}


def _parse_scene_discuss_json_ng(raw: str) -> dict:
    """Small chat models sometimes wrap JSON in a markdown fence or add
    stray text despite being told not to -- this tolerates the fence and
    otherwise requires the response to parse as the documented shape."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"scene discussion returned unparseable JSON: {raw[:500]!r}") from e

    if not isinstance(data, dict) or not isinstance(data.get("options"), list) or not data["options"]:
        raise RuntimeError(f"scene discussion JSON had no usable 'options': {raw[:500]!r}")

    cleaned = []
    for i, opt in enumerate(data["options"][:3]):
        if not isinstance(opt, dict):
            continue
        prompt = str(opt.get("prompt") or "").strip()
        if not prompt:
            continue
        label = str(opt.get("label") or "").strip() or chr(65 + i)
        try:
            seconds = float(opt.get("seconds"))
        except (TypeError, ValueError):
            seconds = 3.0
        cleaned.append({
            "label": label,
            "prompt": prompt,
            "seconds": max(0.5, min(30.0, seconds)),
        })
    if not cleaned:
        raise RuntimeError(f"scene discussion options had no usable prompt text: {raw[:500]!r}")

    return {"discussion": str(data.get("discussion") or "").strip(), "options": cleaned}


def _discuss_next_scene_vision_ng(turns: list, images: list, config: LtxConfig,
                                   seed: Optional[int] = None) -> dict:
    """Vision-aware sibling of discuss_next_scene_ng's text-only path --
    runs LTX_VISION_SCENE_DISCUSS_HELPER via the venv's own python, same
    in-memory base64 image handoff as _chat_with_gemma_vision_ng."""
    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_VISION_SCENE_DISCUSS_HELPER.is_file():
        raise FileNotFoundError(
            f"vision scene-discuss helper script not found at {LTX_VISION_SCENE_DISCUSS_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for scene discussion not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [python, str(LTX_VISION_SCENE_DISCUSS_HELPER), "--gemma", gemma, "--seed", str(seed)]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            input=json.dumps({"turns": turns, "images": images}),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"vision scene discussion failed (rc={proc.returncode}): "
            f"{proc.stdout[-500:]}")

    marker = "\nReply: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(
            f"vision scene discussion produced no 'Reply:' output: "
            f"{proc.stdout[-500:]}")
    raw = proc.stdout[idx + len(marker):].strip()

    parsed = _parse_scene_discuss_json_ng(raw)
    return {"raw": raw, **parsed}


# Same wording as ltx_scene_chat_helperNG.py / ltx_vision_chat_helperNG.py's
# own SYSTEM_PROMPT -- duplicated here (not imported) because those are
# standalone scripts meant to run inside the ltx-2-mlx venv as a
# subprocess, not modules for the main app's own Python environment to
# import. Keep the two in sync by hand if either wording changes.
_ASK_RACHEL_INSTRUCTION = (
    " You have a friend, Rachel, who can look things up in the real world "
    "for you -- current events, live data, anything your training can't "
    "know or might have wrong. If you genuinely need that, write a line "
    "starting with exactly 'ASK_RACHEL:' followed by your question, and "
    "nothing else in your reply. Only do this when you actually need "
    "outside information -- not for creative or storyboarding questions "
    "you can already answer yourself."
)
_GEMMA_CHAT_SYSTEM_PROMPT = (
    "You are Gemma, chatting with a filmmaker who is using you elsewhere in "
    "this app to storyboard shots for an image-to-video render. Right now "
    "they just want to talk -- about the project, an idea, or anything else "
    "on their mind. Reply naturally and conversationally, in plain text. Do "
    "not force the conversation toward shot options, JSON, or any fixed "
    "format unless they specifically ask for one."
) + _ASK_RACHEL_INSTRUCTION
_GEMMA_VISION_CHAT_SYSTEM_PROMPT = (
    "You are Gemma, chatting with a filmmaker who is using you elsewhere in "
    "this app to storyboard shots for an image-to-video render. Right now "
    "they just want to talk -- about the project, an idea, or anything else "
    "on their mind, sometimes with a reference image attached. Reply "
    "naturally and conversationally, in plain text. Do not force the "
    "conversation toward shot options, JSON, or any fixed format unless "
    "they specifically ask for one."
) + _ASK_RACHEL_INSTRUCTION


def _chat_with_gemma_server_ng(turns: list, config: LtxConfig,
                                seed: Optional[int] = None,
                                images: Optional[list] = None) -> Optional[dict]:
    """Try the standalone gemma_server.sh (mlx_vlm.server) first. Returns
    None -- not raises -- when nothing's listening, so callers fall back
    to the existing subprocess helpers unchanged; only raises once the
    server has actually accepted the connection and then failed, since a
    running-but-erroring server is a real failure worth surfacing rather
    than silently masking behind a slow subprocess retry."""
    import requests

    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        return None

    system_prompt = _GEMMA_VISION_CHAT_SYSTEM_PROMPT if images else _GEMMA_CHAT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    for i, turn in enumerate(turns):
        content = turn["content"]
        if images and i == len(turns) - 1 and turn.get("role") == "user":
            parts = [{"type": "text", "text": content}]
            for b64 in images:
                parts.append({"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content = parts
        messages.append({"role": turn["role"], "content": content})

    try:
        resp = requests.post(
            f"{LTX_GEMMA_SERVER_URL}/v1/chat/completions",
            json={
                "model": gemma,
                "messages": messages,
                "max_tokens": 768,
                "temperature": 0.7,
                **({"seed": seed} if seed is not None else {}),
            },
            timeout=(LTX_GEMMA_SERVER_CONNECT_TIMEOUT_S, LTX_ENHANCE_TIMEOUT_S),
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return None

    if resp.status_code != 200:
        raise RuntimeError(f"Gemma server chat failed (HTTP {resp.status_code}): {resp.text[-500:]}")

    reply = resp.json()["choices"][0]["message"]["content"]
    if not reply or not reply.strip():
        raise RuntimeError("Gemma server chat produced empty output")
    return {"reply": reply.strip()}


_ASK_RACHEL_MARKER = "ASK_RACHEL:"


def _extract_ask_rachel_ng(reply: str) -> Optional[str]:
    """Pull the question out of a Gemma reply that used the ASK_RACHEL
    delegation convention (see _ASK_RACHEL_INSTRUCTION), or None if she
    didn't ask for anything this turn. Takes everything from the marker
    to the end of the reply as the question, in case she wrote more
    after it despite being told not to -- confirmed live that she
    sometimes keeps chatting past it, which drags Rachel into a longer
    tangential answer than a strict single-line extraction would allow.
    Left this way on purpose rather than trimmed to the first line: it's
    a real personality quirk, not just noise, and worth living with
    rather than engineering away."""
    idx = reply.find(_ASK_RACHEL_MARKER)
    if idx == -1:
        return None
    question = reply[idx + len(_ASK_RACHEL_MARKER):].strip()
    return question or None


def ask_rachel_ng(question: str) -> str:
    """Forward one delegated question to the same Hermes agent gateway
    Rachel's own chat view (routes/rachelNG.py) talks to -- reusing its
    auth/config, but as a plain synchronous call instead of that route's
    browser-facing SSE stream. This is the hand-off Gemma's ASK_RACHEL
    convention exists for: she has no reliable structured tool-calling of
    her own (confirmed live -- asked to call a tool, she wrote a fake
    Python implementation instead of a real call), so delegation goes
    through plain text on both sides rather than any API-level tool
    contract.

    Deliberately does NOT send routes/rachelNG.py's X-Hermes-Session-Id/
    X-Hermes-Session-Key headers -- those exist specifically to load
    Rachel's persistent memory/context onto a turn (see that route's own
    comment), which a one-off factual lookup doesn't need and shouldn't
    pay for. Per Hermes' own API docs, /v1/chat/completions is stateless
    by default without them -- the full conversation is just whatever's
    in `messages`, nothing more -- while the full toolset (web search
    etc.) is still available since that's bound to the API-server
    config, not to session continuity."""
    import requests
    from configNG import get_hermes_api_key, get_hermes_base_url, get_hermes_model

    base_url = get_hermes_base_url()
    if not base_url:
        raise RuntimeError("Hermes gateway not configured (hermes_base_url is empty)")

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {get_hermes_api_key()}",
        },
        json={
            "model": get_hermes_model(),
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        # Same generous read timeout as routes/rachelNG.py, and for the
        # same reason -- the agent can sit for minutes on the first token
        # (a long tool-use loop plus a big context preamble).
        timeout=(10, 600),
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    return answer.strip()


def chat_with_gemma_ng(turns: list, config: LtxConfig,
                        seed: Optional[int] = None,
                        images: Optional[list] = None) -> dict:
    """Free-form, unconstrained chat against the enhance Gemma checkpoint,
    with one round of ASK_RACHEL delegation: if her reply uses that
    convention (see _ASK_RACHEL_INSTRUCTION), the question is forwarded
    to Rachel via ask_rachel_ng, the answer is folded back in as one more
    turn, and Gemma is asked once more for a final reply -- no further
    delegation on that second pass, so this can never loop. `turns` is
    the caller's full conversation so far -- a list of {"role": "user"|
    "assistant", "content": str} dicts, excluding the system prompt
    (_generate_gemma_reply_ng's paths prepend their own). Returns
    {"reply": str}, plus {"delegated": {"question": str, "answer": str}}
    when a hand-off happened, so callers/logs can see it occurred.

    images, when given, is a list of base64-encoded image strings (no
    "data:" prefix) belonging to the CURRENT turn only -- see
    _generate_gemma_reply_ng for why only the newest turn's images can
    ever matter."""
    result = _generate_gemma_reply_ng(turns, config, seed=seed, images=images)
    question = _extract_ask_rachel_ng(result["reply"])
    if question is None:
        return result

    try:
        answer = ask_rachel_ng(question)
    except Exception as e:
        # Rachel unreachable/misconfigured -- surface Gemma's raw reply
        # (marker text and all) rather than losing the turn entirely.
        result["rachel_error"] = str(e)
        return result

    followup_turns = turns + [
        {"role": "assistant", "content": result["reply"]},
        {"role": "user", "content": f"[Rachel says: {answer}]"},
    ]
    final = _generate_gemma_reply_ng(followup_turns, config, seed=seed, images=None)
    final["delegated"] = {"question": question, "answer": answer}
    return final


def _generate_gemma_reply_ng(turns: list, config: LtxConfig,
                              seed: Optional[int] = None,
                              images: Optional[list] = None) -> dict:
    """One Gemma turn, no delegation handling -- tries the standalone
    gemma_server.sh first (see _chat_with_gemma_server_ng), falling back
    to LTX_SCENE_CHAT_HELPER (same venv-python subprocess pattern as
    discuss_next_scene_ng) when nothing's listening there. Returns
    {"reply": str}.

    images, when given, is a list of base64-encoded image strings (no
    "data:" prefix) belonging to the CURRENT turn only -- via the server
    path above, or _chat_with_gemma_vision_ng as a subprocess fallback.
    Only the current turn's images can matter: mlx_vlm's chat template
    places image tokens on just the last user message (see
    ltx_vision_chat_helperNG.py's docstring), so an image attached
    earlier in the conversation was never going to be visible to Gemma on
    a later call anyway -- callers should only ever pass images alongside
    the newest turn."""
    server_result = _chat_with_gemma_server_ng(turns, config, seed=seed, images=images)
    if server_result is not None:
        return server_result

    if images:
        return _chat_with_gemma_vision_ng(turns, images, config, seed=seed)

    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_SCENE_CHAT_HELPER.is_file():
        raise FileNotFoundError(
            f"scene-chat helper script not found at {LTX_SCENE_CHAT_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for free chat not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [python, str(LTX_SCENE_CHAT_HELPER), "--gemma", gemma, "--seed", str(seed)]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            input=json.dumps(turns),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"chat failed (rc={proc.returncode}): {proc.stdout[-500:]}")

    marker = "\nReply: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(f"chat produced no 'Reply:' output: {proc.stdout[-500:]}")
    return {"reply": proc.stdout[idx + len(marker):].strip()}


def _chat_with_gemma_vision_ng(turns: list, images: list, config: LtxConfig,
                                seed: Optional[int] = None) -> dict:
    """Vision-aware sibling of chat_with_gemma_ng's text-only path -- runs
    LTX_VISION_CHAT_HELPER via the venv's own python. `images` (base64
    strings) travel over the subprocess's stdin alongside `turns`, never
    touching disk: mlx_vlm's own image loader accepts a base64 data URI
    directly, so there's nothing a temp file would buy here that the
    bytes already sitting in memory don't -- per this repo's no-disk-cache
    principle (only real exports belong on disk)."""
    python = _resolve_ltx_python_ng()
    if not python:
        raise FileNotFoundError(
            f"ltx-2-mlx venv python not found at {LTX_DEFAULT_PYTHON}")
    if not LTX_VISION_CHAT_HELPER.is_file():
        raise FileNotFoundError(
            f"vision-chat helper script not found at {LTX_VISION_CHAT_HELPER}")
    gemma = _resolve_ltx_enhance_gemma_ng(config)
    if not gemma:
        raise FileNotFoundError(
            f"Chat Gemma-3 checkpoint for free chat not found at "
            f"{config.enhance_gemma_path or LTX_DEFAULT_ENHANCE_GEMMA}")

    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    cmd = [python, str(LTX_VISION_CHAT_HELPER), "--gemma", gemma, "--seed", str(seed)]

    with _LTX_SUBPROCESS_LOCK:
        proc = subprocess.run(
            cmd,
            input=json.dumps({"turns": turns, "images": images}),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(LTX_REPO_DIR),
            timeout=LTX_ENHANCE_TIMEOUT_S,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"vision chat failed (rc={proc.returncode}): {proc.stdout[-500:]}")

    marker = "\nReply: "
    idx = proc.stdout.rfind(marker)
    if idx == -1:
        raise RuntimeError(f"vision chat produced no 'Reply:' output: {proc.stdout[-500:]}")
    return {"reply": proc.stdout[idx + len(marker):].strip()}
