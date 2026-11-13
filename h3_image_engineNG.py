"""NG: MiniMax-H3-Image -- a picture from words, or an edit of 1 to 6 reference pictures (Picture 1 is the one that is
edited, the others are named in the prompt). One small input -> output file that runs the vendored runner
(vendor/minimax_h3_mlx/scripts/generate_h3_image.py) as a subprocess in the Hailuo H3 venv, a peer of klein_engineNG.py.

The 22 GB pack is read in place from Phosphene's models folder (H3_MODELS_ROOT/minimax-h3-image, made by its own
download button); the text encoder is the one the H3 video lane already uses. Nothing is downloaded or copied here.

Licence: the model's own community licence excludes the EU, UK, Republic of Korea and US (see the pack's LICENSE).
"""
from __future__ import annotations

import json
import os
import queue
import random
import signal
import subprocess
import threading
import time
from pathlib import Path

import h3_engineNG as h3
import ltx_engineNG as ltx

RUNNER = h3.H3_VENDOR_DIR / "scripts" / "generate_h3_image.py"
PACK_DIR = h3.H3_MODELS_ROOT / "minimax-h3-image"
PACK_FILES = ("dit/adaln_cache.safetensors", "vae/diffusion_pytorch_model.safetensors", "pdd_fused_heads.safetensors")
TIERS = {"draft": 4.0, "standard": 8.0, "high": 16.0}       # megapixels (the model's floor is 4)
SHAPES = ("match", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")   # "match" = the shape of Picture 1 (edit only)
MAX_REFS = 6                    # the engine takes 16; Phosphene's Studio stops at 6 to keep time and memory near measured
TIMEOUT_S = {"draft": 1800, "standard": 3600, "high": 7200}
ATTEMPTS = 3                    # macOS's GPU watchdog kills a render at random; a fresh process is the cure
RESULT_NOTE = "MiniMax-H3-Image"


class H3ImageCancelled(RuntimeError):
    pass


def h3_image_health_ng() -> dict:
    """Is everything on disk that a render needs. Never launches the model."""
    problems = []
    if not h3.H3_DEFAULT_PYTHON.is_file():
        problems.append("the Hailuo H3 install is missing")
    if not RUNNER.is_file():
        problems.append("the vendored H3-Image runner is missing")
    missing = [f for f in PACK_FILES if not (PACK_DIR / f).is_file()]
    if missing:
        problems.append("the MiniMax-H3-Image model is not downloaded: use its download button in Phosphene's Images page")
    if not h3.H3_COMPACT_ROOT.is_dir():
        problems.append("the H3 text encoder is missing")
    if not h3.H3_TEXT_CONFIG.is_file():
        problems.append("the H3 text encoder config is missing")
    return {"ok": not problems, "problems": problems, "tiers": dict(TIERS), "shapes": list(SHAPES),
            "max_refs": MAX_REFS, "name": RESULT_NOTE}


def validate_ng(prompt: str, n: int, shape: str, tier: str, refs) -> None:
    if not str(prompt).strip():
        raise ValueError("write a prompt: say what to make, or what to change in Picture 1")
    if not 1 <= int(n) <= 8:
        raise ValueError("n must be 1 to 8")
    if tier not in TIERS:
        raise ValueError("quality must be one of: " + ", ".join(TIERS))
    if shape not in SHAPES:
        raise ValueError("shape must be one of: " + ", ".join(SHAPES))
    if len(refs) > MAX_REFS:
        raise ValueError(f"at most {MAX_REFS} reference pictures")
    if shape == "match" and not refs:
        raise ValueError("'same as picture' needs a reference picture")
    for r in refs:
        if not Path(r).is_file():
            raise ValueError(f"reference picture not found: {Path(r).name}")


def _command(prompt: str, png: Path, metrics: Path, shape: str, mp: float, seed: int, refs) -> list:
    cmd = [str(h3.H3_DEFAULT_PYTHON), str(RUNNER), prompt, "-o", str(png), "--root", str(PACK_DIR),
           "--text-dir", str(h3.H3_COMPACT_ROOT), "--text-config", str(h3.H3_TEXT_CONFIG),
           "--aspect", "1:1" if shape == "match" else shape, "--mp", f"{mp:g}", "--seed", str(seed),
           "--metrics", str(metrics)]
    for r in refs:
        cmd += ["--reference", str(r)]
    if shape == "match":
        cmd.append("--match-reference")
    return cmd


def _run_one(cmd: list, env: dict, timeout_s: float, on_log, should_stop) -> None:
    """One picture, in a fresh process each attempt; retried when macOS's GPU watchdog stopped it."""
    last = ""
    for attempt in range(1, ATTEMPTS + 1):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                env=env, cwd=str(h3.H3_VENDOR_DIR), start_new_session=True)
        lines: "queue.Queue" = queue.Queue()
        threading.Thread(target=lambda: ([lines.put(l.rstrip()) for l in proc.stdout], lines.put(None)),
                         daemon=True).start()
        tail, started = [], time.monotonic()
        try:
            while True:
                if should_stop and should_stop():
                    raise H3ImageCancelled("stopped")
                if time.monotonic() - started > timeout_s:
                    raise RuntimeError("the picture took too long and was stopped")
                try:
                    line = lines.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line.strip():
                    tail = (tail + [line])[-10:]
                    if on_log:
                        on_log(line)
            rc = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                proc.wait(timeout=10)
        if rc == 0:
            return
        last = " | ".join(tail)[-300:]
        if attempt < ATTEMPTS and ltx._watchdog_death_ng(rc, tail):
            if on_log:
                on_log(f"macOS stopped the render (attempt {attempt}/{ATTEMPTS}); starting it again")
            continue
        break
    raise RuntimeError(f"{RESULT_NOTE} failed: {last}" if last else f"{RESULT_NOTE} failed")


def generate_h3_image_ng(prompt: str, n: int, shape: str, tier: str, output_dir, seed=None, refs=(),
                         on_log=None, should_stop=None, on_result=None) -> list:
    """Make `n` pictures (seeds seed, seed+1, ...). Returns [{png_path, seed, width, height, elapsed_seconds}], calling
    `on_result(result)` as each lands. Raises H3ImageCancelled when `should_stop()` turns true."""
    refs = [str(r) for r in refs]
    validate_ng(prompt, n, shape, tier, refs)
    health = h3_image_health_ng()
    if not health["ok"]:
        raise RuntimeError("; ".join(health["problems"]))
    seed = int(seed) if seed not in (None, "") else random.randrange(2 ** 31 - 1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = h3._clean_subprocess_env_ng()
    stamp = int(time.time() * 1000)
    results = []
    for i in range(int(n)):
        png, metrics = output_dir / f"h3_{stamp}_{i:02d}.png", output_dir / f"h3_{stamp}_{i:02d}.json"
        cmd = _command(prompt, png, metrics, shape, TIERS[tier], seed + i, refs)
        started = time.monotonic()
        _run_one(cmd, env, TIMEOUT_S[tier], on_log, should_stop)
        if not png.is_file() or png.stat().st_size == 0:
            raise RuntimeError(f"{RESULT_NOTE} finished but wrote no picture")
        info = json.loads(metrics.read_text()) if metrics.is_file() else {}
        results.append({"png_path": str(png), "seed": seed + i, "width": int(info.get("width") or 0),
                        "height": int(info.get("height") or 0), "elapsed_seconds": round(time.monotonic() - started, 1)})
        if on_result:
            on_result(results[-1])
    return results
