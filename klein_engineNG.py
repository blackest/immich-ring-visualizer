"""immichRingNG -- FLUX 3 Klein pictures (text to image, or edit from up to ten reference pictures).

A peer of hidream_engineNG.py: one small input -> output file that runs the model in a subprocess.
The worker (klein_workerNG.py) is run with the python of Phosphene's FLUX 3 install, which holds
mflux and MLX, and reads the weights from the Hugging Face cache Phosphene downloaded them into. Both
are found, never installed, here; Phosphene's own Klein code is not used, so its Update cannot change
what this does (the same reason the H3 runner is vendored).
"""

import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "klein_workerNG.py"
MAX_REFS = 10
MAX_PIXELS = 1024 * 1024
RESULT_PREFIX = "KLEIN_RESULT "
# "<variant>-q<bits>" -> the Hugging Face repo the weights come from
REPOS = {f"{variant}-q{bits}": f"mflux-community/flux-3-image-klein{'-base' if variant == 'base' else ''}-mflux-q{bits}"
         for variant in ("distilled", "base") for bits in (8, 6, 4)}
DEFAULT_MODEL = "distilled-q8"


class KleinCancelled(RuntimeError):
    pass


def _phosphene_root() -> Path:
    env = os.environ.get("PHOSPHENE_ROOT")
    return Path(env).expanduser() if env else Path("/Volumes/AI/Pinkio/api/phosphene.git")


def worker_python() -> Path:
    env = os.environ.get("PHOSPHENE_FLUX3_ROOT")
    root = Path(env).expanduser() if env else _phosphene_root() / "flux3-worker"
    return root / "env" / "bin" / "python"


def cache_roots() -> list:
    """Hugging Face caches to look in: Phosphene's own (where it downloads Klein), then the
    ones this shell's environment names, then the default."""
    env = os.environ
    found = [_phosphene_root() / "cache" / "HF_HOME" / "hub"]
    if env.get("HF_HUB_CACHE"):
        found.append(Path(env["HF_HUB_CACHE"]).expanduser())
    if env.get("HF_HOME"):
        found.append(Path(env["HF_HOME"]).expanduser() / "hub")
    found.append(Path.home() / ".cache" / "huggingface" / "hub")
    return found


def snapshot_ng(model: str):
    """The folder of the newest complete snapshot of `model` in the cache, or None."""
    repo = REPOS.get(model)
    if repo is None:
        raise ValueError(f"unknown Klein model {model!r} (one of {', '.join(REPOS)})")
    for root in cache_roots():
        snaps = root / ("models--" + repo.replace("/", "--")) / "snapshots"
        found = [p for p in snaps.iterdir() if (p / "manifest.json").is_file()
                 and (p / "transformer" / "0.safetensors").is_file()] if snaps.is_dir() else []
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
    return None


def klein_health_ng() -> dict:
    py = worker_python()
    models = [m for m in REPOS if snapshot_ng(m) is not None]
    return {"ready": py.is_file() and bool(models), "worker_python": str(py), "worker_found": py.is_file(),
            "models": models, "default": DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)}


def validate_ng(width: int, height: int, refs, n: int) -> None:
    if width < 256 or height < 256 or width % 16 or height % 16 or width * height > MAX_PIXELS:
        raise ValueError("Klein sizes are multiples of 16, at least 256, and at most 1024 x 1024 pixels")
    if len(refs) > MAX_REFS:
        raise ValueError(f"Klein takes at most {MAX_REFS} reference pictures")
    if not 1 <= n <= 16:
        raise ValueError("make 1 to 16 pictures at a time")


def _clean_env() -> dict:
    drop = {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"}
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith(("Malloc", "MFLUX_"))}
    env.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1", MLX_ENABLE_TF32="0")
    return env


def generate_klein_ng(prompt: str, n: int, width: int, height: int, output_dir, seed: int, refs=(),
                      model: str = DEFAULT_MODEL, on_log=None, should_stop=None, on_result=None) -> list:
    """Render `n` pictures (seeds seed, seed+1, ...). Returns [{png_path, seed, width, height,
    elapsed_seconds}], calling `on_result(result)` as each lands. Raises KleinCancelled when
    `should_stop()` turns true (the worker is killed)."""
    refs = [str(r) for r in refs]
    validate_ng(width, height, refs, n)
    py = worker_python()
    if not py.is_file():
        raise RuntimeError(f"the FLUX 3 worker is not installed ({py} not found): install FLUX 3 from Phosphene")
    snap = snapshot_ng(model)
    if snap is None:
        raise RuntimeError(f"the Klein weights for {model} are not in the Hugging Face cache: download them in Phosphene")
    os.makedirs(output_dir, exist_ok=True)
    request = {"model": str(snap), "variant": model.split("-")[0], "prompt": prompt,
               "seeds": [seed + i for i in range(n)], "width": width, "height": height,
               "output_dir": str(output_dir), "refs": refs}
    proc = subprocess.Popen([str(py), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", env=_clean_env(),
                            start_new_session=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: ([lines.put(l.rstrip()) for l in proc.stdout], lines.put(None)),
                     daemon=True).start()
    results, tail = [], []
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.close()
        started = time.monotonic()
        while True:
            if should_stop and should_stop():
                raise KleinCancelled("Klein stopped")
            if time.monotonic() - started > 1800 + 900 * n:
                raise RuntimeError("Klein timed out")
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith(RESULT_PREFIX):
                results.append(json.loads(line[len(RESULT_PREFIX):]))
                if on_result:
                    on_result(results[-1])
            else:
                tail = (tail + [line])[-15:]
                if on_log:
                    on_log(line)
        if proc.wait(timeout=30):
            raise RuntimeError("Klein did not finish: " + "\n".join(tail[-6:]))
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
    if len(results) != n or any(not Path(r["png_path"]).is_file() for r in results):
        raise RuntimeError("Klein returned fewer pictures than asked for; check the log")
    return results
