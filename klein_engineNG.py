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
# Output size limit. Measured 2026-11-10 with a 1024 px reference: peak memory is ~13 GB per output megapixel
# (1024x576 7.8 GB / 137 s, 1536x864 17.4 GB / 216 s, 2048x1152 30.8 GB / 367 s), so 2048x1152 is a comfortable top on a 96 GB Mac.
MAX_PIXELS = int(os.environ.get("KLEIN_MAX_PIXELS", 2048 * 1152))
# A reference picture is shrunk to this long side before Klein sees it. Measured 2026-11-10: a 4032x3024 phone photo
# made a 1024x576 render 4.9x slower (41 s/step vs 8.3) and used 22 GB instead of 8 GB, with no sign it helps the likeness.
REF_MAX_SIDE = int(os.environ.get("KLEIN_REF_MAX_SIDE", 1024))
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


OPTION_KEYS = ("steps", "guidance", "match_ref", "low_ram", "vae_tile", "loras")
MAX_LORAS = 4


def validate_options_ng(options, variant: str, refs) -> dict:
    """The optional settings, checked and reduced to the ones that apply; {} when none were given."""
    options = dict(options or {})
    unknown = set(options) - set(OPTION_KEYS)
    if unknown:
        raise ValueError("unknown Klein setting: " + ", ".join(sorted(unknown)))
    out = {}
    if options.get("steps") is not None:
        if not isinstance(options["steps"], int) or not 1 <= options["steps"] <= 100:
            raise ValueError("steps is a whole number from 1 to 100")
        out["steps"] = options["steps"]
    if options.get("guidance") is not None and variant == "base":
        if not 0 <= float(options["guidance"]) <= 20:
            raise ValueError("guidance is from 0 to 20")
        out["guidance"] = float(options["guidance"])
    if options.get("match_ref") and refs:
        out["match_ref"] = True
    if options.get("low_ram") is False:
        out["low_ram"] = False
        tile = options.get("vae_tile")
        if tile is not None:
            if not isinstance(tile, int) or tile < 128 or tile % 16:
                raise ValueError("the decode tile size is a multiple of 16, at least 128")
            out["vae_tile"] = tile
    if options.get("loras"):
        import klein_lorasNG
        wanted = options["loras"]
        if not isinstance(wanted, list) or len(wanted) > MAX_LORAS:
            raise ValueError(f"at most {MAX_LORAS} LoRAs at a time")
        out["loras"] = []
        for item in wanted:
            entry = klein_lorasNG.find_lora_ng(item.get("name") if isinstance(item, dict) else None)
            if entry is None:
                raise ValueError("unknown FLUX 3 LoRA: " + str(item.get("name") if isinstance(item, dict) else item))
            scale = float(item.get("scale", entry["recommended_strength"]))
            if not 0 <= scale <= 3:
                raise ValueError("a LoRA strength is from 0 to 3")
            if entry["distilled_only"] and variant == "base":
                raise ValueError(f"{entry['title']} only works with the distilled model, not base")
            out["loras"].append({"name": entry["name"], "scale": scale})
    return out


def validate_ng(width: int, height: int, refs, n: int, match_ref: bool = False) -> None:
    if match_ref and refs:
        pass  # the first reference sets the size
    elif width < 256 or height < 256 or width % 16 or height % 16 or width * height > MAX_PIXELS:
        raise ValueError(f"Klein sizes are multiples of 16, at least 256, and at most {MAX_PIXELS / 1e6:.1f} megapixels (2048 x 1152 at most)")
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
                      model: str = DEFAULT_MODEL, on_log=None, should_stop=None, on_result=None,
                      options=None) -> list:
    """Render `n` pictures (seeds seed, seed+1, ...). Returns [{png_path, seed, width, height,
    elapsed_seconds}], calling `on_result(result)` as each lands. Raises KleinCancelled when
    `should_stop()` turns true (the worker is killed)."""
    refs = [str(r) for r in refs]
    options = validate_options_ng(options, model.split("-")[0], refs)
    validate_ng(width, height, refs, n, options.get("match_ref", False))
    py = worker_python()
    if not py.is_file():
        raise RuntimeError(f"the FLUX 3 worker is not installed ({py} not found): install FLUX 3 from Phosphene")
    snap = snapshot_ng(model)
    if snap is None:
        raise RuntimeError(f"the Klein weights for {model} are not in the Hugging Face cache: download them in Phosphene")
    os.makedirs(output_dir, exist_ok=True)
    request = {"model": str(snap), "variant": model.split("-")[0], "prompt": prompt,
               "seeds": [seed + i for i in range(n)], "width": width, "height": height,
               "output_dir": str(output_dir), "refs": refs, **options}
    if options.get("loras"):  # names were checked above; only now do they become paths
        import klein_lorasNG
        request["loras"] = [[klein_lorasNG.find_lora_ng(l["name"])["path"], l["scale"]] for l in options["loras"]]
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
