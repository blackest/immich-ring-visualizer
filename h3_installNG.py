"""Self-service install recipe for the H3 (MiniMax) engine -- builds the
list of engine_installNG.InstallStep the "Install now" button runs, so
h3_engineNG.py's H3_REPO_DIR/H3_MODELS_ROOT resolve to a real install.

Ported from Phosphene's scripts/pinokio/h3_checkout.sh + install_h3.js's
venv/deps steps (MIT licensed). Unlike LTX and Music, the weights fetch
itself is NOT ported -- it shells out to the freshly-cloned repo's OWN
scripts/download_selected.py, which is upstream MiniMax-H3-MLX code, not
Phosphene's. What IS ported is the step Phosphene calls SEPARATELY
afterward and easy to miss: h3_engineNG.py's default model ("h3q8", the
AUTO default since Phosphene v4.8.0) needs a locally quantized DiT
(H3_DIT_Q8_DIR) that download_selected.py never produces -- it's built on
this machine from the downloaded bf16 DiT via the repo's OWN
scripts/quantize_stream.py, then validated by actually loading it in a
fresh subprocess (mirroring h3_build_q8.sh's own reasoning: the quantizer's
own process can SIGKILL on a 36 GB Mac after writing valid shards, so
"shards exist" and "shards were proven loadable" are marked separately).
Skipping this step would leave a real download looking complete while
h3_health_ng() still reports dit_ok=False.

H3 IS AN EXPLICIT OPT-IN, never bundled into a default "set everything up"
flow: ~75 GB, a 36 GB RAM floor (see ram_ok below, checked by the route
before this job is even started), and MiniMax's own Community License
(territory restrictions) -- the frontend only starts this after a license
checkbox is ticked (see routes/h3NG.py's confirm_license requirement).

Code: https://github.com/mrbizarro/minimax-h3-mlx.git, pinned to the exact
engine commit Phosphene validates each release against (H3_PIN_SHA, reached
via the ref H3_PIN_REF -- both copied from Phosphene's h3_checkout.sh, THE
one place its pin lives; bump them together). The venv's packages install
under h3_constraints.txt (a copy of Phosphene's scripts/pinokio/
h3_constraints.txt, MIT) via `uv pip install -c`, because the runner's own
requirements.txt lines are only floors and a new PyPI release once broke
every fresh install. See H3_SETUP.md for the plain-CLI equivalent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from engine_installNG import InstallStep, run_logged
from h3_engineNG import H3_DIT_BF16, H3_DIT_Q8_DIR, H3_MODELS_ROOT, H3_REPO_DIR

H3_URL = "https://github.com/mrbizarro/minimax-h3-mlx.git"
H3_PIN_SHA = "3f6c75c09ccfa336a8c211b787d6a2350d83ef7a"  # Phosphene v4.19.x engine pin
H3_PIN_REF = "phosphene/v4.19.0"  # the ref on origin that keeps the pin reachable
H3_LOCAL_BRANCH = "phosphene-pinned"  # same local branch name Phosphene uses
H3_CONSTRAINTS = Path(__file__).resolve().parent / "h3_constraints.txt"
H3_MIN_RAM_BYTES = 36_000_000_000  # same floor as Phosphene's h3_preflight.sh


def ram_ok() -> Tuple[bool, Optional[str]]:
    """H3's own memory floor -- checked separately from (and in addition
    to) engine_installNG.mac_apple_silicon_ok, since even a qualifying
    Apple Silicon Mac can be too small for even the reduced-RAM Q8 engine.
    Fail-open (returns ok=True) if hw.memsize can't be read, same
    reasoning as Phosphene's own preflight: a check that can't read the
    hardware must never be what blocks an otherwise-fine install."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                              capture_output=True, text=True, timeout=5)
        mem_bytes = int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return True, None
    if mem_bytes < H3_MIN_RAM_BYTES:
        gb = mem_bytes // 1_000_000_000
        return False, (
            f"Hailuo H3 needs at least a 36 GB Mac (this Mac has {gb} GB) -- "
            f"even the reduced-RAM Q8 engine peaks around 25.6 GiB while "
            f"rendering. Nothing to install here.")
    return True, None


def _git_head(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
        capture_output=True, text=True, check=True).stdout.strip()


def _step_clone(on_log: Callable[[str], None]) -> None:
    dest = H3_REPO_DIR
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_logged(["git", "clone", H3_URL, str(dest)], on_log=on_log)

    run_logged(["git", "remote", "set-url", "origin", H3_URL], cwd=str(dest), on_log=on_log)
    def have_pin() -> bool:
        return subprocess.run(["git", "cat-file", "-e", f"{H3_PIN_SHA}^{{commit}}"],
                              cwd=str(dest), capture_output=True).returncode == 0

    # The ref first (one round trip, keeps the pin reachable locally); the
    # bare commit as a fallback if the ref was renamed upstream.
    for ref in (H3_PIN_REF, H3_PIN_SHA):
        if have_pin():
            break
        subprocess.run(["git", "fetch", "--force", "origin", ref],
                       cwd=str(dest), capture_output=True)
    if not have_pin():
        raise RuntimeError(
            f"could not fetch the pinned H3 engine commit {H3_PIN_SHA[:7]} from {H3_URL}")
    if (dest / "minimax_h3_mlx").is_dir():
        run_logged(["git", "reset", "--hard", "HEAD"], cwd=str(dest), on_log=on_log)
    run_logged(["git", "checkout", "--force", "-B", H3_LOCAL_BRANCH, H3_PIN_SHA],
               cwd=str(dest), on_log=on_log)
    if not (dest / "scripts" / "quantize_stream.py").is_file():
        raise RuntimeError(
            f"the pinned H3 engine {H3_PIN_SHA[:7]} has no scripts/quantize_stream.py -- "
            f"the Q8 build step cannot run from this tree.")
    on_log(f"minimax-h3-mlx pinned at {_git_head(dest)[:12]}")


def _venv_python(repo_dir: Path) -> Path:
    return repo_dir / ".venv" / "bin" / "python"


def _step_venv(on_log: Callable[[str], None]) -> None:
    dest = H3_REPO_DIR
    py = _venv_python(dest)
    healthy = py.is_file() and subprocess.run([str(py), "-c", "import sys"]).returncode == 0
    if healthy:
        on_log("venv healthy -- reusing it.")
        return
    if shutil.which("uv") is None:
        raise RuntimeError(
            "`uv` not found on PATH -- install it first (https://docs.astral.sh/uv/).")
    run_logged(["uv", "venv", "--python", "3.11", "--seed", ".venv"], cwd=str(dest), on_log=on_log)


def _step_deps(on_log: Callable[[str], None]) -> None:
    dest = H3_REPO_DIR
    py = str(_venv_python(dest))
    # -c: the constraints file only forbids a resolve from leaving the
    # known-good set (and moves a drifted package back); it installs nothing.
    run_logged(["uv", "pip", "install", "--python", py, "-c", str(H3_CONSTRAINTS),
                "-r", "requirements.txt"], cwd=str(dest), on_log=on_log)


def _step_weights(on_log: Callable[[str], None]) -> None:
    dest = H3_REPO_DIR
    py = str(_venv_python(dest))
    # download_selected.py appends "models/" to --root itself, and
    # H3_MODELS_ROOT already IS that "models/" directory (see
    # h3_engineNG.py's _resolve_h3_models_root_ng) -- so its parent is
    # what this flag actually wants.
    run_logged([py, "scripts/download_selected.py", "--root", str(H3_MODELS_ROOT.parent)],
               cwd=str(dest), on_log=on_log)


def _step_build_q8(on_log: Callable[[str], None]) -> None:
    dest = H3_REPO_DIR
    py = str(_venv_python(dest))
    pack = H3_DIT_Q8_DIR
    marker = pack / ".built_ok"

    if marker.is_file():
        on_log("Q8 engine already built and validated -- skipping.")
        return
    if not H3_DIT_BF16.is_file():
        raise RuntimeError(f"bf16 DiT not found at {H3_DIT_BF16} -- weights step must run first.")

    on_log("Building the reduced-RAM Q8 engine (~5 min, one time)...")
    run_logged([py, "scripts/quantize_stream.py", "--src", str(H3_DIT_BF16), "--out", str(pack)],
               cwd=str(dest), on_log=on_log)

    # Validated in a FRESH process, same reasoning as h3_build_q8.sh: the
    # quantizer's own process can hold the bf16 source mmap AND the Q8
    # intermediates at once and get SIGKILLed on a 36 GB Mac right after
    # writing every shard correctly -- a clean load_dit() afterward, with
    # nothing else resident, is the only thing that actually proves the
    # pack works. Only mark .built_ok once this succeeds.
    on_log("Validating the Q8 engine in a fresh process...")
    run_logged([py, "-c",
                "import sys; from minimax_h3_mlx.load import load_dit; "
                f"load_dit({str(pack)!r}, verbose=False); print('LOAD OK')"],
               cwd=str(dest), on_log=on_log)
    marker.touch()
    on_log("Q8 engine built and validated.")


def build_steps() -> List[InstallStep]:
    return [
        InstallStep("clone", _step_clone),
        InstallStep("venv", _step_venv),
        InstallStep("deps", _step_deps),
        InstallStep("weights", _step_weights),
        InstallStep("build_q8", _step_build_q8),
    ]
