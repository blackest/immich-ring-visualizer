"""Self-service install recipe for the LTX (Animate) engine -- builds the
list of engine_installNG.InstallStep the "Install now" button runs, so
ltx_engineNG.py's LTX_REPO_DIR/LTX_WEIGHTS_ROOT resolve to a real install
without needing Phosphene (see PHOSPHENE_DECOUPLING_PLAN.md).

Ported from Phosphene's scripts/pinokio/ltx_checkout.sh, ltx_venv.sh, and
install.js's own `uv pip install` sequence (MIT licensed) -- same fork,
same pinned tag, same core dependency set. Trimmed relative to install.js:
this app's ltx_engineNG.py is a fixed-tier CLI wrapper (always
--two-stages-hq, no live preview, no Remix/Upscale-x2, no agent/push-
notification features), so the handful of Phosphene-panel-only packages in
its own dependency list (smolagents, litellm, pywebpush) are left out --
they exist for Phosphene's own agent tooling, not for anything
ltx_engineNG.py's subprocess calls actually import or invoke. mlx-vlm IS
kept: ltx_vision_chat_helperNG.py and its siblings run inside this same
venv and do import it.

Weights come from ltx_pack_fetchNG.py (its own module -- see that file for
why the weights specifically still come from Phosphene's own GitHub
release infrastructure, and why that isn't "depending on Phosphene").

Code: https://github.com/mrbizarro/ltx-2-mlx.git, tag v0.14.19+ltx25.7.
See LTX_SETUP.md for the plain-CLI equivalent of everything below.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, List

import ltx_pack_fetchNG
from engine_installNG import InstallStep, run_logged
from ltx_engineNG import LTX_REPO_DIR, LTX_WEIGHTS_ROOT

LTX_FORK_URL = "https://github.com/mrbizarro/ltx-2-mlx.git"
LTX_PIN_TAG = "v0.14.19+ltx25.7"

# Pinned versions matter here, not just "latest": mlx 0.31.2 measurably
# regresses LTX's own audio output by ~22 dB (see
# PHOSPHENE_DECOUPLING_PLAN.md), and ltx-core-mlx's own pyproject pins a
# Python>=3.11 floor the packages below are built against.
_CORE_PINS = ("mlx==0.31.1", "mlx-lm==0.31.1", "mlx-metal==0.31.1", "transformers>=5.0.0,<5.13.0")
_LTX_PACKAGES = ("./packages/ltx-core-mlx", "./packages/ltx-pipelines-mlx", "./packages/ltx-trainer")


def _git_head(repo_dir: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
        capture_output=True, text=True, check=True).stdout.strip()


def _step_clone(on_log: Callable[[str], None]) -> None:
    dest = LTX_REPO_DIR
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_logged(["git", "clone", LTX_FORK_URL, str(dest)], on_log=on_log)

    # Forced refspec (not the `tag` shortcut, which ignores --force) so a
    # retagged release can't leave a stale local tag pointing at the wrong
    # commit -- ltx_checkout.sh's comment documents this as a real bug it
    # fixes, not defensive-programming fluff.
    run_logged(["git", "remote", "set-url", "origin", LTX_FORK_URL], cwd=str(dest), on_log=on_log)
    run_logged(["git", "fetch", "--force", "origin",
                f"+refs/tags/{LTX_PIN_TAG}:refs/tags/{LTX_PIN_TAG}"], cwd=str(dest), on_log=on_log)
    # An app-managed checkout can carry local edits across a pin move (a
    # patch step touching tracked source, an interrupted previous install)
    # -- reset before checkout rather than let a dirty tree block it.
    run_logged(["git", "reset", "--hard", "HEAD"], cwd=str(dest), on_log=on_log)
    run_logged(["git", "checkout", LTX_PIN_TAG], cwd=str(dest), on_log=on_log)

    want = subprocess.run(["git", "rev-parse", f"{LTX_PIN_TAG}^{{commit}}"], cwd=str(dest),
                           capture_output=True, text=True, check=True).stdout.strip()
    got = _git_head(dest)
    if want != got:
        raise RuntimeError(f"HEAD is {got} but pin {LTX_PIN_TAG!r} is {want}")
    on_log(f"ltx-2-mlx pinned to {LTX_PIN_TAG} ({got[:12]})")


def _step_venv(on_log: Callable[[str], None]) -> None:
    dest = LTX_REPO_DIR
    py311 = dest / "env" / "bin" / "python3.11"
    healthy = py311.is_file() and subprocess.run(
        [str(py311), "-c", "import sys"]).returncode == 0
    if healthy:
        on_log("venv healthy -- reusing it.")
        return
    if shutil.which("uv") is None:
        raise RuntimeError(
            "`uv` not found on PATH -- install it first (https://docs.astral.sh/uv/).")
    run_logged(["uv", "venv", "--python", "3.11", "--seed", "env"], cwd=str(dest), on_log=on_log)


def _step_deps(on_log: Callable[[str], None]) -> None:
    dest = LTX_REPO_DIR
    py = str(dest / "env" / "bin" / "python")
    constraints = dest / "pip-build-constraints.txt"

    run_logged(["uv", "pip", "install", "--python", py, *_CORE_PINS], cwd=str(dest), on_log=on_log)

    base_cmd = ["uv", "pip", "install", "--python", py]
    if constraints.is_file():
        base_cmd += ["--build-constraints", str(constraints)]
    run_logged([*base_cmd, *_LTX_PACKAGES], cwd=str(dest), on_log=on_log)
    # The workspace install above links its members EDITABLE; a codec patch
    # step elsewhere in this project (if any is ever added) would otherwise
    # find real git-tracked source instead of site-packages. Reinstalling
    # --no-deps here is what install.js itself does to close that class of
    # bug -- see its own comment for the exact failure it fixes.
    run_logged([*base_cmd, "--reinstall", "--no-deps", *_LTX_PACKAGES], cwd=str(dest), on_log=on_log)

    run_logged(["uv", "pip", "install", "--python", py, "--no-deps", "mlx-vlm==0.4.4"],
               cwd=str(dest), on_log=on_log)


def _step_weights(on_log: Callable[[str], None]) -> None:
    for repo_key in ("q8_25", "hq_25", "gemma4_25"):
        ltx_pack_fetchNG.fetch_pack(repo_key, LTX_WEIGHTS_ROOT, on_log)
    on_log("LTX-2.5 weights ready.")


def build_steps() -> List[InstallStep]:
    return [
        InstallStep("clone", _step_clone),
        InstallStep("venv", _step_venv),
        InstallStep("deps", _step_deps),
        InstallStep("weights", _step_weights),
    ]
