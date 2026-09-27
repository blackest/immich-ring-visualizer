"""Self-service install recipe for the Music (YuE2) engine -- builds the
list of engine_installNG.InstallStep the "Install now" button runs, so
music_engineNG.py's MUSIC_REPO_DIR/MUSIC_MODELS_ROOT resolve to a real
install without requiring Phosphene (see PHOSPHENE_DECOUPLING_PLAN.md for
why that dependency was already removed from the render path itself --
this closes the remaining "how do you even get yue2-mlx" gap).

Ported from Phosphene's scripts/pinokio/music_clone.sh, music_checkout.sh,
music_venv.sh, music_sync.sh, music_fetch.py (MIT licensed) -- same repo,
same pinned commit, same two Hugging Face weight sources. Trimmed relative
to the original: Phosphene's fetch does its own atomic shard-staging and a
signed receipt file on top of what huggingface_hub's own resumable
hf_hub_download already gives for free; this keeps the sha256 verification
(the part that actually catches a corrupt/truncated download) and drops
the extra bookkeeping layer, since this app only ever has one install of
this engine to manage, not a fleet of them.

Code: https://github.com/vanch007/mlx-Yue, pinned commit (matches
Phosphene's scripts/music/engine_pin.txt as of this writing).
Weights: HF vanch007/mlx-Yue2-3B (generator) + m-a-p/YuE2-Vae (decoder),
both pinned revisions -- see MUSIC_SETUP.md for the plain-CLI equivalent
of everything below.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, List

from engine_installNG import InstallStep, run_logged
from music_engineNG import MUSIC_MODELS_ROOT, MUSIC_REPO_DIR

MUSIC_CLONE_URL = "https://github.com/vanch007/mlx-Yue"
MUSIC_PIN_SHA = "9253ed133343406947bde7b67d43c7a63fb39d99"

# repo, revision, prefix, {filename: (bytes, sha256 | None)} -- sha256 is
# only pinned for the one file Phosphene's own installer pins (the VAE's
# model.safetensors); the rest are trusted via the pinned HF revision
# itself, same trust boundary hf_hub_download always gives a caller.
_GENERATOR = {
    "repo": "vanch007/mlx-Yue2-3B", "revision": "fa66d203dd56d7e033e05ee8b32269768984c4cc",
    "files": ("LICENSE", "THIRD_PARTY_NOTICES.md", "ar-8bit.safetensors", "ar-bf16.safetensors",
              "config.json", "licenses/SnakeBeta-NVIDIA-MIT.txt",
              "licenses/stable-audio-tools-MIT.txt", "nar-bf16.safetensors", "qwen.tiktoken"),
}
_VAE = {
    "repo": "m-a-p/YuE2-Vae", "revision": "95535e72a97bc0f09b8ada125d26b4009428c0e8",
    "files": ("model.safetensors", "config.json", "weights_manifest.json",
              "LICENSE", "THIRD_PARTY_NOTICES.md"),
}
_VAE_MODEL_SHA256 = "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _step_clone(on_log: Callable[[str], None]) -> None:
    dest = MUSIC_REPO_DIR
    if (dest / ".git").is_dir():
        on_log(f"{dest} already a checkout -- skipping clone.")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_logged(["git", "clone", "--no-checkout", MUSIC_CLONE_URL, str(dest)], on_log=on_log)

    run_logged(["git", "fetch", "--force", "origin", MUSIC_PIN_SHA], cwd=str(dest), on_log=on_log)
    run_logged(["git", "checkout", "--force", "--detach", "FETCH_HEAD"], cwd=str(dest), on_log=on_log)
    got = _git_head(dest)
    if got != MUSIC_PIN_SHA:
        raise RuntimeError(f"checked out {got}, expected pinned {MUSIC_PIN_SHA}")
    on_log(f"yue2-mlx pinned to {MUSIC_PIN_SHA}")


def _git_head(repo_dir: Path) -> str:
    import subprocess
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
        capture_output=True, text=True, check=True).stdout.strip()


def _venv_python(repo_dir: Path) -> Path:
    return repo_dir / ".venv" / "bin" / "python"


def _step_venv(on_log: Callable[[str], None]) -> None:
    dest = MUSIC_REPO_DIR
    py = _venv_python(dest)
    import subprocess
    healthy = py.is_file() and subprocess.run(
        [str(py), "-c", "import sys; assert sys.version_info[:2] == (3, 12)"],
    ).returncode == 0
    if healthy:
        on_log("venv already Python 3.12 -- reusing it.")
        return
    if shutil.which("uv") is None:
        raise RuntimeError(
            "`uv` not found on PATH -- install it first (https://docs.astral.sh/uv/), "
            "it's what builds and syncs this engine's venv.")
    run_logged(["uv", "venv", "--clear", "--python", "3.12", str(dest / ".venv")], on_log=on_log)


def _step_deps(on_log: Callable[[str], None]) -> None:
    dest = MUSIC_REPO_DIR
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(dest / ".venv"))
    run_logged(["uv", "sync", "--frozen", "--no-dev", "--extra", "transcription",
                "--project", str(dest)], env=env, on_log=on_log)


def _fetch_hf_files(spec: dict, dest_dir: Path, on_log: Callable[[str], None],
                     pinned_sha256: dict | None = None) -> None:
    from huggingface_hub import hf_hub_download

    pinned_sha256 = pinned_sha256 or {}
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        target = dest_dir / name
        expected = pinned_sha256.get(name)
        if target.is_file() and (expected is None or _sha256(target) == expected):
            on_log(f"keep {name} (already present)")
            continue
        on_log(f"fetch {name} from {spec['repo']} @ {spec['revision']}")
        got = Path(hf_hub_download(
            repo_id=spec["repo"], revision=spec["revision"], filename=name,
            local_dir=str(dest_dir)))
        if expected is not None and _sha256(got) != expected:
            raise RuntimeError(f"{name}: sha256 mismatch after download -- corrupt fetch")


def _step_weights(on_log: Callable[[str], None]) -> None:
    _fetch_hf_files(_GENERATOR, MUSIC_MODELS_ROOT / "generator", on_log)
    _fetch_hf_files(_VAE, MUSIC_MODELS_ROOT / "vae", on_log,
                     pinned_sha256={"model.safetensors": _VAE_MODEL_SHA256})
    on_log("YuE2 weights ready.")


def build_steps() -> List[InstallStep]:
    return [
        InstallStep("clone", _step_clone),
        InstallStep("venv", _step_venv),
        InstallStep("deps", _step_deps),
        InstallStep("weights", _step_weights),
    ]
