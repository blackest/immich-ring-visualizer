"""Fetch the LTX-2.5 weight packs Ring Visualizer's ltx_engineNG.py needs.

Trimmed, attributed adaptation of Phosphene's scripts/fetch_pack_release.py
(MIT licensed). Upstream LTX-2.5 is gated on Hugging Face and Phosphene's
own HF token is read-only, so its maintainer mirrors its own quantization
as **GitHub release assets** on a release of the public `mrbizarro/Phosphene`
repo -- a normal release-binary download, not a dependency on Phosphene the
running app (see PHOSPHENE_DECOUPLING_PLAN.md / LTX_SETUP.md).

Kept from the original: the manifest-driven shard reassembly (a GitHub
release asset caps at 2 GiB; the 11+ GB transformer ships sharded) with
per-shard AND per-file sha256 verification and Range-request resume -- this
is the part worth porting faithfully, since a truncated multi-GB download
silently producing a broken model is exactly the failure class it exists to
prevent. Dropped: Phosphene's own multi-pack registry (required_files.json)
and its own progress-sidecar/receipt bookkeeping -- this file only ever
manages the fixed 3 packs below, hardcoded, so there's nothing to look up
and no fleet of installs to reconcile.

Stdlib only, on purpose (matches the original) -- no new dependency for
what's ultimately a manifest fetch + HTTP GETs with resume.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

_RELEASE_REPO = "mrbizarro/Phosphene"
_RELEASE_TAG = "weights-ltx25-v1"
_USER_AGENT = "immichring"
_CHUNK = 1 << 20
_ATTEMPTS = 3
_TIMEOUT_S = 30

# repo-key -> (local dir name under LTX_WEIGHTS_ROOT, manifest asset name).
# q8_25 + hq_25 both land in the SAME directory -- that's intentional, it's
# how Phosphene itself assembles its "high" tier: q8_25 is the base engine
# (connector/vae/vocoder/upscalers), hq_25 adds the full-quality DiT
# (transformer-dev.safetensors) + the distilled refine LoRA
# ltx_engineNG.py's LTX_DISTILLED_LORA expects to find alongside it.
# gemma4_25 is the paired text encoder, its own separate directory.
#
# Deliberately NOT fetched here (not needed by this app's fixed "high"-tier
# CLI invocation, see ltx_engineNG.py's own docstring): `tae` (Phosphene's
# live-preview decoder -- this app has no live preview) and `ic_upscale_x2`
# (Phosphene's Remix->Upscale feature -- not wired up here either).
PACKS = {
    "q8_25": ("ltx-2.5-mlx-q8", "q8_25__phosphene_release_manifest.json"),
    "hq_25": ("ltx-2.5-mlx-q8", "hq_25__phosphene_release_manifest.json"),
    "gemma4_25": ("gemma4-12b-ltx25-q4", "gemma4_25__phosphene_release_manifest.json"),
}


def _asset_url(asset: str) -> str:
    return f"https://github.com/{_RELEASE_REPO}/releases/download/{_RELEASE_TAG}/{asset}"


def _fetch_json(url: str) -> dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            last_err = e
            if attempt < _ATTEMPTS:
                time.sleep(5 * attempt)
    raise RuntimeError(f"could not read release manifest at {url}: {last_err}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _open_ranged(url: str, offset: int):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    resp = urllib.request.urlopen(req, timeout=_TIMEOUT_S)
    return resp, bool(offset) and resp.status == 206


def _download_shard(url: str, dest: Path, expect_bytes: int, expect_sha: str,
                     label: str, on_log: Callable[[str], None]) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            have = dest.stat().st_size if dest.exists() else 0
            if have > expect_bytes:
                dest.unlink()
                have = 0
            if have < expect_bytes:
                resp, resumed = _open_ranged(url, have)
                mode = "ab" if resumed else "wb"
                if not resumed:
                    have = 0
                last_report = 0.0
                with resp, open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last_report > 2.5:
                            pct = (100.0 * have / expect_bytes) if expect_bytes else 100.0
                            on_log(f"{label}: {have >> 20}/{expect_bytes >> 20} MB ({pct:.1f}%)")
                            last_report = now
            if _sha256_file(dest) != expect_sha or dest.stat().st_size != expect_bytes:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"{label}: shard verification failed")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError) as e:
            last_err = e
            if attempt < _ATTEMPTS:
                on_log(f"{label}: attempt {attempt}/{_ATTEMPTS} failed ({e}), retrying")
                time.sleep(5 * attempt)
    raise RuntimeError(f"{label}: giving up after {_ATTEMPTS} attempts -- {last_err}")


def _assemble_file(name: str, spec: dict, dest_dir: Path, on_log: Callable[[str], None]) -> None:
    target = dest_dir / name
    want_bytes = int(spec["bytes"])
    want_sha = spec["sha256"]
    if target.exists() and target.stat().st_size == want_bytes and _sha256_file(target) == want_sha:
        on_log(f"keep {name} (already verified)")
        return

    parts_dir = dest_dir / ".immichring_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    partial = dest_dir / (name + ".partial")
    for idx, shard in enumerate(spec["shards"]):
        label = f"{name} [{idx + 1}/{len(spec['shards'])}]"
        shard_path = parts_dir / shard["asset"]
        _download_shard(_asset_url(shard["asset"]), shard_path,
                         int(shard["bytes"]), shard["sha256"], label, on_log)
        with open(partial, "ab") as out, open(shard_path, "rb") as src:
            shutil.copyfileobj(src, out, _CHUNK)
        shard_path.unlink(missing_ok=True)

    size = partial.stat().st_size
    if size != want_bytes:
        raise RuntimeError(f"{name}: assembled {size} bytes, manifest declares {want_bytes}")
    got = _sha256_file(partial)
    if got != want_sha:
        raise RuntimeError(f"{name}: assembled sha256 mismatch")
    partial.replace(target)
    on_log(f"{name}: {size >> 20} MB, sha256 verified")


def fetch_pack(repo_key: str, weights_root: Path, on_log: Callable[[str], None]) -> None:
    local_dir_name, manifest_asset = PACKS[repo_key]
    dest_dir = weights_root / local_dir_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = _asset_url(manifest_asset)
    on_log(f"{repo_key}: manifest {url}")
    manifest = _fetch_json(url)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError(f"release manifest for {repo_key!r} declares no files")
    for name, spec in files.items():
        _assemble_file(name, spec, dest_dir, on_log)
