"""NG: discovers RefMod reference files for the H3 view.

Sibling to h3_lora_libraryNG.py: what RefMods exist on disk, whether each one
can be used, and resolving a UI-supplied name to a real path (a client-supplied
path never goes straight into a subprocess argv).

A RefMod (ComfyUI-MiniMaxH3Mod format) is a saved, already-normalised video
latent plus a JSON `refmod_meta` header; a `_reflora` file packs one as a
bundle next to a LoRA. Only the visual members are used -- audio RefMods are
refused (voice transfer does not work). Files are only listed: nothing here
reads tensor data, just the safetensors header.

Folder: `<H3_MODELS_ROOT>/refmods`, or LTX_H3_REFMODS if set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import h3_engineNG as h3
import h3_loraNG

PATCH_H = PATCH_W = 2  # the DiT's (1, 2, 2) patch: rows = T * (H/2) * (W/2)
_LORA_KEYS = (".lora_down.weight", ".lora_A.weight")


def refmods_dir_ng() -> Path:
    env = os.environ.get("LTX_H3_REFMODS")
    return Path(env).expanduser() if env else h3.H3_MODELS_ROOT / "refmods"


def _member_rows(member: dict) -> Optional[int]:
    """Packed rows one visual member costs, or None if its geometry is unusable."""
    try:
        t = int(member.get("latent_t", 1))
        h, w = int(member["latent_h"]), int(member["latent_w"])
    except (KeyError, TypeError, ValueError):
        return None
    if t < 1 or h < 2 or w < 2 or h % PATCH_H or w % PATCH_W:
        return None
    return t * (h // PATCH_H) * (w // PATCH_W)


def _entry_ng(path: Path) -> dict:
    out = {"name": path.stem, "path": str(path), "kind": "", "images": 0, "videos": 0,
           "rows": 0, "size_mb": round(path.stat().st_size / 1e6, 2), "has_lora": False,
           "description": "", "ok": False, "reason": ""}
    try:
        _hlen, header = h3_loraNG._header(path)
    except (OSError, ValueError, UnicodeDecodeError):
        out["reason"] = "not a readable safetensors file"
        return out
    meta_raw = (header.get("__metadata__") or {}).get("refmod_meta")
    if not meta_raw:
        out["reason"] = "no RefMod metadata in this file"
        return out
    try:
        meta = json.loads(meta_raw)
    except ValueError:
        out["reason"] = "RefMod metadata is not valid JSON"
        return out
    members = meta.get("members") if meta.get("kind") == "bundle" else [meta]
    members = [m for m in (members or []) if isinstance(m, dict)]
    visual = [m for m in members if m.get("kind") in ("image", "video")]
    out["kind"] = "bundle" if meta.get("kind") == "bundle" else str(meta.get("kind") or "")
    out["images"] = sum(m.get("kind") == "image" for m in visual)
    out["videos"] = sum(m.get("kind") == "video" for m in visual)
    out["has_lora"] = any(k.endswith(_LORA_KEYS) for k in header)
    out["description"] = str(meta.get("description") or "")[:200]
    if not visual:
        out["reason"] = "audio-only RefMod (voice transfer is not supported)"
        return out
    sizes = [_member_rows(m) for m in visual]
    if any(s is None for s in sizes):
        out["reason"] = "a member's latent size is not usable (needs even height and width)"
        return out
    out["rows"] = sum(sizes)
    out["ok"] = True
    return out


def list_h3_refmods_ng() -> list:
    """Every .safetensors in the RefMod folder (recursive), runnable first."""
    root = refmods_dir_ng()
    if not root.is_dir():
        return []
    entries = []
    for p in sorted(root.rglob("*.safetensors")):
        if p.is_file() and "runner-layout" not in p.parts:
            entries.append(_entry_ng(p))
    entries.sort(key=lambda e: (not e["ok"], e["name"].lower()))
    return entries


def resolve_h3_refmod_ng(name: str) -> Optional[dict]:
    """The runnable library entry for `name`, or None. Names come from the
    listing only, so a client can never name an arbitrary path."""
    for e in list_h3_refmods_ng():
        if e["ok"] and e["name"] == name:
            return e
    return None
