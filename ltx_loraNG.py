"""NG: discovers optional style/character LoRA checkpoints for the
Animate (LTX) workflow.

Sibling to ltx_engineNG.py rather than folded into it -- that file only
owns turning an already-resolved path + strength into the CLI's --lora
flag. This owns the other half: what LoRAs exist on disk, and resolving
a UI-supplied name to a real path (never trust a client-supplied path
straight into a subprocess argv).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ltx_engineNG import LTX_WEIGHTS_ROOT

LTX_LORAS_DIR = LTX_WEIGHTS_ROOT / "loras"

# Sits alongside the real video-LoRA checkpoints in this folder but isn't
# one -- a voice-clone LoRA pair (*.audio.safetensors + *.voice.mp3).
_SKIP_SUFFIXES = (".audio.safetensors",)


def list_ltx_loras_ng() -> list[dict]:
    """Every video LoRA checkpoint available to the Animate workflow,
    sorted by name. Returns [] (not an error) if the folder doesn't
    exist -- the dropdown just stays empty, same as any other missing
    optional asset in this app."""
    if not LTX_LORAS_DIR.is_dir():
        return []
    out = []
    for p in sorted(LTX_LORAS_DIR.glob("*.safetensors")):
        if p.name.endswith(_SKIP_SUFFIXES):
            continue
        out.append({"name": p.stem, "path": str(p)})
    return out


def resolve_ltx_lora_path_ng(name: str) -> Optional[str]:
    """Look up a dropdown-supplied name against the real directory
    listing rather than trusting a client-supplied path outright --
    the result goes straight into ltx_engineNG's subprocess argv, so
    only ever hand back a path this app itself found on disk."""
    name = (name or "").strip()
    if not name:
        return None
    for entry in list_ltx_loras_ng():
        if entry["name"] == name:
            return entry["path"]
    return None
