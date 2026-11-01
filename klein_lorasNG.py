"""immichRingNG -- the FLUX 3 LoRAs Klein can use.

Looks in `<Phosphene>/mlx_models/loras/flux3/`: Phosphene's own LoRA collection point, in a subfolder because
its top level holds LTX LoRAs and Phosphene lists those. (The H3 view reads a different folder,
hailuo-h3/models/loras.) A file
is listed only when its safetensors header looks like a FLUX 3 LoRA; nothing here reads tensor data.
A name from the browser is looked up in this listing -- a client-supplied path never reaches the model.
Optional `<name>.json` sidecar: name, recommended_strength, trigger_words, notes, kind.
"""

import json
import os
import struct
from pathlib import Path
from typing import Optional

SUBFOLDER = "flux3"
FLUX3_KEY_MARKERS = ("content_mode_blocks", "single_blocks", "mode_blocks")


def loras_dir() -> Path:
    env = os.environ.get("KLEIN_LORAS_DIR")
    if env:
        return Path(env).expanduser()
    from klein_engineNG import _phosphene_root
    return _phosphene_root() / "mlx_models" / "loras" / SUBFOLDER


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 64 * 1024 * 1024:
            raise ValueError("header too large")
        return json.loads(f.read(n))


def _sidecar(path: Path) -> dict:
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _entry(path: Path) -> Optional[dict]:
    try:
        header = _header(path)
        size = path.stat().st_size
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return None
    keys = [k for k in header if k != "__metadata__"]
    declared = str((header.get("__metadata__") or {}).get("ss_base_model_version", "")).lower()
    if not any(k.endswith(".lora_A.weight") for k in keys):
        return None
    if declared != "flux3" and not any(m in k for k in keys for m in FLUX3_KEY_MARKERS):
        return None
    meta = _sidecar(path)
    strength = meta.get("recommended_strength")
    triggers = meta.get("trigger_words")
    return {
        "name": path.stem,
        "path": str(path),
        "size_mb": round(size / 1e6),
        "title": meta.get("name") if isinstance(meta.get("name"), str) else path.stem,
        "kind": meta.get("kind") if isinstance(meta.get("kind"), str) else "style",
        "notes": meta.get("notes") if isinstance(meta.get("notes"), str) else "",
        "trigger_words": [t for t in triggers if isinstance(t, str)] if isinstance(triggers, list) else [],
        "recommended_strength": float(strength) if isinstance(strength, (int, float))
                                and not isinstance(strength, bool) else 1.0,
        # a LoRA that also adapts the distilled model's target time embedding cannot load on the base model
        "distilled_only": any("ref_time_in" in k for k in keys),
    }


def list_loras_ng() -> list:
    folder = loras_dir()
    if not folder.is_dir():
        return []
    found = [_entry(p) for p in sorted(folder.glob("*.safetensors"))]
    return [e for e in found if e]


def find_lora_ng(name: str) -> Optional[dict]:
    name = (name or "").strip()
    return next((e for e in list_loras_ng() if e["name"] == name), None) if name else None
