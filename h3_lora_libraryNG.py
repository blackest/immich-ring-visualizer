"""NG: discovers user style/character LoRAs for the H3 view.

Sibling to h3_loraNG.py (which only knows how to lay a file's qkv rows out
for the runner) and the H3 counterpart of ltx_loraNG.py: what LoRAs exist on
disk, whether each one is a layout this app can run, and resolving a
UI-supplied name to a real path (never trust a client-supplied path straight
into a subprocess argv).

The folder is `<H3_MODELS_ROOT>/loras`, the same one Phosphene imports into,
so a LoRA added there (with its `<name>.json` sidecar of trigger words and
recommended strength) shows up here too. Files are only listed -- nothing
here reads tensor data, just the safetensors header.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h3_engineNG as h3
import h3_lora_convertNG as h3_lora_convert
import h3_loraNG

H3_LORAS_DIR = h3.H3_MODELS_ROOT / "loras"

# diffusers / PEFT namespace: split attention projections and the diffusers
# feed-forward path. Its alpha is NOT stored in the file, so running it at
# strength 1.0 would be silently wrong -- refused rather than guessed.
_DIFFUSERS_MARKERS = (".to_q.", ".to_k.", ".to_v.", ".to_out.", ".ff.net.")
# kohya / sd-scripts naming carries `.alpha` scalars with the same trap.
_KOHYA_SUFFIX = ".lora_down.weight"
_LORA_A_SUFFIX = ".lora_A.weight"
DEFAULT_STRENGTH = 1.0


def _compat_ng(path: Path) -> Optional[str]:
    """None if the file is runnable, else one plain sentence saying why not."""
    try:
        _hlen, header = h3_loraNG._header(path)
    except (OSError, ValueError, UnicodeDecodeError):
        return "not a readable safetensors file"
    keys = [k for k in header if k != "__metadata__"]
    if any(k.endswith(_KOHYA_SUFFIX) for k in keys):
        return "kohya-format LoRA (alpha is stored separately) -- import it through Phosphene to convert it"
    if any(m in k for k in keys for m in _DIFFUSERS_MARKERS):
        return "diffusers/PEFT-format LoRA (alpha is not in the file) -- not runnable on H3"
    if not any(k.endswith(_LORA_A_SUFFIX) for k in keys):
        return "no LoRA weights found in this file"
    return None


def _sidecar_ng(path: Path) -> dict:
    """Trigger words / recommended strength from Phosphene's `<stem>.json`,
    when there is one. Anything unreadable just means no extras."""
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(meta, dict):
        return {}
    triggers = meta.get("trigger_words")
    strength = meta.get("recommended_strength")
    return {
        "title": meta.get("name") if isinstance(meta.get("name"), str) else None,
        "trigger_words": [t for t in triggers if isinstance(t, str)] if isinstance(triggers, list) else [],
        "recommended_strength": float(strength) if isinstance(strength, (int, float))
                                and not isinstance(strength, bool) else None,
    }


def list_h3_loras_ng() -> list[dict]:
    """Every LoRA file in the library, sorted by name. Returns [] (not an
    error) if the folder doesn't exist. Incompatible files are still listed,
    with ok=False and a reason, so the UI can say why instead of hiding them."""
    if not H3_LORAS_DIR.is_dir():
        return []
    out = []
    for p in sorted(H3_LORAS_DIR.glob("*.safetensors")):
        reason = _compat_ng(p)
        extras = _sidecar_ng(p)
        try:
            size = p.stat().st_size
        except OSError:
            continue
        out.append({
            "name": p.stem,
            "path": str(p),
            "size_mb": round(size / 1e6),
            "ok": reason is None,
            "reason": reason,
            "convertible": reason is not None and h3_lora_convert.is_kohya_ng(p),
            "title": extras.get("title"),
            "trigger_words": extras.get("trigger_words", []),
            "recommended_strength": extras.get("recommended_strength") or DEFAULT_STRENGTH,
        })
    return out


def resolve_h3_lora_path_ng(name: str) -> Optional[str]:
    """Look up a dropdown-supplied name against the real directory listing.
    Returns the path only for a runnable file this app itself found on disk,
    else None."""
    name = (name or "").strip()
    if not name:
        return None
    for entry in list_h3_loras_ng():
        if entry["name"] == name:
            return entry["path"] if entry["ok"] else None
    return None
