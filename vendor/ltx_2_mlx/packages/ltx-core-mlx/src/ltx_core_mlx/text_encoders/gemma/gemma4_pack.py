"""Load a Gemma 4 text tower from the pack shape ``quantize_ltx.py`` produces.

The text encoder ships to users quantized, built by ``scripts/quantize_ltx.py``
with the ``ltx-te-gemma`` recipe. That recipe was read off the pack LTX-2.3
already ships (``mlx_models/gemma-3-12b-it-4bit``) rather than invented, so this
loader targets the same layout by construction:

* **sharded** — many ``*.safetensors`` plus ``model.safetensors.index.json``;
* the quantized set is every attention and MLP projection in the language tower
  **plus ``embed_tokens``** — an ``nn.Embedding``, which the DiT's
  ``apply_quantization`` deliberately will not touch (it only quantizes
  ``nn.Linear``), hence the dedicated predicate here;
* vision and audio towers are left unquantized and, here, never loaded at all;
* the ``quantization`` block lives in **``config.json``**, not a sidecar —
  ``quantize_ltx.py`` writes it there specifically because that is where mlx-lm
  reads it from, and this loader follows the same rule.

``group_size``/``bits`` are read from ``config.json`` when present and otherwise
derived from the tensor shapes, so a pack built at a different group size loads
without a code change — the same policy ``utils/weights.derive_quant_params``
applies to the DiT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.text_encoders.gemma.gemma4 import (
    Gemma4TextConfig,
    Gemma4TextTower,
    sanitize_weights,
)
from ltx_core_mlx.utils.weights import derive_quant_params

#: Filenames that are never model shards.
_NON_SHARD_STEMS = {"quantize_config", "split_model", "embedded_config"}


def read_config(model_path: str | Path) -> dict[str, Any]:
    """Read ``config.json`` from a text-encoder directory."""
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json at {config_path}")
    return json.loads(config_path.read_text())


def find_shards(model_path: str | Path) -> list[Path]:
    """Every weight shard in a pack, index-ordered when an index exists.

    The index is authoritative when present because ``quantize_ltx.py`` writes
    one; globbing is the fallback for a single-file or hand-assembled encoder.
    """
    path = Path(model_path)
    if path.is_file():
        return [path]

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except (json.JSONDecodeError, OSError):
            weight_map = {}
        named = {path / name for name in weight_map.values()}
        # The 2.3 pack ships a STALE index inherited from its bf16 parent,
        # naming shards that do not exist (documented in quantize_ltx.py). Keep
        # only files actually on disk, and fall through if that leaves nothing.
        present = sorted(p for p in named if p.exists())
        if present:
            return present

    shards = sorted(
        p for p in path.glob("*.safetensors") if p.stem not in _NON_SHARD_STEMS
    )
    if not shards:
        raise FileNotFoundError(f"no *.safetensors found in {path}")
    return shards


def load_shard_weights(model_path: str | Path) -> dict[str, mx.array]:
    """Load and merge every shard. Duplicate keys across shards are an error."""
    merged: dict[str, mx.array] = {}
    for shard in find_shards(model_path):
        for key, value in mx.load(str(shard)).items():
            if key in merged:
                raise ValueError(
                    f"key {key!r} appears in more than one shard of {model_path}; "
                    "the pack is inconsistent and would load ambiguously"
                )
            merged[key] = value
    return merged


def quantization_from_config(config: dict[str, Any]) -> dict[str, int] | None:
    """Read the ``quantization`` block ``quantize_ltx.py`` writes to config.json."""
    block = config.get("quantization")
    if isinstance(block, dict) and "bits" in block and "group_size" in block:
        return {"bits": int(block["bits"]), "group_size": int(block["group_size"])}
    return None


def _quantized_module_paths(weights: dict[str, mx.array]) -> set[str]:
    """Module paths carrying packed weights, identified by their ``.scales``."""
    return {key.rsplit(".scales", 1)[0] for key in weights if key.endswith(".scales")}


def _derive_from_shapes(
    model: nn.Module,
    weights: dict[str, mx.array],
    quantized: set[str],
) -> tuple[int, int] | None:
    """Derive ``(bits, group_size)`` against the model's own float shapes.

    Packed columns and scales columns together only pin down ``bits *
    group_size``; the float parameter still carries the true ``in_features``,
    which splits it. Same reasoning — and same helper — as the DiT path.
    """
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    for path in sorted(quantized):
        weight_key, scales_key = f"{path}.weight", f"{path}.scales"
        if weight_key not in weights or scales_key not in weights:
            continue
        if weight_key not in params or scales_key in params:
            continue
        try:
            return derive_quant_params(
                int(params[weight_key].shape[-1]),
                int(weights[weight_key].shape[-1]),
                int(weights[scales_key].shape[-1]),
            )
        except ValueError:
            continue
    return None


def apply_pack_quantization(
    model: Gemma4TextTower,
    weights: dict[str, mx.array],
    config: dict[str, Any] | None = None,
) -> tuple[int, int] | None:
    """Quantize exactly the modules the pack shipped quantized.

    Returns the ``(bits, group_size)`` applied, or ``None`` for a bf16 pack.

    Unlike the DiT helper this accepts ``nn.Embedding`` as well as ``nn.Linear``,
    because the ``ltx-te-gemma`` recipe quantizes ``embed_tokens``. Quantizing a
    module the pack left in bf16 — or skipping one it packed — both end in a
    shape mismatch at ``load_weights``, which is the loud outcome we want.
    """
    quantized = _quantized_module_paths(weights)
    if not quantized:
        return None

    declared = quantization_from_config(config or {})
    derived = _derive_from_shapes(model, weights, quantized)

    if derived is not None:
        bits, group_size = derived
    elif declared is not None:
        bits, group_size = declared["bits"], declared["group_size"]
    else:
        raise ValueError(
            "the pack carries quantized tensors but neither config.json nor the "
            "tensor shapes pin down (bits, group_size); refusing to guess"
        )

    def predicate(path: str, module: nn.Module) -> bool:
        return path in quantized and isinstance(module, (nn.Linear, nn.Embedding))

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return bits, group_size


def load_gemma4_tower(
    model_path: str | Path,
    config: dict[str, Any] | None = None,
    strict: bool = True,
) -> Gemma4TextTower:
    """Build and load a Gemma 4 text tower from a pack directory.

    Args:
        model_path: Directory holding ``config.json`` + shards, or a single file
            (``config`` must then be supplied).
        config: Pre-read config, if the caller already has it.
        strict: Passed to ``load_weights``. Leave it ``True``: a missing tensor
            in a text encoder is a randomly-initialised module that renders
            plausible garbage, which is the failure mode this port exists to
            refuse.

    Returns:
        A loaded, evaluated :class:`Gemma4TextTower`.
    """
    raw_config = config if config is not None else read_config(model_path)
    tower_config = Gemma4TextConfig.from_dict(raw_config)
    model = Gemma4TextTower(tower_config)

    weights = sanitize_weights(load_shard_weights(model_path), tower_config)
    apply_pack_quantization(model, weights, raw_config)

    model.load_weights(list(weights.items()), strict=strict)
    mx.eval(model.parameters())
    return model


__all__ = [
    "apply_pack_quantization",
    "find_shards",
    "load_gemma4_tower",
    "load_shard_weights",
    "quantization_from_config",
    "read_config",
]
