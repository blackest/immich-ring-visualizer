"""Config-keyed text-encoder loading — Gemma 3 (LTX-2.3) or Gemma 4 (LTX-2.5).

LTX-2.3 encodes prompts with **stock, ungated** ``mlx-community/gemma-3-12b-it-4bit``.
LTX-2.5 replaces it with a **custom 26 GB Gemma 4 12B fine-tune** that Lightricks
ships inside its own gated repo, with the dual-linear text projection relocated
*into* the text-encoder file ("with-proj") instead of living in the connector.

Two consequences the rest of the library has to stop assuming:

1. ``model_type`` is no longer ``gemma3``. The 2.5 encoder declares
   ``gemma4_unified`` — a multimodal Gemma 4 of which we want the **text tower
   only** (its vision and audio towers must never be loaded).
2. The projection's input width is no longer the hardcoded ``3840 * 49``. It is
   ``hidden_size * (num_hidden_layers + 1)``, derived from the encoder's own
   config, exactly as the merged ComfyUI implementation derives it
   (``comfy/text_encoders/lt.py``, commit 57ce8e1a). For Gemma 4 12B the numbers
   happen to land on the same 3840 x 49, which is a coincidence worth *not*
   relying on: the moment a checkpoint moves, a hardcoded 188160 loads a
   correctly-shaped, wrongly-strided projection and produces plausible garbage.

This module is the seam. It does not itself implement a Gemma 4 tower — see
:func:`resolve_text_tower`, which fails loud and actionable when the installed
runtime cannot build one, rather than silently falling back to Gemma 3 and
encoding every 2.5 prompt with the wrong text encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

# Checkpoint ``model_type`` -> the mlx-lm architecture module able to build its
# TEXT tower. ``gemma4_unified`` is the encoder-free multimodal packaging; the
# text tower inside it is a plain Gemma 4, so it resolves to the same module.
MODEL_TYPE_ALIASES: dict[str, str] = {
    "gemma3": "gemma3",
    "gemma3_text": "gemma3_text",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
    "gemma4_unified": "gemma4",
}

# Towers we can build with the pinned runtime today. Kept explicit rather than
# probed-and-hoped: a wrong text encoder is not a crash, it is a quality
# regression nobody can attribute.
_KNOWN_GEMMA3 = {"gemma3", "gemma3_text"}


@dataclass(frozen=True)
class TextProjectionSpec:
    """Shape of the text projection, detected from the weights themselves.

    Mirrors ``comfy.text_encoders.lt.sd_detect``: the projection type, both
    output widths and both bias flags are read off the state dict rather than
    assumed, so a checkpoint that moves any of them loads without a code edit.
    """

    projection_type: str  # "dual_linear" | "single_linear"
    video_dim: int
    audio_dim: int | None = None
    video_bias: bool = True
    audio_bias: bool = True
    input_dim: int | None = None

    @property
    def is_dual(self) -> bool:
        return self.projection_type == "dual_linear"


@dataclass(frozen=True)
class TextEncoderSpec:
    """Everything the loader needs to build the right encoder for a checkpoint."""

    model_type: str
    """Raw ``model_type`` as declared by the encoder's config.json."""
    architecture: str
    """Resolved mlx-lm architecture (``gemma3`` / ``gemma4``)."""
    hidden_size: int
    num_hidden_layers: int
    generation: tuple[int, int]
    """(2, 3) for the Gemma 3 lineage, (2, 5) for the Gemma 4 one."""

    @property
    def num_gemma_layers(self) -> int:
        """Hidden states per token: the embedding output plus every layer."""
        return self.num_hidden_layers + 1

    @property
    def projection_input_dim(self) -> int:
        """``hidden_size * (num_hidden_layers + 1)`` — never a hardcoded 188160."""
        return self.hidden_size * self.num_gemma_layers


def detect_text_projection(state_dict_keys, prefix: str = "") -> TextProjectionSpec | None:
    """Detect the text projection's shape from checkpoint keys.

    ``state_dict_keys`` may be a mapping (shapes are then read from it) or a
    bare iterable of names (shapes come back ``None``/defaults).

    Returns ``None`` when no projection is present — which is the honest answer
    for a bare Gemma tower with the projection still living in the connector.
    """
    shapes: dict = state_dict_keys if hasattr(state_dict_keys, "get") else {}
    keys = set(state_dict_keys)

    def shape_of(key):
        value = shapes.get(key)
        return tuple(value.shape) if value is not None and hasattr(value, "shape") else None

    video_key = f"{prefix}text_embedding_projection.video_aggregate_embed.weight"
    audio_key = f"{prefix}text_embedding_projection.audio_aggregate_embed.weight"
    if video_key in keys and audio_key in keys:
        v_shape = shape_of(video_key)
        a_shape = shape_of(audio_key)
        return TextProjectionSpec(
            projection_type="dual_linear",
            video_dim=v_shape[0] if v_shape else 4096,
            audio_dim=a_shape[0] if a_shape else 2048,
            video_bias=f"{prefix}text_embedding_projection.video_aggregate_embed.bias" in keys,
            audio_bias=f"{prefix}text_embedding_projection.audio_aggregate_embed.bias" in keys,
            input_dim=v_shape[1] if v_shape else None,
        )

    for key in (
        f"{prefix}text_embedding_projection.weight",
        f"{prefix}text_embedding_projection.aggregate_embed.weight",
    ):
        if key in keys:
            shape = shape_of(key)
            return TextProjectionSpec(
                projection_type="single_linear",
                video_dim=shape[0] if shape else 3840,
                audio_dim=None,
                video_bias=key.removesuffix("weight") + "bias" in keys,
                input_dim=shape[1] if shape else None,
            )
    return None


def read_encoder_config(model_path) -> dict:
    """Read an mlx text-encoder directory's ``config.json``. ``{}`` if absent."""
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _text_config(config: dict) -> dict:
    """Gemma 4's multimodal configs nest the text tower under ``text_config``."""
    for key in ("text_config", "language_model_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and "hidden_size" in nested:
            return nested
    return config


def detect_text_encoder(model_path) -> TextEncoderSpec:
    """Build a :class:`TextEncoderSpec` from an encoder directory.

    Unknown ``model_type`` values fall through to the Gemma 3 lineage with the
    dims read from the config, which is the conservative choice: the dims are
    what the projection depends on, and they are read, not guessed.
    """
    config = read_encoder_config(model_path)
    model_type = str(config.get("model_type", "gemma3"))
    text = _text_config(config)

    architecture = MODEL_TYPE_ALIASES.get(model_type, model_type)
    generation = (2, 5) if architecture.startswith("gemma4") else (2, 3)

    return TextEncoderSpec(
        model_type=model_type,
        architecture=architecture,
        hidden_size=int(text.get("hidden_size", 3840)),
        num_hidden_layers=int(text.get("num_hidden_layers", 48)),
        generation=generation,
    )


def resolve_text_tower(spec: TextEncoderSpec) -> str:
    """Return the architecture to build, or raise with a fix.

    Deliberately loud. Silently loading a Gemma 3 tower for a checkpoint that
    declares Gemma 4 would produce a model that runs, renders, and is subtly
    wrong in a way no user could diagnose — the single worst outcome available
    here. A clear failure that names the missing piece is strictly better.

    Gemma 4 resolves to the **vendored** tower in
    :mod:`ltx_core_mlx.text_encoders.gemma.gemma4`, not to ``mlx_lm.models``.
    That is not a stylistic choice: no mlx-lm release provides a correct gemma4
    at the ``mlx`` version this project pins. See that module's docstring for
    the version matrix.
    """
    if spec.architecture in _KNOWN_GEMMA3:
        return spec.architecture

    if spec.architecture == "gemma4":
        try:
            import importlib

            importlib.import_module("ltx_core_mlx.text_encoders.gemma.gemma4")
        except ImportError as exc:  # pragma: no cover — vendored, always present
            raise NotImplementedError(
                "This checkpoint's text encoder declares "
                f"model_type={spec.model_type!r} (Gemma 4), and the vendored "
                "Gemma 4 tower could not be imported. Falling back to the Gemma 3 "
                "tower would encode every prompt with the wrong text encoder and "
                "degrade quality silently, so this refuses instead."
            ) from exc
        return "gemma4"

    raise NotImplementedError(
        f"Unsupported text-encoder model_type {spec.model_type!r}. Known types: "
        f"{sorted(MODEL_TYPE_ALIASES)}."
    )


def load_text_encoder(model_path, spec: TextEncoderSpec | None = None):
    """Load the text tower for ``model_path``, whichever generation it is.

    Returns a loaded :class:`~ltx_core_mlx.text_encoders.gemma.encoders.base_encoder.GemmaLanguageModel`.
    Gemma 3 takes exactly the path it took before this module existed.
    """
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    spec = spec or detect_text_encoder(model_path)
    architecture = resolve_text_tower(spec)  # raises before any weights are touched

    encoder = GemmaLanguageModel(model_path, architecture=architecture)
    encoder.load()
    return encoder


def feature_extractor_kwargs(
    spec: TextEncoderSpec,
    projection: TextProjectionSpec | None = None,
) -> dict:
    """Build the kwargs that size the connector to this encoder.

    ``caption_channels`` and ``num_gemma_layers`` come from the encoder config;
    the video/audio widths come from the projection weights when available.
    Together these replace the hardcoded ``3840 * 49 -> 4096 / 2048``.
    """
    kwargs = {
        "caption_channels": spec.hidden_size,
        "num_gemma_layers": spec.num_gemma_layers,
    }
    if projection is not None:
        kwargs["video_dim"] = projection.video_dim
        if projection.audio_dim is not None:
            kwargs["audio_dim"] = projection.audio_dim
    return kwargs


def detect_projection_from_file(path, prefix: str = "") -> TextProjectionSpec | None:
    """Detect the projection by reading a safetensors file's tensor shapes."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        weights = mx.load(str(path))
    except Exception:  # noqa: BLE001 — an unreadable file is "no projection"
        return None
    return detect_text_projection(weights, prefix=prefix)


__all__ = [
    "MODEL_TYPE_ALIASES",
    "TextEncoderSpec",
    "TextProjectionSpec",
    "detect_projection_from_file",
    "detect_text_encoder",
    "detect_text_projection",
    "feature_extractor_kwargs",
    "load_text_encoder",
    "read_encoder_config",
    "resolve_text_tower",
]
