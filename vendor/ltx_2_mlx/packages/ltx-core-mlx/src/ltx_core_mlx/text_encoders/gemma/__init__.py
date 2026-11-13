"""Gemma text encoders — Gemma 3 (LTX-2.3) and Gemma 4 (LTX-2.5).

Which one a checkpoint wants is a config question, not a constant: see
``loader.detect_text_encoder``.
"""

from ltx_core_mlx.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextConfig, Gemma4TextTower
from ltx_core_mlx.text_encoders.gemma.gemma4_pack import load_gemma4_tower
from ltx_core_mlx.text_encoders.gemma.loader import (
    TextEncoderSpec,
    TextProjectionSpec,
    detect_text_encoder,
    detect_text_projection,
    load_text_encoder,
    resolve_text_tower,
)

__all__ = [
    "Embeddings1DConnector",
    "Gemma4TextConfig",
    "Gemma4TextTower",
    "GemmaFeaturesExtractorV2",
    "GemmaLanguageModel",
    "TextEncoderSpec",
    "TextProjectionSpec",
    "detect_text_encoder",
    "detect_text_projection",
    "load_gemma4_tower",
    "load_text_encoder",
    "resolve_text_tower",
]
