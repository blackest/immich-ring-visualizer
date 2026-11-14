"""immichRingNG -- the app's own folder of model weights.

`/Volumes/AI/ImmichRingModels` (or IMMICHRING_MODELS_ROOT) holds `mlx_models/` (LTX, H3, music, speech, LoRAs, characters)
and `HF_HOME/hub` (Klein's Hugging Face cache): clones of what Phosphene/Pinokio downloaded, so nothing here depends on
either of them. Each engine resolver looks here first and only then at the old Phosphene/Pinokio locations. Nothing is
created or downloaded by this module; a folder that does not exist yet simply reads as "not there".
"""

import os
from pathlib import Path
from typing import Optional

DEFAULT_ROOT = Path("/Volumes/AI/ImmichRingModels")


def models_root() -> Path:
    env = os.environ.get("IMMICHRING_MODELS_ROOT")
    return Path(env).expanduser() if env else DEFAULT_ROOT


def mlx_models() -> Optional[Path]:
    """`<root>/mlx_models` if it exists, else None."""
    p = models_root() / "mlx_models"
    return p if p.is_dir() else None


def hf_hub() -> Optional[Path]:
    """`<root>/HF_HOME/hub` if it exists, else None."""
    p = models_root() / "HF_HOME" / "hub"
    return p if p.is_dir() else None
