"""immichRingNG -- Pika H3 Flex ("Flex"), a six-forward speed option for H3.

Pika Labs' few-step version of MiniMax-H3: a rank-128 adapter plus the "re-noise" sampler it was
distilled for. The runner has the sampler (`--pika-flex`, vendored from minimax-h3-mlx `bef6d03`); the
adapter is the file Phosphene's `scripts/fetch_h3_flex.py` builds on this Mac from Pika's own release
(`models/pika-h3-flex/`), which this app finds but does not fetch. It is passed as an ordinary
`--lora PATH:1.0`; `--pika-flex` itself sets the ladder, the shifts and the step count, so the engine's
`steps` only has to agree (FLEX_STEPS = 7 knots = 6 forwards).

Flex is text or first-frame video only: Pika's reference adapter needs a separate trunk, so no RefMods
and no pictures-as-references. It does not combine with the other speed options (turbo, Faster Best,
Fast), and the re-noise sampler cannot use `--step-cache`. Licence: the adapter is a MiniMax H3 Model
Derivative (MiniMax H3 Community License, shown beside the file); Pika's code is Apache-2.0.
"""

from pathlib import Path
from typing import Optional

import h3_loraNG as h3_lora

FLEX_DIRNAME = "pika-h3-flex"
FLEX_FILE = "pika_h3_flex_fl2va_phosphene.safetensors"
FLEX_BYTES = 1956166104
FLEX_STEPS = 7  # knots including the terminal 0: six forwards
FLEX_FLAG = "--pika-flex"
FLEX_CREDIT = ("Pika H3 Flex by Pika Labs (Mellis, Inc.), running on Apple Silicon (MLX) through "
               "minimax-h3-mlx. Powered by MiniMax H3.")


def resolve_flex_ng(models_root) -> Optional[str]:
    """The built adapter at its exact size (never the 2.8 GB upstream file), or None."""
    path = Path(models_root) / FLEX_DIRNAME / FLEX_FILE
    try:
        return str(path) if path.is_file() and path.stat().st_size == FLEX_BYTES else None
    except OSError:
        return None


def runner_supports_flex_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, FLEX_FLAG)


def flex_argv_ng(adapter: str) -> list:
    """The runner argv: the sampler preset and the adapter at 1.0 (alpha == rank, nothing folded)."""
    return [FLEX_FLAG, "--lora", f"{adapter}:1.0"]


def check_flex_ng(*, turbo, v13, tristep, refmods, references, model) -> None:
    """Refuse what Flex cannot do, before anything is queued."""
    if turbo or v13 or tristep:
        raise ValueError("Flex is its own speed option -- not with turbo, Faster Best or Fast")
    if refmods or references or model == "h3ref":
        raise ValueError("Flex has no RefMod or reference-picture mode (Pika's reference adapter needs "
                         "a different base) -- use Fast for those")
