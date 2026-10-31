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

import os

import h3_loraNG as h3_lora

FLEX_DIRNAME = "pika-h3-flex"
FLEX_FILE = "pika_h3_flex_fl2va_phosphene.safetensors"
FLEX_BYTES = 1956166104
FLEX_REF2VA_FILE = "pika_h3_flex_ref2va_phosphene.safetensors"
FLEX_REF2VA_BYTES = 1956166152  # scripts/build_pika_ref2va_adapter.py's result (the FL2VA one plus 48 bytes of notice text)
FLEX_REF2VA_ENV = "RINGVIZ_H3_FLEX_REF2VA"
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


def resolve_flex_ref2va_ng(models_root) -> Optional[str]:
    """The Ref2VA adapter (scripts/build_pika_ref2va_adapter.py) at its exact size, or None: the env override, then
    the H3 models folder's pika-h3-flex/, then ~/models/h3-ref2va/pika-flex-ref2va/."""
    paths = []
    if os.environ.get(FLEX_REF2VA_ENV):
        paths.append(Path(os.environ[FLEX_REF2VA_ENV]).expanduser())
    paths.append(Path(models_root) / FLEX_DIRNAME / FLEX_REF2VA_FILE)
    paths.append(Path.home() / "models" / "h3-ref2va" / "pika-flex-ref2va" / FLEX_REF2VA_FILE)
    for path in paths:
        try:
            if path.is_file() and path.stat().st_size == FLEX_REF2VA_BYTES:
                return str(path)
        except OSError:
            continue
    return None


def resolve_flex_for_model_ng(models_root, model) -> Optional[str]:
    """The adapter that fits `model`'s trunk: Ref2VA's for "h3ref", FL2VA's for the others. The two are not
    interchangeable (Pika trained each on its own trunk)."""
    return resolve_flex_ref2va_ng(models_root) if model == "h3ref" else resolve_flex_ng(models_root)


def runner_supports_flex_ng(runner) -> bool:
    return h3_lora.runner_has_flag_ng(runner, FLEX_FLAG)


def flex_argv_ng(adapter: str) -> list:
    """The runner argv: the sampler preset and the adapter at 1.0 (alpha == rank, nothing folded)."""
    return [FLEX_FLAG, "--lora", f"{adapter}:1.0"]


def check_flex_ng(*, turbo, v13, tristep, refmods, references, model) -> None:
    """Refuse what Flex cannot do, before anything is queued."""
    if turbo or v13 or tristep:
        raise ValueError("Flex is its own speed option -- not with turbo, Faster Best or Fast")
    if (refmods or references) and model != "h3ref":
        raise ValueError("Flex takes RefMods and reference pictures only on the Ref2VA model (H3 Ref2VA Q8) -- "
                         "pick that model, or use Fast")
