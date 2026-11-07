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

# Flex Turbo: Pika's own four-forward LoRA (Pika-Labs/Pika-H3-Flex-Turbo-LoRA, formerly akhaliq's), Flex fused with
# LightX2V's Turbo distill, FL2VA trunk only. Run as published: euler over five knots, shifts 6/3, the file at 1.0, and
# the runner re-orders its q/k/v and fc1 halves at load. The recipe is the repo's own; the file is Phosphene's pinned
# revision (dcacd93a), so the size and sha256 match what its H3 Flex Turbo picker installs.
FLEX_TURBO_FILE = "pika_h3_flex_turbo4_fl2va_lite_r32.safetensors"
FLEX_TURBO_BYTES = 310189392
FLEX_TURBO_SHA256 = "7c60a2ff31f7f56c1f59020c2441fa0532c7b3fd8a510fbd0f806327021f5367"
FLEX_TURBO_URL = ("https://huggingface.co/akhaliq/Pika-H3-Flex-Turbo-LoRA/resolve/"
                  "dcacd93a50f714ba1cd61c7ca2cc5f65e9a1ec60/pika_h3_flex_turbo_4step_fl2va_lite_r32.safetensors")
FLEX_TURBO_KNOTS = "1000,750,500,250,0"
FLEX_TURBO_STEPS = 5  # knots including the terminal 0: four forwards
FLEX_TURBO_VIDEO_SHIFT = "6"
FLEX_TURBO_AUDIO_SHIFT = "3"
FLEX_TURBO_RUNNER_FLAGS = ("--sigma-knots", "--video-shift", "--audio-shift", "--lora-qkv-permute", "--lora-fc1-swap")
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


def resolve_flex_turbo_ng(models_root) -> Optional[str]:
    """The Turbo LoRA at its exact size beside the Flex adapter, or None."""
    path = Path(models_root) / FLEX_DIRNAME / FLEX_TURBO_FILE
    try:
        return str(path) if path.is_file() and path.stat().st_size == FLEX_TURBO_BYTES else None
    except OSError:
        return None


def runner_supports_flex_turbo_ng(runner) -> bool:
    return all(h3_lora.runner_has_flag_ng(runner, flag) for flag in FLEX_TURBO_RUNNER_FLAGS)


def flex_turbo_argv_ng(adapter: str) -> list:
    """The runner argv: the sampler, the ladder, the shifts, and the adapter at 1.0 with its layout flags. The step
    count (FLEX_TURBO_STEPS) is passed by the engine like every other option's."""
    return ["--sampler", "euler", "--sigma-knots", FLEX_TURBO_KNOTS,
            "--video-shift", FLEX_TURBO_VIDEO_SHIFT, "--audio-shift", FLEX_TURBO_AUDIO_SHIFT,
            "--lora", f"{adapter}:1.0", "--lora-qkv-permute", "--lora-fc1-swap"]


def check_flex_turbo_ng(*, flex, turbo, v13, tristep, refmods, references, model) -> None:
    """Refuse what Flex Turbo cannot do, before anything is queued: it is FL2VA only (text or first frame)."""
    if flex:
        raise ValueError("Flex and Flex Turbo are separate speed options -- pick one")
    if turbo or v13 or tristep:
        raise ValueError("Flex Turbo is its own speed option -- not with turbo, Faster Best or Fast")
    if model == "h3ref" or refmods or references:
        raise ValueError("Flex Turbo is the FL2VA adapter: text or a first frame only, no RefMods or reference "
                         "pictures -- pick H3 (not Ref2VA), or use Flex")


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
