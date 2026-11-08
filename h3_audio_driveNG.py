"""immichRingNG -- H3 audio drive ("lip-sync"): a soundtrack the picture is generated to.

The vendored runner's `--audio-drive AUDIO` holds the given audio in the target audio rows, so the video is
generated to it -- a face mouths the words, a band plays the song. Each 5 s window gets its own slice, so a
chained clip is not limited to one window. The delivered audio is that same file, never the VAE round trip.

Measured here (2026-11-08, 640x384, 16 steps): a speech WAV drove a clear, articulated mouth that stopped
when the voice stopped, with and without a character LoRA (the LoRA's own reference rows are not used; its
weights are -- see h3_loraNG). Slow: about 19-20 min per 5 s at that size, because the audio rows join the
packed sequence.

A RefMod (or Ref2VA reference) CAN drive a face: the runner's refusal was lifted 2026-11-08 after a test (the driven
audio is a frozen target, separate from the RefMod's condition rows; the mouth followed the words and the identity
held with no LoRA and no first frame). What it does not combine with: `--extend-from` ("extend, then lip-sync"). A first frame, a last frame, keyframes, LoRAs and
speed adapters are not refused by the runner; speed adapters with audio drive are untested.

This module only checks the request and builds the argv; the engine and job queue call it.
"""

from pathlib import Path
from typing import Optional

import h3_loraNG as h3_lora

AUDIO_DRIVE_FLAG = "--audio-drive"
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
MAX_AUDIO_BYTES = 200 * 1024 * 1024  # a clip this long is minutes of audio, far past any usable render


def clean_ext_ng(ext: str) -> str:
    """The upload's extension if it is an audio type we accept, else '.wav' (ffmpeg sniffs the real format)."""
    ext = (ext or "").strip().lower()
    return ext if ext in AUDIO_EXTS else ".wav"


def check_audio_drive_ng(audio_drive_path: Optional[str], *, refmods=None, references=None,
                         extend_from=None, runner=None, check_file: bool = True) -> None:
    """Refuse a combination the runner cannot do, before the model loads. No-op when no audio is given."""
    if audio_drive_path is None:
        return
    if extend_from is not None:
        raise ValueError("audio drive does not combine with extend -- extend first, then lip-sync")
    if check_file and not Path(audio_drive_path).is_file():
        raise FileNotFoundError(f"H3 audio-drive file not found at {audio_drive_path}")
    if runner is not None and not h3_lora.runner_has_flag_ng(runner, AUDIO_DRIVE_FLAG):
        raise ValueError("this H3 runner has no --audio-drive -- update the minimax-h3-mlx runner")


def audio_drive_argv_ng(audio_drive_path: Optional[str]) -> list:
    return [AUDIO_DRIVE_FLAG, str(audio_drive_path)] if audio_drive_path else []
