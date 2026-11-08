"""immichRingNG -- cut one voice take into one piece per line, at its pauses.

Why: cloning each line separately gave three different voices from one reference (median pitch 113, 92 and
115 Hz on the knight clips). One Speech pass over all of a character's lines keeps one voice and one delivery;
this cuts that single take back into per-line audio so each piece can drive its own clip.

The text-to-speech leaves a pause at the end of each sentence (0.3-0.55 s in the measured take) and almost none at
commas (about 0.06 s), so the cuts are the (N-1) longest quiet gaps. If the take has fewer clear gaps than the lines
need, the script is too run-together and we say so instead of cutting through a word.

numpy only; WAV in and out through the stdlib `wave` module (the Speech engine writes plain PCM WAV).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import numpy as np

HOP_S = 0.01             # envelope resolution
MIN_GAP_S = 0.20         # a gap shorter than this is not a sentence break
PAD_S = 0.12             # kept either side of a cut so a word is never clipped
SILENCE_REL = 0.04       # a frame is quiet when its level is under this fraction of the loud (95th percentile) level
# A clone often leaves a breathy hiss in its pauses (measured: ~5-15% of the loud level), above the strict level.
# cut_points_ng tries these in turn and uses the first that finds enough pauses.
SILENCE_STEPS = (0.04, 0.08, 0.12, 0.18)
WINDOW_S = 5.0           # the clip length a piece has to fit


class SceneAudioError(ValueError):
    pass


def read_wav_ng(path) -> tuple:
    """(mono float32 samples in -1..1, sample rate). 16-bit PCM only, which is what the Speech engine writes."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SceneAudioError("expected 16-bit PCM audio")
        sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav_ng(path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _envelope(x: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(sr * HOP_S))
    n = len(x) // hop
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)


def find_gaps_ng(x: np.ndarray, sr: int, rel: float = SILENCE_REL) -> list:
    """[(start_s, end_s)] of interior quiet stretches of at least MIN_GAP_S, in time order. `rel` is the quiet level as
    a fraction of the loud (95th percentile) level."""
    env = _envelope(x, sr)
    if len(env) == 0:
        return []
    quiet = env < max(float(np.percentile(env, 95)) * rel, 1e-4)
    gaps, i, n = [], 0, len(env)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if i > 0 and j < n and (j - i) * HOP_S >= MIN_GAP_S:      # interior only: not the lead-in or the tail
                gaps.append((i * HOP_S, j * HOP_S))
            i = j
        else:
            i += 1
    return gaps


def cut_points_ng(x: np.ndarray, sr: int, pieces: int) -> list:
    """The `pieces`-1 cut times (centre of the longest quiet gaps), in time order."""
    if pieces <= 1:
        return []
    gaps: list = []
    for rel in SILENCE_STEPS:
        gaps = find_gaps_ng(x, sr, rel)
        if len(gaps) >= pieces - 1:
            break
    if len(gaps) < pieces - 1:
        raise SceneAudioError(
            f"found {len(gaps)} clear pause(s) in the voice but {pieces - 1} are needed between {pieces} lines -- "
            f"end each line with a full stop so there is a breath between them")
    longest = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[: pieces - 1]
    return sorted((a + b) / 2.0 for a, b in longest)


def split_take_ng(x: np.ndarray, sr: int, pieces: int, max_s: float = WINDOW_S, group_sizes=None) -> list:
    """The take cut into `pieces` arrays (lines). Each is trimmed of its lead/tail silence (a little breathing room
    kept) and must fit `max_s`; raises SceneAudioError, naming the piece, when it does not.

    `group_sizes` (sentences per line, summing to the number of sentences spoken) makes a line that holds several
    sentences keep its own internal pauses: the take is cut at EVERY sentence break first, then only the breaks
    between lines are used."""
    if group_sizes is not None:
        if len(group_sizes) != pieces or any(g < 1 for g in group_sizes):
            raise SceneAudioError("the line grouping does not match the number of lines")
        every = cut_points_ng(x, sr, sum(group_sizes))
        between, acc = [], 0
        for g in group_sizes[:-1]:
            acc += g
            between.append(every[acc - 1])
        cuts = [0.0] + between + [len(x) / sr]
    else:
        cuts = [0.0] + cut_points_ng(x, sr, pieces) + [len(x) / sr]
    out = []
    for k in range(pieces):
        a, b = int(cuts[k] * sr), int(cuts[k + 1] * sr)
        seg = x[a:b]
        env = _envelope(seg, sr)
        if len(env) == 0 or float(env.max()) < 1e-4:
            raise SceneAudioError(f"line {k + 1} came out silent")
        loud = np.where(env > max(float(np.percentile(env, 95)) * SILENCE_REL, 1e-4))[0]
        lo = max(0, int((loud[0] * HOP_S - PAD_S) * sr))
        hi = min(len(seg), int(((loud[-1] + 1) * HOP_S + PAD_S) * sr))
        seg = seg[lo:hi]
        if len(seg) / sr > max_s:
            raise SceneAudioError(
                f"line {k + 1} runs {len(seg) / sr:.1f} s but a clip is {max_s:.0f} s -- shorten that line "
                f"or split it in two")
        out.append(seg)
    return out


def match_loudness_ng(pieces: list, target_rms: Optional[float] = None) -> list:
    """Scale every piece to one speech level (the median of the pieces' own, unless a target is given)."""
    levels = [float(np.sqrt((p ** 2).mean())) for p in pieces]
    target = target_rms or float(np.median(levels))
    out = []
    for p, lv in zip(pieces, levels):
        gain = min(target / max(lv, 1e-6), 4.0)
        out.append(np.clip(p * gain, -1.0, 1.0))
    return out
