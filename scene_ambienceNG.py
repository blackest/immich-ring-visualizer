"""immichRingNG -- ambience under a scene's dialogue: a sound bed per setting.

H3 makes sound as well as picture, so a bed is one short text-to-video render of the EMPTY place (no people, no speech,
no music) whose audio track is kept and whose picture is thrown away. The bed goes under every clip in that setting at a
fixed distance below the speech (default -22 dB), faded 30 ms in and out so a cut never clicks. A bed is one 5 s window,
and a clip is never longer, so one bed covers every clip of its setting.

This module only builds the prompt and does the audio work (ffmpeg + numpy); the render itself is an ordinary H3 job
queued by scene_jobsNG.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

BED_DB = -22.0           # the bed's speech-relative level
FADE_S = 0.03
BED_RATE = 32000         # H3's audio rate
MAX_SETTING = 160


class AmbienceError(RuntimeError):
    pass


def ambience_prompt_ng(setting: str) -> str:
    """An H3 prompt in its own native sections whose picture is an empty place and whose soundscape is the place's
    sound: the model is told plainly there are no people, no speech and no music."""
    setting = " ".join((setting or "").split())[:MAX_SETTING]
    if not setting:
        raise AmbienceError("an ambience bed needs a setting to describe")
    return (f"integrated_multimodal_description: [Shot 1] Live-action, cinematic, an empty place, {setting}, no people "
            f"visible, the camera drifts very slowly, natural light.\\n"
            f"overall_soundscape: The natural ambient sound of {setting}: room tone and environmental sounds, "
            f"layered and continuous, no speech, no voices, no singing.\\n"
            f"non_diegetic_music: N/A")


def _run(cmd: list) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AmbienceError((r.stderr.strip().splitlines() or ["ffmpeg error"])[-1])


def extract_audio_ng(mp4_path: str, wav_path: str) -> str:
    """The clip's sound as a mono 32 kHz wav."""
    if not Path(mp4_path).is_file():
        raise AmbienceError("the ambience render has no file")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4_path), "-vn", "-ac", "1", "-ar", str(BED_RATE), str(wav_path)])
    if not Path(wav_path).is_file() or Path(wav_path).stat().st_size < 2000:
        raise AmbienceError("the ambience render has no sound")
    return str(wav_path)


def level_ng(path: str) -> float:
    """RMS of the louder part of a file's audio (the quiet gaps do not count), 0..1."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(x) == 0:
        raise AmbienceError("no audio to measure")
    loud = x[np.abs(x) > 0.1 * float(np.abs(x).max() or 1.0)]
    return float(np.sqrt((loud ** 2).mean())) if len(loud) else float(np.sqrt((x ** 2).mean()))


def mix_ambience_ng(clip_mp4: str, bed_wav: str, out_mp4: str, keep_s: float, bed_db: float = BED_DB) -> str:
    """The clip with the bed under its sound: the bed is scaled so its own level sits `bed_db` below the clip's speech
    level, trimmed to `keep_s`, faded, and summed in; the picture is copied untouched."""
    voice, bed = level_ng(clip_mp4), level_ng(bed_wav)
    gain = min((voice * 10 ** (bed_db / 20.0)) / max(bed, 1e-6), 8.0)
    f = FADE_S
    graph = (f"[1:a]atrim=0:{keep_s:.3f},asetpts=PTS-STARTPTS,volume={gain:.5f},"
             f"afade=t=in:d={f},afade=t=out:st={max(0.0, keep_s - f):.3f}:d={f}[b];"
             f"[0:a]apad[v];[v][b]amix=inputs=2:duration=shortest:normalize=0[a]")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(clip_mp4), "-i", str(bed_wav), "-filter_complex", graph,
          "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_mp4)])
    return str(out_mp4)
