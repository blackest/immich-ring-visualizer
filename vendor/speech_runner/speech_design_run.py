#!/usr/bin/env python3
"""Qwen3-TTS VoiceDesign: a voice made from a written description. Ours, not Phosphene's.

Phosphene's own runner (speech_run.py, byte-identical in this folder) offers the nine preset speakers and cloning from a
clip. VoiceDesign is the third checkpoint of the same model family ("a gravelly older man, slow and weary"): no sample at
all, the description steers who speaks. It lives in its own file so the vendored runner stays a byte copy; this one imports
that runner's helpers (script splitting, silence trimming, WAV writing, speed change, the [speech] protocol) and speaks the
same protocol, so the app drives it exactly like the other kinds.

A designed voice is a fresh draw every time: the same words and description give a different speaker per seed. To keep a
voice, save a take you like and use it as a clone reference (the app's "save this voice").

    speech_design_run.py --pack mlx_models/speech --design "a warm, slow, older male voice" --text-file t.txt --out o.wav
"""

import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_run as base  # noqa: E402  (the vendored runner; importing it also clears the PyTorch env switches)

MAX_DESIGN_CHARS = 300


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="mlx_models/speech (holds voicedesign/)")
    ap.add_argument("--design", required=True, help="the voice, in words")
    ap.add_argument("--language", default="english")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    lines = base.split_lines(Path(a.text_file).read_text(encoding="utf-8", errors="replace"))
    if not lines:
        base.fail(base.RC_INPUT, "The script is empty. Write at least one line.")
    if sum(len(l) for l in lines) > base.MAX_CHARS:
        base.fail(base.RC_INPUT, f"The script is longer than {base.MAX_CHARS} characters. Split it into shorter takes.")
    if not 0.5 <= a.speed <= 2.0:
        base.fail(base.RC_INPUT, "Speed must be between 0.5 and 2.0.")
    design = " ".join(a.design.split())
    if not 8 <= len(design) <= MAX_DESIGN_CHARS:
        base.fail(base.RC_INPUT, f"Describe the voice in 8 to {MAX_DESIGN_CHARS} characters.")
    model_dir = Path(a.pack) / "voicedesign"
    if not (model_dir / "config.json").is_file():
        base.fail(1, f"The voice-design model is not installed at {model_dir}.")

    base.say("stage", "load")
    t0 = time.time()
    import numpy as np
    import mlx.core as mx
    mx.set_cache_limit(256 * 2**20)
    from mlx_audio.tts.utils import load_model
    model = load_model(str(model_dir))
    load_s = time.time() - t0
    seed = a.seed if a.seed >= 0 else int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    mx.random.seed(seed)

    pieces = []
    gap = np.zeros(int(base.LINE_GAP_S * base.SAMPLE_RATE), dtype=np.float32)
    t1 = time.time()
    for i, line in enumerate(lines):
        base.say("line", f"{i + 1}/{len(lines)}")
        if base.low_memory():
            base.fail(base.RC_MEMORY, "Stopped before the Mac ran out of memory. Close other apps and try again.")
        chunks = [np.asarray(r.audio, dtype=np.float32).reshape(-1)
                  for r in model.generate_voice_design(text=line, language=a.language.title(), instruct=design)]
        if not chunks:
            base.fail(1, f"The model returned no audio for line {i + 1}.")
        audio = base.trim_silence(np.concatenate(chunks), pad_s=0.0 if len(lines) > 1 else base.EDGE_PAD_S)
        if pieces:
            pieces.append(gap)
        pieces.append(audio)
    gen_s = time.time() - t1
    audio = np.concatenate(pieces)
    if len(lines) > 1:
        pad = np.zeros(int(base.EDGE_PAD_S * base.SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([pad, audio, pad])
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio * (10 ** (base.PEAK_DBFS / 20) / peak)

    base.say("stage", "write")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial" + out.suffix)
    if abs(a.speed - 1.0) > 1e-3:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.wav"
            base.write_wav(raw, audio)
            base.change_speed(raw, partial, a.speed)
    else:
        base.write_wav(partial, audio)
    with wave.open(str(partial), "rb") as w:
        seconds = w.getnframes() / float(w.getframerate())
    os.replace(partial, out)
    done = {"audio_seconds": round(seconds, 3), "sample_rate": base.SAMPLE_RATE, "lines": len(lines), "seed": seed,
            "load_seconds": round(load_s, 2), "generate_seconds": round(gen_s, 2),
            "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "rss_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}
    base.say("done", json.dumps(done))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
