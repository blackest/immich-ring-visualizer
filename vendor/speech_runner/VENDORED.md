# speech_run.py, vendored for immichRingNG

The Speech runner `speech_engineNG.py` runs as a subprocess (`speech-mlx/.venv/bin/python vendor/speech_runner/speech_run.py ...`).
It lives here so a Phosphene update cannot change what this app runs. Only the venv (`speech-mlx/.venv`) and the weights
(`mlx_models/speech`: `customvoice/` and `base/`) are shared with Phosphene.

**Source:** `scripts/speech/speech_run.py` from Phosphene v5.0.3 (commit `7dc644d`, https://github.com/mrbizarro/phosphene),
**MIT licence**, copyright (c) 2026 Mr. Bizarro. Copied byte for byte on 2026-11-08; no local changes.

Model: Qwen3-TTS 12 Hz 1.7B (Apache-2.0, the Qwen team) through mlx-audio (MIT), 8-bit MLX conversions by mlx-community.

Protocol on stdout, one line each: `[speech] stage <name>`, `[speech] line <i>/<n>`, `[speech] done {json}`, `[speech] error <sentence>`.
Exit codes: 0 ok, 2 bad input, 3 memory guard, 1 anything else.

## speech_design_run.py is ours

`speech_run.py` above is a byte copy of Phosphene's runner and stays one. `speech_design_run.py` is **our own** small runner
for the third Qwen3-TTS checkpoint, **VoiceDesign** (`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`, Apache-2.0, 4.2 GB,
unpacked into `<speech pack>/voicedesign`): a voice made from a written description, no sample. It imports `speech_run.py`'s
helpers (script splitting, silence trimming, WAV writing, speed change, the `[speech]` protocol) so the app drives it like
the other kinds, and calls `model.generate_voice_design(text, language, instruct)` from the mlx-audio already in the speech
venv. Checked 2026-11-08 through the app: "a deep, gravelly older man, slow and weary" came out at a median 90 Hz over 5.5 s,
"a bright, quick, cheerful young woman" at 308 Hz over 3.1 s, and Whisper read back the words of both.
