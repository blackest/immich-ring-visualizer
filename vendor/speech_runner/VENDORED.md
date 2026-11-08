# speech_run.py, vendored for immichRingNG

The Speech runner `speech_engineNG.py` runs as a subprocess (`speech-mlx/.venv/bin/python vendor/speech_runner/speech_run.py ...`).
It lives here so a Phosphene update cannot change what this app runs. Only the venv (`speech-mlx/.venv`) and the weights
(`mlx_models/speech`: `customvoice/` and `base/`) are shared with Phosphene.

**Source:** `scripts/speech/speech_run.py` from Phosphene v5.0.3 (commit `7dc644d`, https://github.com/mrbizarro/phosphene),
**MIT licence**, copyright (c) 2026 Mr. Bizarro. Copied byte for byte on 2026-11-08; no local changes.

Model: Qwen3-TTS 12 Hz 1.7B (Apache-2.0, the Qwen team) through mlx-audio (MIT), 8-bit MLX conversions by mlx-community.

Protocol on stdout, one line each: `[speech] stage <name>`, `[speech] line <i>/<n>`, `[speech] done {json}`, `[speech] error <sentence>`.
Exit codes: 0 ok, 2 bad input, 3 memory guard, 1 anything else.
