# The engines: what runs, from where (checked 2026-11-13)

Every engine runs as a subprocess. **All of the code that runs is now in this repository** (`vendor/` and the app's own
scripts); what is still outside is each engine's Python environment and the model weights.

| Engine | Code that runs | Python env | Interpreter | Env patched by Phosphene? |
|---|---|---|---|---|
| LTX (Animate, upscale x2, make-vertical, scene/Gemma helpers) | `vendor/ltx_2_mlx` (fork at `v0.14.19+ltx25.7`) | `vendor/ltx_2_mlx/env` (ours) | uv 3.11 in `~/.local/share/uv` (ours) | **Yes**: two files in the ltx packages, carried into the vendored copy |
| Music (`lyra`) | `vendor/yue2_mlx` (mlx-Yue `9253ed1`) | `vendor/yue2_mlx/.venv` (ours) | uv 3.14.6 (ours) | No |
| H3 (Enhanced H3, Talk, H3-Image) | `vendor/minimax_h3_mlx` (runner and package) | `Pinkio/.../minimax-h3-mlx/.venv` | **Pinokio's** uv 3.11 | No (80 packages identical to a clean build) |
| Speech | `vendor/speech_runner` | `PhospheneNative/speech-mlx/.venv` | uv 3.12 in `~/.local/share/uv` (ours) | No (35 packages) |
| Klein | `klein_workerNG.py` (ours; uses `mflux` from the env) | `Pinkio/.../flux3-worker/env` | **Pinokio's** uv 3.12 | No (57 packages) |
| Make-vertical, x2 upscale, H3 upscale | `vendor/vertical_outpaint`, `upscale_x2NG.py`, `h3_upscaleNG.py` (ours) | the LTX env | | |

**Weights** are still read from Pinokio: `mlx_models` (LTX, H3, music, speech, LoRAs) and Klein's Hugging Face cache.

## How this was checked

- **Which code runs.** Each runner's imports were run and every loaded module listed: none comes from a Phosphene or Pinokio
  source tree (H3: 219 modules, 16 from `vendor/minimax_h3_mlx`; speech: none outside the env). The H3 env does not even have
  `minimax_h3_mlx` installed; the runner runs with its vendored folder as the working directory.
- **Patches inside the envs.** Each env was diffed, file by file, against a clean build of its exact pins. LTX differed (see
  `ltx_2_mlx/VENDORED.md`: the running code is Phosphene's patched `site-packages` copy, not the git tree). H3, speech, Klein and
  the old music env had **no** code differences, so they can be rebuilt from their pin lists without losing anything.

## What is left to separate

1. **Interpreters and env folders for H3 and Klein.** They borrow Pinokio's Pythons and live inside Pinokio. Rebuilding them
   from `uv pip freeze` pins is mechanical (nothing is patched) but each needs a real render to prove it.
2. **Weights.** Cloning `mlx_models` (about 340 GB) into an app-owned root costs no space on APFS; the Pinokio copy has things
   `PhospheneNative/models` lacks (the 22 GB `minimax-h3-image` pack, `speech/voicedesign`, `characters`).
3. **The resolver defaults** that still point at `/Volumes/AI/Pinkio/...` as a last fallback.
4. **Python versions.** LTX is on 3.11, H3 on 3.11, Klein and speech on 3.12. Music is on 3.14.6 and passed; the rest are
   untested on 3.14 and the combined-environment finding applies (music's exact pins must be held).
