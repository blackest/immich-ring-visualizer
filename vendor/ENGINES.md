# The engines: what runs, from where (checked 2026-11-13)

Every engine runs as a subprocess. **The code, the Python environment and the interpreter of every engine are now in this
repository** (`vendor/` and the app's own scripts). What is still outside is the model weights.

| Engine | Code that runs | Python env (git-ignored, built from the committed lock) | Interpreter | Env patched by Phosphene? |
|---|---|---|---|---|
| LTX (Animate, upscale x2, make-vertical, scene/Gemma helpers) | `vendor/ltx_2_mlx` (fork at `v0.14.19+ltx25.7`) | `vendor/ltx_2_mlx/env` (`build_env.sh`) | uv 3.14.6 | **Yes**: two files in the ltx packages, carried into the vendored copy |
| Music (`lyra`) | `vendor/yue2_mlx` (mlx-Yue `9253ed1`) | `vendor/yue2_mlx/.venv` (`build_venv.sh`) | uv 3.14.6 | No |
| H3 (Enhanced H3, Talk, H3-Image) | `vendor/minimax_h3_mlx` (runner and package) | `vendor/minimax_h3_mlx/.venv` (`build_venv.sh`) | uv 3.14.6 | No (80 packages) |
| Speech | `vendor/speech_runner` | `vendor/speech_runner/.venv` (`build_venv.sh`) | uv 3.14.6 | No (35 packages) |
| Klein | `klein_workerNG.py` (ours; uses `mflux` from the env) | `vendor/flux3_worker/env` (`build_env.sh`) | uv 3.14.6 | No (57 packages) |
| Make-vertical, x2 upscale, H3 upscale | `vendor/vertical_outpaint`, `upscale_x2NG.py`, `h3_upscaleNG.py` (ours) | the LTX env | | |

Each resolver (`ltx_engineNG`, `music_engineNG`, `h3_engineNG`, `speech_engineNG`, `klein_engineNG`) prefers the vendored
env once it is built, then falls back to the old Phosphene/Pinokio locations, and an environment variable still overrides
everything. Rebuild any env with its script; the exact package set is the `requirements.lock` beside it.

**Weights** are still read from Pinokio: `mlx_models` (LTX, H3, music, speech, LoRAs) and Klein's Hugging Face cache.

## How this was checked

- **Which code runs.** Each runner's imports were run and every loaded module listed: none comes from a Phosphene or Pinokio
  source tree (H3: 219 modules, 16 from `vendor/minimax_h3_mlx`; speech: none outside the env).
- **Patches inside the envs.** Each env was diffed, file by file, against a clean build of its exact pins. Only LTX differed
  (see `ltx_2_mlx/VENDORED.md`: the running code is Phosphene's patched `site-packages` copy, not the git tree). H3, speech, Klein
  and the old music env had **no** code differences, so rebuilding them from their locks lost nothing.
- **Same output as before.** Each vendored env was run through the app's own generate function against the env it replaces, same
  prompt, seed and size, on an M2 Max:

  | Engine | Result |
  |---|---|
  | LTX text-to-video and image-to-video, and one job through the HTTP API | every frame and the audio identical |
  | Moving every env to Python 3.14.6 with the same pins (LTX, H3, speech, Klein; music was done first) | speech WAV byte-identical, Klein pixel-identical, LTX t2v/i2v and H3 every frame and the audio identical |
  | Music (Python 3.12 against 3.14.6) | every intermediate array identical; audio within 119 dB (numpy rounding in the final decode) |
  | Speech (preset voice) | the WAV file byte-identical |
  | Klein (512x512, seed 21) | every pixel identical (only embedded EXIF/XMP metadata differs) |
  | H3 (Q8, 6 steps, 576x320, 73 frames) | every frame and the audio identical |

## What is left to separate

1. **Weights.** Cloning `mlx_models` (about 340 GB) into an app-owned root costs no space on APFS; the Pinokio copy has things
   `PhospheneNative/models` lacks (the 22 GB `minimax-h3-image` pack, `speech/voicedesign`, `characters`). Then Klein's Hugging Face
   cache (about 19 GB).
2. **The resolver defaults** that still point at `/Volumes/AI/Pinkio/...` as a last fallback, and the one setting that names a root.
3. **Python versions.** All five engines are on 3.14.6 with their original exact pins. The combined-environment finding still applies if they are ever merged into one env: music's exact pins (`mlx` 0.32.2, `transformers` 5.10.4 ...) must be held.
4. **Installers.** "Install H3" on a machine without the weights needs `scripts/download_selected.py` and `scripts/quantize_stream.py`,
   which the vendored H3 folder does not carry.
