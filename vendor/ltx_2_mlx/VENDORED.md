# ltx-2-mlx, vendored for immichRingNG

`ltx_engineNG.py` and its helpers (Animate, upscale x2, make-vertical, the scene and Gemma helpers) run this engine as a
subprocess. It lives here so a Phosphene or Pinokio update cannot change what this app runs. Only the model weights
(`mlx_models`) are still shared.

**Source:** https://github.com/mrbizarro/ltx-2-mlx (a fork of dgrauet/ltx-2-mlx), tag `v0.14.19+ltx25.7`, commit `bf419b6f7`,
**MIT licence**, copyright (c) 2025 dgrauet (`LICENSE`). Taken from the Phosphene/Pinokio checkout on 2026-11-13.

**What is here:** `packages/` (`ltx-core-mlx`, `ltx-pipelines-mlx`, `ltx-trainer`), `pyproject.toml`, `uv.lock`, `README.md`,
`LICENSE`. Left out on purpose: `docs`, `tests`, `scripts`, `CHANGELOG.md`, `CLAUDE.md`.

## Not byte-for-byte the tag: two files, exactly as the running env had them

The code that renders is not the git tree. Phosphene's installer builds the packages into the env's `site-packages` and its
updater then patches those copies. On 2026-11-13 the running copies differed from the tag in two files, so those two are taken
from the running copies (everything else in all three packages is identical to what ran, checked file by file):

1. `ltx_core_mlx/text_encoders/gemma/encoders/base_encoder.py`: when a prompt is longer than the token limit it keeps the
   **first** tokens, not the last (upstream keeps the last). This matters for our capped 256-token fallback.
2. `ltx_core_mlx/model/video_vae/video_vae.py`: the ffmpeg line reads `LTX_OUTPUT_PIX_FMT` / `LTX_OUTPUT_CRF` from the
   environment and adds `-movflags +faststart`. With the variables unset it is lossless (yuv444p, crf 0). The app always sets
   them from the Output preset, so its clips are unaffected; this only matters for a hand-run command.

`ltx_codecNG.apply_codec_patch_ng` still runs before each render; it now finds its marker in this file and reports "already".

## Third-party packages: one difference, not carried over

The env's 33,000 files were diffed against a clean build of the same pins. Besides pip's own certificate bundle, the only
difference is `mflux/models/qwen/model/qwen_transformer/qwen_transformer.py`: Phosphene's optional "FBCache" speed patch for
the Qwen image model, a no-op unless `MFLUX_FB_CACHE=1`. immichring does not use Qwen-Image from this env, so it is left out.

## The env

`env/` here is git-ignored. Build it with `./build_env.sh`: Python 3.11, the 136 exact packages in `requirements.lock`
(installed with `--no-deps`, because Phosphene installed `mlx-vlm` that way and the freeze is the complete list), and the three
packages editable from `packages/`. `build-constraints.txt` pins `hatchling<1.32`, as Phosphene does: the package
`pyproject.toml` files point their readme outside the package folder, which hatchling 1.32 refuses.

The installer in `ltx_installNG.py` does nothing in this folder (`engine_installNG.app_vendored`).

## Where it diverges from upstream, and why it is frozen

As of 2026-11-13 the fork is 40 commits ahead of and 137 behind dgrauet/ltx-2-mlx. The fork is LTX 2.5 support (sampler, duration
head, unfused LoRAs, live preview, diffusion decoder); upstream has since added its own, implemented differently, so moving
back is a re-port, not a merge. The 2.5 high-quality path the app uses (`--two-stages-hq`, `--distilled-lora`) behaves the way
the fork made it. Do not move this without comparing renders.

## Checked 2026-11-13 (M2 Max, LTX-2.5 q8, `--two-stages-hq`)

Same prompt, seed 21, 384x256, 25 frames, the app's own `generate_ltx_video_ng` (including the capped-prompt wrapper), once
with this folder as the engine and once with the Phosphene/Pinokio engine it replaces:

- text-to-video: every video frame bit-identical (PSNR infinite), audio stream checksum identical (100 s here, 86 s there)
- image-to-video from a reference image: the same, and the clip differs sharply from the text-only one, so the image did condition it
- one job submitted through the app's HTTP API (`/api/ng/videogen/generate`) on this engine: bit-identical to the Pinokio render

The scripts that run on this interpreter (the Gemma and scene helpers, upscale x2, the H3 upscale, make-vertical) import the
same third-party packages here as under the old env. Only text-to-video and image-to-video were rendered end to end.
