# minimax-h3-mlx, vendored for immichRingNG

The H3 video runner (`scripts/generate_staged.py` and the `minimax_h3_mlx` package) that
`h3_engineNG.py` runs. It lives here so that Phosphene's updater, which moves its own copy of the
engine to an exact commit every time you press Update, cannot take away the flags this app needs
(`--ref2va-reference`, `--refmod`, `--lazy-adaln`, `--adaln-cache-dir`).

**Licence:** Apache-2.0 (`LICENSE`, unchanged). Original work by the minimax-h3-mlx authors
(https://github.com/mrbizarro/minimax-h3-mlx; the app's fork is blackest/minimax-h3-mlx). Files we
changed or added are listed below, as the licence asks.

**Base:** commit `3f6c75c` ("fix: bound media sources, stream decoded frames, honour --extend-frames"),
the commit Phosphene v4.19.x pins (`scripts/pinokio/h3_checkout.sh`, ref `phosphene/v4.19.0`).

**Taken from upstream after the base:** `bef6d03` ("feat: Pika H3 Flex sampler (--pika-flex)", Phosphene v4.19.15): `scheduler.py` re-noise sampler, `--sigma-knots --sampler --video-shift --audio-shift --pika-flex` in `generate_staged.py`, `tests/test_pika_flex.py`. Applied with `git apply` (clean) on 2026-10-30; README and `run_tests.sh` changes of that commit are not vendored.

**Updated 2026-11-08 to Phosphene v5.0.7's pin (`59da1474b3`)**, 19 commits after the base above: low-memory streaming
(`dit_stream.py`, `memory_plan.py`, `tensor_stream.py`, `vae_stream.py`, `--stream-dit`, `--dit-resident`), the streamed text
encoder, the float16-mixed video decoder (default; `H3_VAE_DTYPE=float32` gives the old picture exactly), upstream's own
`--reference` pictures (`reference.py`) and their tests. Merged by hand with a three-way merge against `bef6d03`: `load_dit` takes
both `lazy_adaln` and `stream`; `generate_staged.py` keeps our `--refmod` / `--ref2va-reference` packing beside upstream's
`reference_grids`; `text_encoder.encode` passes `references=` only when given, so upstream's tests' stubs still fit.
Checked by real renders against the old runner (same seed): audio bit-identical, picture within 0.4-0.5/255 (the decoder),
identical with `H3_VAE_DTYPE=float32`; `--stream-dit` output identical with the denoise peak 4 GB instead of 22 GB.
The image-generation modules of that pin (`ph3_image*.py`, `generate_image.py`, ...) are not vendored; `video_vae_v2.py` is
kept but no longer the default decoder.

**Ours on top of that:** `--refmod` / `--ref2va-reference` now compose with `--audio-drive` (one entry removed from the
refusal list in `generate_staged.py`). Tested: a RefMod-only render with a driven speech line, the mouth followed the words,
the identity held, output audio equal to the driven file. A first frame, keyframes, extend and chained windows are still refused.

**Added on top of the base** (six commits from the fork's `reference-flag` branch, 2026-10-27 to 28):

| commit | what |
|---|---|
| `7f2c5e5` | `ref2va_packing.py`: the packed layout for reference rows |
| `f81276d` | `refmod_io.py`, `--refmod`: read a saved RefMod and condition on it |
| `defddd2` | `--lazy-adaln`: build the AdaLN cache block by block (less memory) |
| `8fa033e` | `--adaln-cache-dir`: reuse AdaLN tables across ordinary renders |
| `3e507c0` | `refmod_encode.py`, `--reference`: encode pictures straight from the file |
| `1166cff` | `--reference` takes clips; `<Video N>` presentation with 2 fps timestamped blocks |

Changed files: `minimax_h3_mlx/{adaln,draft_cache,load,text_encoder,scheduler}.py`, `scripts/generate_staged.py`.
New files: `minimax_h3_mlx/{ref2va_packing,refmod_encode,refmod_io}.py`, `scripts/make_refmod.py`
(make a RefMod from pictures; the app's "Make RefMod" runs it), `scripts/refmod_roundtrip.py`
(checks our encoder against a library file; see the plan for its inconclusive result).

**Not vendored:** the benches, the quantizer and build scripts (Phosphene's install builds the Q8
weights), the parity tests (they need reference weights), docs and the reference folder.

**How it runs:** with the Hailuo H3 venv's python from Phosphene's install (it holds mlx and the locked
package set); the runner puts its own folder first on `sys.path`, so nothing is installed. Models come
from Phosphene's models folder as before. Tests: `vendor/minimax_h3_mlx/run_tests.sh` (12 logic tests).

**Moving to a newer engine:** when Phosphene moves its pin, diff the two commits
(`git diff 3f6c75c..<new> -- minimax_h3_mlx scripts/generate_staged.py` in the fork), bring the changes
here, re-run `run_tests.sh`, then render once. Nothing here changes when Phosphene updates.

**Renamed 2026-10-31:** our `--reference` and `--reference-size` are now `--ref2va-reference` and `--ref2va-reference-size`
(and `args.reference` / `args.reference_size` are `args.ref2va_reference` / `args.ref2va_reference_size`). The upstream
`phosphene/v5.0` branch has its own `--reference IMAGE` (a face or character sheet on the FL2VA model, given to every window),
which is a different feature under the same name; renaming ours means a later sync from upstream cannot collide with it.
Our flag is for the Ref2VA DiT: pictures and clips, one window.

**VAE v2 (2026-11-07):** `minimax_h3_mlx/video_vae_v2.py` is a verbatim copy of `minimax_h3_mlx/video_vae.py`
from Phosphene's engine pin `cd07c3c` (`phosphene/v5.0-image3`, Apache-2.0): mixed-precision blocks, compiled blocks,
`decode_chunks`. It is OPT-IN (`H3_VAE_IMPL=v2`, the "VAE v2" checkbox); the default decoder is still `video_vae.py`.
Changed to select it: `load.py` (`_video_vae_classes`) and `scripts/generate_staged.py` (module pick at import,
`decode_video` uses `decode_chunks` when the model has it).
