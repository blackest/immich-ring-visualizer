# Enhanced H3: compiled briefs, per-window prompts, and full references

**Status (2026-10-28): steps 1-3 below are built** (per-window prompts, the Enhanced H3
pane, render through the H3 queue). The runner fork, Phosphene and ComfyUI have not
been changed. Tested: text-only and picture+audio briefs, refine, a 5 s i2v draft
render. Not tested: video clips in the tray, a 15 s three-window chain from the pane,
a spoken line through the pane.

## The goal, in one sentence

Let the H3 view write its prompt with OpenH3-IR's compiler, render clips longer
than one window with a different prompt per window, and eventually pass up to 9
images, 3 videos and audio as references, without needing ComfyUI running.

## Findings from the test session

- **Compiler.** `h3ir` (OpenH3-IR, Apache-2.0) is a standalone library with a CLI
  and an HTTP service. The ComfyUI nodes only wrap it. `h3ir compile` works
  against the Gemma MLX server (`H3IR_LLM_URL=http://127.0.0.1:8080/v1`,
  `H3IR_LLM_MODEL=/Users/john/models/gemma4-26b-mlx`). Its CLI only takes
  `--image`; videos and audio go through its Python API or `h3ir serve`.
  `--out` writes the prompt text, not JSON.
- **Brief vs plain sentence.** Text-to-video: very different films, same woman
  (same seed). The brief pushes style (colour grade, invented detail), driven
  mostly by creativity=extreme and the Director profile. Image-to-video: both good;
  the brief was judged slightly better.
- **Chain bug (fixed).** immichring sent one prompt to every 5 s window, so a
  two-shot brief repeated its opening. Fixed by `chain_prompts` (commit `d09753e`).
  Verified: window 2 became a new shot, the join was clean, and a spoken line
  (`<d>[English] ...</d>`) landed in the close-up only.
- **ComfyUI route.** The OpenH3-IR example workflow is Compile node -> a Render
  subgraph of core nodes (BasicGuider, SamplerCustomAdvanced `res_multistep`,
  beta scheduler 20 steps, VAEDecode + VAEDecodeAudio, CreateVideo 24 fps).
  Core `MiniMaxH3ReferenceToVideo` takes 9 images, 3 videos (2-15 s), 3 video
  soundtracks, 3 standalone audios. `MiniMaxH3ImageToVideo` (fl2va model) takes
  first/last frame only. On this Mac (GGUF, MPS, no speed LoRA, 736x416, 5 s):
  ~137 s/step, ~46 min per clip. The MLX Fast 3-step path is ~12 min for 10 s.
  The gap is the step count, not the framework.
- **Text encoder on CPU.** On Apple Silicon `text_encoder_device()` falls back to
  CPU (vram_state SHARED), ~2 min to encode one prompt. Fix written in
  `custom_nodes/Silicon-Implants/text_encoder_device_mps.py`; needs a ComfyUI
  restart, untested.

## What the H3 runner already has (fork: blackest/minimax-h3-mlx, Apache-2.0)

| Piece | Status |
|---|---|
| Video VAE encoder (`Encoder3d`, `encode()`) | exists; `refmod_encode.encode_images` uses it |
| Audio VAE encoder (`AudioVAE.encode`) | exists |
| Reference packing, 9 images / 3 videos / 3 audios (`ref2va_packing.py`) | exists; takes geometry only |
| Qwen3-VL text encoder | `build_request` writes `<Picture i>: ` for keyframes only |
| `mlx_vlm` 0.7.8 `qwen3_vl` video input (`pixel_values_videos`) | in the library; runner passes `video_grid_thw=None` |
| `--refmod` (`generate_staged.py` ~line 1027) | saved latents only, visual only, single window, no other media |

The real orchestrator is `scripts/generate_staged.py` (2,476 lines).
`pipeline.py` has no reference or chain support.

## Behaviour to match (from ComfyUI's `comfy/text_encoders/minimax.py` docstring)

Read for behaviour only. Do not copy code: ComfyUI is GPL-3.0.

    t2va:   <prompt>
    fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
    ref2va: per condition, request order, 1-based ordinals per type:
              image -> "<Picture i>: " <vision block>
              audio -> "<Audio j>: "            (audio never enters Qwen)
              video -> "<Video k>: " then per 2-frame block at 2 fps
                       "<T.T seconds>" <vision block (2 frames)>
            then <prompt>

Order is images, then videos (a video's soundtrack `<Audio j>` label comes
right before its `<Video k>`), then standalone audio. `h3ir`'s asset order
follows the same rule, so the two stay in agreement.

One difference to test: in ComfyUI an image reference goes to Qwen **and** the
VAE; in the runner's RefMod path only the latent is used. Possibly why RefMods
behave differently from the official ref2va. Unverified.

## Build order

1. **DONE: per-window prompts.** `chain_prompts` through `h3_engineNG.py`,
   `h3_jobsNG.py`, `routes/h3NG.py` (`d09753e`).
2. **DONE: `enhancedh3` pane** (`routes/enhancedh3NG.py`, `static/enhancedh3NG.js`; follow `ADDING_A_PANEL.md`). Sentence + optional
   keyframe + the compiler controls; "Write brief" calls `h3ir` and shows the
   result in editable boxes, one per window, split at `[Shot N]`; Render sends
   `chain_prompts`. Known limit: the compiler places shots by time (shot 2 at
   7.5 s) but windows switch at 5 s. Compiler warnings (e.g. mood words in the
   music line) should be shown. Memory: compile first, unload the LLM, then render.
3. **3-window chain (15 s)** to check seams over two joins. 15 s was the
   longest length the original runner authors validated.
4. **Runner fork: references.** Order: (a) text presentation for an ordered item
   list; (b) video through Qwen3-VL (riskiest: must match the 2-frame block
   layout); (c) raw media -> VAE encode -> `condition_rows` + `ref_geoms`, audio
   rows included; (d) repeatable `--ref-image` / `--ref-video` / `--ref-audio`;
   (e) carry reference rows into every chained window (largest change; today's
   code refuses it).
5. **Verify against ComfyUI** with the same inputs: token sequence, packed row
   counts, reference latents.
6. **Fallback until step 4 is done:** reference-heavy jobs go through the
   existing ComfyUI bridge (`routes/comfyNG.py`) with the OpenH3-IR workflow.
   Needs the encoder patch and a speed LoRA first, and the bridge may not suit
   the OpenH3-IR custom nodes (media tray, director profile). Untested.

## Open questions

- Does MLX Qwen3-VL's video path match ComfyUI's block layout exactly?
- Memory with 12 references on the q8 DiT (96 GB, with the LLM unloaded)?
- Do references keep their effect in later chain windows?
- Does the plain sentence beat the brief for image-to-video over more shots?
  Only one comparison so far, same seed.
- Splitting a brief per window: cut at `[Shot N]`, or compile each window
  separately with `--shots 1`?
- Should the compiler call go through `h3ir serve` (own process) or the CLI?
  CLI works today; a pane should keep the LLM address configurable.

## Observed issues (our own, add as found)

Phosphene's prompt guards (`h3_prompt_guards.py`, MIT: ellipsis, spoken, hands,
nouns) are NOT mirrored. They were tuned on Phosphene's own planner prompts; ours
come from h3ir + Gemma. Add one only after we reproduce the problem it targets.
A "not sung" guard would also be wrong for music videos, so spoken vs sung should
be an explicit per-clip choice in the pane, not a hidden rule.

| Date | Test | Result |
|---|---|---|
| 2026-10-27 | Gantry chain, close-up line `<d>[English] Where are you, Marko?</d>`, no "not sung" sentence. Seed 617403820, draft, Fast 3-step, 10 s, 2 windows (job a3ceaa2abbf0) | Line spoken, with the urgency asked for. No singing. One case only. |

Ideas (untested): music video via `--audio-drive` using a song from the Music
pane (YuE2); check immichring passes the flag through first.

## Status and next steps (2026-10-28, end of a long session)

**Built and committed:** pane (brief, per-window prompts, render, progress bar,
refresh-safe state), Extend (`--extend-from`, commit 4c9ae5b), director/cinematographer
style drafts (DIRECTOR_PROFILES.md, untested).

**Findings worth keeping**
- Extend works when the new section gets a compiled brief. A one-line hand-written prompt
  lost the face; the same source, seed and a Gemma-compiled prompt held identity
  (jobs af306abbe17b vs 602dff368f9d, Draft 640x384, ~6.5 min per 5 s).
- Native 640x384 Draft already gives good face detail. Phosphene's "Upscale & Face Fix"
  (LTX 2.3 x2 spatial upscaler, 1 refine step from the clip, weights in
  Pinkio/.../mlx_models/ltx-2.3-mlx-q8/) is the quality route to 1280x768; the runner's
  `hires_refine.py` is a latent refine that needs a `--save-stage-a` cache (chain/extend
  compatibility unchecked). Neither is wired into immichring.
- Gemma crashes the compiler on video clips; Qwen3-VL-8B-8bit reads clips but its brief
  came back `degraded`.
- H3's trained lengths are 124 and 362 frames (compiler `/options`), so a 6 s window
  (141 frames) is not a trained length; 362 (~15 s single pass) is, and is untested here.
- Only the first-frame picture and the extend clip reach the renderer. Other pictures,
  clips and sounds are described to the compiler, which writes `<Picture N>` labels the
  renderer cannot honour. Warn or stop sending them.
- Compiler quirks seen: prompts under the 350-token band; later windows can open with
  "The camera cuts to"; a continuation can end "fades to black"; each window repeats a
  "music begins" line (may restart the score at each seam, unheard).

**Next (user approved this order):** 1. last frame (done 2026-10-28: first+last, 10 s chain, extend+last and compiled brief all end on the still), 2. keyframes (done 2026-10-28: first + 2.5 s mid + last on a 5 s clip, mid frame matches; untested on a chain), 3. replace in a video.
- Runner side is ready: `--last-frame PATH` (lands on the last delivered frame),
  `--keyframe PATH@WHEN` (frame index or `2.5s`, repeatable, routed to the window that
  owns it), alignment sentence added automatically unless the prompt already starts with
  "For the target video" / "How the reference pictures". Extend allows `--last-frame` but
  not `--first-frame` or `--keyframe`. `--refmod` allows none of them.
- Plumbing to do: `h3_engineNG.generate_h3_video_ng` (`last_frame_path`, `keyframes`
  [(path, when)], validate files and the `when` pattern, refuse with RefMod and
  with extend+keyframes), `h3_jobsNG` (write `last.*` / `kf_N.*` into the job dir; never
  take client paths), `routes/h3NG.py` (`last_file`, `keyframe_files` + `keyframe_times`),
  pane (role `frame_anchor_last` -> `last_file`; a stills-at-time control for mid
  keyframes that goes to the renderer only). Then test: first+last on a 5 s clip, last
  frame on the final window of a 10 s chain, then Extend + last frame. How `--last-frame`
  behaves across chained windows is untested.
- Replace in a video: the compiler has `edit_source`, `replacement_subject`,
  `placed_subject` roles, but the runner has no edit/replace mode (nothing in
  generate_staged.py). It needs runner-fork work (reference video/image conditioning),
  which is the existing "raw references" step below. Plan it as its own task.

### Replace in a video: ComfyUI test and fork design (2026-10-28)

**Test (ComfyUI, `MiniMaxH3ReferenceToVideo`, ref2va int8 DiT, GGUF Q4_K_M encoder, 640x384,
124 frames, 20 steps, seed 1234, one hand-written prompt).** Source: 5 s of a bedroom render, one
woman on a bed. References: that clip as `<Video 1>`, a portrait of another woman as `<Picture 1>`.
Result: scene, camera, pose and timing held; her clothes changed to the picture's black leather
jacket and pendant; her face and loose hair stayed the source woman's, so identity did not
transfer. One seed and one prompt only, with `ref_image_size` left at `match`, so it is unknown
whether `max` or a compiler-written prompt would carry the face. Cost: ~85 min for 5 s on this Mac
(about 4.4 min per step, DiT quantization emulated), against minutes in the MLX runner. The NVFP4
text encoder cannot run on MPS (float8 block scales); the GGUF encoder does. Silicon-Implants puts
the encoder on MPS globally. The compiler 500s on a real `edit_source` clip (the Gemma/video crash
again), so an edit prompt had to be written by hand. ComfyUI has no edit mode: an edit is
Ref2VA with the clip and the replacement picture as references and a prompt that says what changes.
Its Fun ControlNet patch is a separate masked inpaint route; no controlnet weights are installed.

**What the runner has.** `ref2va_packing.py` packs reference blocks from geometry alone;
`--refmod` supplies saved latents (single window, no other media); `conditioning.encode_video_rows`
and `encode_audio_rows` already encode a source clip for `--extend-from` and `--video-to-audio`;
the text encoder (`text_encoder.build_request`) takes pictures only (`<Picture N>: ` + a vision
block tagged video). The upstream reference under `reference/diffusers/modular/packing_ref2va.py`
defines the full presentation and decoders, so build from that.

**Fork work, in order (each testable alone):**
1. `--reference PATH` (repeatable, order is meaning). Kind from the file type: image, video, audio.
   Images go through `refmod_encode.encode_images` (cover-crop onto a canvas sized like the
   reference's `resolve_reference_image_size`); a clip through the video VAE as 17n+5 frames
   (`decode_reference_video`, `video_latents_for_frames`) with its own soundtrack through
   `encode_audio_rows` when it has one; audio on its own likewise. Output feeds the existing
   `build_ref2va_packed_sequence`, so the `--refmod` row path is reused, not duplicated.
2. Text encoder presentation for references: `<Picture i>: ` + vision block, `<Audio j>: ` alone
   (audio never enters Qwen; a clip's soundtrack label comes right before its `<Video k>: `), and
   per clip `<Video k>: ` then, per 2-frame block at 2 fps, `<T.T seconds>` + a 2-frame vision
   block. Mean of the pair's two timestamps, formatted `{:.1f}` (rounds half to even, so the first
   block reads `<0.2 seconds>`). Needs a video processor path beside the image one.
3. Guards in `media_mode_errors`: `--reference` composes with no `--first-frame`, `--last-frame`,
   `--keyframe`, `--extend-from`, `--audio-drive`, `--video-to-audio`, no chaining, and needs a
   Ref2VA `--dit`. Limits 9 pictures, 3 clips, 3 sounds, 12 in all; a clip at most the window.
4. A `--reference-size match|max` switch (the reference-image sizing ComfyUI exposes); default
   `match`.

**Test order for the fork:** (a) one picture reference only, to prove the encoder labels and rows
against a known `--refmod` of the same picture; (b) picture + clip, the edit above with the same
seed, compared with the ComfyUI output; (c) `max` sizing and a prompt that names the face and hair.
Bytes will not match ComfyUI (different stack); judge on scene hold, clothing and face.

**immichring after that.** Add `reference_files` + `reference_roles` to `/api/ng/h3/generate`
(files into the job dir as `ref_N.*`, never client paths; refuse with first/last/keyframes/extend
exactly as the runner does), a Runner flag probe, and send `edit_source` and the replacement
picture from the pane. Two gaps found on the way: immichring's `_upload_assets` in
`routes/enhancedh3NG.py` drops the `replaces` field (h3ir's own client comment describes the same
bug), and the compiler 500s on a real edit clip, so until that is fixed the edit prompt is
written by hand or by a Qwen3-VL compile.

**Also queued:** character-note box (saved text prepended to every sentence), a
`director_profile` box (format not yet confirmed from the h3ir source; do NOT run a
whole-disk `find` to locate the package, look in the venv that runs `h3ir serve`),
effort-level comparison of briefs, a scene-planner above the compiler, managed
`h3ir serve` / Gemma launcher, standalone packaging (the H3 code is mostly
self-contained: engine, jobs, logs, lora, ref2va plus routes; ties are configNG,
model_offloadNG, install helpers and video_analysisNG).

Do not edit any `.py` while a render runs (the Flask reloader kills the job).

## Status 2026-10-29 (written before a reboot; read this first)

**Done and committed.** immichring: last frame `48ba9e3`, timed keyframes `997cf5c`, replace design
`d9b4188`, video tab converts AV1/VP9 downloads to H.264 `3d74a52` (QuickTime/Preview cannot play AV1 on
an M2), `gemma_server.sh` fixed `aee9a89`, `upscale_x2NG.py` WIP `964f82f`. Fork (`minimax-h3-mlx`),
branch `reference-flag`: `--reference` pictures `3e507c0`, clips `1166cff`. The branch lives in a
worktree at `/Volumes/AI/worktrees/minimax-h3-mlx-reference-flag` (tests: `.venv/bin/python
tests/test_reference_encode.py` there). It is NOT merged: the live install is still on `adaln-cache`, so
immichring cannot use `--reference` yet. The install has an untracked `refmod_encode.py` that the branch
now tracks; delete the untracked copy before merging.

**What the tests showed.** `--reference` picture: identity carried well (portrait, 5.5 min). Picture + clip
edit (bedroom clip + portrait, hand-written prompt, seed 1234): clothes changed, scene held, face stayed the
source woman's, identical failure to ComfyUI (which took ~85 min, 20 steps; fork 12.5 min, Fast 3-step).
Untried: `--reference-size max` plus a prompt naming face and hair. Not built: a clip's soundtrack
(`<Audio j>` label and audio rows). `test_text_encoder_parity` needs `torch`, absent from the fork venv, so
the `build_request` refactor is only indirectly covered. Clips for reference are in `~/Movies/h3-fork-tests/`.

**Models moved to `/Volumes/AI/models/` (symlinks left at `/Users/john/models/<name>`):** Qwen3.8-27B-4bit,
Qwen3-Coder-Next-4bit, h3-ref2va (all verified, originals removed; internal free 28 GB -> 190 GB).
`gemma4-26b-mlx` is copied and verified at `/Volumes/AI/models/gemma4-26b-mlx` but NOT swapped: the original
is still at `/Users/john/models/gemma4-26b-mlx`. To finish (server stays up, launchd restarts it):
`mv /Users/john/models/gemma4-26b-mlx /Users/john/models/gemma4-26b-mlx.old && ln -s
/Volumes/AI/models/gemma4-26b-mlx /Users/john/models/gemma4-26b-mlx`, then `kill` the :8080 server (PID from
`pgrep -f "mlx_vlm.server.*8080"`), check `curl 127.0.0.1:8080/v1/models`, then `rm -rf` the `.old`. The
Qwen3-VL-8B-8bit was already in `/Volumes/AI/models/`. These models need `/Volumes/AI` mounted.

**Services.** `com.john.mlx-server` (LaunchDaemon, KeepAlive): Gemma 4 26B on :8080 via
`/Users/john/mlxenv/bin/python`, model path `/Users/john/models/gemma4-26b-mlx`. `com.ringviz.gemma-server`
(LaunchAgent): `gemma_server.sh`, Gemma 3 12B on :8811, had been crash-looping on a missing `uvicorn` (180k
failures, 82 MB `~/Library/Logs/gemma-server.log`, safe to clear); fixed by running under `mlxenv`. Kokoro on
:8880. `h3ir serve` on :8700 needs `H3IR_LLM_URL=http://127.0.0.1:8080/v1` and
`H3IR_LLM_MODEL=/Users/john/models/gemma4-26b-mlx`; I did not find how it is launched, so restart it by hand
from `/Volumes/AI/ComfyUI` (`venv/bin/h3ir serve --host 127.0.0.1 --port 8700`). ComfyUI was on :8182 (killed).
The :8080 Gemma server held 17 GB of the 26 GB swap, which was full.

**2x upscale / face fix, outside Phosphene.** Phosphene's "Upscale & Face Fix" is a `generate_restore` job
to its warm helper, which wraps upstream `ICLoraPipeline.generate_and_save(source_video=..., refine_steps=3)`
with the LTX 2.5 q8 pack, the `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2` adapter and Gemma 4 12B
(`gemma4-12b-ltx25-q4`) as encoder; all on `/Volumes/AI/Pinkio/.../mlx_models`. The helper cannot be imported
(its main loop is module-level), so `upscale_x2NG.py` calls `ICLoraPipeline` directly with the helper's
environment settings (run it with `ltx-2-mlx/env/bin/python3.11`). It crashed twice, right after the text
encoder loaded, on a Metal "Impacting Interactivity" watchdog error, once before and once after adding the
M1/M2 buffer limits (`MLX_MAX_OPS_PER_BUFFER`/`MLX_MAX_MB_PER_BUFFER` 10; `--short-gpu` uses 1). Likely cause:
full swap. Retry after the reboot. Sizes: 640x384 is 5:3, so x2 = 1280x768 is not 720p; a 640x352 draft gives
1280x704, a 960x544 draft gives 1920x1088. Phosphene's "720p fit" is a Lanczos cover-and-crop (up to 8% trim).
Idea: trace which library lines run during one upscale with stdlib `sys.settrace` (no `coverage` in that env).

**Next, in order.**
1. After reboot: `/Volumes/AI` mounted; :8080, :8811, :8700, immichring :5050 up; swap low.
2. Finish the Gemma move (above).
3. Retry `upscale_x2NG.py` on `~/Movies/h3-fork-tests/edit_fork.mp4`; time it; then the trace.
4. `--reference-size max` + face/hair prompt test; encode the same picture as a RefMod and as `--reference`
   at the same size and compare the rows numerically.
5. Qwen/Gemma: a second `h3ir serve` (:8701) on a Qwen vision model; immichring sends a brief there when
   the tray holds a clip. First test whether Qwen avoids the compiler 500 on the bedroom clip. (Gemma's clip
   output is rejected by the compiler; Qwen's is accepted but its story is weaker.)
6. immichring for references: `reference_files` on `/api/ng/h3/generate`, job files, pane, runner flag probe;
   an in-app RefMod builder (pick ~8 images with the image picker, encode with `refmod_encode`, save to the
   RefMod library; one stacked block, unlike 8 separate `--reference` pictures); a 640x352 draft preset; an
   "Upscale 2x" step after a render; keyframes on a chained 10 s clip; click through the "keyframe at" field.
7. Fork: soundtrack references, then merge `reference-flag`.
8. Bugs: `_upload_assets` in `routes/enhancedh3NG.py` drops `replaces`; the compiler 500s on a real
   `edit_source` clip; clear the 82 MB log.

The user prefers smaller, modular ports over wrapping Phosphene or ComfyUI code; prefer calling the library
class directly (about 25 lines for the upscale) to importing a large wrapper.

## Licences

Ring Viz: PolyForm Small Business 1.0.0 (from 2026-08-23). H3 runner: Apache-2.0
(keep its LICENSE/NOTICE on any files taken from it). `h3ir`: Apache-2.0.
Phosphene: MIT. ComfyUI: GPL-3.0, so use it as a behavioural reference only.
Not legal advice.
