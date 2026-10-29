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

## Update 2026-10-29, later (steps 1-3 done, step 5 mostly done, step 4 changed)

**Done.** Reboot checks passed. Gemma move finished: `gemma4-26b-mlx` is now a symlink to
`/Volumes/AI/models/`, :8080 restarted on it, the 15 GB `.old` removed (internal free 259 GB).
`upscale_x2NG.py` crashed a third time on the Metal watchdog with buffers at 10 and swap at 0 (so not
swap); with `--short-gpu` (1 op per buffer) it ran clean: 640x384 -> 1280x768, 124 frames, audio kept,
8 min 35 s (3 denoise steps at ~2 min 18 s, VAE decode 44 s). One op per buffer is now the default on
M1/M2 (`ba99d3f`). Output: `~/Movies/h3-fork-tests/edit_fork_x2.mp4`.

**`--reference-size max` is not worth testing.** It means upstream's 2048 px short edge, about 10x the
pixels of a 640x384 render. At 1.5 min it had 24.7 GB resident, 52 GB wired and swap climbing, so it was
stopped. The flag only takes `match|max`; if size is ever worth testing, add a numeric value (768 or
1024) in `refmod_encode.reference_size` and the argparse choice.

**The compiler and video.** `h3ir` does not send the model a video: `analyse.py` samples three frames
(10/50/90%) and sends them as `image_url` stills, so the model infers the motion between three panels.
Both Gemma 4 and Qwen have a `video_token_id`; neither is used. What differed is how well each followed
the compiler's format on the bedroom `edit_source` clip:
- Gemma 4 26B: card `lighting` came back as a dict, `style._tidy()` called `.strip()` on it, 500.
- Qwen3-VL-8B: cards fine, but the written brief failed R24 (and R2, T12 on a retry) twice, so the
  compiler returned its fallback draft, which kept BOTH women. Unusable for an edit.
- Qwen3.8-27B-4bit (`/Volumes/AI/models/`, `Qwen3_5ForConditionalGeneration`): status `ready`, a proper
  edit brief (`<Subject 1>` portrait woman preserved, `<Subject 2>` motion/timing only).
A possible Gemma fix is coercing non-string card fields; untested. The "heretic" Qwen models are not on
disk (the cached `Youssofal/Qwen3.6-27B-...Heretic` has only config and index; the 32B
`llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic` has GGUF and bf16 but no MLX build).

**Edit test, compiled brief (seed 1234, 640x384, 124 frames, Fast 3-step, Ref2VA Q8, `match`).**
`--reference` portrait (`pg-z-image-turbo_00367_.png`) + the bedroom clip
(`exportsNG/_h3gen/602dff368f9d/extend_src.mp4`), prompt = the 27B's brief
(`~/Movies/h3-fork-tests/edit_prompt_compiled.txt`). 13.6 min, ~20 GB active, swap peaked ~7.5 GB. Result
(`edit_compiled.mp4`): face, straight dark shoulder-length hair and jacket all match the portrait; room,
moon, window, headboard, camera and seated pose held. The earlier hand-written run kept the source
woman's face and wavy hair. Caveats: one seed; the brief and my added face/hair wording changed together
(a hand-written run with that sentence at `match` would say which did it); the satin skirt vanished and
her legs are bare, because the brief says nothing below the jacket (a compiler gap).

**Wired into immichring (uncommitted until now).** `h3ir_serveNG.sh gemma|qwen` starts the compiler on
:8700 (Gemma 26B on :8080) or :8701 (Qwen3.8-27B on :8082, started if not up). New setting
`h3ir_video_base_url` (:8701). `routes/enhancedh3NG.py` sends the uploads and the brief to :8701 when any
asset is a video, remembers which compiler wrote each brief for refine, and the "isn't running" error
names the right launch command. Tests: `tests/test_enhancedh3_routingNG.py` (8). The launcher itself was
only syntax-checked. The Qwen server holds ~15 GB: stop it (`pkill -f "mlx_vlm.server.*--port 8082"`)
before a render.

**Still open.** `h3ir serve` is stopped (start with `./h3ir_serveNG.sh gemma`). Steps 4 (RefMod vs
`--reference` row comparison), 6, 7, 8 above; clear the 82 MB `~/Library/Logs/gemma-server.log`.

## Update 2026-10-29, evening: references in the app, and two 30 s boxer chains

**References are in the app** (`82e04c7`, `h3_referencesNG.py`; fork `reference-flag` fast-forwarded into the
install, not pushed). Verified: a render through `/api/ng/h3/generate` is byte-identical to the command-line
one. Found in the browser and fixed: a first picture defaulted to a first-frame role (now `subject` once a clip
is in the tray, unless the user chose); Seconds defaulted to 10 (a reference brief is one 5 s window, so it is
capped at 5.167 with a note); the compiler's refusal showed as a Python dict. Also fixed (`38e5b9b`): the length
cap compares frames, so the compiler's 30.677 s (736 frames, 6 windows, same as 30.0) is accepted. A swap wants
roles `edit_source` (clip) + `replacement_subject` (picture) + a note naming who it replaces; the compiler refuses
`replacement-target-unnamed` otherwise. Per-window references are untested: a reference render is one window only.

**How a chain really works** (read from `generate_staged.py` ~2290 and the job metrics). Window 1 starts from
`--first-frame` if given. Each later window gets the previous window's last frame as its "first" keyframe
(delivered frames = 124 + 5 x 123 = 739, trimmed to 736; audio crossfade one frame). `chain_context_frames` is 0
in the app: no motion or voice history crosses a join, so a window restarts the action from one still (window 2
re-raised the head window 1 had just lowered). The runner's comment says carried context brings motion AND voice
across; it is in steps of 17 frames, not exposed by the app, untested. Timed keyframes ARE per-window anchors
inside a chain (a keyframe at frame 0 of a later window is skipped; the runner prepends an alignment line to the
window prompt), so a face picture can be pinned at chosen times; it pins the whole frame, so only where the shot
matches. `<Picture N>` in a window after the first points at nothing (only window 1 has the picture).

**The boxer test** (intent "An old boxer reminices about his former career and the fight he lost , improvise
dialogue", 29.5 s, bold, tarantino, effort max, seed 42, line "I should have won", draft 640x384, Fast).
Run 1, no picture: ~33 min; style, voice and pacing held; the compiler split the supplied line one or two words
per window ("I should" / "have"), a drawn-out word worked; the face changed from silver-haired/clean (window 1)
to bearded after a cut to the hands. Re-render of scene 1 as a reference render (bearded frame, 5.8 min) kept the
face but copied the reference's whole frontal framing, not the low-angle close-up asked for, and window 2 of run
1 opens on the OLD silver man for ~1.1 s (it continues the old window 1, and "have" is spoken over it), so a
hand-spliced replacement would flash him: replace-a-window needs window 2 re-rendered from the new last frame
(Extend). Run 2: first-frame = the bearded frame, same settings, 34.0 min, 719 frames: bearded man held through
the hands (windows 3-4, ~10 s, long, same text twice) and the cut back to the face (window 5, relit with a blue
window drawn from a wind-on-glass sound cue). Gemma compiled it in ~30 s. Backups: `~/Movies/h3-fork-tests/
boxer_30s/` and `boxer_30s_v2/`; the view's saved state is backed up in `boxer_30s/enhancedh3_state_backup.json`.

**Compiler facts.** Creativity is an h3ir input (restrained/balanced/bold/extreme; default balanced); speech is
licensed from bold. h3ir 0.4.1 500s on a first-frame picture plus a dialogue entry; the app works around it by
writing the line into the sentence. With 3 shots over 6 windows the same shot text is repeated in two windows
(flagged "carries on the previous shot"; the line is spoken twice, which here worked as emphasis). Setting Shots
to 6 gives one shot per window.

**Ideas, not built.** Expose `--chain-context-frames`; per-window references (one reference job per window,
face repeated, then a join); a replace-a-window-and-re-extend step in the app; strip dangling `<Picture N>`
labels from later windows; a second speaker / off-screen word via Extend (text prompt, no compiler); a hint for
swap roles; numeric `--reference-size`; reference audio in the runner.

## Licences

Ring Viz: PolyForm Small Business 1.0.0 (from 2026-08-23). H3 runner: Apache-2.0
(keep its LICENSE/NOTICE on any files taken from it). `h3ir`: Apache-2.0.
Phosphene: MIT. ComfyUI: GPL-3.0, so use it as a behavioural reference only.
Not legal advice.
