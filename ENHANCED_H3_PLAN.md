# Enhanced H3: compiled briefs, per-window prompts, and full references

**Status: PLAN (study done, 2026-10-27). Only step 1 below is built.** Nothing in
the H3 runner fork, Phosphene or ComfyUI has been changed for this.

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
2. **`enhancedh3` pane** (follow `ADDING_A_PANEL.md`). Sentence + optional
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

## Licences

Ring Viz: PolyForm Small Business 1.0.0 (from 2026-08-23). H3 runner: Apache-2.0
(keep its LICENSE/NOTICE on any files taken from it). `h3ir`: Apache-2.0.
Phosphene: MIT. ComfyUI: GPL-3.0, so use it as a behavioural reference only.
Not legal advice.
