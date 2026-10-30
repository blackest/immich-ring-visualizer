# Handoff for another Claude: H3 prompt compiler + adaLN notes (2026-10-27)

Written by a Claude chat session (phone, read-only access to a few folders on
this Mac Studio). I have NOT run, installed or edited any code. Everything
below is either read from files/pages or reasoned from what John said. Items
marked (unverified) are guesses. John's standing rule: do not build or edit
anything until he explicitly says so.

## 1. The problem
John's H3 renders from plain prompts like "man at a table" come out flat.
H3 expects a structured prompt, and nothing in his app builds one.

## 2. What his app does today (read from ~/immich-app/repo/immichring)
- routes/h3NG.py takes the prompt form field (required, non-empty) and hands
  it on unchanged.
- h3_engineNG.py generate_h3_video_ng() passes the prompt verbatim as the
  first argv of the minimax-h3-mlx runner, with --dit, --compact-root,
  --frames, --height/--width, --steps, --seed, optional --first-frame,
  speed adapters, --lora, and --refmod via h3_ref2vaNG.refmod_argv_ng().
- grep for integrated_multimodal_description / soundscape / "<Picture" in the
  app code found nothing: no prompt structuring anywhere.
- h3_ref2vaNG.py validates refmods before launch (needs the h3ref model, no
  chained windows, no first-frame, no turbo/v1.3, max 9 images + 3 videos).

## 3. The H3 prompt format (from MiniMax's official guide + skill; not run)
Three named sections:
  integrated_multimodal_description: [Shot 1] style, composition, subject,
     action, camera, dialogue, diegetic sound. Later shots: "At 00:03.500,
     the camera cuts to ...". First shot has no timestamp.
  overall_soundscape: ambience
  non_diegetic_music: score or N/A
Rules: dialogue verbatim; camera moves as actions with amplitude/speed; say
"static" explicitly if wanted; one dominant action per shot; concrete detail
over words like "cinematic". Reference labels (<Picture 1>, <Video 1>,
<Audio 1>) apply when references are passed as media.
Sources: huggingface.co/MiniMaxAI/MiniMax-H3 docs/VIDEO_PROMPT_WRITING_GUIDE_
base_en.md ; github.com/MiniMax-AI/MiniMax-H3 skills/h3-prompt-writing.

## 4. Correction about refmods
Earlier in the chat I guessed missing <Picture N> labels explained weak
refmod results. From PR minimax-h3-mlx#2, a RefMod is a saved VAE latent
packed into the sequence as extra rows, not a picture the prompt points at.
So the labels may matter less. A same-seed A/B (plain vs compiled brief, with
the refmod on) would show whether the brief helps at all.

## 5. OpenH3-IR (the compiler found via AiKonFlux's YouTube video)
- Repo: github.com/ruashots/open-h3-ir (Apache-2.0, v0.4.1, "young project,
  pin a commit"). Video: youtu.be/zd97ZAMEjJA (Gemma/Qwen via LM Studio in
  ComfyUI). Note: its README says H3's own licence excludes EU/UK/Korea/US.
  John said licensing is not a worry.
- Turns plain text + optional image/video refs into a validated H3 brief via
  any OpenAI-compatible LLM endpoint. Compiler needs no GPU. ~14k lines in
  h3ir/, deps: fastapi, uvicorn, pydantic, httpx, tiktoken.
- Interfaces: CLI (h3ir doctor / compile / budget / directors / eval /
  controls), HTTP (h3ir serve, docs/calling-the-api.md), ComfyUI nodes.
- Needs a vision-capable model to read reference pictures. Cannot hear audio.
- It snaps duration to H3's frame grid (10 s -> 243 frames = 10.125 s).
  CHECK this against _h3_chain_plan_ng and any frame-count-keyed adaLN cache.
- Cloned to ~/immich-app/repo/open-h3-ir (tarball; a failed first git clone
  left ~/immich-app/_to_delete_open-h3-ir-partial-clone for John to delete).
  NOT installed. RUN_ME.md in the same folder has the Terminal steps. The
  chat's shell is a Linux VM, so venv/pip must run natively on macOS.
- Candidate endpoints: Rachel's 27B (address unknown, check it can see
  images), gemma_server.sh (mlx_vlm.server, gemma-3-12b-it-4bit, port 8811,
  12B so briefs may be weaker), Ollama (:11434/v1), LM Studio (:1234/v1).
- Suggested integration if it proves worthwhile: run `h3ir serve` as a
  separate process and POST from routes/h3NG.py, showing the brief in the
  prompt box before rendering. Not started.

## 6. Test plan John agreed is worth trying (not yet run)
Same seed, 3-4 real shots incl. one refmod shot: plain sentence vs compiled
brief. If the brief wins, wiring it in is worth it. Compile first, unload the
LLM, then render (GPU memory).

## 7. adaLN cache notes (from John, discussed in chat; code not seen by me)
- Ref2VA bf16 ~66 GB; Q8 build ~44-47 GB on disk with adaLN kept bf16 (~1/3).
  adaLN builds a small cache; without it resident weights ~22 GB.
- Cache is per workflow and depends on settings, NOT on the prompt, so
  compiled briefs can reuse it.
- John says adaLN is currently lazy-loaded and he has enough to use the cache.
- Suggestions I made (unverified against his code): (a) a cache miss should
  fail loudly or build only on request, not silently load adaLN on a low-RAM
  Mac; (b) the cache key must include every setting that affects it (steps,
  size, frame count, speed adapter); (c) test with/without adaLN, same seed,
  compare frames and peak memory; (d) check --metrics records peak memory.
- Build peak used most of 96 GB; converting shard by shard may lower it
  (unverified). 64 GB "just about" runs the full 44 GB version.

## 8. Phosphene (mrbizarro)
Repo: github.com/mrbizarro/Phosphene (MIT). Issue #63 tracks multi-image /
Ref2VA. John's own comments there (as blackest) cover PR minimax-h3-mlx#2
(--refmod), timings, bf16 vs Q8, the 8-image RefMod + turbo + 2x face result
(~11 min total), and the adaLN idea. mrbizarro has not committed to a panel
build or timeline; says peak RAM is the real gate. John mentioned a Phosphene
update shortly before this note; I did not read what changed. Worth checking
before updating his own H3 runner or h3_constraints.txt pins.

## 9. Open questions for John
- Which endpoint to use for the compiler; does Rachel's 27B support vision?
- Does the adaLN cache key cover frame count and speed adapter?
- Wire the compiler in via HTTP, or just use it by hand?
