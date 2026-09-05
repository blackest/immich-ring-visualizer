# Decoupling character-sheet generation from Phosphene

**Status:** planning doc, nothing built yet. Written at the end of a session
that (a) built the current Phosphene-dependent integration and (b) tuned
Phosphene's character-sheet prompt/seed handling. Read this whole file
before writing code — several things below are open decisions, not settled
answers.

## The goal, in one sentence

Take the character-sheet generation logic that currently lives inside
Phosphene (a separate app/process) and port it directly into Ring
Visualizer, so Ring Visualizer can generate a 3-view character sheet from
one photo without Phosphene running at all.

## Why

Ring Visualizer currently drives this feature by making HTTP calls to a
Phosphene process running on `127.0.0.1:8198` (see `config.py`'s
`PHOSPHENE_BASE_URL`). That's a live runtime dependency on a separate,
somewhat fragile app — Phosphene is a single 27k-line Python file
(`mlx_ltx_panel.py`) that Pinokio manages, and switching branches or
restarting it can break things out from under Ring Visualizer (this
happened once already this session). John wants to know how hard it'd be
to remove that dependency. Short answer worked out this session: fairly
doable, roughly a day or two, because the actual image-generation step
turns out to already be decoupled from Phosphene's own code — it's a
subprocess call to a standalone tool. See "What Phosphene actually depends
on" below.

## Current state (what exists today)

**Ring Visualizer side** (this repo, branch `refactor/js-module-split`) —
the Phosphene-dependent integration, all of which this plan replaces:
- `config.py` — `PHOSPHENE_BASE_URL = os.environ.get("PHOSPHENE_BASE_URL", "http://127.0.0.1:8198")`
- `ring_viz.py` — imports and registers `phosphene_bp`
- `routes/phosphene.py` (305 lines) — Flask blueprint: `GET /api/phosphene/status`,
  `POST /api/phosphene/characters` (create a draft), `POST /api/phosphene/characters/<id>/sheet/generate`,
  `POST /api/phosphene/sheet-from-asset` (Immich asset → draft + sheet in one call),
  `POST /api/phosphene/sheet-from-upload` (video frame / disk file, via multipart upload)
- `static/phosphene-sheet.js` (137 lines) — panel logic; branches on
  `lastNeighborsRender.centerId` (real Immich asset) vs. `lastVideoRingState`
  with the `'__anchor__'` sentinel (video frame / folder image)
- `templates/index.html` — the `#phosphene-sheet-panel` block (name input,
  generate button, "choose photo from disk" button, status/result area)

**Phosphene side** (`/Volumes/AI/Pinkio/api/phosphene.git`, branch
`local-dev`, currently at commit `1efcf86`) — this is what gets ported, NOT
deleted from Phosphene. Phosphene keeps working exactly as it does today;
we're copying logic out, not removing it from there.

## What Phosphene actually depends on (the important discovery)

`generate_character_sheet()` calls `agent_image_engine.generate(...)`,
which for the `hidream_inline` engine (the one this feature uses) shells
out to a **standalone HiDream install that already lives outside
Phosphene**, at `~/HIDREAM-O1-MLX-LAB-active` (or wherever
`HIDREAM_LAB_DIR` env var points, or `~/AI/HIDREAM-O1-MLX-LAB-active`,
etc. — see `_resolve_hidream_lab_dir()` in `image_engine.py:1555`). It's a
venv + a generator script, invoked as a subprocess with `--prompt`,
`--seed`, `--ref-images`, etc. Phosphene doesn't run the model in-process;
it just knows how to build the right command line and read the PNGs back
off disk.

That means porting this feature does **not** require reimplementing any ML
or duplicating Phosphene's 27k-line file. It requires porting:
1. A small amount of character bookkeeping (bundle.json, the prompt
   template, image compositing) — plain Python, no ML.
2. A ~200-line subprocess wrapper that calls the *same* standalone HiDream
   install Phosphene calls. Once ported, Ring Visualizer talks to HiDream
   directly — Phosphene drops out of the loop entirely for this feature.

## Part 1 — what to port: character bookkeeping + prompt (from `mlx_ltx_panel.py`)

All line numbers below are as of Phosphene commit `1efcf86` on `local-dev`
and **will drift** — treat them as a starting point for grep, not gospel.

- `CHARACTER_SHEET_VIEWS` (line ~23841) — the 3-view dict: `front`,
  `profile_left`, `three_quarter`, each mapped to a camera/pose phrase.
- `_character_sheet_view_prompt(view_phrase, wardrobe="")` (line ~23872) —
  builds the per-view prompt. **This was just tuned tonight** to fix real,
  observed defects (see "Things learned this session" below) — port the
  current version, don't rewrite it from scratch.
- `_compose_character_sheet_row(image_paths, out_path)` (line ~23918) —
  PIL: lays the 3 rendered views into one horizontal strip image.
- `_character_safe_id(value)` (line ~3294) and `_CHARACTERS_ID_RE` (line
  ~3284) — trigger-name validation (alphanumeric/space/underscore/hyphen).
- `create_draft_character(trigger, source_image_path, *, name="",
  pronoun="", subject_noun="")` (line ~3523) and
  `DraftCharacterExistsError` (line ~3518) — creates a character record
  from one photo: validates the trigger and image, copies the image to
  `<characters_dir>/<trigger>/avatar.<ext>`, writes `bundle.json`
  (`schema: "phosphene/character_bundle@1"`).
- `generate_character_sheet(character_id, *, engine_override="hidream_inline",
  views=None, wardrobe="", seed=-1, anchor_chain=True)` (line ~23954) — the
  orchestrator: resolves the character's reference image, loops over the 3
  views, calls the image engine once per view (see Part 2), chains the
  rendered front view as a second reference for later views
  (`anchor_chain`, fixes hair-color drift on side angles — read the
  comment at the call site, it documents a real measured failure), then
  composites and writes `sheet.png` + `sheet.json` atomically.
- `CharacterSheetBusyError` (line ~23866) and `_CHARACTER_SHEET_ENGINES`
  (line ~23859) — the busy-GPU error and the allowlist of engines that
  actually condition on a reference image (only these can build a sheet;
  text-only engines would render a stranger).
- `LORAS_DIR` (line ~1359) and `_CHARACTERS_CACHE_PATH` (line ~3286) —
  Phosphene's path constants. **Don't copy these verbatim** — Ring
  Visualizer uses its own storage location — see "Where character data
  lives" below.

You likely do **not** need `list_characters()`'s full `include_drafts`
machinery (trained-vs-draft LoRA discovery, `ltx_compatible` fusion
checks) — that exists because Phosphene manages a whole
character-to-LoRA-training lifecycle. Ring Visualizer's use case is
simpler: one photo in, one sheet out. A minimal record lookup (does
`<dir>/<trigger>/bundle.json` exist, what's its avatar path) is probably
enough — see "Where character data lives" below for the confirmed shape.

## Part 2 — what to port: the HiDream subprocess engine (from `image_engine.py`)

From `image_engine.py` (current HEAD `4e1dde6`):

- `_generate_hidream(prompt, n, width, height, output_dir, base_seed,
  config, refs=None, on_log=None)` (line ~1631) — the actual subprocess
  call. Resolves the venv python and model path, snaps resolution to
  HiDream's trained aspect ratios, builds the `--ref-images` /
  `--editing-scheduler` / `--fb-cache` command line, runs it with a
  watchdog timeout (default 1500s via `PHOSPHENE_HIDREAM_TIMEOUT_S`),
  parses the resulting PNGs. For a character sheet, always call with
  `n=1` (one full-res image per call — a batched `n>1` call would give
  every candidate the SAME prompt, which is wrong across different views).
- Helpers it needs: `_resolve_hidream_lab_dir()` (line ~1555),
  `_resolve_hidream_python(config)` (line ~1608),
  `_resolve_hidream_model(config)` (line ~1615),
  `_snap_to_trained_resolution(width, height)` (line ~1597),
  `_patch_align(value, patch=32)` (line ~1621),
  `_clean_subprocess_env()` (line ~84) — strips macOS `Malloc*` env vars
  that otherwise spam stderr from HiDream's subprocess tree.
- Constants: `HIDREAM_LAB_DIR`, `HIDREAM_DEFAULT_PY`, `HIDREAM_DEFAULT_MODEL`,
  `HIDREAM_GENERATE_SCRIPT`, `HIDREAM_PATCH_SIZE` (=32),
  `HIDREAM_TRAINED_RESOLUTIONS` (line ~1587).
- The relevant fields off `ImageEngineConfig` (dataclass, line ~157):
  `hidream_python_path`, `hidream_model_path`, `hidream_recipe` (`"dev"`),
  `hidream_steps` (6), `hidream_noise_scale` (7.5), `hidream_noise_clip_std`
  (2.5), `hidream_guidance_scale` (0.0), `hidream_editing_scheduler`
  (`"flow_match"`), `hidream_fb_cache` (True), `hidream_fb_threshold`
  (0.15), `hidream_fb_keep_last` (8). You don't need the whole dataclass —
  just these fields with the same defaults, or a plain dict/simple class.
- `ImageJobCancelled` (line ~106) — only meaningful if Ring Visualizer adds
  its own cancel/stop button; otherwise skip it and let the watchdog
  timeout be the only abort path.
- **Skip** `_register_active_proc` / `_unregister_active_proc` — that's
  wiring for Phosphene's own `/stop` endpoint cascading `killpg` across
  every active job. Not needed unless Ring Visualizer builds an equivalent
  cancel feature.

## Where character data lives (decided)

Ring Visualizer already has a convention for this: every analysis job
(video, folder, Immich selection) exports its curated training images to
`EXPORT_DIR/<source_name>/` (see `config.py`'s `EXPORT_DIR` and
`routes/export.py`'s `dest_dir = os.path.join(EXPORT_DIR, source_name)` —
`source_name` is a free-form label: a video filename, `"folder_set"`,
an auto-generated `"immich_selection_N"`, or whatever the user typed).
There's a real example on disk today at `exports/rippedley/`, a flat
folder of 34 curated `frNNNNN_simX.XX.png` crops.

Character-sheet data follows the *same* convention rather than mirroring
Phosphene's `mlx_models/characters/<trigger>/` tree: it uses the same
`<name>` the character-sheet panel already collects (the optional
"Character name" field, i.e. the `trigger`), and lives at
`EXPORT_DIR/<name>/character/`:

    exports/<name>/
      fr00002_sim0.39.png          <- existing: curated training crops (untouched)
      ...
      character/
        avatar.png                 <- the single reference photo used
        bundle.json                <- {trigger, name, pronoun, subject_noun, schema}
        sheet.png                  <- composited 3-view strip
        sheet.json                 <- per-view prompt/seed/path metadata
        sheet_views/
          front/*.png
          profile_left/*.png
          three_quarter/*.png

The `character/` subfolder keeps sheet-generation artifacts out of the
flat crop list a LoRA trainer would glob over, while still sitting right
next to that name's curated training images — one folder per character/
project, same as today. If John later wants the 3 sheet-view renders
folded directly into the flat training-crop list (not just organized
alongside it), that's a small follow-up (copy or symlink them up a
level) — don't assume it, ask first.

This also means the "trained-vs-draft LoRA" distinction from Phosphene's
`list_characters()` doesn't apply here at all — there's no LoRA-training
lifecycle in Ring Visualizer to track. A "character" here is just: does
`exports/<name>/character/bundle.json` exist yet.

## GPU contention (decided)

No file lock, no patch to Phosphene. John's call: "don't run both at once"
as a plain convention is enough — the same level of rigor already used for
the LoRA-training/WindowServer GPU crash on the Mac Studio M2 Max (fixed
by locking the screen during training, not by building real coordination).
Concretely: don't kick off a character-sheet generation in Ring Visualizer
while Phosphene is mid-render or mid-training, and vice versa. Nothing to
build for this — just don't add a "convenient" feature like auto-retry-on-
failure that would paper over a real GPU-memory clash by silently retrying
into it.

## One more thing that doesn't need deciding

The existing panel/JS (`static/phosphene-sheet.js`, the
`#phosphene-sheet-panel` HTML block) already handles all three sources
(Immich asset, video frame, disk file) correctly — that UI layer doesn't
need to change. Only the *backend* routes it calls need to stop proxying
to Phosphene and start calling the ported code directly.

**All decisions are now settled — this file is ready to execute from
without stopping to ask John anything, unless something in the real code
contradicts an assumption above.**

## Suggested build order

1. Create a new module in this repo (e.g. `character_sheet.py`) with the
   ported Part 1 logic (prompt template, `create_draft_character`,
   `generate_character_sheet`, compositing) plus a new `hidream_engine.py`
   (or similar) with the ported Part 2 subprocess wrapper. Keep them
   separate — one is pure bookkeeping, the other is the engine call — the
   way Phosphene itself keeps `image_engine.py` separate from
   `mlx_ltx_panel.py`.
2. Skip the GPU lock — per "GPU contention (decided)" above, this is
   handled by convention (don't run both at once), not code.
3. Rewrite `routes/phosphene.py` (rename it — it's no longer calling
   Phosphene) to call the new local module instead of making HTTP
   requests, writing to `exports/<name>/character/` as shown above. The
   route *shapes* (`sheet-from-asset`, `sheet-from-upload`) can probably
   stay identical since the frontend already expects them.
4. Remove `PHOSPHENE_BASE_URL` from `config.py` and the blueprint's HTTP
   client code once nothing calls it.
5. Port (or write fresh, matching the pattern) tests equivalent to
   Phosphene's `test_character_sheet.py` / `test_character_draft.py` —
   they're a good model for what to cover: prompt content, seed handling,
   view dedup/validation, busy-GPU handling, sidecar JSON schema, atomic
   sheet writes.
6. Test all three sources end-to-end against the real HiDream install
   before considering Phosphene fully removable as a dependency.

## Things learned this session worth carrying over

- **The prompt was just tuned** (Phosphene commit `1efcf86`) to fix two
  real defects seen in an actual render: skin tone drifted warmer/more tan
  than the reference across all 3 views, and the profile_left view
  loosened a pulled-back hairstyle to hair worn down even with correct
  anchor-chaining (`refs_ignored: false` — the reference images WERE
  passed through, the model just deviated on the loosest-specified
  attribute at a harder pose). Port the current prompt text, not an older
  version.
- **Seed handling was changed the same session**: `generate_character_sheet`
  now resolves ONE seed for the whole sheet (either the caller's explicit
  seed, or one random draw at `seed=-1`) and uses it for every view,
  instead of the old `seed + i` per-view offset (or an uncontrolled random
  draw per view). Port this behavior, not the old per-view-offset version.
- `anchor_chain=True` (default) feeds every view after the first the
  rendered front view as a second reference alongside the avatar — this
  fixes a real, previously-observed hair-*color* drift on side angles.
  There's a documented trade-off (side views can under-rotate the head)
  — read the comment at the `generate_character_sheet` call site.
- Each HiDream call for a sheet view uses `n=1`, one full-resolution image
  per call — never batch multiple views into one `n>1` call.
- Character-sheet generation deliberately does NOT queue behind other GPU
  jobs on the Phosphene side — it fails fast (`CharacterSheetBusyError`,
  429) rather than waiting, because a sheet is 3+ full renders and
  silently queueing behind e.g. a 12-minute video render is worse than an
  honest busy error. Worth keeping that philosophy in the port.
- A real render of 3 views took about 28 minutes total (~28 steps at
  ~20s/step per view × 3 sequential views) on the reference hardware —
  make sure whatever HTTP timeout Ring Visualizer's own frontend uses
  accounts for that (this bit Ring Visualizer once already, when it was
  still calling Phosphene over HTTP with too short a timeout).
