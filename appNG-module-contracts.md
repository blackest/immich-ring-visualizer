# appNG.js — Module Contracts (verified against source only)

Status: planning doc. Every entry below has been read directly from
immich-ring-visualizer (dev-ng branch) — nothing here is from discussion or
memory of past design talk. Where an earlier version of this doc guessed
wrong (noted inline), the guess is corrected, not kept.

Rule for every module: declared input, declared output, no reaching into
another module's state. Reusable/toolkit-tier modules must not know about
"projects" or "characters" at all — that knowledge stays in the app-tier
assembly code (CharacterProject / ProjectManager).

Note: Rachel is independently extracting from this same source. Where our
cuts land in the same place (see initguing/initchar below), that's a good
sign the boundary is real. Compare contracts (entry/exit shape), not
implementation, when merging.

---

## Confirmed toolkit-tier modules

### generate_hidream_ng (hidream_engineNG.py)
Already correctly isolated — no change needed.
- Entry: `(prompt, n, width, height, output_dir, base_seed, config, refs?)`
- Exit: list of generated image paths (one subprocess call, model loads once)
- n=1 always in current character-sheet usage (batched n>1 would give every
  candidate the same prompt, so per-shot generation calls this with n=1)

### Nearest-to-target picker (currently "PosePicker")
CORRECTION: earlier version of this doc described a 7x7 fixed reference
grid — that was carried over from unbuilt discussion, not the real code.
Verified actual implementation: `buildPickerCellsNG` hardcodes a **3x3**
grid — nearest 9 candidates to a dialed-in yaw/pitch target (default =
candidate pool centroid), tolerance-filtered, bulk-selectable.
- Entry: candidate pool (scored/pose-tagged) + target {yaw, pitch} + grid
  size (should be parameterized, not hardcoded to 9, for reuse)
- Exit: user's selected subset of the pool
- Toolkit-tier once genericized: nothing here needs to know "pose" specifically
  — it's "N closest scored items to a dialed target," reusable for anything
  with a 2D (or n-D) similarity space.

### FrameScrubber (currently inline in ProjectManager.renderFramePreview + drawFrame)
Frame-by-frame stepping through already-extracted frames, canvas-drawn,
with a *separate* `<audio>` element manually kept in sync (this is
deliberate, per existing code comment — avoids needing real video seek
for frame-accurate stepping).
- Entry: `{ frames[], audioObjectUrl, currentFrame }`
- Exit: current frame index + play/pause state (events)
- Owns: canvas draw (`drawFrame` logic), audio-sync timer, step/seek math
- Currently tangled: reads `project.video.*` directly, writes to
  module-level DOM els (`previewCanvasEl`, `frameCounterEl`, etc.) instead
  of a supplied render target.

### VideoPopout (currently `PlaybackModal`)
A real `<video>` element with native audio — mechanically different from
FrameScrubber (this is NOT the same module wearing two hats; two
different players for two different jobs, confirmed by source comment).
Handles two sources: raw source clip, and reconstructed
rejected-frames-blanked build.
- Entry: `{ src, fps, title, rangeStart?, rangeEnd? }`
- Exit: play state changes; exposes `stepFrame(delta)`, `play/pause/seek`
- Currently tangled: `open()`/`openBuild()` take a whole `project` and pull
  `project.video.objectUrl`, `project.playback.url`, `project.name`,
  `project.video.fps` directly; `stepFrame()` re-looks-up the project from
  `ProjectManager.projects` by id just to read fps. Writes directly to
  module-level DOM els (`playbackVideoEl`, `playbackModalTitleEl`, etc.)
  Needs a thin adapter at each of the two call sites to extract plain
  values before handing off, rather than the player reaching into `project`
  itself.

### PoseClassifier / IdentityScorer
Not yet located/verified against source this session — still open. Do not
assume prior description of these (per-bin expected-similarity table etc.)
is real code; that was design discussion, not confirmed implementation.
Verify before relying on it.

---

## App-tier state machine (confirmed via renderMain/renderStage/renderVideoStage)

`renderMain()` is already, functionally, close to a correct StageController
decision tree — verified from source:
- no project open -> placeholder ("press + to start one")
- project exists, no task picked -> placeholder, bottom-row task buttons
  disabled until a project exists (`renderBottomBar`: `btn.disabled = !active`)
- task picked (video/folderzip/immich), no analysis yet -> placeholder
  prompting to run analysis
- analysis done -> hands off to `renderStage()`

Flaw: fused to specific DOM elements (`mainPlaceholderEl`, `stageWrapEl`,
`sidebarEl`) instead of returning a state name and letting a separate
renderer do the showing/hiding. Cut line: separate the state decision
(pure) from the DOM toggle (thin renderer).

`renderStage()` — clean dispatcher (immich-neighbor-browse vs.
video/folderzip). Barely needs touching.

`renderVideoStage()` — THE actual monolith-inside-the-monolith. One
function currently does at least four unrelated jobs:
1. HUD text derivation (mode label, anchor label, match %)
2. Sidebar "currently selected" panel update
3. Mode branch: pose-list view vs. ring view (two different visualizations
   sharing one function)
4. Ring view's DOM construction — band circles, center node, per-result
   node placement, hover-preview wiring, all built imperatively inline

Cut lines: HUD updater, sidebar panel updater, PoseListView (already
semi-separate via `renderPoseListNG`, just needs the branch removed from
here), and a proper RingVisualizer component (anchor + banded/scored
results + ringScale in, rendered nodes + hover/select events out — this
one is toolkit-tier too: "place items by score, closest = center" isn't
face-specific).

Not yet fully read: rest of `renderVideoStage` past the point covered this
session, and `renderImmichStage` — pick up here next.

---

## initguing.js / initchar.js (Rachel's split — independently converged)

Matches the app-tier boundary found above:
- `initguing.js` = the "no project" shell/blank-state branch of
  `renderMain()`, as its own file instead of inline.
- `initchar.js` = the `+`-pressed -> create project -> name it transition
  (`createProject()` + naming UI), pulled out as its own unit with an
  explicit handoff instead of living inline in ProjectManager.

Both independently extracted from the same appNG.js source, not
coordinated in advance — treat convergence here as a signal the boundary
is real. When comparing any future piece against Rachel's version, compare
entry/exit contracts first; matching contracts = interchangeable even if
implementations differ.

---

## Next up (not started)
- Finish reading renderVideoStage (past current point) + renderImmichStage
- Locate and verify PoseClassifier / IdentityScorer against real source
  (currently unconfirmed — do not build against the earlier design-only
  description)
- Person Clusters block (not yet read this session)
- Export settings (`gatherExportParamsNG`) (not yet read this session)
