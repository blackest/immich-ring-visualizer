# Project Structure

## Overview

Immich Ring Visualizer is a Flask backend (`ring_viz.py`) plus a browser
frontend (`templates/index.html` + `static/*.js`) for curating LoRA
training datasets from an Immich photo/video library. It uses Immich's
Postgres/pgvector database and InsightFace (buffalo_l) for face
detection/embedding, plus video frame analysis.

## Backend

- **`ring_viz.py`** — Flask app: routes, Immich pgvector queries,
  face/pose detection, video frame extraction (PyAV — reads directly
  from uploaded bytes, no disk writes), export/crop pipeline.

## Frontend

- **`templates/index.html`** — page markup, loads the three JS files
  below via `url_for('static', ...)` in this exact order.
- **`static/style.css`** — styling.
- **`static/viz-render.js`**
- **`static/media-ingest.js`**
- **`static/selection-ui.js`**

### History of the JS split

The frontend logic originally lived in one large `static/app.js`. It
was first split into `appa.js`/`appb.js` purely by size (roughly even
halves, ~132,000 characters combined) so editing one area didn't
require holding the whole file in context. That split was never
concern-based — it was just "first half / second half".

It was later reorganized into the three concern-based files below, on
branch `refactor/js-module-split` (commits `f461646`, `c3d624e`, off
`main` at `12d17fa`). This reorg was done in two stages:

1. **Bootstrap refactor (no file split yet).** The original
   `appa.js`/`appb.js` relied on ~27 pieces of top-level "glue" code
   (event-listener wiring, IIFEs) whose correctness depended on
   *script load order* — e.g. `init()` was called partway through
   `appa.js`, so all of `appb.js`'s wiring implicitly ran after it.
   That glue was wrapped into named `wireXxx()` functions and called
   explicitly, in the exact original order, from one `bootstrap()`
   function at the end. This made script load order stop mattering
   for correctness — only "are all functions defined by the time
   `bootstrap()` runs" matters, which any file split satisfies.
   Verified (mechanically, and by hand-testing in browser) to be a
   pure reorganization: same 166 top-level statements, same execution
   order, no functional changes.

2. **Concern-based split.** With load-order no longer fragile, the
   code was split by concern into the three files below. Verified
   mechanically: every top-level name (function or variable) assigned
   to exactly one file, no name split across files where doing so
   would break execution-order assumptions (checked specifically for
   top-level variable initializers that reference something defined
   in a different file).

### What's in each file

**`viz-render.js`** — ring rendering and its supporting UI controls:
- Core ring layout/sizing (`sizeForSim`, `radiusForSim`, `render`)
- Fisheye lens effect (`applyFisheye`, `attachLensEffect`)
- Pose/scale picker grids (build/setup/render/sync for both)
- Squeeze & sharpness cutoff controls, ranked-sort wiring
  (`applySqueeze`, `metricValueForRing`)
- Video ring rendering, pose list, pose-list scrubber
  (`renderVideoRing`, `renderPoseList`, `setupPoseListScrubber`,
  `flashHighlightFrame`)

**`media-ingest.js`** — getting frames/video/images into the tool:
- File/dropzone handling, frame seeking, shot-scale classification
- Frame playback controls, audio sync
- Video analysis + polling (`startVideoAnalysis`, `pollAnalysis`)
- Folder/zip loader (`startFolderAnalysis`)
- Immich integration: neighbor loading, running/re-running Immich
  analysis, adding/removing Immich nodes from the ring

**`selection-ui.js`** — selection, export, and app shell:
- Selection state (`selectedFrames`, save/restore to storage)
- Export parameter form (`getExportParams`)
- Selection modal (open/close/render, pose-scatter view inside it,
  pose detection for selected items)
- Immich selection bar, search modal
- App init, hover preview, thumbnail/preview URL helpers, sidebar
  detail panel, result sorting
- **`init()`** and **`bootstrap()`** — `bootstrap()` is the very last
  thing that runs; it calls every `wireXxx()` setup function (across
  all three files) plus `init()`, in the original execution order.
  Because it lives in the last-loaded file, all three files' functions
  are guaranteed to be defined by the time it runs.

### Load order

`index.html` loads them in this order — **this order matters**,
since `bootstrap()` (in `selection-ui.js`) must load last:

```html
<script src="{{ url_for('static', filename='viz-render.js') }}"></script>
<script src="{{ url_for('static', filename='media-ingest.js') }}"></script>
<script src="{{ url_for('static', filename='selection-ui.js') }}"></script>
```

### Notes for future edits

- These are plain global-scope scripts (not ES modules) — all three
  files share one global namespace, same as the original `app.js`.
  A name only needs to be unique once across all three files.
- Because `bootstrap()` centralizes all setup calls, adding a new
  top-level `addEventListener`/wiring block should follow the same
  pattern: wrap it in a named `wireXxx()` function in the appropriate
  file, and add a call to it in `bootstrap()` (in `selection-ui.js`)
  in the position that matches when it should run relative to the
  existing calls — rather than relying on script load order again.
- No file currently holds a dedicated "shared state" module — each
  piece of state lives in whichever file's functions use it. If a
  future change needs the same state read/written from two of these
  three files, that's a signal it may be time to split out a
  `state.js` loaded first, rather than duplicating or awkwardly
  cross-referencing state across files.
