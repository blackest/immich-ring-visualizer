# Ring Visualizer — Routes, Tabs & Render Queue Design Notes

Working notes from a design discussion (2026-09-04) on restructuring the app's
navigation and data model. Nothing here is built yet — this is capturing
decisions and open questions before implementation starts.

## Status / handover

As of 2026-09-04, this is **pure planning — nothing has been built**. No
NG files exist yet, no route has been added, no code has changed. This
document is the complete plan to start from.

- Repo: `~/immich-app/repo/immichring` on the Mac Studio.
- **Branch plan:** get `main` current first — resolve/commit whatever's
  outstanding (uncommitted changes on disk, any unmerged branch work) so
  `main` is a true, clean baseline — then create/switch to a `dev` branch
  off that for all NG work, rather than building on a stale `main` or
  continuing to pile onto an old feature branch.
- **Cleanup as part of getting `main` current:** the v1-v5 UI-variant
  prototypes (`index1-5.html`/`style-v1-5.css`) turned out to actually be
  wired and working (confirmed 2026-09-04, contradicting the earlier
  assumption they were dormant) — **remove their routes from routes.py**
  now that immichRingNG is the actual direction. Whether the underlying
  template/CSS files get deleted outright or kept around dormant in case
  anything's worth salvaging for NG is still open.
- Ground rule for this project (and in general): don't build without being
  explicitly told to. This document is background/context, not a
  standing instruction to start implementing.

## Why

The app has grown by bolting features onto a single shared curation view
(character-sheet generation panel, collapsible shot grid, HiDream status
banner, etc.). Five unwired UI-variant prototypes already sit in the repo
(`index1-5.html`/`style-v1-5.css`) aiming at a fullscreen tabbed layout but
were never routed. This session worked through what the actual navigation
and data model should be before touching layout/CSS.

## Reference points

- **DaVinci Resolve** — page-per-workflow-stage navigation (Media/Cut/Edit/
  Color/Fairlight/Deliver). Mapped here onto Curate/Character/Deliver/Log as
  distinct top-level pages, each a full workspace for one stage of work.
- **ComfyUI** — two ideas borrowed: (1) tab-per-workflow, each tab holding
  independent state; (2) nothing executes until explicitly triggered
  (Queue Prompt) — reinforces the "no automatic fetches" rule below.
- **Print spooler** — model for the render/generation queue (Deliver page).

## Core principle: nothing fetches or generates automatically

Current bug: the app drags in a random, unvetted batch of Immich images by
default, and a plain page refresh re-triggers this fetch (it's a load-time
side effect, not a deliberate action). Both are wrong.

Rule going forward: no Immich query, no generation, no fetch of any kind
happens except as a direct, explicit user action. A refresh should restore
whatever was actually on screen (or land on an empty/safe state) — never
silently re-run a query.

## Top-level pages

1. **Curate** — the ring/matching work (Immich search, video analysis,
   folder/zip import all feed one shared ring/scoring view). Not yet
   discussed in detail — landing behavior TBD beyond "starts empty, no
   auto-fetch."
2. **Character** — character-sheet generation workspace. Tabbed,
   ComfyUI-style — see below.
3. **Deliver** — the render/generation queue. Print-spooler model — see
   below. Open question: does this also host LoRA training-run tracking, or
   is training a separate lane that just happens to compete for the same
   GPU?
4. **Log** — top-level page (not a sub-section of Deliver) holding fault/
   event history for the whole app, not just render jobs — scope (curate
   ingestion failures, export errors, etc.) not fully nailed down yet.

**Not yet discussed:** Settings page contents, Curate page specifics.

## Character tabs

- Each open character (Mary, Paul, Ringo, Jane, Fred, ...) is a tab, styled
  after ComfyUI's per-workflow tabs — independent state per tab.
- `+` adds a new character tab.
- Switching tabs = switching context; no reload needed if already open.
- Closing a tab explicitly **saves** that character's durable state (the
  shopping list + traits — not necessarily in-progress UI state like scroll
  position or an unconfirmed edit) and releases any associated in-memory
  assets.
- **Open / unresolved:** exact new-tab flow (placeholder label you rename
  in place, vs. prompting for a name before the tab exists); whether closed
  tabs can be quickly reopened (browser-style "reopen last closed tab") or
  closed is just closed.

## Layout: three-axis navigation

Within the Character page, navigation splits into three independent axes —
answering "who," "what," and "how" separately rather than one deep menu:

- **Top bar — project/character tabs** (the "who"). Covered above.
- **Bottom bar — class of action / task** (the "what"). Selects which
  broad task you're doing for the active project. Current concrete buttons:
  **Immich**, **Video**, **Folder/Zip** (the three ingest sources) — more
  buttons will be added as other tasks (Shots, Traits, Export, ...) get
  defined. Only about 4 buttons fit comfortably; if the category count
  grows toward a dozen, some overflow handling (grouping, scrolling, a
  "more" menu) will be needed — **parked, not decided**.
- **Left rail — context-aware tooling** (the "how"). Which buttons appear
  on the left rail depends on whichever bottom-bar task is currently
  selected — e.g. selecting the Ingest-type task shows ingest-related
  buttons here. Clicking a left-rail button slides out a column with that
  button's specific tools/options.

Modeled on tools already in daily use: Resolve's page selector famously
lives at the bottom, not the top; the left-rail-plus-sliding-context-panel
is the same shape as VS Code's activity bar. Judged feasible specifically
*because* it mirrors familiar patterns rather than inventing a new one.

Two things flagged as needing care in execution, not open design
questions: the bottom→left causal link needs to read as obviously
intentional (matching highlight/color) or it'll feel like "why did my
tools just change"; and screen real estate on the iPad (three chrome
regions before content even starts) needs checking once there's something
to actually test — a double-column variant was floated for iPad
specifically, deferred until then ("we shall see").

**Collapsible bar:** a single toggle button (top or bottom) should be able
to slide a bar closed to reclaim space, particularly for iPad. **Open:**
which bar this applies to (bottom task bar, left rail, or an independent
toggle for each), and whether "collapsed" means fully hidden or shrunk to
a thin strip.

**Per-tab memory:** each character tab should remember which task/page it
was last on, so switching away and back restores context instead of
resetting to a default. **Open:** whether this needs to survive an actual
tab close/reopen (saved as part of the project's durable state) or only
persists across switches within the same live session.

## Data model: "shopping list" per project

A project (a character, or a curation session) is represented as a list of
wanted items rather than a live-loaded batch of images. This avoids the
auto-fetch problem by construction — there's nothing to load until
something is actually on the list.

- **Definites** — specific known shots (e.g. the existing preset shots like
  profile_left, three_quarter).
- **Fill** — open-ended target, not a fixed count. Could be 1, 15, or up to
  ~500 (largest LoRA dataset size mentioned). Satisfied by curated real
  photos and/or generated shots, whichever's appropriate.
- Each item has **two independent fulfillment states**, not one: image
  present/absent, and caption present/absent. A project can be
  shot-complete but caption-incomplete.
- **Assets list** per project tracks byte size of resident items, so total
  RAM commitment across all open projects can be checked against what the
  machine actually has (explicitly: an 8GB Mac vs. a 96GB Mac shouldn't be
  treated the same).

### RAM vs. disk — why the two project types behave differently

- **Character projects**: generated shots already live on disk
  (`exports/`) — the accepted "expensive to redo" exception to the
  no-disk-writes principle. An inactive/cold character tab is cheap: just
  JSON + thumbnails read from disk, no recomputation.
- **Curation sessions**: ingestion (video frames, Immich downloads, folder/
  zip uploads) is deliberately RAM-only, never written to disk, because
  it's cheap to redo. So a cold/inactive curation tab has nothing to
  reload from disk — coming back to it means re-running the search/decode.
  Accepted as fine (it was called cheap to redo for a reason), but the tab
  needs to retain its search/ingestion parameters so it can silently redo
  itself rather than showing a blank state with no memory of what it was.

### Hot-project cap and eviction

Motivating case: multiple characters each loading their own video for
ingestion means the RAM cost isn't unique to "curation sessions" as a
separate project type — a character tab doing video-based ingest carries
the same RAM weight as a curation session would.

- Built as a **configurable cap** (a number of simultaneously "hot"
  projects allowed), not hardcoded — the practical starting point is 2,
  but because the cap is a parameter driving generic logic rather than
  special-cased slots, raising it later (toward 3, 5, ...) is a config
  change, not a redesign.
- **Real memory pressure is the actual backstop**, regardless of the
  configured number: if the cap is set higher than the hardware can
  support (e.g. 50 on a Mac Mini), the app doesn't fail — it silently
  frees ("pages out") a tab on switching once real pressure demands it,
  overriding the configured cap.
- **Eviction is silent**, no user-facing notice. Framed as equivalent to
  OS-level paging: since durable state is already saved (shopping list +
  traits for a character; search/ingest parameters for a curation-style
  session), evicting the RAM-resident copy isn't data loss, it's just
  reclaiming memory. "Paging in" a cold project later costs a cheap disk
  read (character projects) or a real re-fetch/re-decode (curation-style
  ingestion), consistent with the RAM-vs-disk distinction above.

### Storage: SQLite, not plain JSON files

Decided to move from per-project JSON files to SQLite. Reasoning:

- Cross-project queries ("everything still missing a caption," RAM/disk
  committed across all open projects) are awkward over a folder of JSON
  files, trivial with real queries.
- Partial updates (mark one item's caption done) shouldn't require
  rewriting an entire multi-hundred-item JSON blob.
- SQLite specifically fits a local single-user app well: single file, no
  server process, ships in Python's stdlib — doesn't add the dependency/
  ops weight a "real" database would.
- **Schema flexibility for character traits**: the set of traits that
  matter (eye color, age, gender, etc.) is still being discovered over
  time, so those don't get fixed columns. Split: stable, well-understood
  structure (list items, image/caption state, asset sizes) as real
  relational tables; open-ended character traits as a flexible JSON
  attributes column per character, using SQLite's JSON functions so it's
  still queryable without having pre-declared every possible column.

## Render/generation queue (Deliver page)

Modeled on a print spooler — a visible job list, not a hidden background
process.

- **Global, not per-tab.** Reflects that there's one real shared resource
  (GPU / HiDream engine), same contention concern already handled between
  HiDream and Phosphene via a lockfile.
- **FIFO by default**, with a **priority choice made at submission time**:
  a job goes to the back of the queue, or is flagged to jump the line —
  not a separate drag-to-reorder UI.
- **Granularity is per-shot, not per-batch.** Aborting mid-render only
  loses the one shot in progress (matches the existing truncated-PNG
  validity check — a partial render is discarded, not composited); the
  rest of that character's remaining shots just sit un-rendered, ready to
  resume later.
- **Abort vs. wait**: when a priority job wants to jump in, the choice is
  presented explicitly — abort whatever's currently rendering (losing that
  one partial shot) or let it finish first before the priority job starts.
- **Capacity context**: ~10 min/shot means roughly 48 renders possible over
  an unattended 8-hour stretch (same pattern as leaving LoRA training runs
  going overnight with the screen locked). This points toward wanting a
  bulk "fill toward target" action — queue up however many shots a
  character's fill-quota still needs in one go, rather than queuing shots
  one at a time — but this was floated, not fully confirmed.

### Status indicator (on the Deliver tab button itself, not a separate panel)

- **Number** = queue depth (count of items waiting).
- **Color bar**:
  - Default — idle, nothing rendering (covers both "queue empty" and
    "queue has items but nothing started yet" the same way).
  - Green — actively rendering.
  - Red — fault logged. Covers two different underlying situations that
    read the same color: a transient "one job failed, keep going, check
    when free" fault, vs. a persistent "engine unreachable, nothing can
    run until restarted" state. These behave differently (transient vs.
    sticks until manual restart) but don't need visually distinct colors,
    since the Log page holds the actual detail.
- **On fault**: log it, but keep processing other queued jobs if the
  engine can still run them. Only a fully-down engine actually halts
  everything.

## GPU contention / dispatch gating

Before dispatching the next queued job, check whether the GPU is actually
free.

- Decided **against** relying solely on the existing lockfile handshake
  (built specifically for HiDream vs. Phosphene coordination), because
  other local processes can compete for the same hardware unpredictably —
  e.g. Rachel (Gemma via Ollama) actively inferring wasn't part of that
  handshake at all.
- Instead: **check actual GPU load/utilization directly**, so a new
  contender showing up later doesn't need to be hand-registered anywhere.
- **Backoff**: deliberately coarse, not tight polling — check no more
  often than once every 5–10 minutes while waiting for the GPU to free up.

## Implementation approach

This new layout goes in as a **secondary route** (working name
**immichRingNG**) alongside the existing app — a new page/route, not a
replacement of what currently works — same spirit as the unwired
`index1-5.html` prototypes already in the repo, but this time actually
served and iterated on rather than left dormant. Once the route/page shell
exists, the next layer of work is the "glue": wiring the bottom-bar/
left-rail buttons to actual functions (ingest calls, etc.) rather than the
page being static chrome.

**Hard rule on existing modules:** for this stage of development, original
files are **read-only**. NG starts as a **full duplicate**, not
a selective one. Every file NG needs gets copied to an NG-suffixed twin
(`video.py` → `videoNG.py`) from day one, including logic that hasn't
changed — NG does not import from or call into the original modules at
all. The codebase is small enough that the duplication cost is trivial
(even 50MB of code is nothing), and the point isn't storage efficiency,
it's removing any reason to have an original file open while working on
NG — that's what actually prevents an accidental edit slipping into the
current app. The only time an original file gets touched is a genuine fix
to the *current* app (index.html and what it depends on) — a real bug in
what's live today, unrelated to NG. NG-motivated changes and current-app
maintenance fixes never share a commit or a file. Old modules are expected
to be retired once their NG twin has proven out and replaces them — this
isn't meant to be permanent parallel maintenance.

## Status log

- **Video ingest tab isolation — confirmed working.** Suspected bug: Sue's
  and Mary's tabs both showing the same video. Root cause investigation
  found the actual `CharacterProject` state (video, videoFile, job, ring)
  was already correctly isolated per-instance; the observed symptom was
  stale/leftover `localStorage` state from before the class-per-project
  model existed. Clearing `immichRingNG:state` and creating fresh tabs
  resolved it. Confirmed clean across repeated tab switches with two
  different videos loaded.
- **Real bug found and fixed along the way:** `detectionNG.py`'s
  `get_face_app_ng()` returns one shared `FaceAnalysis` (InsightFace)
  model instance. Harmless in the original single-project app, but NG
  runs each project's analysis in its own background thread so two
  projects can analyze in parallel — meaning two threads could call into
  that same ONNXRuntime session concurrently, which isn't safe and can
  cross-contaminate results between jobs. Fixed with a lock
  (`_face_app_lock_ng` in `stateNG.py`) around the `face_app.get(...)`
  calls in `video_analysisNG.py`. Job orchestration stays fully parallel;
  only the actual model inference call is serialized. Worth deliberately
  testing: load videos into two tabs and hit "Start Analysis" on both
  within a second of each other, confirm both results look sane.
- **Playback pop-out — ported.** The original app's "Build playback
  (rejected frames blanked)" feature (`routes/video.py`'s
  `build_playback`/`playback_file`) is now ported to
  `routes/videoNG.py` as near-identical NG twins
  (`build-playback`/`playback-file`, `_analysis_jobs_ng` /
  `configNG.FRAME_STORE`). Frontend differs from the original by design:
  instead of embedding the result inline in the left rail, it opens in a
  modal (`PlaybackModal` in `appNG.js`) so it's not fighting the ring/list
  for space. The modal auto-closes if the owning tab is switched away
  from or closed, and tracks fps/jobId per the project that built it, so
  it can't end up showing one project's clip while a different tab is
  active. Playback state isn't persisted to `localStorage` (the built
  file lives in `FRAME_STORE`, a server tempdir that doesn't survive a
  restart, and a new analysis/video load invalidates any previous build,
  matching how `ring` is already treated).

## Open questions / not yet decided

- Curate page: landing behavior specifics beyond "no auto-fetch."
- Settings page: contents not discussed.
- New-tab creation flow (placeholder-then-rename vs. name-first prompt).
- Whether closed character tabs are reopenable.
- Whether Deliver's queue unifies shot rendering and LoRA training, or
  keeps training as a separate lane.
- Bulk "fill toward target" queuing action — floated, not confirmed.
- Exact character-attribute schema — deliberately left open/evolving by
  design (that's the point of the JSON attributes column).
- Collapsible bar: which bar it applies to, and fully-hidden vs.
  shrunk-to-a-strip.
- Bottom-bar overflow once task categories exceed ~4 visible slots.
- Left-rail slide-out mechanics: pushes main content aside, or floats over
  it?
- Per-tab last-viewed-page memory: durable (survives tab close) vs.
  session-only.
- iPad-specific double-column layout — deferred until there's a build to
  test against.
