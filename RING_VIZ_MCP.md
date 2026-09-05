# An MCP interface to Ring Visualizer — discussion

This is a discussion doc, not a commitment or a spec — capturing what an
MCP server *for* Ring Visualizer would do, why it'd be worth having, and
what it would actually take to build, so there's something concrete to
react to next time this comes up.

Note the direction: this is the opposite of the earlier ComfyUI
slot-override idea. That was Ring Visualizer becoming a *client* of
another tool (driving an arbitrary ComfyUI workflow). This is Ring
Visualizer becoming a *server* — exposing its own capabilities as MCP
tools so Claude (or any other MCP client) can drive **it**. Unrelated
except in spirit; both are "let something else operate this app."

---

## What Ring Visualizer actually is today

Worth restating plainly, since the MCP surface should mirror what's
really here, not just the character-sheet feature that's been the focus
lately. Per `PROJECT_STRUCTURE.md`: *"a Flask backend plus a browser
frontend for curating LoRA training datasets from an Immich photo/video
library. It uses Immich's Postgres/pgvector database and InsightFace
(buffalo_l) for face detection/embedding, plus video frame analysis."*

Concretely, six blueprints, 35 routes:

| Blueprint | Routes | Does |
|---|---|---|
| `immich_bp` | 11 | Query Immich's pgvector DB for faces/people: `random_face`, `find_by_filename`, `asset_face_pose`, `neighbors` (similarity search), `person_clusters`, `person_assets`, plus `analyze_immich` and thumbnail/preview proxies |
| `video_bp` | 7 | Ingest and face-analyze a video in memory (no disk writes — PyAV reads straight from uploaded bytes): `preview_video`, `analyze_video`, `analysis_status`, `frame_file`, `build_playback`, `playback_file` |
| `folder_bp` | 1 | `analyze_folder` — the same face analysis, over a local folder of images instead of a video |
| `export_bp` | 4 | The curation pipeline: `export_job`, `export_immich_assets`, and export-preview proxies — this is where a "training dataset" actually gets written to `exports/<name>/` |
| `phosphene_bp` | 11 | Character-sheet generation (this session's work): draft characters, job-queued sheet generation, reroll, poll, per-shot prompt/thumbnail serving |
| `main_bp` | 1 | Serves the page itself |

So the app is really: **find a face → confirm it's the right person
across a library/video/folder → curate a crop set → (now) generate a
synthetic turnaround dataset from one photo of them.** Character-sheet
generation is the newest link in a chain that already existed.

---

## What an MCP interface would do

Expose a subset of the above as MCP tools, so an agent can drive that
same chain by name/description instead of clicking through the ring UI.
Roughly three tiers, in order of how well they already fit the
request/poll shape MCP tool calls want:

**Tier 1 — already job-queued, translates almost 1:1** (`phosphene_bp`,
plus `video_bp`/`export_bp`'s job endpoints): `generate_character_sheet`,
`get_sheet_job_status`, `reroll_shot`, `analyze_video`,
`get_analysis_status`, `export_immich_assets`. These already return a
job id immediately and get polled — exactly the shape an MCP tool for a
long-running action should have. Minimal translation work.

**Tier 2 — synchronous lookups, straightforward wrapping**
(`immich_bp`'s query routes, `person_clusters`, `person_assets`,
`neighbors`, `find_by_filename`, `analyze_folder`): fast, read-only or
near-read-only, map cleanly to a single MCP tool call each.

**Tier 3 — binary/streaming responses, not tool calls at all**
(`thumb`, `preview`, `framefile`, playback files): these serve images
and video, not something an MCP *tool result* should carry. Leave these
as plain HTTP the frontend keeps hitting directly; at most, an MCP tool
returns a URL for one of these rather than the bytes.

---

## Why it'd be useful

- **Natural-language driving of a genuinely clicky UI.** *"Find photos
  of the person in this reference shot across my library, then generate
  an extended character sheet, cinematic style, and tell me when it's
  done"* collapses several manual steps (search → confirm → open panel →
  set options → wait → check back) into one request an agent can carry
  out and report back on.

- **Unattended / overnight orchestration.** Character-sheet extended
  presets and video analysis are both GPU-bound jobs that can run for
  hours. That's exactly the kind of thing tedious to babysit by hand but
  natural for an agent to kick off and check on periodically — including
  a scheduled Claude session that starts a job, then wakes up later to
  poll it and report status, rather than someone needing to leave a
  browser tab open.

- **Cross-capability composition that doesn't exist today.** Right now
  "find this person → curate their crops → generate a dataset for them"
  is three separate UI panels. An agent with MCP tools for all three
  could chain them in one request instead of the user doing so by hand.

- **Using Ring Visualizer from outside its own browser tab** — from
  Claude Desktop, Claude Code, or any other MCP client, without the web
  UI being the only way in.

---

## What would actually need to change

**1. It has to be a thin client of the running Flask app, not a second
process importing the same modules.** This is the one hard constraint,
and it comes straight out of decisions already made this session:
`character_sheet.py`'s `_SHEET_LOCK` and `sheet_jobs.py`'s in-memory
`_JOBS` dict are both process-local. A second Python process that
imported `character_sheet`/`sheet_jobs` directly would get its *own*
independent lock and job registry — meaning an MCP-triggered render and
a browser-triggered render could run at the same time, which is exactly
the GPU-contention problem the app's single-process-lock design exists
to prevent, and MCP-started jobs wouldn't show up when polling from the
web UI (or vice versa). So: the MCP server has to be a separate small
process that talks to `ring_viz.py` over plain HTTP — `POST
/api/phosphene/characters/<id>/sheet/generate`, `GET
/api/phosphene/sheet-jobs/<id>`, etc. — exactly like the browser does.
One process stays the single source of truth for "is a render running."

**2. A new, separate entrypoint** — something like `mcp_server.py` at
the repo root, not a Flask blueprint (it isn't part of the Flask app at
all, it's a second server sitting next to it) — plus the `mcp` Python
SDK added to `requirements.txt`. Runs alongside `ring_viz.py`, not
instead of it.

**3. Long-running jobs need a start tool + a separate poll tool**, not
one blocking call. Costs nothing extra to design — `sheet_jobs.py` and
`video_bp`'s job endpoints already work this way — but it's worth
naming explicitly: an MCP tool that blocks for up to 2.5 hours on an
extended-preset render is a bad tool. `generate_character_sheet` returns
a job id immediately; `get_sheet_job_status` is a separate tool a caller
polls.

**4. A GPU-cost guard rail is worth having before this is agent-facing.**
Today a stray double-click in the UI just fails fast
(`CharacterSheetBusyError`, 429) — cheap. An agent that misreads a
result and retries `generate_character_sheet` in a loop could tie up
the GPU for hours before a human notices. Worth deciding whether the
extended preset specifically needs an explicit confirmation step (a
tool that reports "this will take ~2.5 hours, call `confirm_and_start`
to actually begin") rather than firing immediately like the UI button
does.

**5. Bind to localhost only, same as the app already assumes.** Ring
Visualizer has no auth of its own today (it's a single-user local
tool — only its calls *out* to Immich need an API key). The MCP server
inherits that same trust boundary rather than inventing new auth; it
just shouldn't listen on anything but localhost.

**6. Scope the first version narrow.** Wrapping all 35 routes
mechanically on day one isn't the right first step. The character-sheet
piece (tier 1 above) is the most obviously agent-friendly slice — it's
fresh, already job-queue-shaped, and self-contained — and a couple of
the Tier 2 lookups (`neighbors`, `person_assets`) cover the "find this
person" half of the workflow described above. That's enough for the
composed use case in the "why useful" section without touching video
analysis, folder analysis, or export yet.

**7. Testing follows the same pattern already used all session** — spin
up the Flask app via its test client (or point the MCP server at a real
running instance) and verify each MCP tool wrapper translates its
arguments and results correctly, engine calls mocked out the same way
`hidream_engine.generate_hidream` has been throughout.

---

## Open questions, not answered here

- Exact tool list and names for a first version (proposed above at the
  "why useful" level, not pinned down as an API).
- Whether `get_sheet_job_status`'s response should be trimmed for an
  MCP tool result (it currently returns full per-shot prompts + a log
  tail — fine for a browser poll, possibly more than an agent needs
  back every call).
- Whether this MCP server should be something Claude sessions connect
  to automatically when working in this repo, or something started
  manually alongside `ring_viz.py`.
