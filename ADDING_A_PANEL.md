# Adding a new panel (bottom-bar task)

## Status / handover

Written after building two new panels back-to-back in one session —
**Hd-Multi** (`hdmulti`, one-off HiDream edit/multi-ref jobs) and
**ComfyUI** (`comfy`, arbitrary ComfyUI workflow from a PNG's embedded
metadata). Both follow the same shape as the earlier `generate` /
`chat` / `rachel` / `videogen` tasks. This is that shape, written down
so the next one goes faster and skips the mistakes made this time.

See [APP_ARCHITECTURE_NOTES.md](APP_ARCHITECTURE_NOTES.md) for the
overall page/tab/layout model this fits into, and
[MODULARIZATION_PLAN.md](MODULARIZATION_PLAN.md) for the (separate,
pre-NG) `appNG.js` split-up effort. Neither of those documents this
specific "how do I add one more bottom-bar task" checklist — this file
is that checklist.

## What a "panel" is

A panel (internally, a "task") is a bottom-bar button
(`data-task="<name>"`) that, when active, takes over the **entire**
left rail and main stage with its own self-contained UI — it does not
plug into the shared ring/sidebar/selection machinery at all. `generate`,
`chat`, `rachel`, `videogen`, `hdmulti`, and `comfy` are all this shape.
(Contrast with `video`/`immich`/`folder`, which share the common rail
body and ring-based main stage — that's a different, older pattern not
covered here.)

Each panel owns:
- one rail pane: `#ng-<name>-pane`
- one main region: `#ng-<name>-main`
- one JS module: `static/<name>NG.js`, exposing `window.<Name>NG = { sync }`
- usually its own backend blueprint: `routes/<name>NG.py`

## Checklist

### 1. Backend: routes file + state + blueprint registration

Create `routes/<name>NG.py` with its own `Blueprint` (e.g.
`<name>NG_bp`). Look at `routes/hdmultiNG.py` or `routes/comfyNG.py` for
the shape: a `POST /api/ng/<name>/generate`-style kickoff route that
starts a background job and returns a `jobId`, a
`GET /api/ng/<name>/status/<job_id>` poll route, and (if the result is
binary, e.g. an image) a `GET /api/ng/<name>/result/<job_id>` route
that streams it back.

Two job patterns showed up so far, pick whichever matches what you're
wrapping:
- **You own the compute** (Hd-Multi: calls into `hidream_engineNG`
  directly, in-process) — run it in a background thread, track
  progress in an in-memory dict (`stateNG.py`), same as the existing
  `generateNG`/`videogenNG` job queues. Reuse the relevant lock (e.g.
  `character_sheetNG._SHEET_LOCK_NG` for anything touching the HiDream
  subprocess) so you don't fight other panels for the same GPU.
- **You're proxying an already-async external service** (ComfyUI: has
  its own `/prompt` submit + `/history/<id>` poll + `/view` fetch) —
  your status/result routes can just forward to the real service's own
  job semantics instead of running your own thread. Simpler, and correct
  as long as the external service already serializes its own GPU use.

If the job needs any durable per-job state, add a dict for it in
`stateNG.py` (e.g. `_hdmulti_jobs_ng = {}`, `_comfy_jobs_ng = {}`,
`_comfy_extracts_ng = {}`). If it needs a new on-disk directory or a
configurable remote address, add it in `configNG.py` (see
`HDMULTI_DIR` and `comfyui_base_url` / `get_comfyui_base_url()` /
`NG_ADDRESS_SETTINGS` for the pattern — the latter gets you a free
"Addresses" entry in the settings modal, editable without a restart).

Register the blueprint in `ring_viz.py`:
```python
from routes.<name>NG import <name>NG_bp  # NG: new blueprint, no changes to existing ones
...
app.register_blueprint(<name>NG_bp)  # NG: /api/ng/<name>/* (one-line description), additive only
```

### 2. Frontend: JS module

Create `static/<name>NG.js` as an IIFE exposing exactly one global:
```js
window.<Name>NG = { sync: sync };
```
`sync(active)` is called on every `ProjectManager.render()` (see step
4) and is the *only* entry point the rest of the app calls. Inside it:
- look up your elements fresh each call (`refreshEls()` pattern — cheap,
  and survives the DOM not existing yet on first load)
- compute `var on = !!(active && active.task === "<name>");`
- show/hide your pane and main: `els.pane.style.display = on ? "" : "none";` (same for `els.main`)
- if `on`, also force-hide `#ng-controls-pane` (the shared bottom controls
  strip) — every one of these full-takeover panels does this
- lazily call your own `init()` (wire up event listeners once, guarded
  by an `inited` flag) — don't wire listeners at load time, since your
  DOM elements are template-rendered and may not exist yet when the
  script first runs, and `sync()` may be called many times before the
  panel is ever opened

For the actual generate → poll → show-result flow, `comfyNG.js` and
`hdmultiNG.js` are the two reference implementations — copy whichever
is closer (external-service-proxy vs. own-job-queue, matching whichever
backend pattern you picked in step 1).

Add the script tag in `templates/indexNG.html`, **after** `videogenNG.js`
and **before** `bootstrapWiringNG.js` (which fires the first
`ProjectManager.render()` — your module must be fully defined before
that first call, since `render()` unconditionally checks
`window.<Name>NG`):
```html
<script src="{{ url_for('static', filename='videogenNG.js') }}"></script>
<script src="{{ url_for('static', filename='<name>NG.js') }}"></script>
<script src="{{ url_for('static', filename='bootstrapWiringNG.js') }}"></script>
```

### 3. Template: task button + pane + main region

Add the bottom-bar button, near the other task buttons:
```html
<button class="ng-task-btn" data-task="<name>">Display Name</button>
```

Add `#ng-<name>-pane` as a sibling inside the rail `<aside>` (near
`#ng-hdmulti-pane`/`#ng-comfy-pane`), `style="display: none"` by
default — it can be as small as a one-paragraph explainer if all the
real controls live in the main region instead (that's what both
Hd-Multi and ComfyUI do).

Add `#ng-<name>-main` as a sibling inside `<main id="ng-main">` (near
`#ng-hdmulti-main`/`#ng-comfy-main`), also `display: none` by default.

### 4. Wire into `projectManagerNG.js`

This file has one three-way OR-chain that every full-takeover panel
must be added to, appearing in **three separate places** — grep for
`"videogen"` to find all of them (as of this doc: lines ~410, ~437 in
the render-flow area, and ~922 in `renderLeftRail()`). Add
`|| active.task === "<name>"` to each:
- in `renderMain()`, the guard that hides the shared video-preview
  player when a full-takeover panel is active
- the equivalent guard a few lines later
- in `renderLeftRail()`, the guard that collapses `#ng-leftrail-body`
  (the shared rail content) so your `#ng-<name>-pane` gets the full
  rail height instead of splitting it 50/50

Then add your own sync call in `render()`, alongside the others
(`GenerateNG`, `ChatNG`, `RachelNG`, `VideoGenNG`, `HdMultiNG`,
`ComfyNG`):
```js
if (window.<Name>NG) window.<Name>NG.sync(active);
```
Order among these doesn't matter *unless* your panel also needs to
force-hide `#ng-controls-pane` — `VideoGenNG.sync` documents why it
must run last among the ones that existed at the time (it's the only
one whose hide is unconditional-when-active, so it "wins" regardless of
order against ones that only show it back). If yours also does an
unconditional hide, put it last for the same reason; if not, position
doesn't matter.

### 5. CSS: the flex-fill rule — do not skip this

Every `#ng-<name>-main` needs the same rule the other main regions have,
or the panel renders correctly but looks broken — packed into a
narrow column-width box instead of filling the stage:
```css
#ng-<name>-main {
  flex: 1 1 auto;
  min-width: 0;
  overflow-y: auto;
  padding: 20px 16px 90px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
```
This was forgotten for `#ng-hdmulti-main` the first time — everything
worked functionally, but the user's own screenshot showed it "packed in
pretty tight" with huge unused space beside it on a wide screen. Add
this proactively, before you ever get to testing layout — it was added
that way for `#ng-comfy-main` afterward and there was nothing to fix.

## Quick reference: minimal file list for a new panel

- `routes/<name>NG.py` — new
- `static/<name>NG.js` — new
- `ring_viz.py` — import + register blueprint
- `templates/indexNG.html` — task button, `#ng-<name>-pane`,
  `#ng-<name>-main`, script tag
- `static/projectManagerNG.js` — 3x OR-chain, 1x sync call
- `static/styleNG.css` — at minimum, the `#ng-<name>-main` flex-fill
  rule, plus whatever panel-specific classes your UI needs
- `stateNG.py` / `configNG.py` — only if the job needs durable
  in-memory state or a configurable address

## After any template edit: restart, don't just re-fetch

`ring_viz.py` runs with `debug=False, use_reloader=True`. Jinja
templates are compiled and cached in-process — editing
`templates/indexNG.html` alone will **not** show up on reload, even
with cache-busting, because the process never restarts on an `.html`
change. `touch ring_viz.py` (or edit any `.py` file) to trigger the
existing reloader watcher and pick up template changes. This came up
for every panel built so far and is easy to forget mid-iteration.
