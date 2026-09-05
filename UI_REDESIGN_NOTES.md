# Settings-panel / responsive-layout redesign — 5 parallel variants

**Written:** 2026-09-03, same session as `progress-report1.md` item 3.
**Status:** all 5 built and structurally verified (balanced tags/braces, `git diff` confirms `templates/index.html` and `static/style.css` are byte-identical to before — nothing about the default app changed). **Not yet visually verified in a browser** — the running Flask process has `debug=False` (no auto-reload), so **`ring_viz.py` needs to be restarted** before `/v1`–`/v5` will resolve (they 404 against the currently-running process since the routes were added to `routes/main.py` after it started).

## What each variant is, and where to load it once restarted

The live app is untouched at `/` (`index.html` + `style.css`, exactly as before). Five alternates, each its own template + stylesheet, each reachable at its own route:

- **`/v1`** (`templates/index1.html`, `static/style-v1.css`) — Resolve-style hard-cut pages, top tab bar: **Explore / Reference / Character / Shots / Logs**. Only one page visible at a time; Explore shows the ring + today's left tool panel; the other four take the right sidebar fullscreen and show only their own slice of it.
- **`/v2`** (`index2.html`, `style-v2.css`) — same idea but only **4 tabs**: Explore / **Build** (Reference + Character merged into one two-column page — config on the left, the reference photo + match grid on the right, so you can pick a photo and configure the sheet without leaving the page) / Shots / Logs.
- **`/v3`** (`index3.html`, `style-v3.css`) — same 5-page breakdown as v1, but: tab bar pinned to the **bottom** (Resolve's actual placement), the Character page is a **two-column split with the reference photo blown up large** on the right (today it's a 56×56px thumbnail), and the Shots page keeps a slim reference-photo strip plus a compact "Generate" button rather than hiding config entirely.
- **`/v4`** (`index4.html`, `style-v4.css`) — a different interaction model: **no tab bar, no navigation**. The default view is pixel-identical to today's app. A small floating pill at the bottom center has 4 "⤢ Focus" buttons (Reference/Character/Shots/Logs); clicking one blows that section up fullscreen, clicking again (or "✕ Exit focus") returns to normal. Zero risk to the everyday view, opt-in expansion only.
- **`/v5`** (`index5.html`, `style-v5.css`) — the conservative option: **desktop layout is 100% unchanged** from the live app (no pages, no tabs, no new interaction model at all). Only `@media` breakpoints were added (the current app has zero `@media` queries anywhere — confirmed root cause of the iPad breakage). Under 1200px width, the two fixed side panels become slide-out drawers (two small toggle buttons appear, "☰ Tools" / "▤ Sheet") instead of fixed 340px overlays that cover the ring; the ring gets the full viewport by default. This directly fixes the reported iPad/tablet squeeze with the smallest possible change, at the cost of not addressing the "too much crammed in one column" complaint at desktop width.

## How they were built (why the default app is safe)

Every variant is **CSS-only + a few lines of vanilla JS**, layered on top of a straight copy of the original `index.html`/`style.css`. No element was moved, renamed, or deleted — the exact same `#left-panel`, `#stage`, `#sidebar`, `#phosphene-sheet-panel`, `#phosphene-shot-grid`, etc. IDs exist in every file, and all four original `<script src>` includes (`viz-render.js`, `media-ingest.js`, `selection-ui.js`, `phosphene-sheet.js`) are untouched and unmodified — so every bit of existing interactive behavior (ring rendering, search, video analysis, pose/shot pickers, export, sheet generation, reroll, ranked-match selection) should keep working exactly as it does today, in every variant. Layout changes are done entirely via new CSS rules scoped to `body[data-page="..."]` (v1/v2/v3) or `body.focus-*` (v4) or `@media` (v5), which override the base rules by specificity/source-order — nothing in the original `style.css` was edited. Two small, purely additive class attributes (`class="ph-heading"`, `class="ph-selects"`) were added to two previously-anonymous `<div>`s inside the character-sheet panel, only so the new CSS could target them cleanly; nothing else in the HTML structure changed. `routes/main.py` only had 5 new `@main_bp.route(...)` functions appended — the original `/` route is byte-for-byte what it was.

## Known minor wrinkles (not bugs, just worth knowing)

- On pages/focus-states where "more settings" is forced open (Character/Build pages), the "▸ more settings" toggle button is still visible but inert (clicking it does nothing, since the CSS forces it open with `!important`). Harmless, but a real design would probably hide that toggle in those states too.
- The "Shots" pages show `#phosphene-sheet-img` even before anything has been generated (empty `src=""`), which will render as a broken-image icon until a sheet exists. Cosmetic only.
- None of this has been visually confirmed yet — restart `ring_viz.py`, then check each of `/v1` through `/v5` (and the iPad-width behavior specifically, per the original ask) before picking one, or mixing ideas from more than one.

## Suggested next step

Restart the app, open each route, click around (page tabs / focus buttons / drawer toggles), and say which direction feels right — or which pieces of which variants to combine (e.g. v3's big reference-photo idea + v2's merged Build page + v5's drawer approach for narrow widths are all independently portable ideas, not locked to their variant).
