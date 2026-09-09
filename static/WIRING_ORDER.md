# Wiring order for the ngrefactor split

Replace indexNG.html's single script tag:

    <script src="{{ url_for('static', filename='appNG.js') }}"></script>

with these 13, in this exact order (each relies on names declared by
earlier ones being in shared page-script scope -- no IIFE wrappers, no
namespace object, just plain top-level declarations across files, same
as how appNG.js's own single IIFE worked internally):

    <script src="{{ url_for('static', filename='domRefsManifestNG.js') }}"></script>
    <script src="{{ url_for('static', filename='ringDisplayHelpersNG.js') }}"></script>
    <script src="{{ url_for('static', filename='exportSettingsNG.js') }}"></script>
    <script src="{{ url_for('static', filename='personClustersNG.js') }}"></script>
    <script src="{{ url_for('static', filename='poseListAndFrameHelpersNG.js') }}"></script>
    <script src="{{ url_for('static', filename='chartAndStaticPreviewNG.js') }}"></script>
    <script src="{{ url_for('static', filename='pickersNG.js') }}"></script>
    <script src="{{ url_for('static', filename='characterProjectNG.js') }}"></script>
    <script src="{{ url_for('static', filename='projectManagerNG.js') }}"></script>
    <script src="{{ url_for('static', filename='videoNG.js') }}"></script>
    <script src="{{ url_for('static', filename='bootstrapWiringNG.js') }}"></script>
    <script src="{{ url_for('static', filename='initguing.js') }}"></script>
    <script src="{{ url_for('static', filename='initchar.js') }}"></script>

Verified: all 13 files pass `node -c` individually and concatenated
together in this order (no syntax errors, no duplicate top-level
declarations). This only checks syntax -- it does NOT prove the app
actually runs correctly in a browser (DOM timing, actual element IDs
matching indexNG.html's markup, etc. are untested). This is the first
real test; expect to find and fix runtime issues the same way we found
and fixed the drawFrame/setPlayingVisual bug in Codex's split.

## videoNG.js consolidation (post-launch cleanup)
playbackModalNG.js no longer exists -- its entire content (the
PlaybackModal object) moved into videoNG.js, along with video-specific
pieces pulled out of domRefsManifestNG.js, characterProjectNG.js,
poseListAndFrameHelpersNG.js, projectManagerNG.js, and
bootstrapWiringNG.js's wiring tail. This consolidates "the video player"
(which had ended up scattered across six files) into one dedicated
module. See videoNG.js's own header for the full list of what moved
from where and why load order places it right after projectManagerNG.js
and right before bootstrapWiringNG.js (must load after the class/object
it patches, must load before bootstrapWiringNG.js's tail calls
ProjectManager.render()).

## What was fixed to make this wiring possible
- pickersNG.js: removed its `(function(){...})()` IIFE wrapper and
  `window.pickersNG = {...}` export block -- it's the only one of the 11
  files that had been wrapped; unwrapped so its functions join the same
  shared script scope as everything else.
- bootstrapWiringNG.js: removed an orphaned trailing `})();` (the
  original appNG.js's own closing brace, meaningless once this chunk
  stands alone) and fixed a doc-comment typo (`frames*/immich*`
  accidentally closed the JSDoc comment early with a literal `*/`).
- personClustersNG.js: removed a duplicate `let lastClusterRowsNG = [];`
  (was declared both in my header and already in the extracted body).
- projectManagerNG.js: added `window.ProjectManager = ProjectManager;`
  as an explicit bridge, since `initguing.js`/`initchar.js` check
  `window.ProjectManager` directly, and a top-level `const` is reachable
  as a bare identifier by later scripts but does NOT automatically become
  a `window` property (only `var`/`function` declarations do that).

## Known likely rough edges (not yet tested)
- Every extracted file's own doc header lists what it assumes is already
  loaded/exists -- worth a quick reread of each if something specific
  breaks, since the header names the exact coupling points.
- `bootstrapWiringNG.js` runs its wiring unconditionally at parse time
  (same as the original appNG.js), not wrapped in a deferred function --
  fine as long as it's the LAST script tag, after every DOM element and
  every other function it references already exists.
- pickersNG.js: removed its `(function(){...})()` IIFE wrapper and
  `window.pickersNG = {...}` export block -- it's the only one of the 11
  files that had been wrapped; unwrapped so its functions join the same
  shared script scope as everything else.
- bootstrapWiringNG.js: removed an orphaned trailing `})();` (the
  original appNG.js's own closing brace, meaningless once this chunk
  stands alone) and fixed a doc-comment typo (`frames*/immich*`
  accidentally closed the JSDoc comment early with a literal `*/`).
- personClustersNG.js: removed a duplicate `let lastClusterRowsNG = [];`
  (was declared both in my header and already in the extracted body).
- projectManagerNG.js: added `window.ProjectManager = ProjectManager;`
  as an explicit bridge, since `initguing.js`/`initchar.js` check
  `window.ProjectManager` directly, and a top-level `const` is reachable
  as a bare identifier by later scripts but does NOT automatically become
  a `window` property (only `var`/`function` declarations do that).

## Known likely rough edges (not yet tested)
- Every extracted file's own doc header lists what it assumes is already
  loaded/exists -- worth a quick reread of each if something specific
  breaks, since the header names the exact coupling points.
- `bootstrapWiringNG.js` runs its wiring unconditionally at parse time
  (same as the original appNG.js), not wrapped in a deferred function --
  fine as long as it's the LAST script tag, after every DOM element and
  every other function it references already exists.
