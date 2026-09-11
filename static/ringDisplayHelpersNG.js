/**
 * ringDisplayHelpersNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 39-84 (ring math + shot-
 * scale bands) and 390-410 (hover preview), reordered together here since
 * they're all small display-math/DOM-feedback helpers used throughout
 * renderVideoStage() (see projectManagerNG.js's Cut Line 2).
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS -- THE CLOSEST THING IN appNG.js TO A REAL "RingVisualizer"
 * TOOLKIT-TIER MODULE (per appNG-module-contracts.md's suggested future
 * cut for renderVideoStage). None of these functions know about "faces"
 * or "characters" specifically -- they operate on generic {similarity}-
 * shaped or {vertFillPct/bboxRatio}-shaped objects. This is the genuine
 * toolkit-tier candidate, unlike the pose/scale pickers (see
 * pickersNG.js's correction of the same over-optimistic claim there).
 *
 *   sizeForSim(sim) -- maps a 0-1 similarity score to a pixel size between
 *     MIN_SIZE (34) and CENTER_SIZE (120) -- "closer match = bigger node"
 *
 *   radiusForSim(sim, ringScale) -- maps a 0-1 similarity score to a
 *     distance-from-center in pixels, scaled by ringScale (the "multiplier
 *     that moves rings in/out from center" John described) and the
 *     viewport's smaller dimension. Depends on window.innerWidth/Height
 *     at call time -- not passed in, a real (if minor) global read.
 *
 *   thumbUrlFor(r) -- trivial accessor, r.thumbUrl || ""
 *
 *   SHOT_SCALE_BANDS / shotScaleForNG(r) -- classifies a frame's face-
 *     height % of frame into one of 7 named false-color bands (Extreme
 *     wide -> Extreme close-up), ported verbatim from media-ingest.js's
 *     SHOT_SCALE_BANDS. Used by the chart's shot-scale strip (see
 *     renderNGChart in playbackAndChartNG.js) AND by the scale picker
 *     (pickersNG.js) independently -- another instance of the same
 *     underlying concept implemented in more than one place, though this
 *     one at least shares the classifier function itself.
 *
 *   SQUEEZE_DEFAULTS -- per-ring-sort-metric default for the "min-sim
 *     squeeze" slider (sim: 65, yaw: 20, pitch: 20, roll: 20, blur: 65) --
 *     switching sort metric resets the squeeze slider to a sensible
 *     starting point for that metric's scale, UNLESS the user already
 *     dragged the slider this session (that "already touched" tracking
 *     lives on the project instance -- see characterProjectNG.js's
 *     squeezeUserOverridden field, not in this file).
 *
 *   showHoverPreview(r) / hideHoverPreview() -- the ring/list hover-to-
 *     preview panel: debounced (80ms) via a MODULE-LEVEL `hoverTimer`
 *     (not per-project -- there is only ever one hover preview panel
 *     regardless of which project tab is active, which is correct since
 *     the user only hovers one thing at a time).
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 *   - window.innerWidth/innerHeight -- read directly in radiusForSim()
 *   - Module-level DOM globals defined elsewhere in appNG.js, captured
 *     once: hoverImg, hoverCaption, hoverPanel (used only by
 *     showHoverPreview/hideHoverPreview -- the math functions above have
 *     NO DOM coupling at all, which is exactly why they're the genuine
 *     toolkit-tier candidate).
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. sizeForSim/radiusForSim/thumbUrlFor/SHOT_SCALE_BANDS/
 *      shotScaleForNG/SQUEEZE_DEFAULTS need NOTHING beyond this file --
 *      they could be used standalone right now if imported.
 *   2. showHoverPreview/hideHoverPreview need hoverImg/hoverCaption/
 *      hoverPanel to exist first.
 *   3. Remove the corresponding lines from appNG.js.
 */

  const CENTER_SIZE = 120;
  const MIN_SIZE = 34;
  const MAX_RADIUS_VW = 42;
  const BAND_COUNT = 8;

  function sizeForSim(sim) {
    const t = Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
    return MIN_SIZE + t * (CENTER_SIZE - MIN_SIZE);
  }

  function radiusForSim(sim, ringScale) {
    const minDim = Math.min(window.innerWidth, window.innerHeight);
    const maxR = minDim * (MAX_RADIUS_VW / 100) * ringScale;
    const t = 1 - Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
    return (90 * ringScale) + t * (maxR - (90 * ringScale));
  }

  function thumbUrlFor(r) {
    return r.thumbUrl || "";
  }

  // ---- shot-scale bands (from media-ingest.js's SHOT_SCALE_BANDS, ported
  // verbatim -- classifies a frame's face-height % of frame into a fixed
  // false-color bucket for the chart's shot-scale strip) ----
  const SHOT_SCALE_BANDS = [
    { max: 0.05, label: "Extreme wide",       color: "#8a5fd4" },
    { max: 0.12, label: "Full shot",          color: "#4a7fd4" },
    { max: 0.20, label: "Cowboy/American",    color: "#4ad4c4" },
    { max: 0.25, label: "Medium",             color: "#4ad46a" },
    { max: 0.35, label: "Medium close-up",    color: "#d4c04a" },
    { max: 0.50, label: "Close-up",           color: "#d4824a" },
    { max: Infinity, label: "Extreme close-up", color: "#d4544a" },
  ];
  function shotScaleForNG(r) {
    const pct = typeof r.vertFillPct === "number" ? r.vertFillPct
      : (typeof r.bboxRatio === "number" ? Math.sqrt(r.bboxRatio) : null);
    if (pct === null) return null;
    const band = SHOT_SCALE_BANDS.find((b) => pct <= b.max) || SHOT_SCALE_BANDS[SHOT_SCALE_BANDS.length - 1];
    return { pct, ...band };
  }

  // min-sim squeeze defaults per ring-sort metric (from viz-render.js's
  // SQUEEZE_DEFAULTS) -- switching sort metric resets the squeeze slider
  // to a sensible starting point for that metric's scale, unless the
  // person has already dragged the slider themselves this session.
  const SQUEEZE_DEFAULTS = { sim: 65, yaw: 20, pitch: 20, roll: 20, blur: 65 };
  // showHoverPreview()/hideHoverPreview() used to be duplicated here --
  // this file loads after domRefsManifestNG.js, so the copy here silently
  // won (last top-level `function` declaration at shared script scope
  // wins) and shadowed domRefsManifestNG.js's version, including its
  // face-box overlay call (see [[face-box-overlay]]). Removed; the real
  // definitions live in domRefsManifestNG.js, which already has
  // hoverImg/hoverCaption/hoverPanel/hoverTimer captured.


// ---- exposed for other modules (renderVideoStage, pickersNG, chart) to call ----
window.ringDisplayHelpersNG = {
  sizeForSim, radiusForSim, thumbUrlFor, SHOT_SCALE_BANDS, shotScaleForNG,
  SQUEEZE_DEFAULTS, showHoverPreview, hideHoverPreview,
  CENTER_SIZE, MIN_SIZE, MAX_RADIUS_VW, BAND_COUNT,
};

