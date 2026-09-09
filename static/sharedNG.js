// sharedNG.js
// Shared constants and helper functions used across the NG editor.

(function () {
  "use strict";

  window.AppNG = window.AppNG || {};

  const STORAGE_KEY = "immichRingNG:state";
  const CENTER_SIZE = 120;
  const MIN_SIZE = 34;
  const MAX_RADIUS_VW = 42;
  const BAND_COUNT = 8;
  const SQUEEZE_DEFAULTS = { sim: 65, yaw: 20, pitch: 20, roll: 20, blur: 65 };

  const SHOT_SCALE_BANDS = [
    { max: 0.05, label: "Extreme wide", color: "#8a5fd4" },
    { max: 0.12, label: "Full shot", color: "#4a7fd4" },
    { max: 0.2, label: "Cowboy/American", color: "#4ad4c4" },
    { max: 0.25, label: "Medium", color: "#4ad46a" },
    { max: 0.35, label: "Medium close-up", color: "#d4c04a" },
    { max: 0.5, label: "Close-up", color: "#d4824a" },
    { max: Infinity, label: "Extreme close-up", color: "#d4544a" },
  ];

  function sizeForSim(sim) {
    const t = Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
    return MIN_SIZE + t * (CENTER_SIZE - MIN_SIZE);
  }

  function radiusForSim(sim, ringScale) {
    const minDim = Math.min(window.innerWidth, window.innerHeight);
    const maxR = minDim * (MAX_RADIUS_VW / 100) * ringScale;
    const t = 1 - Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
    return 90 * ringScale + t * (maxR - 90 * ringScale);
  }

  function thumbUrlFor(r) {
    return (r && r.thumbUrl) || "";
  }

  function shotScaleForNG(r) {
    const pct =
      typeof r.vertFillPct === "number"
        ? r.vertFillPct
        : typeof r.bboxRatio === "number"
          ? Math.sqrt(r.bboxRatio)
          : null;
    if (pct === null) return null;
    const band =
      SHOT_SCALE_BANDS.find((b) => pct <= b.max) ||
      SHOT_SCALE_BANDS[SHOT_SCALE_BANDS.length - 1];
    return { pct, ...band };
  }

  window.AppNG.shared = {
    STORAGE_KEY,
    CENTER_SIZE,
    MIN_SIZE,
    MAX_RADIUS_VW,
    BAND_COUNT,
    SHOT_SCALE_BANDS,
    SQUEEZE_DEFAULTS,
    sizeForSim,
    radiusForSim,
    thumbUrlFor,
    shotScaleForNG,
  };
})();
