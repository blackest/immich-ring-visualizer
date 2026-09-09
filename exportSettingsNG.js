/**
 * exportSettingsNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 89-207.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 *   gatherExportParamsNG() -- reads the Export Settings panel (crop mode,
 *     resize, margin, upscale-cap checkbox+value) into a plain params
 *     object matching the Python backend's image_opsNG.py's
 *     _export_params_from_body_ng. Ported from selection-ui.js's
 *     getExportParams(). THIS IS A WIDELY-CALLED FUNCTION -- referenced
 *     from characterProjectNG.js (exportFrames/exportSelectedImmichAssets)
 *     and projectManagerNG.js, both already extracted. Any real split
 *     needs this file loaded before those.
 *
 *   applyResolutionSummaryNG(summary) -- takes a job's resolutionSummary
 *     (reported once analysis/polling picks it up -- see
 *     characterProjectNG.js's poll()) and reflects it into the Export
 *     Settings panel (likely min/max resolution hints -- see full body
 *     below for exact DOM writes).
 *
 *   wireExportSettingsNG() -- attaches the panel's own input event
 *     listeners (crop mode changes, checkbox toggles, etc.) -- call once
 *     at startup.
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 *   - A number of module-level DOM element globals (export panel's own
 *     inputs/checkboxes) captured via document.getElementById calls
 *     INSIDE these functions themselves (gatherExportParamsNG calls
 *     document.getElementById("ng-export-upscale") directly rather than
 *     using a captured-once module-level const, unlike most of the rest
 *     of appNG.js's pattern) -- worth noting as a minor inconsistency in
 *     the source's own conventions, not something changed here.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after the Export Settings panel's DOM exists.
 *   2. Remove the corresponding lines from appNG.js.
 *   3. characterProjectNG.js and projectManagerNG.js both need this file
 *      loaded first once they're wired for real.
 */

  function gatherExportParamsNG() {
    const upscaleOn = document.getElementById("ng-export-upscale").checked;
    return {
      width: parseInt(document.getElementById("ng-export-width").value, 10) || 512,
      height: parseInt(document.getElementById("ng-export-height").value, 10) || 512,
      cropMode: document.getElementById("ng-export-crop-mode").value,
      minFacePx: parseFloat(document.getElementById("ng-export-min-face").value) || 0,
      margin: parseFloat(document.getElementById("ng-export-margin").value) || 2.2,
      interp: document.getElementById("ng-export-interp").value,
      upscale: upscaleOn,
      maxUpscale: upscaleOn ? (parseFloat(document.getElementById("ng-export-max-upscale").value) || null) : null,
      padMode: upscaleOn ? document.getElementById("ng-export-pad-mode").value : "none",
      native: document.getElementById("ng-export-native").checked,
    };
  }

  // Show/hide the margin/cap/clip rows depending on crop mode + upscale,
  // wire size presets, and keep the "Match source" resolution note in
  // sync. Ported from selection-ui.js's export-settings IIFE +
  // media-ingest.js's applyResolutionSummary/wireMatchSourceResBtn.
  let lastResolutionSummaryNG = null;
  function applyResolutionSummaryNG(summary) {
    if (summary !== undefined) lastResolutionSummaryNG = summary;
    summary = lastResolutionSummaryNG;
    const note = document.getElementById("ng-export-source-res-note");
    const matchBtn = document.getElementById("ng-export-match-source-btn");
    if (!note || !matchBtn) return;
    if (!summary) {
      note.style.display = "none";
      matchBtn.style.display = "none";
      return;
    }
    const nativeOn = document.getElementById("ng-export-native").checked;
    const exportW = parseInt(document.getElementById("ng-export-width").value, 10);
    const exportH = parseInt(document.getElementById("ng-export-height").value, 10);
    const downsampling = !nativeOn && (summary.modeWidth > exportW || summary.modeHeight > exportH);

    let text;
    if (summary.uniform) {
      text = `Source: all ${summary.totalCount} images are ${summary.modeWidth}\u00d7${summary.modeHeight}`;
    } else {
      text = `Source: mostly ${summary.modeWidth}\u00d7${summary.modeHeight} (${summary.modeCount}/${summary.totalCount}), range ${summary.minWidth}\u2013${summary.maxWidth} \u00d7 ${summary.minHeight}\u2013${summary.maxHeight}`;
    }
    if (downsampling) {
      text += ` \u2014 <span style="color:#e0a94a;">exporting at ${exportW}\u00d7${exportH} throws away resolution \u2014 tick "native resolution" below to keep it</span>`;
    } else if (nativeOn) {
      text += ` \u2014 native resolution export is on, source pixels are kept`;
    }
    note.innerHTML = text;
    note.style.display = "block";
    matchBtn.style.display = (!nativeOn && (downsampling || exportW !== summary.modeWidth || exportH !== summary.modeHeight)) ? "inline-block" : "none";
    matchBtn.dataset.w = summary.modeWidth;
    matchBtn.dataset.h = summary.modeHeight;
  }

  function wireExportSettingsNG() {
    const cropModeSel = document.getElementById("ng-export-crop-mode");
    const marginRow = document.getElementById("ng-export-margin-row");
    const maxUpscaleRow = document.getElementById("ng-export-max-upscale-row");
    const padRow = document.getElementById("ng-export-pad-row");
    const upscaleCb = document.getElementById("ng-export-upscale");
    function syncRows() {
      const isFace = cropModeSel.value === "face";
      const showCap = isFace && upscaleCb.checked;
      marginRow.style.display = isFace ? "flex" : "none";
      maxUpscaleRow.style.display = showCap ? "flex" : "none";
      padRow.style.display = showCap ? "flex" : "none";
    }
    cropModeSel.addEventListener("change", syncRows);
    upscaleCb.addEventListener("change", syncRows);
    syncRows();

    document.querySelectorAll(".ng-export-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("ng-export-width").value = btn.dataset.w;
        document.getElementById("ng-export-height").value = btn.dataset.h;
        applyResolutionSummaryNG();
      });
    });

    const nativeCb = document.getElementById("ng-export-native");
    const widthInput = document.getElementById("ng-export-width");
    const heightInput = document.getElementById("ng-export-height");
    function syncNativeState() {
      const on = nativeCb.checked;
      // native mode still uses width/height as an aspect ratio for face/center
      // crop modes, so keep them enabled but dim them to signal the meaning
      // shifted from 'exact output size' to 'output aspect ratio'.
      [widthInput, heightInput].forEach((el) => { el.style.opacity = on ? "0.55" : "1"; });
      document.querySelectorAll(".ng-export-preset-btn").forEach((btn) => { btn.style.opacity = on ? "0.4" : "1"; });
    }
    nativeCb.addEventListener("change", () => { syncNativeState(); applyResolutionSummaryNG(); });
    syncNativeState();

    const matchBtn = document.getElementById("ng-export-match-source-btn");
    matchBtn.addEventListener("click", () => {
      widthInput.value = matchBtn.dataset.w;
      heightInput.value = matchBtn.dataset.h;
      matchBtn.style.display = "none";
    });

    widthInput.addEventListener("input", () => applyResolutionSummaryNG());
    heightInput.addEventListener("input", () => applyResolutionSummaryNG());
  }

  // ---- Person Clusters, ported from selection-ui.js's IIFE inside
  // wireImmichSearchModal(). Pure-DB query (via dbNG.py -> Immich's
  // Postgres, same instance the rest of NG's Immich search hits) --
  // ranks named persons by how tightly their tagged faces cluster, no
  // analysis job or video/folder ingest involved. Clicking a person row
  // loads their tagged assets into a thumbnail grid; single-click toggles
  // an asset into the active project's export selection, double-click
  // loads it as a face-similarity anchor (same as a Search result).
  //
  // `lastClusterRows` is deliberately a plain module-level list, not
  // per-project state: the clusters themselves are a property of the
  // Immich library, not of any one character tab, so there's nothing to
  // isolate per-project here -- only the resulting export selection
  // (project.selectedAssetIds) is per-project.

// ---- exposed for other modules (CharacterProjectNG, ProjectManagerNG, generateNG's settings panel) to call ----
window.exportSettingsNG = { gatherExportParamsNG, applyResolutionSummaryNG, wireExportSettingsNG };

