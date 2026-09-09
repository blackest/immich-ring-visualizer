/**
 * pickersNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines ~2737-3026 (as of this
 * extraction). Pulled out for study/documentation purposes as part of
 * decomposing appNG.js into separate assemblies -- see
 * appNG-module-contracts.md.
 *
 * STATUS: NOT YET WIRED. This file is not currently loaded or referenced
 * anywhere. It is a faithful copy-out, not a working module -- see
 * "TO MAKE THIS ACTUALLY RUN" at the bottom for what's still needed.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * Two near-identical "nearest-N-to-target" pickers:
 *   - Pose Picker  -- nearest 9 frames to a dialed-in {pitch, yaw} target
 *   - Scale Picker -- nearest 9 frames to a dialed-in face-scale % target
 *
 * Both search the current project's analyzed pool (project.ring.baseResults)
 * for the 9 closest candidates to a target value, in a 3x3 grid that keeps
 * a frame in its slot as long as it's still in the nearest-9 (so the grid
 * doesn't visually reshuffle on every tiny slider nudge). Grid cells are
 * built once and reused across projects/re-renders; the pool/displayed-slot
 * state lives per-project.
 *
 * Originally described in appNG-module-contracts.md as a single
 * near-toolkit-tier module ("nearest-to-target picker"), generic enough to
 * be reusable for any N-dimensional similarity space. THAT WAS AN
 * OVERSTATEMENT of the real code, corrected here: there are actually TWO
 * separate, hand-duplicated implementations (pose = 2D pitch/yaw distance,
 * scale = 1D percentage distance), sharing only buildPickerCellsNG() and
 * updatePickerCellVisualNG(). Nothing has been genericized in this pass --
 * per instruction, this is a straight lift, not a refactor. Whether to
 * merge these into one generic module is a decision for later.
 *
 * ============================================================
 * COUPLING POINTS (everything this code currently reaches into)
 * ============================================================
 * This is NOT toolkit-tier as extracted -- it is fused to "project" and
 * "ProjectManager" concepts throughout. A genuinely reusable version would
 * need all of the following turned into explicit parameters/callbacks:
 *
 *  READS from `project` (passed in as an argument, so at least that part
 *  is already explicit):
 *   - project.ring.baseResults       -- the analyzed candidate pool
 *   - project.isActive               -- whether to touch the empty/controls
 *                                        DOM visibility for this project
 *   - project.posePickerPool / project.scalePickerPool        (written)
 *   - project.posePickerDisplayed / project.scalePickerDisplayed (written)
 *   - project._posePickerPitchTarget / _posePickerYawTarget   (read+write)
 *   - project._scalePickerTarget                              (read+write)
 *   - project._posePickerToleranceOn / _posePickerTolerance   (read+write)
 *   - project._scalePickerToleranceOn / _scalePickerTolerance (read+write)
 *   - project.selectedFrames.has() / .add()  -- selection state
 *   - project.toggleFrameSelection()         -- selection mutator
 *
 *  READS from module-level DOM globals, captured once at file load (NOT
 *  passed in -- this is the same anti-pattern flagged for FrameScrubber/
 *  VideoPopout in appNG-module-contracts.md):
 *   - #ng-pose-picker-grid / -empty / -controls / -pitch / -yaw /
 *     -pitch-num / -yaw-num / -tolerance-enable / -tolerance-val /
 *     -count / -select-btn
 *   - #ng-scale-picker-grid / -empty / -controls / -slider / -num /
 *     -tolerance-enable / -tolerance-val / -count / -select-btn
 *   - #ng-export-min-face (setupScalePickerNG reads this directly to
 *     apply the same "too distant to be useful" floor Export Settings
 *     uses -- a real cross-feature coupling, not just a DOM lookup)
 *
 *  CALLS OUT to globals not defined in this file:
 *   - ProjectManager.getActive()
 *   - ProjectManager.render()
 *
 * ============================================================
 * CALL-SITE WIRING (how appNG.js currently drives this)
 * ============================================================
 * From appNG.js's per-project render tick (~line 1871-1884):
 *
 *   // Pose Picker / Shot Scale Picker: re-run setup (reset sliders to the
 *   // pool's centroid, reset sticky grid slots) only when this project's
 *   // ring identity actually changed since we last looked -- not on
 *   // every render tick (polling, unrelated toggles, etc.).
 *   if (active.ring !== active._posePickerRingRef) {
 *     active._posePickerRingRef = active.ring;
 *     setupPosePickerNG(active);
 *   }
 *   if (active.ring !== active._scalePickerRingRef) {
 *     active._scalePickerRingRef = active.ring;
 *     setupScalePickerNG(active);
 *   }
 *   renderPosePickerGridNG(active);
 *   renderScalePickerGridNG(active);
 *
 * Note the `active.ring !== active._posePickerRingRef` identity check is
 * itself a piece of implicit contract: it's how the caller knows setup
 * should re-run (new ring analysis landed) vs. skip (unrelated re-render).
 * That check lives in appNG.js's render loop, NOT in this file -- so this
 * file's "setup" functions have no defense against being called when they
 * shouldn't be; they trust the caller to have already decided that.
 *
 * At startup, appNG.js also calls (once, unconditionally):
 *   wirePosePickerControlsNG();
 *   wireScalePickerControlsNG();
 * to attach the slider/button event listeners.
 *
 * ============================================================
 * ENTRY/EXIT CONTRACT (as extracted, project-coupled form)
 * ============================================================
 *   setupPosePickerNG(project)   -- in: project. out: none (mutates project
 *                                    fields + DOM visibility as a side effect)
 *   renderPosePickerGridNG(project) -- in: project. out: none (mutates DOM,
 *                                    reads/writes project fields)
 *   wirePosePickerControlsNG()   -- in: nothing (closes over module-level
 *                                    DOM + ProjectManager). out: none.
 *                                    Attaches event listeners; call once.
 *   (scale picker: same three-function shape, mirrored)
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load this file as its own <script> in indexNG.html, after the DOM
 *      elements above exist and after ProjectManager is defined on window
 *      (same load-order fragility already flagged in initchar.js).
 *   2. Delete the corresponding block from appNG.js (lines ~2737-3026,
 *      the two `const ...CellsNG = buildPickerCellsNG(...)` calls, and
 *      the setup/render call sites at ~1871-1884, plus the two
 *      wire*ControlsNG() calls at startup) -- NOT done here; appNG.js is
 *      untouched by this extraction so nothing breaks yet. Right now this
 *      file is a second copy, not a replacement.
 *   3. Decide whether to genericize pose+scale into one module now or
 *      later (see "WHAT THIS IS" above).
 */

  // ---- DOM refs (module-level globals, captured once at load) ----
  const posePickerGridEl = document.getElementById("ng-pose-picker-grid");
  const posePickerEmptyEl = document.getElementById("ng-pose-picker-empty");
  const posePickerControlsEl = document.getElementById("ng-pose-picker-controls");
  const posePickerPitchSlider = document.getElementById("ng-pose-picker-pitch");
  const posePickerYawSlider = document.getElementById("ng-pose-picker-yaw");
  const posePickerPitchNum = document.getElementById("ng-pose-picker-pitch-num");
  const posePickerYawNum = document.getElementById("ng-pose-picker-yaw-num");
  const posePickerToleranceEnable = document.getElementById("ng-pose-picker-tolerance-enable");
  const posePickerToleranceVal = document.getElementById("ng-pose-picker-tolerance-val");
  const posePickerCountEl = document.getElementById("ng-pose-picker-count");
  const posePickerSelectBtn = document.getElementById("ng-pose-picker-select-btn");
  const posePickerCellsNG = buildPickerCellsNG(posePickerGridEl);

  const scalePickerGridEl = document.getElementById("ng-scale-picker-grid");
  const scalePickerEmptyEl = document.getElementById("ng-scale-picker-empty");
  const scalePickerControlsEl = document.getElementById("ng-scale-picker-controls");
  const scalePickerSlider = document.getElementById("ng-scale-picker-slider");
  const scalePickerNum = document.getElementById("ng-scale-picker-num");
  const scalePickerToleranceEnable = document.getElementById("ng-scale-picker-tolerance-enable");
  const scalePickerToleranceVal = document.getElementById("ng-scale-picker-tolerance-val");
  const scalePickerCountEl = document.getElementById("ng-scale-picker-count");
  const scalePickerSelectBtn = document.getElementById("ng-scale-picker-select-btn");
  const scalePickerCellsNG = buildPickerCellsNG(scalePickerGridEl);

  // Shared by both pickers -- the only genuinely shared code between them.
  function buildPickerCellsNG(gridEl) {
    const cells = [];
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement("div");
      cell.innerHTML = `<img loading="lazy">`;
      gridEl.appendChild(cell);
      cells.push(cell);
    }
    return cells;
  }

  function updatePickerCellVisualNG(project, cell, it, dist, onClick) {
    cell.style.visibility = "visible";
    const isSelected = project.selectedFrames.has(it.frame);
    const img = cell.querySelector("img");
    if (img.dataset.frame !== String(it.frame)) {
      img.src = it.thumbUrl;
      img.dataset.frame = it.frame;
    }
    img.title = it.titleText(dist, isSelected);
    img.style.borderColor = isSelected ? "var(--ng-accent)" : "var(--ng-border)";
    cell.onclick = onClick;
  }

  // ============================================================
  // POSE PICKER -- nearest 9 by 2D {pitch, yaw} distance
  // ============================================================

  function setupPosePickerNG(project) {
    project.posePickerPool = (project.ring ? project.ring.baseResults : [])
      .filter((r) => r.frameId && typeof r.pitch === "number" && typeof r.yaw === "number")
      .map((r) => ({
        frame: r.frame,
        thumbUrl: r.thumbUrl,
        pitch: r.pitch,
        yaw: r.yaw,
        blur: typeof r.blur === "number" ? r.blur : 0,
        vertFillPct: r.vertFillPct,
        titleText(dist, isSelected) {
          const scaleTxt = typeof this.vertFillPct === "number" ? `, face ${(this.vertFillPct * 100).toFixed(0)}% frame ht` : "";
          return `pitch ${this.pitch.toFixed(1)}, yaw ${this.yaw.toFixed(1)} (\u0394${dist.toFixed(1)} from target), sharpness ${this.blur.toFixed(0)}${scaleTxt} \u2014 click to ${isSelected ? "remove from" : "add to"} selection`;
        },
      }));
    project.posePickerDisplayed = new Array(9).fill(null);

    if (!project.posePickerPool.length) {
      if (project.isActive) { posePickerEmptyEl.style.display = "block"; posePickerControlsEl.style.display = "none"; }
      return;
    }
    if (project.isActive) { posePickerEmptyEl.style.display = "none"; posePickerControlsEl.style.display = "flex"; }

    const pitchVals = project.posePickerPool.map((it) => it.pitch);
    const yawVals = project.posePickerPool.map((it) => it.yaw);
    // start centered on the pool's own centroid rather than an arbitrary 0,
    // same convention the original uses.
    project._posePickerPitchTarget = Math.round(pitchVals.reduce((s, v) => s + v, 0) / pitchVals.length);
    project._posePickerYawTarget = Math.round(yawVals.reduce((s, v) => s + v, 0) / yawVals.length);
  }

  function renderPosePickerGridNG(project) {
    if (!project.posePickerPool.length) {
      posePickerEmptyEl.style.display = "block";
      posePickerControlsEl.style.display = "none";
      return;
    }
    posePickerEmptyEl.style.display = "none";
    posePickerControlsEl.style.display = "flex";

    const pitchTarget = typeof project._posePickerPitchTarget === "number" ? project._posePickerPitchTarget : 0;
    const yawTarget = typeof project._posePickerYawTarget === "number" ? project._posePickerYawTarget : 0;
    posePickerPitchSlider.value = pitchTarget;
    posePickerYawSlider.value = yawTarget;
    posePickerPitchNum.value = pitchTarget;
    posePickerYawNum.value = yawTarget;
    posePickerToleranceEnable.checked = !!project._posePickerToleranceOn;
    posePickerToleranceVal.disabled = !project._posePickerToleranceOn;
    posePickerToleranceVal.value = typeof project._posePickerTolerance === "number" ? project._posePickerTolerance : 5;

    let ranked = project.posePickerPool
      .map((it) => ({ it, dist: Math.hypot(it.pitch - pitchTarget, it.yaw - yawTarget) }))
      .sort((a, b) => a.dist - b.dist);
    if (project._posePickerToleranceOn) {
      ranked = ranked.filter((r) => r.dist <= posePickerToleranceVal.value);
    }
    ranked = ranked.slice(0, 9);
    const rankedFrames = ranked.map((r) => r.it.frame);

    const displayed = project.posePickerDisplayed;
    const keepSlot = displayed.map((frame) => frame !== null && rankedFrames.includes(frame));
    const toPlace = ranked.filter((r) => !displayed.includes(r.it.frame));
    let placeIdx = 0;

    for (let i = 0; i < 9; i++) {
      if (keepSlot[i]) {
        const match = ranked.find((r) => r.it.frame === displayed[i]);
        updatePickerCellVisualNG(project, posePickerCellsNG[i], match.it, match.dist, () => {
          project.toggleFrameSelection(match.it.frame);
          ProjectManager.render();
        });
      } else if (placeIdx < toPlace.length) {
        const { it, dist } = toPlace[placeIdx++];
        displayed[i] = it.frame;
        updatePickerCellVisualNG(project, posePickerCellsNG[i], it, dist, () => {
          project.toggleFrameSelection(it.frame);
          ProjectManager.render();
        });
      } else {
        displayed[i] = null;
        posePickerCellsNG[i].style.visibility = "hidden";
      }
    }

    posePickerCountEl.textContent = project._posePickerToleranceOn
      ? `${ranked.length} within ${posePickerToleranceVal.value}\u00b0 of target (${project.posePickerPool.length} in pool)`
      : `${project.posePickerPool.length} in analyzed pool`;
  }

  function wirePosePickerControlsNG() {
    function onChange() {
      const active = ProjectManager.getActive();
      if (!active) return;
      active._posePickerPitchTarget = Math.max(-90, Math.min(90, parseFloat(posePickerPitchNum.value) || 0));
      active._posePickerYawTarget = Math.max(-90, Math.min(90, parseFloat(posePickerYawNum.value) || 0));
      active._posePickerToleranceOn = posePickerToleranceEnable.checked;
      active._posePickerTolerance = parseFloat(posePickerToleranceVal.value) || 5;
      renderPosePickerGridNG(active);
    }
    posePickerPitchSlider.addEventListener("input", () => { posePickerPitchNum.value = posePickerPitchSlider.value; onChange(); });
    posePickerYawSlider.addEventListener("input", () => { posePickerYawNum.value = posePickerYawSlider.value; onChange(); });
    posePickerPitchNum.addEventListener("input", onChange);
    posePickerYawNum.addEventListener("input", onChange);
    posePickerToleranceEnable.addEventListener("change", onChange);
    posePickerToleranceVal.addEventListener("input", onChange);
    posePickerSelectBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.posePickerDisplayed.filter((f) => f !== null).forEach((frame) => active.selectedFrames.add(frame));
      ProjectManager.render();
    });
  }

  // ============================================================
  // SCALE PICKER -- nearest 9 by 1D face-scale-% distance
  // ============================================================

  function setupScalePickerNG(project) {
    const minFacePxEl = document.getElementById("ng-export-min-face");
    const minFacePx = parseFloat(minFacePxEl ? minFacePxEl.value : 0) || 0;

    project.scalePickerPool = (project.ring ? project.ring.baseResults : [])
      .filter((r) => {
        if (!r.frameId) return false;
        const scaleVal = typeof r.vertFillPct === "number" ? r.vertFillPct : (typeof r.bboxRatio === "number" ? Math.sqrt(r.bboxRatio) : null);
        if (scaleVal === null) return false;
        // honor the same "too distant to be useful" floor Export Settings
        // uses, so this picker doesn't surface frames the export pipeline
        // would just skip anyway. THIS IS A REAL CROSS-FEATURE COUPLING --
        // this picker's pool depends on a DOM value owned by Export
        // Settings, not just a UI-adjacent lookup.
        if (minFacePx > 0 && r.bbox) {
          const faceH = r.bbox[3] - r.bbox[1];
          if (faceH < minFacePx) return false;
        }
        return true;
      })
      .map((r) => ({
        frame: r.frame,
        thumbUrl: r.thumbUrl,
        scalePct: (typeof r.vertFillPct === "number" ? r.vertFillPct : Math.sqrt(r.bboxRatio)) * 100,
        blur: typeof r.blur === "number" ? r.blur : 0,
        titleText(dist, isSelected) {
          return `${this.scalePct.toFixed(0)}% frame ht (\u0394${dist.toFixed(1)} from target), sharpness ${this.blur.toFixed(0)} \u2014 click to ${isSelected ? "remove from" : "add to"} selection`;
        },
      }));
    project.scalePickerDisplayed = new Array(9).fill(null);

    if (!project.scalePickerPool.length) {
      if (project.isActive) { scalePickerEmptyEl.style.display = "block"; scalePickerControlsEl.style.display = "none"; }
      return;
    }
    if (project.isActive) { scalePickerEmptyEl.style.display = "none"; scalePickerControlsEl.style.display = "flex"; }

    const scaleVals = project.scalePickerPool.map((it) => it.scalePct);
    project._scalePickerTarget = Math.round(scaleVals.reduce((s, v) => s + v, 0) / scaleVals.length);
  }

  function renderScalePickerGridNG(project) {
    if (!project.scalePickerPool.length) {
      scalePickerEmptyEl.style.display = "block";
      scalePickerControlsEl.style.display = "none";
      return;
    }
    scalePickerEmptyEl.style.display = "none";
    scalePickerControlsEl.style.display = "flex";

    const target = typeof project._scalePickerTarget === "number" ? project._scalePickerTarget : 30;
    scalePickerSlider.value = target;
    scalePickerNum.value = target;
    scalePickerToleranceEnable.checked = !!project._scalePickerToleranceOn;
    scalePickerToleranceVal.disabled = !project._scalePickerToleranceOn;
    scalePickerToleranceVal.value = typeof project._scalePickerTolerance === "number" ? project._scalePickerTolerance : 10;

    let ranked = project.scalePickerPool
      .map((it) => ({ it, dist: Math.abs(it.scalePct - target) }))
      .sort((a, b) => a.dist - b.dist);
    if (project._scalePickerToleranceOn) {
      ranked = ranked.filter((r) => r.dist <= scalePickerToleranceVal.value);
    }
    ranked = ranked.slice(0, 9);
    const rankedFrames = ranked.map((r) => r.it.frame);

    const displayed = project.scalePickerDisplayed;
    const keepSlot = displayed.map((frame) => frame !== null && rankedFrames.includes(frame));
    const toPlace = ranked.filter((r) => !displayed.includes(r.it.frame));
    let placeIdx = 0;

    for (let i = 0; i < 9; i++) {
      if (keepSlot[i]) {
        const match = ranked.find((r) => r.it.frame === displayed[i]);
        updatePickerCellVisualNG(project, scalePickerCellsNG[i], match.it, match.dist, () => {
          project.toggleFrameSelection(match.it.frame);
          ProjectManager.render();
        });
      } else if (placeIdx < toPlace.length) {
        const { it, dist } = toPlace[placeIdx++];
        displayed[i] = it.frame;
        updatePickerCellVisualNG(project, scalePickerCellsNG[i], it, dist, () => {
          project.toggleFrameSelection(it.frame);
          ProjectManager.render();
        });
      } else {
        displayed[i] = null;
        scalePickerCellsNG[i].style.visibility = "hidden";
      }
    }

    scalePickerCountEl.textContent = project._scalePickerToleranceOn
      ? `${ranked.length} within ${scalePickerToleranceVal.value}% of target (${project.scalePickerPool.length} in pool)`
      : `${project.scalePickerPool.length} in analyzed pool`;
  }

  function wireScalePickerControlsNG() {
    function onChange() {
      const active = ProjectManager.getActive();
      if (!active) return;
      active._scalePickerTarget = Math.max(0, Math.min(100, parseFloat(scalePickerNum.value) || 0));
      active._scalePickerToleranceOn = scalePickerToleranceEnable.checked;
      active._scalePickerTolerance = parseFloat(scalePickerToleranceVal.value) || 10;
      renderScalePickerGridNG(active);
    }
    scalePickerSlider.addEventListener("input", () => { scalePickerNum.value = scalePickerSlider.value; onChange(); });
    scalePickerNum.addEventListener("input", onChange);
    scalePickerToleranceEnable.addEventListener("change", onChange);
    scalePickerToleranceVal.addEventListener("input", onChange);
    scalePickerSelectBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.scalePickerDisplayed.filter((f) => f !== null).forEach((frame) => active.selectedFrames.add(frame));
      ProjectManager.render();
    });
  }

  // setupPosePickerNG/renderPosePickerGridNG/wirePosePickerControlsNG/
  // setupScalePickerNG/renderScalePickerGridNG/wireScalePickerControlsNG
  // are plain top-level function declarations above, so they're already
  // reachable as bare globals from other <script> tags loaded after this
  // one -- no explicit export needed (see WIRING_ORDER.md).
