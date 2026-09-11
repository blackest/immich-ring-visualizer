/**
 * bootstrapWiringNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 3189-3825 -- the tail end
 * of appNG.js's top-level IIFE.
 *
 * STATUS: NOT YET WIRED / NOT REALLY EXTRACTABLE AS A CLEAN MODULE. See
 * "WHY THIS ONE IS DIFFERENT" below before treating this the same as the
 * other extracted files.
 *
 * ============================================================
 * WHY THIS ONE IS DIFFERENT
 * ============================================================
 * Every other file extracted so far has SOME kind of declared contract,
 * even where messy (a class, an object with methods, a function taking
 * `project` in). This file does not -- it is 637 lines of:
 *   - the last handful of small helper functions (wireLeftRailChrome,
 *     wireRightSidebarListToggle, the fisheye-lens-effect wiring,
 *     applyNgFisheye, attachNgLensEffect)
 *   - roughly 60 addEventListener() calls wiring up buttons/inputs across
 *     EVERY feature area of the app (video controls, Immich search,
 *     folder/zip loaders, the selected-frames modal, playback modal,
 *     keyboard shortcuts) directly against module-level DOM globals
 *     declared elsewhere in appNG.js
 *   - the final unconditional startup sequence: wireLeftRailChrome(),
 *     wireRightSidebarListToggle(), wireNgFisheyeLensMouseMove(),
 *     wireNgPoseListLensEffect(), wireExportSettingsNG(),
 *     wirePersonClustersNG(), wirePosePickerControlsNG(),
 *     wireScalePickerControlsNG(), ProjectManager.loadState(),
 *     ProjectManager.render()
 *
 * *** THIS IS appNG.js's ACTUAL BOOTSTRAP SEQUENCE -- it is functionally
 * *** the same job as the bootstrap() function John already built once
 * *** for the pre-NG app (converting appa.js/appb.js's implicit load-
 * *** order-dependent glue into named setup functions called explicitly
 * *** in original order from a single bootstrap()), confirmed working in
 * *** the browser at the time. appNG.js never got that treatment -- this
 * *** tail is the same category of problem recurring: real, working glue
 * *** code, but implicit (bare top-level statements in an IIFE, not a
 * *** named function), and entirely dependent on load order and on every
 * *** other piece of appNG.js already existing by the time it runs.
 *
 * This file is extracted verbatim for completeness (per "exhaust
 * appNG.js"), but it does NOT have a clean input/output contract to
 * document the way the others do -- its entire job IS the wiring-together
 * of everything else. The honest path forward here is the same
 * bootstrap() pattern already proven once: wrap this tail in a named
 * bootstrapNG() function, called last, after every other module (the
 * ones already extracted, plus whatever's still to come) has loaded and
 * registered itself on window.
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 * Effectively everything: ProjectManager, PlaybackModal, all the wire*NG
 * functions from every other extracted file (wireExportSettingsNG,
 * wirePersonClustersNG, wirePosePickerControlsNG, wireScalePickerControlsNG),
 * plus ~50+ module-level DOM element globals not enumerated individually
 * here (ringScaleInput, squeezeSlider, ringSortCbs, sharpEnableCb,
 * sharpSlider, findNeutralBtn, simThresholdInput, blurThresholdInput,
 * cacheFormatPngCb, analysisStartInput, analysisEndInput, startAnalysisBtn,
 * folderRefIndexInput, immich* inputs/buttons, folderFilesInput,
 * zipFileInput, selectedModal elements, frames-related/immich-related select/deselect
 * buttons, rewindBtn/prevFrameBtn/nextFrameBtn/playBtn/stopBtn,
 * popoutVideoBtn, playbackModal* elements, newProjectBtn, loadProjectBtn,
 * stageEl, and more -- see the body below for the exhaustive list).
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. This is the LAST file to load, after literally everything else
 *      extracted in this pass (and everything not yet extracted, if any
 *      remains -- see the running tally).
 *   2. Wrap it in a named bootstrapNG() function -- do not leave it as
 *      bare top-level statements -- and call that function once,
 *      explicitly, from wherever the app's real entry point ends up being.
 *   3. Remove the corresponding lines from appNG.js.
 *   4. This is the natural point to also verify nothing was missed in
 *      the extraction pass -- if appNG.js still has content outside what
 *      became these extracted files plus this tail, that's a sign
 *      something was skipped.
 */

  ringScaleInput.addEventListener("input", (e) => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.ringScale = Number(e.target.value);
    ringScaleVal.textContent = active.ringScale + "%";
    if (active.ring || active.immichRing) ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });
  squeezeSlider.addEventListener("input", (e) => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.squeezeMinPct = Number(e.target.value);
    active.squeezeUserOverridden = true;
    if (active.ring || active.immichRing) ProjectManager.renderStage(active);
    else squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
    ProjectManager.saveState();
  });

  ringSortCbs.forEach((cb) => {
    cb.addEventListener("change", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      if (cb.checked) {
        ringSortCbs.forEach((other) => { if (other !== cb) other.checked = false; });
        active.ringSortMetric = cb.dataset.metric;
        if (!active.squeezeUserOverridden) {
          active.squeezeMinPct = SQUEEZE_DEFAULTS[active.ringSortMetric];
          squeezeSlider.value = active.squeezeMinPct;
        }
      } else {
        // don't allow zero selection -- fall back to similarity
        cb.checked = true;
        return;
      }
      if (active.ring || active.immichRing) ProjectManager.renderStage(active);
      ProjectManager.saveState();
    });
  });

  sharpEnableCb.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.sharpCutoffEnabled = sharpEnableCb.checked;
    sharpControlsEl.style.display = active.sharpCutoffEnabled ? "flex" : "none";
    if (active.ring || active.immichRing) ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });
  sharpSlider.addEventListener("input", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.sharpMinVal = Number(sharpSlider.value);
    if (active.ring || active.immichRing) ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });

  findNeutralBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    ProjectManager.findNeutralPose(active);
  });

  // ---- wiring: Video/Image-Set Analysis controls ----
  simThresholdInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.simThreshold = Number(simThresholdInput.value);
    ProjectManager.saveState();
    ProjectManager.renderLeftRail();
  });
  blurThresholdInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.blurThreshold = Number(blurThresholdInput.value);
    ProjectManager.saveState();
    ProjectManager.renderLeftRail();
  });
  cacheFormatPngCb.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.cacheFormatPng = cacheFormatPngCb.checked;
    ProjectManager.saveState();
  });
  // analysisStartInput/analysisEndInput/startAnalysisBtn wiring moved to
  // videoNG.js -- video-exclusive (folder/zip analysis auto-starts on
  // file selection instead, see loadFolderBtn/loadZipBtn below).
  folderRefIndexInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.folderRefIndex = Math.max(1, Number(folderRefIndexInput.value) || 1);
    ProjectManager.saveState();
  });

  // ---- wiring: Immich batch-analysis controls (mirrors the sim/blur
  // threshold inputs above so they don't require switching off the
  // Immich task to tune -- both sets write to the same project fields
  // and are kept in sync via the renderLeftRail() calls above). ----
  if (immichSimThresholdInput) {
    immichSimThresholdInput.addEventListener("change", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.simThreshold = Number(immichSimThresholdInput.value);
      ProjectManager.saveState();
      ProjectManager.renderLeftRail();
    });
  }
  if (immichBlurThresholdInput) {
    immichBlurThresholdInput.addEventListener("change", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.blurThreshold = Number(immichBlurThresholdInput.value);
      ProjectManager.saveState();
      ProjectManager.renderLeftRail();
    });
  }
  if (immichAnalyzeRefIndexInput) {
    immichAnalyzeRefIndexInput.addEventListener("change", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.immichAnalyzeRefIndex = Math.max(1, Number(immichAnalyzeRefIndexInput.value) || 1);
      ProjectManager.saveState();
    });
  }
  if (immichAnalyzeBtn) {
    immichAnalyzeBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active || !active.selectedAssetIds.size) return;
      if (active.job && active.job.status === "running") return;
      active.startImmichAnalysis(Array.from(active.selectedAssetIds));
    });
  }

  // ---- wiring: sticky Immich selection bar -- Analyze/Export reuse the
  // exact same CharacterProject methods as the Search panel's Analyze
  // button and the Immich matches list's Save-selected button
  // respectively (this bar is just a more prominent second entry point
  // to the same actions, matching the original app's layout); View opens
  // the same combined selected-modal; Clear is equivalent to Deselect
  // All but scoped to this bar for parity with the original. ----
  if (immichAnalyzeSelectedBtn) {
    immichAnalyzeSelectedBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active || !active.selectedAssetIds.size) return;
      if (active.job && active.job.status === "running") return;
      active.startImmichAnalysis(Array.from(active.selectedAssetIds));
    });
  }
  if (immichExportSelectedBtn) {
    immichExportSelectedBtn.addEventListener("click", async () => {
      const active = ProjectManager.getActive();
      if (!active || !active.selectedAssetIds.size) return;
      const prevText = immichExportSelectedBtn.textContent;
      immichExportSelectedBtn.textContent = "Exporting\u2026";
      immichExportSelectedBtn.disabled = true;
      try {
        const result = await active.exportSelectedImmichAssets();
        immichExportSelectedBtn.textContent = result.error
          ? `Error: ${result.error}`
          : `Saved ${result.exported} \u2192 ${result.path}`;
      } catch (e) {
        immichExportSelectedBtn.textContent = `Error: ${e.message}`;
      }
      setTimeout(() => {
        immichExportSelectedBtn.textContent = prevText;
        immichExportSelectedBtn.disabled = ProjectManager.getActive() ? !ProjectManager.getActive().selectedAssetIds.size : true;
      }, 4000);
    });
  }
  if (immichViewSelectedBtn) {
    immichViewSelectedBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      ProjectManager.openSelectedModal(active);
    });
  }
  if (immichClearSelectedBtn) {
    immichClearSelectedBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.selectedAssetIds.clear();
      if (active.immichRing) ProjectManager.renderStage(active);
      ProjectManager.saveState();
    });
  }

  if (immichRandomFaceBtn) {
    immichRandomFaceBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active || active.immichSearching) return;
      active.loadRandomImmichFace();
    });
  }

  // ---- wiring: Load folder / .zip pickers (hidden file inputs, created
  // once and reused -- unlike the per-video "Choose Video..." button,
  // these live in the left rail permanently so a persistent pair of
  // inputs is simpler than videoPickerButton()'s per-render approach) ----
  const folderFilesInput = document.createElement("input");
  folderFilesInput.type = "file";
  folderFilesInput.accept = "image/*";
  folderFilesInput.multiple = true;
  folderFilesInput.style.display = "none";
  document.body.appendChild(folderFilesInput);
  folderFilesInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (active && folderFilesInput.files && folderFilesInput.files.length) {
      active.startFolderAnalysis({ images: Array.from(folderFilesInput.files) });
    }
    folderFilesInput.value = "";
  });
  loadFolderBtn.addEventListener("click", () => folderFilesInput.click());

  const zipFileInput = document.createElement("input");
  zipFileInput.type = "file";
  zipFileInput.accept = ".zip";
  zipFileInput.style.display = "none";
  document.body.appendChild(zipFileInput);
  zipFileInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (active && zipFileInput.files && zipFileInput.files[0]) {
      active.startFolderAnalysis({ zip: zipFileInput.files[0] });
    }
    zipFileInput.value = "";
  });
  loadZipBtn.addEventListener("click", () => zipFileInput.click());

  // ---- wiring: selected-frames modal ----
  const selectedModalOverlay = document.getElementById("ng-selected-modal");
  const selectedModalCloseBtn = document.getElementById("ng-selected-modal-close");
  selectedModalCloseBtn.addEventListener("click", () => { selectedModalOverlay.style.display = "none"; });
  selectedModalOverlay.addEventListener("click", (e) => {
    if (e.target === selectedModalOverlay) selectedModalOverlay.style.display = "none";
  });

  // ---- wiring: ranked-list sort + select all/deselect all ----
  rankedSortRadios.forEach((cb) => {
    cb.addEventListener("change", () => {
      if (!cb.checked) return;
      const active = ProjectManager.getActive();
      if (!active) return;
      if (active.task === "immich") {
        active.immichRankedSortMetric = cb.value;
        if (active.immichRing) ProjectManager.renderStage(active);
      } else {
        active.rankedSortMetric = cb.value;
        if (active.ring) ProjectManager.renderStage(active);
      }
      ProjectManager.saveState();
    });
  });
  framesSelectAllBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active || !active.ring) return;
    active.ring.baseResults.forEach((r) => active.selectedFrames.add(r.frame));
    ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });
  framesDeselectAllBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.selectedFrames.clear();
    if (active.ring) ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });

  // ---- wiring: Immich matches select all / deselect all ----
  immichSelectAllBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active || !active.immichRing) return;
    active.immichRing.baseResults.forEach((r) => active.selectedAssetIds.add(r.assetId));
    ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });
  immichDeselectAllBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.selectedAssetIds.clear();
    if (active.immichRing) ProjectManager.renderStage(active);
    ProjectManager.saveState();
  });

  immichRefreshBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (!active || !active.immichRing || active.immichLoading) return;
    active.recenterImmich(active.immichRing.centerAssetId, active.immichRing.centerFilename);
  });

  immichSaveSelectedBtn.addEventListener("click", async () => {
    const active = ProjectManager.getActive();
    if (!active || !active.selectedAssetIds.size) return;
    const prevText = immichSaveSelectedBtn.textContent;
    immichSaveSelectedBtn.textContent = "Saving\u2026";
    immichSaveSelectedBtn.disabled = true;
    try {
      const result = await active.exportSelectedImmichAssets();
      immichExportResultEl.textContent = result.error
        ? `Error: ${result.error}`
        : `Saved ${result.exported} images \u2192 ${result.path}`;
    } catch (e) {
      immichExportResultEl.textContent = `Error: ${e.message}`;
    }
    immichSaveSelectedBtn.textContent = prevText;
    immichSaveSelectedBtn.disabled = active.selectedAssetIds.size === 0;
  });

  // ---- wiring: Immich filename search (debounced, mirrors the original
  // app's wireSearchInputAndNeighbors) ----
  let immichSearchDebounce = null;
  immichSearchInput.addEventListener("input", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    clearTimeout(immichSearchDebounce);
    const q = immichSearchInput.value;
    immichSearchDebounce = setTimeout(() => active.searchImmich(q), 250);
  });

  // ---- wiring: frame-preview controls, playback modal, and the ←/→
  // keyboard shortcut all moved to videoNG.js -- video-exclusive.

  // ---- left rail chrome: collapse-all / per-section expand / resize /
  // splitter. Ported from selection-ui.js's wireMiscBlock1() (unchanged
  // from the previous NG pass). ----
  function wireLeftRailChrome() {
    const savedWidth = localStorage.getItem("immichRingNG:leftPanelWidth");
    if (savedWidth) leftRailEl.style.width = savedWidth + "px";

    const savedCollapsedAll = localStorage.getItem("immichRingNG:leftPanelCollapsedAll") === "1";
    if (savedCollapsedAll) {
      leftRailEl.classList.add("collapsed-all");
      collapseAllBtn.textContent = "▸";
    }

    // Per-section expand/collapse (Anchor, Video Analysis, Person
    // Clusters, Pose Picker, Shot Scale Picker) -- the persisted
    // controls/frame-preview split percentage this used to also restore
    // here was removed along with the splitter (see below).
    document.querySelectorAll(".panel-section").forEach((sec) => {
      const key = "immichRingNG:section:" + sec.dataset.section;
      const saved = localStorage.getItem(key);
      if (saved === "1") sec.classList.add("expanded");
      if (saved === "0") sec.classList.remove("expanded");
      const header = sec.querySelector(".panel-section-header");
      header.addEventListener("click", () => {
        sec.classList.toggle("expanded");
        localStorage.setItem(key, sec.classList.contains("expanded") ? "1" : "0");
      });
    });

    collapseAllBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const collapsed = leftRailEl.classList.toggle("collapsed-all");
      collapseAllBtn.textContent = collapsed ? "▸" : "◄";
      localStorage.setItem("immichRingNG:leftPanelCollapsedAll", collapsed ? "1" : "0");
    });

    // The controls/frame-preview splitter and its drag-resize logic were
    // removed once Frame Preview moved out of the rail entirely (see
    // videoNG.js's header) -- ng-controls-pane is now the rail body's
    // only content, so there's nothing left to split against.

    let widthDragging = false;
    let startX = 0;
    let startWidth = 0;
    resizeHandleEl.addEventListener("mousedown", (e) => {
      if (leftRailEl.classList.contains("collapsed-all")) return;
      widthDragging = true;
      resizeHandleEl.classList.add("dragging");
      startX = e.clientX;
      startWidth = leftRailEl.getBoundingClientRect().width;
      document.body.style.cursor = "ew-resize";
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!widthDragging) return;
      const newWidth = Math.max(240, Math.min(720, startWidth + (e.clientX - startX)));
      leftRailEl.style.width = newWidth + "px";
    });
    document.addEventListener("mouseup", () => {
      if (!widthDragging) return;
      widthDragging = false;
      resizeHandleEl.classList.remove("dragging");
      document.body.style.cursor = "";
      localStorage.setItem("immichRingNG:leftPanelWidth", Math.round(leftRailEl.getBoundingClientRect().width));
    });
  }

  // ---- right sidebar: "Hide"/"Show" toggle for the ranked-matches list,
  // ported from selection-ui.js's wireMiscBlock2() -- collapses the sort
  // row + both list sections (frames/immich, split apart in NG unlike the
  // original's single #list-body) so the "Currently selected" preview at
  // the top gets the full sidebar height. ----
  function wireRightSidebarListToggle() {
    if (!sidebarEl || !toggleListBtn) return;

    function applyHidden(hidden) {
      sidebarEl.classList.toggle("ng-list-hidden", hidden);
      toggleListBtn.textContent = hidden ? "Show" : "Hide";
      toggleListBtn.title = hidden ? "Show ranked match list" : "Hide ranked match list";
    }

    applyHidden(localStorage.getItem("immichRingNG:listHidden") === "1");
    toggleListBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const hidden = !sidebarEl.classList.contains("ng-list-hidden");
      applyHidden(hidden);
      localStorage.setItem("immichRingNG:listHidden", hidden ? "1" : "0");
    });
  }

  // ---- ring fisheye hover-zoom, ported from viz-render.js's
  // applyFisheye()/wireFisheyeLensMouseMove() -- unchanged constants,
  // just re-targeted at #ng-stage/.ng-node instead of #stage/.node. ----
  const NG_FISHEYE_RADIUS = 160;
  const NG_FISHEYE_MAX_SCALE = 2.0;
  const NG_FISHEYE_MAX_PUSH = 46;
  let ngFisheyeRafPending = false;
  let ngFisheyeLastMouse = null;
  function applyNgFisheye(mx, my) {
    const rect = stageEl.getBoundingClientRect();
    const localX = mx - rect.left;
    const localY = my - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    stageEl.querySelectorAll(".ng-node").forEach((node) => {
      const baseX = parseFloat(node.dataset.baseX || 0);
      const baseY = parseFloat(node.dataset.baseY || 0);
      const nodeScreenX = centerX + baseX;
      const nodeScreenY = centerY + baseY;

      const dx = nodeScreenX - localX;
      const dy = nodeScreenY - localY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < NG_FISHEYE_RADIUS) {
        const t = 1 - dist / NG_FISHEYE_RADIUS;
        const eased = t * t * (3 - 2 * t);
        const scale = 1 + eased * (NG_FISHEYE_MAX_SCALE - 1);
        const push = eased * NG_FISHEYE_MAX_PUSH;

        const angle = Math.atan2(baseY, baseX);
        const pushX = baseX === 0 && baseY === 0 ? 0 : Math.cos(angle) * push;
        const pushY = baseX === 0 && baseY === 0 ? 0 : Math.sin(angle) * push;
        node.style.transform = `translate(-50%, -50%) translate(${pushX}px, ${pushY}px) scale(${scale})`;
        node.style.zIndex = Math.round(10 + eased * 50);
      } else {
        node.style.transform = "translate(-50%, -50%)";
        node.style.zIndex = 1;
      }
    });
    ngFisheyeRafPending = false;
  }
  function wireNgFisheyeLensMouseMove() {
    stageEl.addEventListener("mousemove", (e) => {
      ngFisheyeLastMouse = [e.clientX, e.clientY];
      if (!ngFisheyeRafPending) {
        ngFisheyeRafPending = true;
        requestAnimationFrame(() => applyNgFisheye(...ngFisheyeLastMouse));
      }
    });
    stageEl.addEventListener("mouseleave", () => {
      stageEl.querySelectorAll(".ng-node").forEach((node) => {
        node.style.transform = "translate(-50%, -50%)";
        node.style.zIndex = 1;
      });
    });
  }

  // ---- generic dock-style lens/magnify effect, ported from
  // viz-render.js's attachLensEffect() -- reused for the pose-list-view
  // strip below (unlike the ring's applyNgFisheye, this one re-measures
  // each item's own bounding rect on every move rather than working off
  // dataset base-position offsets, since list items scroll horizontally
  // instead of sitting at fixed polar coordinates). ----
  function attachNgLensEffect(container, itemSelector, { radius = 140, maxScale = 1.6 } = {}) {
    let rafPending = false;
    let lastMouse = null;

    function apply(mx, my) {
      const items = container.querySelectorAll(itemSelector);
      items.forEach((item) => {
        const rect = item.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dist = Math.hypot(mx - cx, my - cy);
        const base = item.dataset.baseTransform || "";
        if (dist < radius) {
          const t = 1 - dist / radius;
          const eased = t * t * (3 - 2 * t);
          const scale = 1 + eased * (maxScale - 1);
          item.style.transform = `${base} scale(${scale})`;
          item.style.zIndex = Math.round(10 + eased * 50);
        } else {
          item.style.transform = base;
          item.style.zIndex = item.dataset.baseZ || 1;
        }
      });
      rafPending = false;
    }

    container.addEventListener("mousemove", (e) => {
      lastMouse = [e.clientX, e.clientY];
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(() => apply(...lastMouse));
      }
    });
    container.addEventListener("mouseleave", () => {
      container.querySelectorAll(itemSelector).forEach((item) => {
        item.style.transform = item.dataset.baseTransform || "";
        item.style.zIndex = item.dataset.baseZ || 1;
      });
    });
  }
  const NG_POSE_LENS_RADIUS = 140;
  const NG_POSE_LENS_MAX_SCALE = 1.6;
  function wireNgPoseListLensEffect() {
    if (poseListViewEl) {
      attachNgLensEffect(poseListViewEl, ".ng-pose-list-item", { radius: NG_POSE_LENS_RADIUS, maxScale: NG_POSE_LENS_MAX_SCALE });
    }
  }

  // newProjectBtn click is wired in initguing.js (Void-state handover to
  // ProjectManager.createProject()) -- do not also bind it here, or every
  // "+" click creates two projects (e.g. "Default" and "Default1").
  taskButtons.forEach((btn) => {
    btn.addEventListener("click", () => ProjectManager.setTask(btn.dataset.task));
  });
  // load/save of a character .json is wired in characterIONG.js (it also
  // enables loadProjectBtn) -- runs before this file in load order.

  wireLeftRailChrome();
  wireRightSidebarListToggle();
  wireNgFisheyeLensMouseMove();
  wireNgPoseListLensEffect();
  wireExportSettingsNG();
  wirePersonClustersNG();
  PickerStageNG.wire();
  ProjectManager.loadState();
  ProjectManager.render();

  // Resume polling for any project that was mid-analysis when a previous
  // render cycle set it up (not applicable right after loadState() since
  // fromPlain() marks an interrupted "running" job as an error -- this is
  // here for symmetry/clarity, not currently reachable).
  //
  // (this file's trailing `})();` from the original appNG.js was removed
  // here -- it closed appNG.js's own top-level IIFE, which this chunk was
  // only ever the tail of; not needed now that this file stands alone in
  // the shared page-script scope alongside the other extracted files)
