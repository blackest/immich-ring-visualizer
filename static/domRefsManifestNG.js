/**
 * domRefsManifestNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 318-473.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * The single biggest structural finding from this whole extraction pass:
 * appNG.js captures roughly 100+ DOM element references as module-level
 * `const` declarations, ALL scoped to the same top-level IIFE, grouped
 * loosely by comment headers (top bar/tabs, left rail chrome, Anchor
 * section, Video/Image-Set Analysis, Frame Preview, playback modal, hover
 * preview, main stage/sidebar, Immich matches, Immich selection bar,
 * Immich search, task-specific page sections) but with NO actual
 * enforced grouping -- any function anywhere in the file can and does
 * reach any of these by closure, which is exactly why every other
 * extracted file in this pass has a "module-level DOM globals, not
 * passed in" coupling note.
 *
 * THIS is the concrete shape of the "112 scattered DOM refs" finding --
 * this file is the actual list. Every other extracted file that said
 * "see full body for exact element names" or "not enumerated
 * exhaustively" is referring to a subset of what's captured here.
 *
 * ============================================================
 * WHY THIS MATTERS FOR THE REBUILD
 * ============================================================ 
 * If/when these pieces get wired for real, this manifest is the thing
 * that has to exist FIRST, before any of: characterProjectNG.js,
 * projectManagerNG.js, pickersNG.js, playbackModalNG.js,
 * exportSettingsNG.js, personClustersNG.js, chartAndStaticPreviewNG.js,
 * poseListAndFrameHelpersNG.js, or bootstrapWiringNG.js. Every one of
 * those files' "TO MAKE THIS ACTUALLY RUN" notes that mentions "module-
 * level DOM globals... not redefined in this extraction" is pointing
 * back at this file.
 *
 * The honest longer-term fix (not attempted here, per "document, don't
 * refine yet"): this flat, ungrouped, closure-shared list is itself a
 * symptom of the same problem as appNG.js as a whole -- there is no
 * enforced boundary between "which refs belong to which feature." A
 * cleaner version would scope each group of refs to the module that
 * actually owns it (e.g. the playback-modal refs living inside
 * playbackModalNG.js itself, not here) rather than one flat shared pool
 * everything can reach into. Not changed in this pass -- this file is a
 * faithful copy of the list as it exists today, so its scale and shape
 * can be seen clearly.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load FIRST, before every other extracted file.
 *   2. Remove the corresponding lines from appNG.js.
 *   3. Longer-term: consider moving each group into the module that
 *      actually owns it, once the modules themselves are real (not
 *      attempted here).
 */


  // ---- DOM refs: top bar / tabs / bottom bar ----
  const tabsEl = document.getElementById("ng-tabs");
  const mainEl = document.getElementById("ng-main");
  const mainPlaceholderEl = document.getElementById("ng-main-placeholder");
  const newProjectBtn = document.getElementById("ng-new-project");
  const loadProjectBtn = document.getElementById("ng-load-project");
  const taskButtons = Array.from(document.querySelectorAll(".ng-task-btn"));
  const videoAudioEl = document.getElementById("ng-video-audio");

  // ---- DOM refs: left rail chrome ----
  const leftRailEl = document.getElementById("ng-leftrail");
  const leftRailEmptyEl = document.getElementById("ng-leftrail-empty");
  const leftRailBodyEl = document.getElementById("ng-leftrail-body");
  const collapseAllBtn = document.getElementById("ng-leftrail-collapse-all");
  const resizeHandleEl = document.getElementById("ng-leftrail-resize-handle");
  const controlsPaneEl = document.getElementById("ng-controls-pane");
  const splitterEl = document.getElementById("ng-splitter");

  // ---- DOM refs: Anchor section ----
  const ringScaleInput = document.getElementById("ng-ring-scale-input");
  const ringScaleVal = document.getElementById("ng-ring-scale-val");
  const squeezeSlider = document.getElementById("ng-ring-squeeze-slider");
  const squeezeVal = document.getElementById("ng-ring-squeeze-val");
  const ringSortCbs = Array.from(document.querySelectorAll(".ng-ring-sort-cb"));
  const sharpEnableCb = document.getElementById("ng-sharp-squeeze-enable");
  const sharpControlsEl = document.getElementById("ng-sharp-squeeze-controls");
  const sharpSlider = document.getElementById("ng-sharp-squeeze-slider");
  const sharpVal = document.getElementById("ng-sharp-squeeze-val");
  const findNeutralBtn = document.getElementById("ng-find-neutral-btn");
  const neutralPoseReadoutEl = document.getElementById("ng-neutral-pose-readout");

  // ---- DOM refs: Video / Image-Set Analysis section ----
  const videoAnalysisBodyEl = document.getElementById("ng-video-analysis-body");
  const analysisStartInput = document.getElementById("ng-analysis-start-sec");
  const analysisEndInput = document.getElementById("ng-analysis-end-sec");
  const simThresholdInput = document.getElementById("ng-sim-threshold");
  const blurThresholdInput = document.getElementById("ng-blur-threshold");
  const cacheFormatPngCb = document.getElementById("ng-cache-format-png");
  const analysisStatusEl = document.getElementById("ng-analysis-status");
  const loadFolderBtn = document.getElementById("ng-btn-load-folder");
  const loadZipBtn = document.getElementById("ng-btn-load-zip");
  const folderRefIndexInput = document.getElementById("ng-folder-ref-index");
  const analysisSectionTitleEl = document.getElementById("ng-analysis-section-title");
  const folderRowEl = document.querySelector(".ng-folder-row");
  const videoRangeRowEl = document.querySelector(".ng-video-range");

  // ---- DOM refs: Frame Preview section ----
  const previewHintEl = document.getElementById("ng-preview-hint");
  const previewCanvasEl = document.getElementById("ng-preview-canvas");
  const previewControlsScrollEl = document.getElementById("ng-preview-controls-scroll");
  const frameCounterEl = document.getElementById("ng-frame-counter");
  const rewindBtn = document.getElementById("ng-btn-rewind-frame");
  const prevFrameBtn = document.getElementById("ng-btn-prev-frame");
  const playBtn = document.getElementById("ng-btn-play-frames");
  const stopBtn = document.getElementById("ng-btn-stop-frames");
  const nextFrameBtn = document.getElementById("ng-btn-next-frame");
  const startAnalysisBtn = document.getElementById("ng-btn-start-analysis");
  const popoutVideoBtn = document.getElementById("ng-btn-popout-video");

  // ---- DOM refs: playback modal (pop-out, rejected frames blanked) ----
  const playbackModalEl = document.getElementById("ng-playback-modal");
  const playbackModalTitleEl = document.getElementById("ng-playback-modal-title");
  const playbackModalCloseBtn = document.getElementById("ng-playback-modal-close");
  const playbackVideoEl = document.getElementById("ng-playback-video");
  const playbackPrevFrameBtn = document.getElementById("ng-playback-prev-frame");
  const playbackNextFrameBtn = document.getElementById("ng-playback-next-frame");

  // ---- DOM refs: hover preview ----
  const hoverPanel = document.getElementById("ng-preview-hover-panel");
  const hoverImg = document.getElementById("ng-preview-hover-img");
  const hoverCaption = document.getElementById("ng-preview-hover-caption");
  let hoverTimer = null;
  function showHoverPreview(r) {
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => {
      hoverImg.src = thumbUrlFor(r);
      const pctText = typeof r.similarity === "number" ? `${(r.similarity * 100).toFixed(1)}%` : "";
      let poseText = "";
      if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
        poseText = `<br>pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
        if (typeof r.blur === "number") poseText += ` &middot; sharpness: ${r.blur.toFixed(0)}`;
        if (typeof r.vertFillPct === "number") poseText += ` &middot; face: ${(r.vertFillPct * 100).toFixed(0)}% frame height`;
      }
      hoverCaption.innerHTML = `${r.filename}${pctText ? ` &mdash; ${pctText}` : ""}${poseText}`;
      hoverPanel.classList.add("active");
    }, 80);
  }
  function hideHoverPreview() {
    clearTimeout(hoverTimer);
    hoverPanel.classList.remove("active");
  }

  // ---- DOM refs: main stage + sidebar ----
  const stageWrapEl = document.getElementById("ng-stage-wrap");
  const stageEl = document.getElementById("ng-stage");
  const poseListViewEl = document.getElementById("ng-pose-list-view");
  const poseListScrubberEl = document.getElementById("ng-pose-list-scrubber");
  const poseScrubSliderEl = document.getElementById("ng-pose-scrub-slider");
  const poseScrubLeftEl = document.getElementById("ng-pose-scrub-left");
  const poseScrubRightEl = document.getElementById("ng-pose-scrub-right");
  const hudModeEl = document.getElementById("ng-hud-mode");
  const hudFilenameEl = document.getElementById("ng-hud-filename");
  const sidebarEl = document.getElementById("ng-sidebar");
  const toggleListBtn = document.getElementById("ng-toggle-list-btn");
  const sidebarCurrentImgEl = document.getElementById("ng-sidebar-current-img");
  const sidebarCurrentFnameEl = document.getElementById("ng-sidebar-current-fname");
  const sidebarCurrentModeEl = document.getElementById("ng-sidebar-current-mode");
  const sidebarCurrentDetailEl = document.getElementById("ng-sidebar-current-detail");
  const framesSectionEl = document.getElementById("ng-frames-section");
  const framesSectionCountEl = document.getElementById("ng-frames-section-count");
  const listBodyFramesEl = document.getElementById("ng-list-body-frames");
  const framesSelectAllBtn = document.getElementById("ng-frames-select-all");
  const framesDeselectAllBtn = document.getElementById("ng-frames-deselect-all");
  const rankedSortRadios = Array.from(document.querySelectorAll(".ng-ranked-sort-cb"));

  // ---- DOM refs: Immich matches sidebar section ----
  const immichSectionEl = document.getElementById("ng-immich-section");
  const immichSectionCountEl = document.getElementById("ng-immich-section-count");
  const listBodyImmichEl = document.getElementById("ng-list-body-immich");
  const immichSelectAllBtn = document.getElementById("ng-immich-select-all");
  const immichDeselectAllBtn = document.getElementById("ng-immich-deselect-all");
  const immichRefreshBtn = document.getElementById("ng-immich-refresh-btn");
  const immichSaveSelectedBtn = document.getElementById("ng-immich-save-selected");
  const immichExportResultEl = document.getElementById("ng-immich-export-result");

  // ---- DOM refs: sticky Immich live-selection bar (selectedAssetIds
  // while browsing neighbors -- ported from templates/index.html's
  // #immich-selection-bar) ----
  const immichSelectionBarEl = document.getElementById("ng-immich-selection-bar");
  const immichSelectionCountEl = document.querySelector("#ng-immich-selection-count span");
  const immichAnalyzeSelectedBtn = document.getElementById("ng-immich-analyze-selected-btn");
  const immichExportSelectedBtn = document.getElementById("ng-immich-export-selected-btn");
  const immichViewSelectedBtn = document.getElementById("ng-immich-view-selected-btn");
  const immichClearSelectedBtn = document.getElementById("ng-immich-clear-selected-btn");


  // ---- DOM refs: Search section (Immich filename search) ----
  const immichSearchInput = document.getElementById("ng-immich-search-input");
  const immichRandomFaceBtn = document.getElementById("ng-immich-random-face-btn");
  const immichSearchStatusEl = document.getElementById("ng-immich-search-status");
  const immichSearchResultsEl = document.getElementById("ng-immich-search-results");
  const immichAnalyzeBtn = document.getElementById("ng-immich-analyze-btn");
  const immichAnalyzeRefIndexInput = document.getElementById("ng-immich-analyze-ref-index");
  const immichSimThresholdInput = document.getElementById("ng-immich-sim-threshold");
  const immichBlurThresholdInput = document.getElementById("ng-immich-blur-threshold");
  const immichAnalysisStatusEl = document.getElementById("ng-immich-analysis-status");
  const framesSectionTitleEl = document.getElementById("ng-frames-section-title");

  // ---- DOM refs: left-rail sections that are task-specific pages ----
  const videoAnalysisSectionEl = document.querySelector('.panel-section[data-section="video-analysis"]');
  const searchSectionEl = document.querySelector('.panel-section[data-section="search"]');

  // =========================================================================
  // CharacterProject -- one instance per open tab.
  // =========================================================================
