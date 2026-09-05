// immichRingNG -- client-side shell + per-project logic.
// NG-only file: no shared code with static/viz-render.js, media-ingest.js,
// selection-ui.js, or phosphene-sheet.js.
//
// Each character tab is a CharacterProject instance (per John's request --
// "each char tab" as "an instance of a class") holding its own video,
// analysis-job, ring, and selection state. Analysis jobs poll independently
// per instance regardless of which tab is active, so two tabs analyzing at
// once genuinely run in parallel -- no queuing at this layer (queuing is
// for the shared-GPU character-sheet generation step, a later slice).
//
// Scope ported so far, from the original app's static/viz-render.js +
// media-ingest.js + selection-ui.js: video load/preview/play (unchanged
// from the previous NG pass), Run Analysis -> face-similarity ring +
// ranked-matches list, similarity-band ring layout, hover preview,
// dblclick export-selection with checkboxes + select all/deselect all,
// Anchor's Ring Scale + Min-sim squeeze, ranked-list sim/sharpness/
// vertical% sort. Deliberately NOT ported yet (left as stubs elsewhere in
// indexNG.html): fisheye lens hover-zoom, the yaw/pitch/roll ring-sort ->
// pose-list-view mode, sharpness-cutoff squeeze, Search, Immich/Folder-Zip
// ingest, Person Clusters, Pose/Shot-Scale pickers, Export Settings +
// selection modal/export pipeline, character-sheet generation.
//
// The left rail's markup (in templates/indexNG.html) is a wholesale port
// of the original app's #left-panel, trimmed section by section as NG
// functionality lands. The collapse/resize/splitter chrome is ported from
// selection-ui.js's wireMiscBlock1(), with its own NG-prefixed localStorage
// keys so it doesn't collide with the original app's own state.

(function () {
  "use strict";

  const STORAGE_KEY = "immichRingNG:state";

  // ---- ring layout constants (from viz-render.js) ----
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

  // ---- DOM refs: Search section (Immich filename search) ----
  const immichSearchInput = document.getElementById("ng-immich-search-input");
  const immichSearchStatusEl = document.getElementById("ng-immich-search-status");
  const immichSearchResultsEl = document.getElementById("ng-immich-search-results");

  // ---- DOM refs: left-rail sections that are task-specific pages ----
  const videoAnalysisSectionEl = document.querySelector('.panel-section[data-section="video-analysis"]');
  const searchSectionEl = document.querySelector('.panel-section[data-section="search"]');

  // =========================================================================
  // CharacterProject -- one instance per open tab.
  // =========================================================================
  class CharacterProject {
    constructor(name) {
      this.id = "p" + ProjectManager.nextId++;
      this.name = name && name.trim() ? name.trim() : ProjectManager.uniquePlaceholderName("default");
      this.task = null;

      // video state (unchanged from the previous NG pass)
      this.video = null; // { previewId, fps, totalFrames, duration, currentFrame, objectUrl, rangeStartSec, rangeEndSec }
      this.videoLoading = false;
      this.videoFile = null; // raw File, needed to re-upload for /api/ng/analyze-video -- not persisted (dies on reload, same as objectUrl)

      // analysis settings, kept per-project so switching tabs doesn't lose them
      this.simThreshold = 0.1;
      this.blurThreshold = 1;
      this.cacheFormatPng = false;

      // Anchor settings
      this.ringScale = 100; // percent
      this.squeezeMinPct = 65;
      this.squeezeUserOverridden = false;
      this.ringSortMetric = "sim"; // sim | yaw | pitch | roll | blur -- switches ring vs pose-list view
      this.sharpCutoffEnabled = false;
      this.sharpMinVal = 0;
      this.folderRefIndex = 1; // 1-based, which image in a folder/zip upload is the reference face

      // analysis job + ring state
      this.job = null; // { jobId, status, error, sourceName, frameCount, passed, failedSim, failedBlur, results, sourceType, refFrame }
      this.ring = null; // { anchorUrl, refFrameIdx, baseResults, sourceType }
      this.playback = null; // { url, fps, jobId } -- rejected-frames-blanked reassembly, popped out in a modal
      this.playbackBuilding = false;
      this.rankedSortMetric = "sim";
      this.selectedFrames = new Set();
      this.staticPreviewFrame = null; // folder/image-set jobs only -- a chart-click result row, since there's no live decode to scrub (see renderFramePreview)

      // Immich ingest state -- independent of the video state above, so a
      // project can have a video ring AND an Immich ring at once, each
      // with its own selection set. Built the same slice-by-slice way as
      // video: search by filename -> pick a face -> pgvector neighbors.
      this.immichSearchQuery = "";
      this.immichSearchResults = []; // transient, not persisted
      this.immichSearching = false;
      this.immichLoading = false;
      this.immichError = null;
      this.immichRing = null; // { centerAssetId, centerFilename, centerPose, mode, baseResults }
      this.immichRankedSortMetric = "sim";
      this.selectedAssetIds = new Set();

      this._pollTimer = null;
      this._playTimer = null;
      this._lastFolderSource = null; // { images, zip } -- last folder/zip upload, kept for "Use as reference & re-analyze"; not persisted (Files don't survive JSON)
    }

    get isActive() {
      return ProjectManager.activeId === this.id;
    }

    stopPlayIfRunning() {
      if (!this._playTimer) return;
      clearInterval(this._playTimer);
      this._playTimer = null;
      // videoAudioEl is a single shared element across all projects -- only
      // pause it if it's still actually pointed at THIS project's video.
      // Regression fix: switching tabs used to leave a playing project's
      // timer running in the background, where it would keep reading/
      // fighting over the shared <audio> element with whichever project
      // became active (reported as "holding the same video").
      if (this.video && videoAudioEl.dataset.objectUrl === this.video.objectUrl) {
        videoAudioEl.pause();
      }
    }

    stopPolling() {
      if (this._pollTimer) {
        clearTimeout(this._pollTimer);
        this._pollTimer = null;
      }
    }

    destroy() {
      this.stopPlayIfRunning();
      this.stopPolling();
      if (this.video && this.video.objectUrl) URL.revokeObjectURL(this.video.objectUrl);
    }

    // ---- persistence (browser-local convenience only, see loadState()/saveState()) ----
    toPlain() {
      return {
        id: this.id,
        name: this.name,
        task: this.task,
        video: this.video,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        cacheFormatPng: this.cacheFormatPng,
        ringScale: this.ringScale,
        squeezeMinPct: this.squeezeMinPct,
        squeezeUserOverridden: this.squeezeUserOverridden,
        ringSortMetric: this.ringSortMetric,
        sharpCutoffEnabled: this.sharpCutoffEnabled,
        sharpMinVal: this.sharpMinVal,
        folderRefIndex: this.folderRefIndex,
        job: this.job,
        ring: this.ring,
        rankedSortMetric: this.rankedSortMetric,
        selectedFrames: Array.from(this.selectedFrames),
        immichRing: this.immichRing,
        immichRankedSortMetric: this.immichRankedSortMetric,
        selectedAssetIds: Array.from(this.selectedAssetIds),
      };
    }

    static fromPlain(data) {
      const p = Object.create(CharacterProject.prototype);
      p.id = data.id;
      p.name = data.name;
      p.task = data.task || null;
      p.video = data.video || null;
      if (p.video) p.video.objectUrl = null; // blob URLs never survive a JSON round-trip
      p.videoLoading = false;
      p.videoFile = null;
      p.simThreshold = typeof data.simThreshold === "number" ? data.simThreshold : 0.1;
      p.blurThreshold = typeof data.blurThreshold === "number" ? data.blurThreshold : 1;
      p.cacheFormatPng = !!data.cacheFormatPng;
      p.ringScale = typeof data.ringScale === "number" ? data.ringScale : 100;
      p.squeezeMinPct = typeof data.squeezeMinPct === "number" ? data.squeezeMinPct : 65;
      p.squeezeUserOverridden = !!data.squeezeUserOverridden;
      p.ringSortMetric = data.ringSortMetric || "sim";
      p.sharpCutoffEnabled = !!data.sharpCutoffEnabled;
      p.sharpMinVal = typeof data.sharpMinVal === "number" ? data.sharpMinVal : 0;
      p.folderRefIndex = typeof data.folderRefIndex === "number" ? data.folderRefIndex : 1;
      p.job = data.job || null;
      // playback builds live in a server tempdir (FRAME_STORE) that doesn't
      // survive a server restart, and the modal itself is a transient UI
      // concern -- neither is worth persisting across a page reload.
      p.playback = null;
      p.playbackBuilding = false;
      p.staticPreviewFrame = null;
      if (p.job && p.job.status === "running") {
        // a page reload orphaned the in-browser poll loop -- the backend
        // job may have finished or may not even exist anymore (server
        // restart). Mark it stale rather than silently polling forever.
        p.job = { ...p.job, status: "error", error: "Analysis was interrupted by a page reload." };
      }
      p.ring = data.ring || null;
      p.rankedSortMetric = data.rankedSortMetric || "sim";
      p.selectedFrames = new Set(Array.isArray(data.selectedFrames) ? data.selectedFrames : []);
      p.immichSearchQuery = "";
      p.immichSearchResults = [];
      p.immichSearching = false;
      p.immichLoading = false;
      p.immichError = null;
      p.immichRing = data.immichRing || null;
      p.immichRankedSortMetric = data.immichRankedSortMetric || "sim";
      p.selectedAssetIds = new Set(Array.isArray(data.selectedAssetIds) ? data.selectedAssetIds : []);
      p._pollTimer = null;
      p._playTimer = null;
      p._lastFolderSource = null;
      return p;
    }

    // ---- video ingest (ported from the previous NG pass, unchanged) ----
    async loadVideo(file) {
      this.videoLoading = true;
      ProjectManager.render();

      const formData = new FormData();
      formData.append("video", file);

      try {
        const res = await fetch("/api/ng/preview-video", { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) {
          alert("Could not load video: " + (data.error || res.status));
          return;
        }
        if (this.video && this.video.objectUrl) URL.revokeObjectURL(this.video.objectUrl);
        const objectUrl = URL.createObjectURL(file);
        this.video = {
          previewId: data.previewId,
          fps: data.fps,
          totalFrames: data.totalFrames,
          duration: data.duration,
          currentFrame: 1,
          objectUrl: objectUrl,
          rangeStartSec: this.video ? this.video.rangeStartSec : null,
          rangeEndSec: this.video ? this.video.rangeEndSec : null,
        };
        this.videoFile = file;
        // a new video invalidates any previous analysis/ring -- avoid
        // showing a ring built from a different clip's frames.
        this.stopPolling();
        this.job = null;
        this.ring = null;
        this.playback = null;
        this.selectedFrames = new Set();
        this.staticPreviewFrame = null;
      } catch (e) {
        alert("Could not load video: " + e.message);
      } finally {
        this.videoLoading = false;
        ProjectManager.render();
      }
    }

    frameTime(frameIdx) {
      const fps = this.video.fps > 0 ? this.video.fps : 24;
      return Math.max(0, (frameIdx - 1) / fps);
    }

    frameFromTime(timeSeconds) {
      const fps = this.video.fps > 0 ? this.video.fps : 24;
      return Math.min(this.video.totalFrames, Math.max(1, Math.floor(timeSeconds * fps) + 1));
    }

    syncAudioToFrame(frameIdx) {
      if (!videoAudioEl.src || videoAudioEl.dataset.objectUrl !== this.video.objectUrl) return;
      try {
        videoAudioEl.currentTime = this.frameTime(frameIdx);
      } catch (e) {
        // seeking before metadata is ready can throw -- harmless, ignore
      }
    }

    stepTo(frameNo) {
      const clamped = Math.max(1, Math.min(this.video.totalFrames, frameNo));
      this.video.currentFrame = clamped;
      if (this.isActive) {
        frameCounterEl.textContent = "Frame: " + clamped + " / " + this.video.totalFrames;
        drawFrame(this, clamped);
      }
      ProjectManager.saveState();
    }

    stepAndSyncAudio(frameNo) {
      this.stopPlayIfRunning();
      if (this.isActive) setPlayingVisual(false);
      this.stepTo(frameNo);
      this.syncAudioToFrame(this.video.currentFrame);
    }

    togglePlay() {
      if (this._playTimer) {
        this.stopPlayIfRunning();
        if (this.isActive) {
          videoAudioEl.pause();
          setPlayingVisual(false);
        }
        return;
      }

      const hasAudio = !!this.video.objectUrl && videoAudioEl.dataset.objectUrl === this.video.objectUrl;
      if (!hasAudio) {
        this.playFramesWithoutAudio();
        return;
      }

      if (this.isActive) setPlayingVisual(true);
      if (this.video.currentFrame >= this.video.totalFrames) this.stepTo(1);
      this.syncAudioToFrame(this.video.currentFrame);

      const playPromise = videoAudioEl.play();
      if (playPromise && playPromise.catch) {
        playPromise.catch(() => this.playFramesWithoutAudio());
      }

      const sampleMs = (this.video.fps > 0 ? 1000 / this.video.fps : 1000 / 24) / 2;
      this._playTimer = setInterval(() => {
        if (videoAudioEl.paused || videoAudioEl.ended) {
          this.stopPlayIfRunning();
          if (this.isActive) setPlayingVisual(false);
          return;
        }
        const target = this.frameFromTime(videoAudioEl.currentTime);
        if (target !== this.video.currentFrame) this.stepTo(target);
        if (target >= this.video.totalFrames) {
          this.stopPlayIfRunning();
          videoAudioEl.pause();
          if (this.isActive) setPlayingVisual(false);
        }
      }, sampleMs);
    }

    playFramesWithoutAudio() {
      if (this.isActive) setPlayingVisual(true);
      const intervalMs = this.video.fps > 0 ? 1000 / this.video.fps : 1000 / 24;
      this._playTimer = setInterval(() => {
        const next = this.video.currentFrame + 1;
        if (next > this.video.totalFrames) {
          this.stopPlayIfRunning();
          if (this.isActive) setPlayingVisual(false);
          return;
        }
        this.stepTo(next);
      }, intervalMs);
    }

    // ---- analysis: Run Analysis with Selected Frame -> ring + ranked matches ----
    async startAnalysis(refFrameOverride) {
      if (!this.video || !this.videoFile) return;
      const refFrame = refFrameOverride != null ? refFrameOverride : this.video.currentFrame;
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null; // a new analysis run invalidates any previous playback build
      this.staticPreviewFrame = null;
      this.job = {
        status: "running",
        sourceType: "video",
        sourceName: this.videoFile.name,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0, passed: 0, failedSim: 0, failedBlur: 0,
        results: [],
        refFrame,
      };
      ProjectManager.render();

      const form = new FormData();
      form.append("video", this.videoFile);
      form.append("simThreshold", this.simThreshold);
      form.append("blurThreshold", this.blurThreshold);
      form.append("refFrame", refFrame);
      form.append("cacheFormat", this.cacheFormatPng ? "png" : "jpg");
      if (this.video.rangeStartSec != null) form.append("startSec", this.video.rangeStartSec);
      if (this.video.rangeEndSec != null) form.append("endSec", this.video.rangeEndSec);

      try {
        const res = await fetch("/api/ng/analyze-video", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok || data.error) {
          this.job = { status: "error", error: data.error || res.status };
          if (this.isActive) ProjectManager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.poll();
      } catch (e) {
        this.job = { status: "error", error: e.message };
        if (this.isActive) ProjectManager.render();
      }
    }

    // ---- analysis: Load folder / .zip -> same pipeline, over stills ----
    async startFolderAnalysis({ images, zip }, refIndexOverride) {
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null;
      this.staticPreviewFrame = null;
      const refIndex = refIndexOverride != null ? refIndexOverride : this.folderRefIndex;
      const sourceName = zip ? zip.name.replace(/\.zip$/i, "") : "folder_set";
      this.job = {
        status: "running",
        sourceType: "folder",
        sourceName,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0, passed: 0, failedSim: 0, failedBlur: 0,
        results: [],
        refFrame: refIndex,
      };
      ProjectManager.render();

      const form = new FormData();
      if (zip) {
        form.append("zip", zip);
      } else {
        images.forEach((f) => form.append("images", f));
        form.append("sourceName", sourceName);
      }
      form.append("simThreshold", this.simThreshold);
      form.append("blurThreshold", this.blurThreshold);
      form.append("refIndex", refIndex);
      form.append("cacheFormat", this.cacheFormatPng ? "png" : "jpg");
      this._lastFolderSource = { images, zip }; // for "Use as reference & re-analyze"

      try {
        const res = await fetch("/api/ng/analyze-folder", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok || data.error) {
          this.job = { status: "error", error: data.error || res.status };
          if (this.isActive) ProjectManager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.job.frameCount = data.imageCount;
        this.poll();
      } catch (e) {
        this.job = { status: "error", error: e.message };
        if (this.isActive) ProjectManager.render();
      }
    }

    async poll() {
      if (!this.job || !this.job.jobId) return;
      try {
        const res = await fetch(`/api/ng/analysis-status/${this.job.jobId}`);
        const data = await res.json();

        if (data.error) {
          this.job.status = "error";
          this.job.error = data.error;
          if (this.isActive) ProjectManager.render();
          return;
        }

        this.job.status = data.status;
        this.job.error = data.error;
        this.job.frameCount = data.frameCount;
        this.job.results = data.results;
        this.job.passed = data.results.filter((r) => r.passed).length;
        this.job.failedSim = data.results.filter((r) => !r.passed && r.failReason === "sim").length;
        this.job.failedBlur = data.results.filter((r) => !r.passed && r.failReason === "blur").length;

        if (data.status === "running") {
          if (this.isActive) ProjectManager.renderLeftRail();
          this._pollTimer = setTimeout(() => this.poll(), 800);
          return;
        }

        if (data.status === "done") {
          this.buildRing(data.results);
        }
        if (this.isActive) ProjectManager.render();
      } catch (e) {
        this.job.status = "error";
        this.job.error = e.message;
        if (this.isActive) ProjectManager.render();
      }
    }

    buildRing(results) {
      const sourceType = this.job.sourceType || "video";
      const baseResults = results
        .filter((r) => r.passed)
        .map((r) => ({
          filename: r.origName || `frame_${r.frame}`,
          frame: r.frame,
          similarity: r.sim,
          thumbUrl: `/api/ng/framefile/${r.frameId}`,
          pitch: r.pitch, yaw: r.yaw, roll: r.roll, blur: r.blur,
          bboxRatio: r.bboxRatio, vertFillPct: r.vertFillPct,
        }))
        .sort((a, b) => b.similarity - a.similarity);

      this.ring = {
        anchorUrl: `/api/ng/framefile/${this.job.jobId}_anchor`,
        refFrameIdx: this.job.refFrame,
        baseResults,
        sourceType,
      };
    }

    // ---- Save kept / Save selected frames to disk ----
    async exportFrames(onlySelected) {
      if (!this.job || !this.job.jobId) return null;
      const body = onlySelected ? { frames: Array.from(this.selectedFrames) } : {};
      const res = await fetch(`/api/ng/export-job/${this.job.jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json();
    }

    // ---- playback: reassemble the clip with rejected frames blanked out,
    // popped out in a modal so it's not fighting the ring for rail space.
    // Ported from the original app's inline "Build playback" button. ----
    async buildPlayback() {
      if (!this.job || this.job.status !== "done" || this.playbackBuilding) return;
      this.playbackBuilding = true;
      if (this.isActive) ProjectManager.render();

      try {
        const res = await fetch(`/api/ng/build-playback/${this.job.jobId}`, { method: "POST" });
        const data = await res.json();
        if (!res.ok || data.error) {
          alert("Could not build playback: " + (data.error || res.status));
          return;
        }
        this.playback = { url: data.url, fps: data.fps || 24, jobId: this.job.jobId };
        if (this.isActive) PlaybackModal.openBuild(this);
      } catch (e) {
        alert("Could not build playback: " + e.message);
      } finally {
        this.playbackBuilding = false;
        if (this.isActive) ProjectManager.render();
      }
    }

    sortedRanked() {
      if (!this.ring) return [];
      const key = this.rankedSortMetric === "sim" ? "similarity" : this.rankedSortMetric;
      const withMetric = this.ring.baseResults.filter((r) => typeof r[key] === "number");
      const withoutMetric = this.ring.baseResults.filter((r) => typeof r[key] !== "number");
      withMetric.sort((a, b) => b[key] - a[key]);
      return [...withMetric, ...withoutMetric];
    }

    // min-sim squeeze (ported from viz-render.js's applySqueeze) + the
    // independent sharpness-cutoff squeeze, both scoped to this project.
    squeezeFiltered(sorted) {
      const cutoff = this.squeezeMinPct / 100;
      const simKept = sorted.filter((r) => (typeof r.similarity === "number" ? r.similarity : 1) >= cutoff);

      if (!this.sharpCutoffEnabled) {
        return { kept: simKept, total: sorted.length };
      }

      // sharpness cutoff only applies to items that actually carry a blur
      // score; nodes with no blur field (e.g. Immich-only) pass through
      // untouched rather than being dropped.
      const kept = simKept.filter((r) => (typeof r.blur !== "number" ? true : r.blur >= this.sharpMinVal));
      return { kept, total: sorted.length };
    }

    toggleFrameSelection(frame) {
      if (this.selectedFrames.has(frame)) this.selectedFrames.delete(frame);
      else this.selectedFrames.add(frame);
      ProjectManager.saveState();
    }

    // ---- Immich ingest: search by filename -> pick a face -> ring of
    // pgvector-nearest neighbors (face embedding, falling back to CLIP
    // image embedding), single-click a node to recenter, dblclick to
    // toggle export selection (mirrors the video ring's dblclick-select). ----
    async searchImmich(query) {
      this.immichSearchQuery = query;
      const trimmed = query.trim();
      if (!trimmed) {
        this.immichSearchResults = [];
        if (this.isActive) ProjectManager.renderLeftRail();
        return;
      }
      this.immichSearching = true;
      if (this.isActive) ProjectManager.renderLeftRail();
      try {
        const res = await fetch(`/api/ng/find-by-filename?name=${encodeURIComponent(trimmed)}`);
        const data = await res.json();
        this.immichSearchResults = Array.isArray(data) ? data : [];
      } catch (e) {
        this.immichSearchResults = [];
      } finally {
        this.immichSearching = false;
        if (this.isActive) ProjectManager.renderLeftRail();
      }
    }

    async loadImmichNeighbors(assetId, filename) {
      this.immichLoading = true;
      this.immichError = null;
      if (this.isActive) ProjectManager.render();

      try {
        const res = await fetch(`/api/ng/neighbors?assetId=${encodeURIComponent(assetId)}&limit=36`);
        const data = await res.json();
        if (!res.ok || data.error) {
          this.immichError = data.error || res.status;
          return;
        }
        const baseResults = data.results
          .filter((r) => r.assetId !== assetId)
          .map((r) => ({
            assetId: r.assetId,
            filename: r.filename,
            similarity: r.similarity,
            thumbUrl: `/api/ng/thumb/${r.assetId}`,
          }));
        this.immichRing = {
          centerAssetId: assetId,
          centerFilename: filename,
          centerPose: null,
          mode: data.mode,
          baseResults,
        };
        // dropping the search results once a ring is built keeps the left
        // rail tidy -- the query text stays so re-searching is easy
        this.immichSearchResults = [];
      } catch (e) {
        this.immichError = e.message;
      } finally {
        this.immichLoading = false;
        if (this.isActive) ProjectManager.render();
      }
      if (this.immichRing && this.immichRing.centerAssetId === assetId) {
        this.loadImmichCenterPose(assetId);
      }
    }

    async loadImmichCenterPose(assetId) {
      // Lazy, best-effort -- pose/blur for the centered asset only, not
      // for every neighbor (that would be one face-detection call per
      // thumbnail just to populate a list). Mirrors the original app's
      // "only called when the match is actually added to the ring".
      try {
        const res = await fetch(`/api/ng/asset-face-pose/${assetId}`);
        const data = await res.json();
        if (!res.ok || data.error) return;
        if (!this.immichRing || this.immichRing.centerAssetId !== assetId) return;
        this.immichRing.centerPose = data;
        if (this.isActive) ProjectManager.renderStage(this);
      } catch (e) {
        // pose is a nice-to-have here -- leave it blank on any failure
      }
    }

    recenterImmich(assetId, filename) {
      this.loadImmichNeighbors(assetId, filename);
    }

    sortedRankedImmich() {
      if (!this.immichRing) return [];
      const key = this.immichRankedSortMetric === "sim" ? "similarity" : this.immichRankedSortMetric;
      const withMetric = this.immichRing.baseResults.filter((r) => typeof r[key] === "number");
      const withoutMetric = this.immichRing.baseResults.filter((r) => typeof r[key] !== "number");
      withMetric.sort((a, b) => b[key] - a[key]);
      return [...withMetric, ...withoutMetric];
    }

    toggleAssetSelection(assetId) {
      if (this.selectedAssetIds.has(assetId)) this.selectedAssetIds.delete(assetId);
      else this.selectedAssetIds.add(assetId);
      ProjectManager.saveState();
    }
  }

  // =========================================================================
  // ProjectManager -- tab bar / active-project orchestration / DOM rendering.
  // =========================================================================
  const ProjectManager = {
    projects: [],
    activeId: null,
    nextId: 1,
    dragSourceId: null,

    getActive() {
      return this.projects.find((p) => p.id === this.activeId) || null;
    },

    isNameTaken(name, excludeId) {
      const lower = name.trim().toLowerCase();
      return this.projects.some((p) => p.id !== excludeId && p.name.toLowerCase() === lower);
    },

    uniquePlaceholderName(base) {
      let candidate = base;
      let n = 1;
      while (this.isNameTaken(candidate, null)) {
        candidate = base + n;
        n++;
      }
      return candidate;
    },

    createProject(name) {
      const project = new CharacterProject(name);
      this.projects.push(project);
      this.activeId = project.id;
      this.render();
    },

    closeProject(id) {
      // Placeholder for "closing saves state" -- no disk-backed persistence
      // layer yet (see APP_ARCHITECTURE_NOTES.md's shopping-list/SQLite
      // plan), so this just removes it from the in-memory tab list for now.
      const closing = this.projects.find((p) => p.id === id);
      if (closing) closing.destroy();
      if (typeof PlaybackModal !== "undefined" && PlaybackModal.projectId === id) PlaybackModal.close();
      const idx = this.projects.findIndex((p) => p.id === id);
      if (idx === -1) return;
      this.projects.splice(idx, 1);
      if (this.activeId === id) {
        this.activeId = this.projects.length ? this.projects[this.projects.length - 1].id : null;
      }
      this.render();
    },

    setActive(id) {
      if (this.activeId === id) return;
      const outgoing = this.getActive();
      if (outgoing) outgoing.stopPlayIfRunning();
      // the playback modal is a pop-out for one specific project's build --
      // don't leave it open showing the outgoing project's clip once a
      // different tab becomes active.
      if (typeof PlaybackModal !== "undefined" && PlaybackModal.projectId && PlaybackModal.projectId !== id) {
        PlaybackModal.close();
      }
      neutralPoseReadoutEl.style.display = "none";
      neutralPoseReadoutEl.innerHTML = "";
      this.activeId = id;
      this.render();
    },

    reorderProject(sourceId, targetId) {
      if (sourceId === targetId) return;
      const fromIdx = this.projects.findIndex((p) => p.id === sourceId);
      const toIdx = this.projects.findIndex((p) => p.id === targetId);
      if (fromIdx === -1 || toIdx === -1) return;
      const [moved] = this.projects.splice(fromIdx, 1);
      this.projects.splice(toIdx, 0, moved);
      this.render();
    },

    renameProject(id, name) {
      const project = this.projects.find((p) => p.id === id);
      if (!project) return;
      const trimmed = name.trim();
      if (!trimmed || trimmed === project.name) return;
      if (this.isNameTaken(trimmed, id)) {
        alert(trimmed + " -- there can only be one.");
        return;
      }
      project.name = trimmed;
    },

    setTask(task) {
      const project = this.getActive();
      if (!project) return;
      if (project.task === "video" && task !== "video") project.stopPlayIfRunning();
      project.task = task;
      this.render();
    },

    // ---- persistence (browser-local convenience: survives a refresh, not
    // a real save/load layer -- see APP_ARCHITECTURE_NOTES.md) ----
    saveState() {
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            projects: this.projects.map((p) => p.toPlain()),
            activeId: this.activeId,
            nextId: this.nextId,
          })
        );
      } catch (e) {
        // localStorage can throw (private mode, quota, disabled) -- non-fatal
      }
    },

    loadState() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (Array.isArray(data.projects)) {
          data.projects.forEach((p) => this.projects.push(CharacterProject.fromPlain(p)));
          this.activeId = data.activeId || null;
          this.nextId = data.nextId || this.projects.length + 1;
        }
      } catch (e) {
        // corrupt/missing state -- just start empty
      }
    },

    // ---- rendering ----
    render() {
      this.renderTabs();
      this.renderBottomBar();
      this.renderLeftRail();
      this.renderMain();
      this.saveState();
    },

    renderTabs() {
      tabsEl.innerHTML = "";
      this.projects.forEach((project) => {
        const tab = document.createElement("div");
        tab.className = "ng-tab" + (project.id === this.activeId ? " ng-tab-active" : "");
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", project.id === this.activeId ? "true" : "false");

        tab.draggable = true;
        tab.addEventListener("dragstart", (e) => {
          this.dragSourceId = project.id;
          tab.classList.add("ng-tab-dragging");
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", project.id);
        });
        tab.addEventListener("dragend", () => {
          this.dragSourceId = null;
          tab.classList.remove("ng-tab-dragging");
        });
        tab.addEventListener("dragover", (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
        });
        tab.addEventListener("drop", (e) => {
          e.preventDefault();
          if (this.dragSourceId) this.reorderProject(this.dragSourceId, project.id);
        });

        const label = document.createElement("span");
        label.className = "ng-tab-label";
        label.textContent = project.name;
        label.title = "Double-click to rename";
        label.addEventListener("dblclick", (e) => {
          e.stopPropagation();
          label.contentEditable = "true";
          label.focus();
          document.execCommand("selectAll", false, null);
        });
        label.addEventListener("blur", () => {
          label.contentEditable = "false";
          this.renameProject(project.id, label.textContent);
          this.render();
        });
        label.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            label.blur();
          }
        });

        const closeBtn = document.createElement("button");
        closeBtn.className = "ng-tab-close";
        closeBtn.textContent = "×";
        closeBtn.title = "Close (saves state)";
        closeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.closeProject(project.id);
        });

        tab.addEventListener("click", () => this.setActive(project.id));
        tab.appendChild(label);
        tab.appendChild(closeBtn);
        tabsEl.appendChild(tab);
      });
    },

    renderBottomBar() {
      const active = this.getActive();
      taskButtons.forEach((btn) => {
        btn.disabled = !active;
        btn.classList.toggle("ng-task-active", !!active && active.task === btn.dataset.task);
      });
    },

    renderMain() {
      const active = this.getActive();

      const showPlaceholder = (text) => {
        mainPlaceholderEl.style.display = "";
        mainPlaceholderEl.textContent = text;
        stageWrapEl.style.display = "none";
        sidebarEl.style.display = "none";
      };

      if (!active) {
        showPlaceholder("No project open. Press + to start one.");
        return;
      }
      if (!active.task) {
        showPlaceholder("Pick Video, Immich, or Folder / Zip below to get started with “" + active.name + "”.");
        return;
      }
      if (active.task === "video") {
        if (!active.ring) {
          showPlaceholder("No analysis yet for “" + active.name + "” — load a video, pick a frame, and press Run Analysis.");
          return;
        }
      } else if (active.task === "immich") {
        if (active.immichLoading && !active.immichRing) {
          showPlaceholder("Loading Immich neighbors…");
          return;
        }
        if (active.immichError && !active.immichRing) {
          showPlaceholder("Immich error: " + active.immichError);
          return;
        }
        if (!active.immichRing) {
          showPlaceholder("Search for a face by filename in the left rail’s Search section to begin.");
          return;
        }
      } else {
        showPlaceholder("Folder / Zip ingest isn’t built yet.");
        return;
      }

      mainPlaceholderEl.style.display = "none";
      stageWrapEl.style.display = "";
      sidebarEl.style.display = "";
      this.renderStage(active);
    },

    renderStage(project) {
      if (project.task === "immich") this.renderImmichStage(project);
      else this.renderVideoStage(project);
    },

    renderVideoStage(project) {
      immichSectionEl.style.display = "none";
      const ring = project.ring;
      const anchorLabel = `Frame ${ring.refFrameIdx} (Anchor)`;
      const metricLabel = { sim: "Similarity", yaw: "Yaw", pitch: "Pitch", roll: "Roll", blur: "Sharpness" }[project.ringSortMetric];
      hudModeEl.textContent = "VIDEO FRAME ANALYSIS (local, not in Immich)";
      hudFilenameEl.textContent = `${anchorLabel} · sorted by ${metricLabel}`;

      // currently-selected panel always shows the anchor for this slice --
      // there's no recenter target since video-frame results carry no
      // assetId (matches the original's behavior for local-frame nodes).
      sidebarCurrentImgEl.src = ring.anchorUrl;
      sidebarCurrentFnameEl.textContent = anchorLabel;
      sidebarCurrentModeEl.textContent = "VIDEO FRAME ANALYSIS (local, not in Immich)";
      sidebarCurrentDetailEl.textContent = "match: 100.0%";

      const sorted = project.sortedRanked();
      const { kept, total } = project.squeezeFiltered(sorted);
      squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        stageEl.style.display = "none";
        poseListViewEl.style.display = "flex";
        poseListScrubberEl.style.display = "flex";
        renderPoseListNG(project, ring.anchorUrl, anchorLabel, kept, (r) => project.toggleFrameSelection(r.frame), (r) => project.selectedFrames.has(r.frame));
        this.renderRankedList(project, kept);
        return;
      }
      stageEl.style.display = "";
      poseListViewEl.style.display = "none";
      poseListScrubberEl.style.display = "none";

      stageEl.innerHTML = "";
      const ringScale = project.ringScale / 100;

      [0.9, 0.7, 0.5, 0.35].forEach((band) => {
        const r = radiusForSim(band, ringScale);
        const ringEl = document.createElement("div");
        ringEl.className = "ng-ring";
        ringEl.style.width = r * 2 + "px";
        ringEl.style.height = r * 2 + "px";
        stageEl.appendChild(ringEl);
      });

      const center = document.createElement("div");
      center.className = "ng-node ng-node-center";
      center.style.width = CENTER_SIZE + "px";
      center.style.height = CENTER_SIZE + "px";
      center.style.transform = "translate(-50%, -50%)";
      center.innerHTML = `<img src="${ring.anchorUrl}">`;
      center.addEventListener("mouseenter", () => showHoverPreview({ filename: "Reference (anchor)", thumbUrl: ring.anchorUrl, similarity: 1 }));
      center.addEventListener("mouseleave", hideHoverPreview);
      stageEl.appendChild(center);

      const bands = Array.from({ length: BAND_COUNT }, () => []);
      kept.forEach((r) => {
        const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
        const bandIdx = Math.min(BAND_COUNT - 1, Math.floor((1 - t) * BAND_COUNT));
        bands[bandIdx].push(r);
      });

      bands.forEach((bandResults, bandIdx) => {
        if (!bandResults.length) return;
        const t = 1 - bandIdx / (BAND_COUNT - 1);
        const avgSim = 0.25 + t * 0.75;
        const radius = radiusForSim(avgSim, ringScale);
        const size = sizeForSim(avgSim);
        const angleOffset = bandIdx * 0.6;

        bandResults.forEach((r, i) => {
          const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          const node = document.createElement("div");
          node.className = "ng-node" + (project.selectedFrames.has(r.frame) ? " ng-node-selected" : "");
          node.style.width = size + "px";
          node.style.height = size + "px";
          node.style.left = `calc(50% + ${x}px)`;
          node.style.top = `calc(50% + ${y}px)`;
          node.style.transform = "translate(-50%, -50%)";
          node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}%`;
          node.innerHTML = `<img src="${thumbUrlFor(r)}">`;

          node.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.toggleFrameSelection(r.frame);
            this.renderStage(project);
          };
          node.addEventListener("mouseenter", () => showHoverPreview(r));
          node.addEventListener("mouseleave", hideHoverPreview);
          stageEl.appendChild(node);
        });
      });

      this.renderRankedList(project, kept);
    },

    renderRankedList(project, kept) {
      listBodyFramesEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.frame = r.frame;
        const pct = (r.similarity * 100).toFixed(1);

        let poseHtml = "";
        if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
          let poseLine = `pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
          if (typeof r.blur === "number") poseLine += ` · sharp: ${r.blur.toFixed(0)}`;
          if (typeof r.vertFillPct === "number") poseLine += ` · ${(r.vertFillPct * 100).toFixed(0)}% frame ht`;
          poseHtml = `<div style="font-size:10px;color:var(--ng-text-dim);margin-top:2px;">${poseLine}</div>`;
        }

        row.innerHTML = `
          <input type="checkbox" class="ng-frame-select-cb" data-frame="${r.frame}" ${project.selectedFrames.has(r.frame) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${thumbUrlFor(r)}" loading="lazy">
          <div class="info">
            <div class="fname">${r.filename}</div>
            <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
            ${poseHtml}
          </div>
          <div class="simpct">${pct}%</div>
        `;
        const cb = row.querySelector(".ng-frame-select-cb");
        cb.addEventListener("change", () => {
          project.toggleFrameSelection(r.frame);
          this.renderStage(project);
        });
        row.addEventListener("mouseenter", () => showHoverPreview(r));
        row.addEventListener("mouseleave", hideHoverPreview);
        listBodyFramesEl.appendChild(row);
      });

      framesSectionEl.style.display = kept.length ? "" : "none";
      framesSectionCountEl.textContent = kept.length ? `(${kept.length})` : "";
    },

    // ---- Immich stage: same ring-layout approach as the video stage, but
    // centered on an Immich asset instead of a video frame, with a
    // click-to-recenter interaction the video ring has no equivalent for
    // (video frames aren't independently-addressable "assets"). Kept as
    // its own method (some duplication vs renderVideoStage) rather than a
    // shared abstraction -- matches this codebase's existing NG-duplicate
    // philosophy and is safer than reshaping the working video path. ----
    renderImmichStage(project) {
      framesSectionEl.style.display = "none";
      const ring = project.immichRing;
      const anchorUrl = `/api/ng/preview/${ring.centerAssetId}`;
      const modeLabel = ring.mode === "clip" ? "CLIP image embedding (no face match found)" : "face embedding";
      const metricLabel = { sim: "Similarity", yaw: "Yaw", pitch: "Pitch", roll: "Roll", blur: "Sharpness" }[project.ringSortMetric];

      hudModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      hudFilenameEl.textContent = `${ring.centerFilename} · sorted by ${metricLabel}`;

      sidebarCurrentImgEl.src = anchorUrl;
      sidebarCurrentFnameEl.textContent = ring.centerFilename;
      sidebarCurrentModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      if (ring.centerPose) {
        const p = ring.centerPose;
        sidebarCurrentDetailEl.textContent =
          `pitch: ${p.pitch.toFixed(1)} yaw: ${p.yaw.toFixed(1)} roll: ${p.roll.toFixed(1)}` +
          (typeof p.blur === "number" ? ` · sharpness: ${p.blur.toFixed(0)}` : "");
      } else {
        sidebarCurrentDetailEl.textContent = "match: 100.0%";
      }

      const sorted = project.sortedRankedImmich();
      const { kept, total } = project.squeezeFiltered(sorted);
      squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        stageEl.style.display = "none";
        poseListViewEl.style.display = "flex";
        poseListScrubberEl.style.display = "flex";
        renderPoseListNG(project, anchorUrl, ring.centerFilename, kept, (r) => project.toggleAssetSelection(r.assetId), (r) => project.selectedAssetIds.has(r.assetId), (r) => project.recenterImmich(r.assetId, r.filename));
        this.renderRankedListImmich(project, kept);
        return;
      }
      stageEl.style.display = "";
      poseListViewEl.style.display = "none";
      poseListScrubberEl.style.display = "none";

      stageEl.innerHTML = "";
      const ringScale = project.ringScale / 100;

      [0.9, 0.7, 0.5, 0.35].forEach((band) => {
        const r = radiusForSim(band, ringScale);
        const ringEl = document.createElement("div");
        ringEl.className = "ng-ring";
        ringEl.style.width = r * 2 + "px";
        ringEl.style.height = r * 2 + "px";
        stageEl.appendChild(ringEl);
      });

      const center = document.createElement("div");
      center.className = "ng-node ng-node-center";
      center.style.width = CENTER_SIZE + "px";
      center.style.height = CENTER_SIZE + "px";
      center.style.transform = "translate(-50%, -50%)";
      center.innerHTML = `<img src="${anchorUrl}">`;
      center.addEventListener("mouseenter", () => showHoverPreview({ filename: ring.centerFilename + " (centered)", thumbUrl: anchorUrl, similarity: 1 }));
      center.addEventListener("mouseleave", hideHoverPreview);
      stageEl.appendChild(center);

      const bands = Array.from({ length: BAND_COUNT }, () => []);
      kept.forEach((r) => {
        const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
        const bandIdx = Math.min(BAND_COUNT - 1, Math.floor((1 - t) * BAND_COUNT));
        bands[bandIdx].push(r);
      });

      bands.forEach((bandResults, bandIdx) => {
        if (!bandResults.length) return;
        const t = 1 - bandIdx / (BAND_COUNT - 1);
        const avgSim = 0.25 + t * 0.75;
        const radius = radiusForSim(avgSim, ringScale);
        const size = sizeForSim(avgSim);
        const angleOffset = bandIdx * 0.6;

        bandResults.forEach((r, i) => {
          const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          const node = document.createElement("div");
          node.className = "ng-node" + (project.selectedAssetIds.has(r.assetId) ? " ng-node-selected" : "");
          node.style.width = size + "px";
          node.style.height = size + "px";
          node.style.left = `calc(50% + ${x}px)`;
          node.style.top = `calc(50% + ${y}px)`;
          node.style.transform = "translate(-50%, -50%)";
          node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}% (click to recenter, dblclick to select)`;
          node.innerHTML = `<img src="${thumbUrlFor(r)}">`;

          node.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.recenterImmich(r.assetId, r.filename);
          });
          node.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.toggleAssetSelection(r.assetId);
            this.renderStage(project);
          };
          node.addEventListener("mouseenter", () => showHoverPreview(r));
          node.addEventListener("mouseleave", hideHoverPreview);
          stageEl.appendChild(node);
        });
      });

      this.renderRankedListImmich(project, kept);
    },

    renderRankedListImmich(project, kept) {
      listBodyImmichEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.assetId = r.assetId;
        const pct = (r.similarity * 100).toFixed(1);

        row.innerHTML = `
          <input type="checkbox" class="ng-asset-select-cb" data-asset="${r.assetId}" ${project.selectedAssetIds.has(r.assetId) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${thumbUrlFor(r)}" loading="lazy">
          <div class="info">
            <div class="fname">${r.filename}</div>
            <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="simpct">${pct}%</div>
        `;
        const cb = row.querySelector(".ng-asset-select-cb");
        cb.addEventListener("change", (e) => {
          e.stopPropagation();
          project.toggleAssetSelection(r.assetId);
          this.renderStage(project);
        });
        row.querySelector("img").addEventListener("click", () => project.recenterImmich(r.assetId, r.filename));
        row.addEventListener("mouseenter", () => showHoverPreview(r));
        row.addEventListener("mouseleave", hideHoverPreview);
        listBodyImmichEl.appendChild(row);
      });

      immichSectionEl.style.display = kept.length ? "" : "none";
      immichSectionCountEl.textContent = kept.length ? `(${kept.length})` : "";
    },

    // ---- Left rail: ported sidebar, shown whenever a project is active ----
    renderLeftRail() {
      const active = this.getActive();

      if (!active) {
        leftRailEmptyEl.style.display = "";
        leftRailBodyEl.style.display = "none";
        return;
      }
      leftRailEmptyEl.style.display = "none";
      leftRailBodyEl.style.display = "flex";

      // The three ingest sources are separate switchable "pages" within
      // the tab -- only the section for the active task is shown.
      videoAnalysisSectionEl.style.display = active.task === "video" ? "" : "none";
      searchSectionEl.style.display = active.task === "immich" ? "" : "none";

      ringScaleInput.value = active.ringScale;
      ringScaleVal.textContent = active.ringScale + "%";
      squeezeSlider.value = active.squeezeMinPct;
      ringSortCbs.forEach((cb) => { cb.checked = cb.dataset.metric === active.ringSortMetric; });
      sharpEnableCb.checked = active.sharpCutoffEnabled;
      sharpControlsEl.style.display = active.sharpCutoffEnabled ? "flex" : "none";
      sharpSlider.value = active.sharpMinVal;
      folderRefIndexInput.value = active.folderRefIndex;
      if (active.task === "immich" && active.immichRing) {
        const { kept, total } = active.squeezeFiltered(active.sortedRankedImmich());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
        sharpVal.textContent = `${active.sharpMinVal} (${kept.length}/${total})`;
      } else if (active.ring) {
        const { kept, total } = active.squeezeFiltered(active.sortedRanked());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
        sharpVal.textContent = `${active.sharpMinVal} (${kept.length}/${total})`;
      } else {
        squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
        sharpVal.textContent = `${active.sharpMinVal} (0/0)`;
      }

      simThresholdInput.value = active.simThreshold;
      blurThresholdInput.value = active.blurThreshold;
      cacheFormatPngCb.checked = active.cacheFormatPng;
      rankedSortRadios.forEach((cb) => {
        const currentMetric = active.task === "immich" ? active.immichRankedSortMetric : active.rankedSortMetric;
        cb.checked = cb.value === currentMetric;
      });

      this.renderVideoAnalysisBody(active);
      this.renderFramePreview(active);
      this.renderAnalysisStatus(active);
      this.renderImmichSearch(active);
    },

    renderImmichSearch(project) {
      if (document.activeElement !== immichSearchInput) {
        immichSearchInput.value = project.immichSearchQuery;
      }
      immichSearchStatusEl.textContent = project.immichSearching ? "Searching…" : "";
      immichSearchResultsEl.innerHTML = "";
      if (!project.immichSearchResults.length) {
        immichSearchResultsEl.style.display = "none";
        return;
      }
      immichSearchResultsEl.style.display = "";
      project.immichSearchResults.forEach((m) => {
        const row = document.createElement("div");
        row.className = "ng-search-result-row";
        row.innerHTML = `<img src="/api/ng/thumb/${m.assetId}" loading="lazy"><span class="ng-search-result-fname">${m.filename}</span>`;
        row.addEventListener("click", () => project.loadImmichNeighbors(m.assetId, m.filename));
        immichSearchResultsEl.appendChild(row);
      });
    },

    renderVideoAnalysisBody(project) {
      videoAnalysisBodyEl.innerHTML = "";

      if (project.videoLoading) {
        videoAnalysisBodyEl.appendChild(placeholder("Loading video..."));
      } else if (project.video) {
        const info = document.createElement("p");
        info.className = "ng-video-hint";
        info.textContent =
          project.video.fps.toFixed(2) + " fps, " +
          project.video.totalFrames + " frames, " +
          project.video.duration.toFixed(1) + "s";
        videoAnalysisBodyEl.appendChild(info);
        videoAnalysisBodyEl.appendChild(videoPickerButton(project, "Replace Video"));
      } else {
        videoAnalysisBodyEl.appendChild(placeholder("No video loaded for this project yet."));
        videoAnalysisBodyEl.appendChild(videoPickerButton(project, "Choose Video..."));
      }

      analysisStartInput.value = project.video && project.video.rangeStartSec != null ? project.video.rangeStartSec : "";
      analysisEndInput.value = project.video && project.video.rangeEndSec != null ? project.video.rangeEndSec : "";

      startAnalysisBtn.disabled = !project.video || project.videoLoading || (project.job && project.job.status === "running");
      startAnalysisBtn.title = !project.video
        ? "Load a video and pick a frame first"
        : (project.job && project.job.status === "running" ? "Analysis already running for this tab" : "");
    },

    renderAnalysisStatus(project) {
      if (!project.job) {
        analysisStatusEl.innerHTML = "";
        return;
      }
      const j = project.job;
      if (j.status === "error") {
        analysisStatusEl.innerHTML = `Error: ${j.error || "unknown error"}`;
        return;
      }
      if (j.status === "running") {
        analysisStatusEl.innerHTML = `
          Processing… ${j.frameCount || 0} images seen<br>
          ${j.passed || 0} kept, ${j.failedSim || 0} low sim, ${j.failedBlur || 0} blurry
          <div class="ng-progress-track"><div class="ng-progress-fill" style="width:${Math.min(100, ((j.frameCount || 0) / 200) * 100)}%"></div></div>
        `;
        return;
      }

      const selCount = project.selectedFrames.size;
      const isVideo = j.sourceType === "video";

      analysisStatusEl.innerHTML = `
        Done — ${j.passed}/${j.frameCount} kept.
        <div id="ng-sim-sparkline-wrap" style="margin-top:8px;"></div>
        <button id="ng-btn-save-kept" class="ng-btn ng-btn-full ng-btn-accent">Save kept frames to disk</button>
        <button id="ng-btn-save-selected" class="ng-btn ng-btn-full ng-btn-accent" ${selCount ? "" : "disabled"}>Save ${selCount} selected frame${selCount === 1 ? "" : "s"} to disk</button>
        <button id="ng-btn-view-selected" class="ng-btn ng-btn-full">View selected</button>
        <div id="ng-export-result" class="ng-stub-note"></div>
      `;

      renderNGChart(project, analysisStatusEl.querySelector("#ng-sim-sparkline-wrap"));

      const exportResultEl = analysisStatusEl.querySelector("#ng-export-result");
      const saveKeptBtn = analysisStatusEl.querySelector("#ng-btn-save-kept");
      const saveSelectedBtn = analysisStatusEl.querySelector("#ng-btn-save-selected");
      const viewSelectedBtn = analysisStatusEl.querySelector("#ng-btn-view-selected");

      saveKeptBtn.onclick = async () => {
        saveKeptBtn.textContent = "Saving…";
        saveKeptBtn.disabled = true;
        try {
          const result = await project.exportFrames(false);
          exportResultEl.textContent = result.error
            ? `Error: ${result.error}`
            : `Saved ${result.exported} frames → ${result.path}`;
        } catch (e) {
          exportResultEl.textContent = `Error: ${e.message}`;
        }
        saveKeptBtn.textContent = "Save kept frames to disk";
        saveKeptBtn.disabled = false;
      };

      if (saveSelectedBtn) {
        saveSelectedBtn.onclick = async () => {
          saveSelectedBtn.textContent = "Saving…";
          saveSelectedBtn.disabled = true;
          try {
            const result = await project.exportFrames(true);
            exportResultEl.textContent = result.error
              ? `Error: ${result.error}`
              : `Saved ${result.exported} frames → ${result.path}`;
          } catch (e) {
            exportResultEl.textContent = `Error: ${e.message}`;
          }
          saveSelectedBtn.textContent = `Save ${project.selectedFrames.size} selected frame${project.selectedFrames.size === 1 ? "" : "s"} to disk`;
          saveSelectedBtn.disabled = project.selectedFrames.size === 0;
        };
      }

      viewSelectedBtn.onclick = () => ProjectManager.openSelectedModal(project);

      // "Pop out playback" only makes sense for a real video source --
      // folder/image-set jobs have no clip for build-playback to
      // reassemble (the backend requires job.videoBytes).
      if (isVideo) {
        const btn = document.createElement("button");
        btn.className = "ng-btn ng-btn-full ng-btn-accent";
        btn.style.marginTop = "6px";
        btn.disabled = project.playbackBuilding;
        btn.textContent = project.playbackBuilding
          ? "Building playback…"
          : project.playback
            ? "Reopen playback"
            : "Pop out playback (rejected frames blanked)";
        btn.addEventListener("click", () => {
          if (project.playback) PlaybackModal.openBuild(project);
          else project.buildPlayback();
        });
        analysisStatusEl.appendChild(btn);
      }
    },

    openSelectedModal(project) {
      const overlay = document.getElementById("ng-selected-modal");
      const grid = document.getElementById("ng-selected-modal-grid");
      const title = document.getElementById("ng-selected-modal-title");
      if (!overlay || !grid) return;

      const renderGrid = () => {
        const items = (project.ring ? project.ring.baseResults : []).filter((r) => project.selectedFrames.has(r.frame));
        title.textContent = `Selected frames (${items.length})`;
        grid.innerHTML = "";
        if (!items.length) {
          grid.appendChild(placeholder("Nothing selected yet — dblclick a ring node or its checkbox to select."));
          return;
        }
        items.forEach((r) => {
          const cell = document.createElement("div");
          cell.className = "ng-selected-modal-cell";
          cell.innerHTML = `
            <img src="${thumbUrlFor(r)}" loading="lazy">
            <div class="ng-selected-modal-cell-info">${r.filename}${typeof r.similarity === "number" ? ` — ${(r.similarity * 100).toFixed(1)}%` : ""}</div>
            <button class="ng-selected-modal-remove" title="Remove from selection">&times;</button>
          `;
          cell.querySelector("img").addEventListener("click", () => {
            if (project.job && project.job.sourceType === "video" && project.video) {
              project.stepAndSyncAudio(r.frame);
            } else {
              showStaticFramePreviewNG(project, r);
            }
          });
          cell.querySelector(".ng-selected-modal-remove").addEventListener("click", () => {
            project.toggleFrameSelection(r.frame);
            if (project.isActive) ProjectManager.render();
            renderGrid();
          });
          grid.appendChild(cell);
        });
      };

      renderGrid();
      overlay.style.display = "flex";
    },

    findNeutralPose(project) {
      const baseResults = project.task === "immich"
        ? (project.immichRing ? project.immichRing.baseResults : null)
        : (project.ring ? project.ring.baseResults : null);
      if (!baseResults) return;

      const { kept } = project.squeezeFiltered(baseResults);
      const pool = kept.filter((r) => typeof r.yaw === "number" && typeof r.pitch === "number" && typeof r.roll === "number");

      if (!pool.length) {
        neutralPoseReadoutEl.style.display = "block";
        neutralPoseReadoutEl.textContent = "No frames with pose data in current working set.";
        return;
      }

      let best = pool[0];
      let bestScore = Math.abs(best.yaw) + Math.abs(best.pitch) + Math.abs(best.roll);
      pool.forEach((r) => {
        const score = Math.abs(r.yaw) + Math.abs(r.pitch) + Math.abs(r.roll);
        if (score < bestScore) { best = r; bestScore = score; }
      });

      neutralPoseReadoutEl.style.display = "block";
      neutralPoseReadoutEl.innerHTML = `
        Most neutral: <b>${best.filename}</b> — yaw ${best.yaw.toFixed(1)}° pitch ${best.pitch.toFixed(1)}° roll ${best.roll.toFixed(1)}° (sim ${(best.similarity * 100).toFixed(1)}%)
        <button type="button" id="ng-use-as-reference-btn" class="ng-btn ng-btn-full ng-btn-accent" style="margin-top:6px;">Use as reference &amp; re-analyze</button>
      `;
      flashHighlightNG(best.frame, best.assetId);

      const useBtn = document.getElementById("ng-use-as-reference-btn");
      useBtn.onclick = () => {
        // this readout describes the *old* anchor's neutral-pose stats,
        // invalidated by the re-analysis it's about to trigger.
        const hideReadout = () => { neutralPoseReadoutEl.style.display = "none"; neutralPoseReadoutEl.innerHTML = ""; };
        const sourceType = project.job ? project.job.sourceType : null;

        if (project.task === "immich") {
          hideReadout();
          project.recenterImmich(best.assetId, best.filename);
        } else if (sourceType === "folder") {
          if (!project._lastFolderSource) {
            useBtn.textContent = "Original folder/zip no longer available — reload it first";
            return;
          }
          hideReadout();
          project.startFolderAnalysis(project._lastFolderSource, best.frame);
        } else {
          if (!project.videoFile) {
            useBtn.textContent = "Original video no longer available — reload it first";
            return;
          }
          hideReadout();
          project.startAnalysis(best.frame);
        }
      };
    },

    renderFramePreview(project) {
      if (!project.video) {
        if (project.staticPreviewFrame) {
          // folder/image-set job with a chart-click preview already shown --
          // redraw it rather than resetting to the "no video" placeholder,
          // since a full render() (selection toggles, tab switches, etc.)
          // must not silently wipe out what the person just clicked to view.
          showStaticFramePreviewNG(project, project.staticPreviewFrame);
          return;
        }
        previewHintEl.textContent = "No video loaded for this project yet.";
        previewCanvasEl.style.display = "none";
        previewControlsScrollEl.style.display = "none";
        return;
      }

      // Play button is a single shared element -- always resync its visual
      // state to whichever project is actually being rendered, rather than
      // trusting whatever the last togglePlay() call left it as.
      setPlayingVisual(!!project._playTimer);

      previewHintEl.textContent = "Reference frame — use ← / → to step one actual video frame";
      previewCanvasEl.style.display = "";
      previewControlsScrollEl.style.display = "";

      if (project.video.objectUrl && videoAudioEl.dataset.objectUrl !== project.video.objectUrl) {
        videoAudioEl.src = project.video.objectUrl;
        videoAudioEl.dataset.objectUrl = project.video.objectUrl;
        videoAudioEl.dataset.projectId = project.id;
        videoAudioEl.load();
      }

      // objectUrl doesn't survive a page reload (blob URLs die with the
      // page), so after a reload there's nothing left to pop out until
      // the video is re-picked -- same constraint the audio-synced
      // scrubber above already lives with.
      popoutVideoBtn.disabled = !project.video.objectUrl;
      popoutVideoBtn.title = project.video.objectUrl
        ? "Pop out the source video, with audio, before any analysis is run"
        : "Video needs to be re-loaded after a page reload before it can be popped out";

      frameCounterEl.textContent = "Frame: " + project.video.currentFrame + " / " + project.video.totalFrames;
      drawFrame(project, project.video.currentFrame);
    },
  };

  // ---- Playback modal: pops out either the untouched source video
  // (opened straight from the objectUrl, before any analysis has run) or
  // the rejected-frames-blanked reassembly built after analysis. Both use
  // a single real <video> element with its own native audio track, so
  // unlike the main frame-by-frame scrubber (canvas + a separate shared
  // <audio>), no manual audio sync is needed here in either mode.
  // Closes itself if that project stops being the active tab, so it can
  // never end up showing one project's clip while another tab is
  // selected. ----
  const PlaybackModal = {
    projectId: null,
    kind: null, // "raw" | "reconstructed"
    rangeStartSec: null,
    rangeEndSec: null,

    open(project) {
      if (!project.video || !project.video.objectUrl) return;
      // A raw pop-out and the main frame-scrubber would otherwise both be
      // playing the same clip's audio at once -- stop the scrubber first.
      project.stopPlayIfRunning();
      if (project.isActive) setPlayingVisual(false);
      this.projectId = project.id;
      this.kind = "raw";
      // Reuse the Analysis Settings start/end-sec range (if the user set
      // one) to bound the raw pop-out too -- previously this only ever
      // affected the /api/ng/analyze-video call, so there was no way to
      // preview e.g. just 19-25s without running a full analysis pass
      // first. loadedmetadata/timeupdate listeners below do the seeking.
      this.rangeStartSec = project.video.rangeStartSec != null ? project.video.rangeStartSec : null;
      this.rangeEndSec = project.video.rangeEndSec != null ? project.video.rangeEndSec : null;
      playbackModalTitleEl.textContent = this.rangeStartSec != null || this.rangeEndSec != null
        ? `${project.name} — source video (${this.rangeStartSec ?? 0}s–${this.rangeEndSec ?? "end"})`
        : `${project.name} — source video`;
      playbackVideoEl.src = project.video.objectUrl;
      playbackModalEl.style.display = "flex";
    },

    openBuild(project) {
      if (!project.playback) return;
      this.projectId = project.id;
      this.kind = "reconstructed";
      this.rangeStartSec = null;
      this.rangeEndSec = null;
      playbackModalTitleEl.textContent = `${project.name} — playback (rejected frames blanked)`;
      playbackVideoEl.src = project.playback.url;
      playbackModalEl.style.display = "flex";
    },

    close() {
      this.projectId = null;
      this.kind = null;
      this.rangeStartSec = null;
      this.rangeEndSec = null;
      playbackModalEl.style.display = "none";
      playbackVideoEl.pause();
      playbackVideoEl.removeAttribute("src");
      playbackVideoEl.load();
    },

    stepFrame(delta) {
      if (playbackModalEl.style.display === "none") return;
      const project = ProjectManager.projects.find((p) => p.id === this.projectId);
      const fps = this.kind === "raw"
        ? ((project && project.video && project.video.fps) || 24)
        : ((project && project.playback && project.playback.fps) || 24);
      playbackVideoEl.pause();
      const step = delta / fps;
      playbackVideoEl.currentTime = Math.max(0, Math.min(playbackVideoEl.duration || Infinity, playbackVideoEl.currentTime + step));
    },
  };

  // ---- match-confidence sparkline + shot-scale strip, ported from
  // media-ingest.js's renderSimSparkline -- reads from the project's own
  // job/results instead of module-level globals ----
  function renderNGChart(project, wrap) {
    const j = project.job;
    if (!wrap || !j || !j.results || !j.results.length) return;

    const results = j.results;
    const threshold = j.simThreshold;
    const blurThreshold = j.blurThreshold;
    const sourceType = j.sourceType;

    const sorted = [...results].sort((a, b) => a.frame - b.frame);

    const W = wrap.clientWidth || 320;
    const H = 90;
    const padL = 4, padR = 4, padT = 8, padB = 4;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const minFrame = sorted[0].frame;
    const maxFrame = sorted[sorted.length - 1].frame;
    const frameSpan = Math.max(1, maxFrame - minFrame);

    const xFor = (frame) => padL + ((frame - minFrame) / frameSpan) * plotW;
    const yFor = (sim) => padT + (1 - Math.max(0, Math.min(1, sim))) * plotH;

    const blurVals = sorted.map((r) => (typeof r.blur === "number" ? r.blur : 0));
    const blurMax = Math.max(1, blurThreshold * 1.4, ...blurVals) * 1.05;
    const yForBlur = (blur) => padT + (1 - Math.max(0, blur) / blurMax) * plotH;

    const linePoints = sorted.map((r) => `${xFor(r.frame).toFixed(1)},${yFor(r.sim).toFixed(1)}`).join(" ");
    const thresholdY = yFor(threshold).toFixed(1);
    const blurThresholdY = yForBlur(blurThreshold).toFixed(1);

    const dots = sorted.map((r) => {
      const cx = xFor(r.frame).toFixed(1);
      const cy = yFor(r.sim).toFixed(1);
      const color = r.passed ? "#7cc4ff" : "#d9534f";
      return `<circle cx="${cx}" cy="${cy}" r="7" fill="transparent" data-frame="${r.frame}" class="ng-spark-hit" style="cursor:pointer;"></circle>` +
             `<circle cx="${cx}" cy="${cy}" r="2" fill="${color}" style="pointer-events:none;"></circle>`;
    }).join("");

    const stripH = 14;
    const segW = Math.max(1, plotW / sorted.length);
    const stripSegs = sorted.map((r) => {
      const scale = shotScaleForNG(r);
      const x = xFor(r.frame).toFixed(1);
      const color = scale ? scale.color : "#2a2a32";
      const title = scale ? `${scale.label} (${(scale.pct * 100).toFixed(0)}%)` : "no scale data";
      return `<rect x="${(x - segW / 2).toFixed(1)}" y="0" width="${segW.toFixed(1)}" height="${stripH}" fill="${color}" data-frame="${r.frame}" class="ng-spark-hit ng-strip-seg" style="cursor:pointer;"><title>frame ${r.frame} — ${title}</title></rect>`;
    }).join("");

    const legend = SHOT_SCALE_BANDS.map((b) =>
      `<span class="ng-shot-scale-legend-item"><span class="ng-shot-scale-swatch" style="background:${b.color};"></span>${b.label}</span>`
    ).join("");

    wrap.innerHTML = `
      <div class="ng-chart-header">
        <span>Match confidence by frame — click a point to jump the preview</span>
        <span class="ng-blur-cutoff-label">·· blur cutoff (${blurThreshold})</span>
      </div>
      <svg width="${W}" height="${H}" class="ng-chart-svg">
        <line x1="${padL}" y1="${thresholdY}" x2="${W - padR}" y2="${thresholdY}"
              stroke="#4a4a55" stroke-width="1" stroke-dasharray="3,3"></line>
        <line x1="${padL}" y1="${blurThresholdY}" x2="${W - padR}" y2="${blurThresholdY}"
              stroke="#d4c04a" stroke-width="1" stroke-dasharray="1,3" opacity="0.8"></line>
        <polyline points="${linePoints}" fill="none" stroke="#5a8fc4" stroke-width="1.5"></polyline>
        ${dots}
      </svg>
      <div class="ng-shot-scale-caption">Shot scale by frame (face height % of frame)</div>
      <svg width="${W}" height="${stripH}" class="ng-strip-svg">${stripSegs}</svg>
      <div class="ng-shot-scale-legend">${legend}</div>
    `;

    wrap.querySelectorAll(".ng-spark-hit").forEach((el) => {
      el.addEventListener("click", () => {
        const frame = parseInt(el.dataset.frame, 10);
        const r = sorted.find((x) => x.frame === frame);
        if (sourceType === "video" && project.video) {
          project.stepAndSyncAudio(frame);
        } else if (r) {
          showStaticFramePreviewNG(project, r);
        }
      });
    });
  }

  // ---- static frame preview for folder/image-set jobs -- there's no
  // live decode to scrub (see MemoryVideo), so a chart click just draws
  // whatever's cached (or already-passed) for that frame straight onto
  // the shared preview canvas ----
  function showStaticFramePreviewNG(project, r) {
    if (!r) return;
    project.staticPreviewFrame = r;
    if (!project.isActive) return;
    previewHintEl.textContent = r.origName || r.filename || `frame ${r.frame}`;
    if (!r.frameId) {
      previewCanvasEl.style.display = "none";
      previewControlsScrollEl.style.display = "none";
      previewHintEl.textContent = `Frame ${r.frame} was rejected (no face / didn't pass thresholds) — no stored image to show.`;
      return;
    }
    const img = new Image();
    img.onload = () => {
      const active = ProjectManager.getActive();
      if (!active || active.id !== project.id) return;
      previewCanvasEl.width = img.naturalWidth;
      previewCanvasEl.height = img.naturalHeight;
      previewCanvasEl.getContext("2d").drawImage(img, 0, 0);
      previewCanvasEl.style.display = "";
      previewControlsScrollEl.style.display = "none";
    };
    img.onerror = () => {
      previewHintEl.textContent = `Could not load stored image for frame ${r.frame}`;
    };
    img.src = `/api/ng/framefile/${r.frameId}?t=${Date.now()}`;
  }

  // ---- pose-list view (yaw/pitch/roll/blur ring-sort modes), ported from
  // viz-render.js's renderPoseList/setupPoseListScrubber -- a horizontal
  // strip sorted by raw signed metric value instead of the radial ring ----
  function renderPoseListNG(project, anchorUrl, anchorLabel, combined, onToggle, isSelectedFn, onRecenter) {
    const metric = project.ringSortMetric;
    poseListViewEl.innerHTML = "";

    const anchorWrap = document.createElement("div");
    anchorWrap.className = "ng-pose-list-anchor";
    anchorWrap.innerHTML = `<img src="${anchorUrl}"><div class="ng-plabel">${anchorLabel}</div>`;
    poseListViewEl.appendChild(anchorWrap);

    const withMetric = combined.filter((r) => typeof r[metric] === "number");
    const withoutMetric = combined.filter((r) => typeof r[metric] !== "number");
    withMetric.sort((a, b) => a[metric] - b[metric]);

    const PITCH_PX_PER_DEG = 1.6;
    const PITCH_CLAMP_DEG = 40;

    withMetric.forEach((r) => {
      const item = document.createElement("div");
      const selected = isSelectedFn(r);
      item.className = "ng-pose-list-item" + (selected ? " ng-pose-list-item-selected" : "");
      if (r.assetId) item.dataset.assetId = r.assetId;
      if (r.frame !== undefined) item.dataset.frame = r.frame;
      const unit = metric === "blur" ? "" : "°";
      const label = metric === "blur" ? "sharp" : metric;
      item.innerHTML = `<img src="${thumbUrlFor(r)}" loading="lazy"><div class="ng-plabel">${label}: ${r[metric].toFixed(1)}${unit}</div>`;

      if (typeof r.pitch === "number") {
        const clamped = Math.max(-PITCH_CLAMP_DEG, Math.min(PITCH_CLAMP_DEG, r.pitch));
        const offsetPx = -clamped * PITCH_PX_PER_DEG;
        item.style.transform = `translateY(${offsetPx}px)`;
      }

      let clickTimer = null;
      item.onclick = () => {
        if (!onRecenter || !r.assetId) return; // no recenter target for local video/folder frames
        clearTimeout(clickTimer);
        clickTimer = setTimeout(() => onRecenter(r), 220);
      };
      item.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearTimeout(clickTimer);
        onToggle(r);
        item.classList.toggle("ng-pose-list-item-selected");
        ProjectManager.saveState();
        // keep the ranked sidebar in sync without a full stage re-render
        // (re-rendering here would rebuild the strip mid-scroll-drag)
        if (project.task === "immich") ProjectManager.renderRankedListImmich(project, combined);
        else ProjectManager.renderRankedList(project, combined);
      };
      item.addEventListener("mouseenter", () => showHoverPreview(r));
      item.addEventListener("mouseleave", hideHoverPreview);
      poseListViewEl.appendChild(item);
    });

    if (withoutMetric.length) {
      const note = document.createElement("div");
      note.className = "ng-pose-list-item ng-pose-list-note";
      note.innerHTML = `<div class="ng-plabel">+${withoutMetric.length} no ${metric} data</div>`;
      poseListViewEl.appendChild(note);
    }

    setupPoseListScrubberNG(metric, withMetric);
  }

  function setupPoseListScrubberNG(metric, withMetric) {
    if (withMetric.length) {
      poseScrubLeftEl.textContent = `${metric}: ${withMetric[0][metric].toFixed(1)}°`;
      poseScrubRightEl.textContent = `${metric}: ${withMetric[withMetric.length - 1][metric].toFixed(1)}°`;
    } else {
      poseScrubLeftEl.textContent = "";
      poseScrubRightEl.textContent = "";
    }

    const maxScroll = () => Math.max(1, poseListViewEl.scrollWidth - poseListViewEl.clientWidth);

    let syncingFromScroll = false;
    poseScrubSliderEl.value = 0;
    poseScrubSliderEl.oninput = () => {
      syncingFromScroll = true;
      poseListViewEl.scrollLeft = (parseFloat(poseScrubSliderEl.value) / 1000) * maxScroll();
      syncingFromScroll = false;
    };
    poseListViewEl.onscroll = () => {
      if (syncingFromScroll) return;
      poseScrubSliderEl.value = Math.round((poseListViewEl.scrollLeft / maxScroll()) * 1000);
    };
    poseListViewEl.onwheel = (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        poseListViewEl.scrollLeft += e.deltaY;
      }
    };
  }

  // ---- flash-highlight a ranked-list row or pose-list item, ported from
  // viz-render.js's flashHighlightFrame ----
  function flashHighlightNG(frame, assetId) {
    const selector = assetId != null
      ? `.ng-pose-list-item[data-asset-id="${CSS.escape(String(assetId))}"], .ng-list-row[data-asset-id="${CSS.escape(String(assetId))}"]`
      : `.ng-pose-list-item[data-frame="${CSS.escape(String(frame))}"], .ng-list-row[data-frame="${CSS.escape(String(frame))}"]`;
    const el = document.querySelector(selector);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    const prevShadow = el.style.boxShadow;
    let flashes = 0;
    const flashInterval = setInterval(() => {
      el.style.boxShadow = flashes % 2 === 0 ? "0 0 0 4px #7cc4ff" : prevShadow;
      flashes++;
      if (flashes > 5) { clearInterval(flashInterval); el.style.boxShadow = prevShadow; }
    }, 200);
  }

  function placeholder(text) {
    const p = document.createElement("p");
    p.className = "ng-placeholder";
    p.textContent = text;
    return p;
  }

  function videoPickerButton(project, label) {
    const btn = document.createElement("button");
    btn.className = "ng-btn";
    btn.textContent = label;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.style.display = "none";
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) project.loadVideo(input.files[0]);
    });

    btn.addEventListener("click", () => input.click());

    const wrap = document.createElement("div");
    wrap.appendChild(btn);
    wrap.appendChild(input);
    return wrap;
  }

  function drawFrame(project, frameNo) {
    const img = new Image();
    img.onload = () => {
      const active = ProjectManager.getActive();
      if (!active || active.id !== project.id) return;
      previewCanvasEl.width = img.naturalWidth;
      previewCanvasEl.height = img.naturalHeight;
      previewCanvasEl.getContext("2d").drawImage(img, 0, 0);
    };
    img.src = "/api/ng/preview-frame/" + project.video.previewId + "/" + frameNo + "?t=" + Date.now();
  }

  function setPlayingVisual(isPlaying) {
    playBtn.classList.toggle("ng-video-play-active", isPlaying);
  }

  // ---- wiring: Anchor controls ----
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
  });
  blurThresholdInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.blurThreshold = Number(blurThresholdInput.value);
    ProjectManager.saveState();
  });
  cacheFormatPngCb.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.cacheFormatPng = cacheFormatPngCb.checked;
    ProjectManager.saveState();
  });
  analysisStartInput.addEventListener("change", () => {
    const project = ProjectManager.getActive();
    if (!project || !project.video) return;
    project.video.rangeStartSec = analysisStartInput.value === "" ? null : Number(analysisStartInput.value);
    ProjectManager.saveState();
  });
  analysisEndInput.addEventListener("change", () => {
    const project = ProjectManager.getActive();
    if (!project || !project.video) return;
    project.video.rangeEndSec = analysisEndInput.value === "" ? null : Number(analysisEndInput.value);
    ProjectManager.saveState();
  });
  startAnalysisBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video && active.videoFile) active.startAnalysis();
  });
  folderRefIndexInput.addEventListener("change", () => {
    const active = ProjectManager.getActive();
    if (!active) return;
    active.folderRefIndex = Math.max(1, Number(folderRefIndexInput.value) || 1);
    ProjectManager.saveState();
  });

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

  // ---- wiring: frame-preview controls (static elements, wired once) ----
  rewindBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video) active.stepAndSyncAudio(1);
  });
  prevFrameBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video) active.stepAndSyncAudio(active.video.currentFrame - 1);
  });
  nextFrameBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video) active.stepAndSyncAudio(active.video.currentFrame + 1);
  });
  playBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video && !active._playTimer) active.togglePlay();
  });
  stopBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video && active._playTimer) active.togglePlay();
  });
  popoutVideoBtn.addEventListener("click", () => {
    const active = ProjectManager.getActive();
    if (active && active.video) PlaybackModal.open(active);
  });

  // ---- wiring: playback modal ----
  playbackModalCloseBtn.addEventListener("click", () => PlaybackModal.close());
  playbackModalEl.querySelector(".ng-playback-modal-backdrop").addEventListener("click", () => PlaybackModal.close());
  playbackPrevFrameBtn.addEventListener("click", () => PlaybackModal.stepFrame(-1));
  playbackNextFrameBtn.addEventListener("click", () => PlaybackModal.stepFrame(1));
  // Range-bounded raw preview: jump to rangeStartSec once the clip is
  // seekable, and stop (rather than rolling on to the rest of the video)
  // once rangeEndSec is reached.
  playbackVideoEl.addEventListener("loadedmetadata", () => {
    if (PlaybackModal.kind === "raw" && PlaybackModal.rangeStartSec != null) {
      playbackVideoEl.currentTime = PlaybackModal.rangeStartSec;
    }
  });
  playbackVideoEl.addEventListener("timeupdate", () => {
    if (PlaybackModal.kind === "raw" && PlaybackModal.rangeEndSec != null
        && playbackVideoEl.currentTime >= PlaybackModal.rangeEndSec) {
      playbackVideoEl.pause();
      playbackVideoEl.currentTime = PlaybackModal.rangeEndSec;
    }
  });

  document.addEventListener("keydown", (e) => {
    if (playbackModalEl.style.display !== "none") {
      if (e.key === "Escape") PlaybackModal.close();
      return;
    }
    const active = ProjectManager.getActive();
    if (!active || !active.video) return;
    if (document.activeElement && ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      active.stepAndSyncAudio(active.video.currentFrame - 1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      active.stepAndSyncAudio(active.video.currentFrame + 1);
    }
  });

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

    const savedControlsPct = parseFloat(localStorage.getItem("immichRingNG:leftControlsPct"));
    if (!Number.isNaN(savedControlsPct)) {
      controlsPaneEl.style.flexBasis = Math.max(24, Math.min(76, savedControlsPct)) + "%";
    }

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

    let splitDragging = false;
    splitterEl.addEventListener("mousedown", (e) => {
      if (leftRailEl.classList.contains("collapsed-all")) return;
      splitDragging = true;
      splitterEl.classList.add("dragging");
      document.body.style.cursor = "ns-resize";
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!splitDragging) return;
      const rect = leftRailBodyEl.getBoundingClientRect();
      const pct = ((e.clientY - rect.top) / rect.height) * 100;
      controlsPaneEl.style.flexBasis = Math.max(24, Math.min(76, pct)) + "%";
    });
    document.addEventListener("mouseup", () => {
      if (!splitDragging) return;
      splitDragging = false;
      splitterEl.classList.remove("dragging");
      document.body.style.cursor = "";
      const rect = leftRailBodyEl.getBoundingClientRect();
      const controlsRect = controlsPaneEl.getBoundingClientRect();
      localStorage.setItem("immichRingNG:leftControlsPct", Math.round((controlsRect.height / rect.height) * 100));
    });

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

  newProjectBtn.addEventListener("click", () => ProjectManager.createProject());
  taskButtons.forEach((btn) => {
    btn.addEventListener("click", () => ProjectManager.setTask(btn.dataset.task));
  });
  loadProjectBtn.disabled = true; // stays disabled until the persistence layer exists

  wireLeftRailChrome();
  ProjectManager.loadState();
  ProjectManager.render();

  // Resume polling for any project that was mid-analysis when a previous
  // render cycle set it up (not applicable right after loadState() since
  // fromPlain() marks an interrupted "running" job as an error -- this is
  // here for symmetry/clarity, not currently reachable).
})();
