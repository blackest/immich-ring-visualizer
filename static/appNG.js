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

  // ---- DOM refs: Video / Image-Set Analysis section ----
  const videoAnalysisBodyEl = document.getElementById("ng-video-analysis-body");
  const analysisStartInput = document.getElementById("ng-analysis-start-sec");
  const analysisEndInput = document.getElementById("ng-analysis-end-sec");
  const simThresholdInput = document.getElementById("ng-sim-threshold");
  const blurThresholdInput = document.getElementById("ng-blur-threshold");
  const cacheFormatPngCb = document.getElementById("ng-cache-format-png");
  const analysisStatusEl = document.getElementById("ng-analysis-status");

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

      // analysis job + ring state
      this.job = null; // { jobId, status, error, sourceName, frameCount, passed, failedSim, failedBlur }
      this.ring = null; // { anchorUrl, refFrameIdx, baseResults, sourceType }
      this.playback = null; // { url, fps, jobId } -- rejected-frames-blanked reassembly, popped out in a modal
      this.playbackBuilding = false;
      this.rankedSortMetric = "sim";
      this.selectedFrames = new Set();

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
      p.job = data.job || null;
      // playback builds live in a server tempdir (FRAME_STORE) that doesn't
      // survive a server restart, and the modal itself is a transient UI
      // concern -- neither is worth persisting across a page reload.
      p.playback = null;
      p.playbackBuilding = false;
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
    async startAnalysis() {
      if (!this.video || !this.videoFile) return;
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null; // a new analysis run invalidates any previous playback build
      this.job = {
        status: "running",
        sourceName: this.videoFile.name,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0, passed: 0, failedSim: 0, failedBlur: 0,
      };
      ProjectManager.render();

      const form = new FormData();
      form.append("video", this.videoFile);
      form.append("simThreshold", this.simThreshold);
      form.append("blurThreshold", this.blurThreshold);
      form.append("refFrame", this.video.currentFrame);
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
        refFrameIdx: this.video ? this.video.currentFrame : 1,
        baseResults,
        sourceType: "video",
      };
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
        if (this.isActive) PlaybackModal.open(this);
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

    // min-sim squeeze (ported from viz-render.js's applySqueeze -- similarity
    // filter only for this slice; the sharpness-cutoff half isn't ported yet)
    squeezeFiltered(sorted) {
      const cutoff = this.squeezeMinPct / 100;
      const kept = sorted.filter((r) => (typeof r.similarity === "number" ? r.similarity : 1) >= cutoff);
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
      hudModeEl.textContent = "VIDEO FRAME ANALYSIS (local, not in Immich)";
      hudFilenameEl.textContent = anchorLabel;

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

      hudModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      hudFilenameEl.textContent = ring.centerFilename;

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
      if (active.task === "immich" && active.immichRing) {
        const { kept, total } = active.squeezeFiltered(active.sortedRankedImmich());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
      } else if (active.ring) {
        const { kept, total } = active.squeezeFiltered(active.sortedRanked());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
      } else {
        squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
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
      analysisStatusEl.innerHTML = `Done — ${j.passed}/${j.frameCount} kept.`;
      const btn = document.createElement("button");
      btn.className = "ng-btn";
      btn.style.marginTop = "6px";
      btn.style.width = "100%";
      btn.disabled = project.playbackBuilding;
      btn.textContent = project.playbackBuilding
        ? "Building playback…"
        : project.playback
          ? "Reopen playback"
          : "Pop out playback (rejected frames blanked)";
      btn.addEventListener("click", () => {
        if (project.playback) PlaybackModal.open(project);
        else project.buildPlayback();
      });
      analysisStatusEl.appendChild(btn);
    },

    renderFramePreview(project) {
      if (!project.video) {
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

      frameCounterEl.textContent = "Frame: " + project.video.currentFrame + " / " + project.video.totalFrames;
      drawFrame(project, project.video.currentFrame);
    },
  };

  // ---- Playback modal: shows the rejected-frames-blanked reassembly for
  // whichever project's playback was most recently built. Closes itself
  // if that project stops being the active tab, so it can never end up
  // showing one project's clip while another tab is selected. ----
  const PlaybackModal = {
    projectId: null,

    open(project) {
      if (!project.playback) return;
      this.projectId = project.id;
      playbackModalTitleEl.textContent = `${project.name} — playback (rejected frames blanked)`;
      playbackVideoEl.src = project.playback.url;
      playbackModalEl.style.display = "flex";
    },

    close() {
      this.projectId = null;
      playbackModalEl.style.display = "none";
      playbackVideoEl.pause();
      playbackVideoEl.removeAttribute("src");
      playbackVideoEl.load();
    },

    stepFrame(delta) {
      if (playbackModalEl.style.display === "none") return;
      const project = ProjectManager.projects.find((p) => p.id === this.projectId);
      const fps = (project && project.playback && project.playback.fps) || 24;
      playbackVideoEl.pause();
      const step = delta / fps;
      playbackVideoEl.currentTime = Math.max(0, Math.min(playbackVideoEl.duration || Infinity, playbackVideoEl.currentTime + step));
    },
  };

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
    if (active.ring || active.immichRing) ProjectManager.renderStage(active);
    else squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
    ProjectManager.saveState();
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

  // ---- wiring: playback modal ----
  playbackModalCloseBtn.addEventListener("click", () => PlaybackModal.close());
  playbackModalEl.querySelector(".ng-playback-modal-backdrop").addEventListener("click", () => PlaybackModal.close());
  playbackPrevFrameBtn.addEventListener("click", () => PlaybackModal.stepFrame(-1));
  playbackNextFrameBtn.addEventListener("click", () => PlaybackModal.stepFrame(1));

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
