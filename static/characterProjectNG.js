/**
 * characterProjectNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 474-1229 (as of this
 * extraction) -- the `class CharacterProject` definition in full.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 * A faithful copy-out for study, not a working module yet -- appNG.js is
 * untouched, still has its own copy of this class.
 *
 * ============================================================
 * WHERE THIS SITS IN THE BUILD ORDER
 * ============================================================
 * Following the same order Rachel used independently (base UI shell ->
 * character instance -> image processing/faceID):
 *
 *   1. Base UI shell    -- initguing.js (void state) + initchar.js (birth
 *                           sequence) -- already extracted, given by John.
 *   2. Character instance (THIS FILE) -- one CharacterProject = one
 *                           character tab (Tom, Mary, default1...), holding
 *                           its own video/job/ring/selection state.
 *   3. Image processing / faceID interface -- the analysis methods living
 *                           INSIDE this class (see below) -- submit a
 *                           source (video file / folder images / Immich
 *                           asset ids) to the backend's face-analysis
 *                           pipeline, poll for completion, receive back
 *                           per-frame {sim, pitch, yaw, roll, blur, bbox}.
 *
 * Layers 2 and 3 are NOT actually separated in the source -- the faceID
 * submit/poll/build-ring methods are methods of CharacterProject itself,
 * sharing `this` (this.job, this.ring, this.videoFile, ...) with all the
 * character-instance bookkeeping (video scrubbing, selection, sort state).
 * They're kept together in this extraction (matching the "as-is, don't
 * refine yet" instruction) but the contract boundary between them is
 * documented below so a future split is easy to reason about.
 *
 * ============================================================
 * WHAT A "CHARACTER INSTANCE" ACTUALLY HOLDS (constructor fields)
 * ============================================================
 * One CharacterProject == one tab == one in-memory character (Tom, Mary,
 * default1, default2...). Fields, grouped by concern:
 *
 *   - identity:            id, name, task
 *   - video state:         video, videoLoading, videoFile
 *   - analysis settings:   simThreshold, blurThreshold, cacheFormatPng
 *   - anchor/ring display: ringScale, squeezeMinPct, ringSortMetric,
 *                           sharpCutoffEnabled, sharpMinVal,
 *                           folderRefIndex, immichAnalyzeRefIndex
 *   - analysis job + ring: job, ring, playback, playbackBuilding,
 *                           rankedSortMetric, selectedFrames,
 *                           staticPreviewFrame
 *   - pose/scale picker:   posePickerPool, posePickerDisplayed,
 *                           _posePickerRingRef, scalePickerPool,
 *                           scalePickerDisplayed, _scalePickerRingRef
 *                           (see pickersNG.js -- this is the project-side
 *                           half of that module's coupling)
 *   - Immich ingest:       immichSearchQuery/Results/Searching/Loading/
 *                           Error, immichRing, immichRankedSortMetric,
 *                           selectedAssetIds, assetPoseCache
 *   - internal/transient:  _pollTimer, _playTimer, _lastFolderSource,
 *                           _lastImmichAssetIds (none of these survive
 *                           reload/persistence -- see toPlain())
 *
 * ============================================================
 * THE FACEID / IMAGE-PROCESSING CONTRACT (layer 3, embedded here)
 * ============================================================
 * Three submit methods, one shared poll loop, one shape-the-results method.
 * All three submit methods converge on the same job/poll/ring shape, just
 * with different sources:
 *
 *   startAnalysis(refFrameOverride)
 *     IN:  this.videoFile (already-loaded video File), this.simThreshold,
 *          this.blurThreshold, this.cacheFormatPng, optional video time
 *          range, optional refFrameOverride (which frame is the reference
 *          face -- defaults to current scrub position)
 *     DOES: POSTs multipart form to /api/ng/analyze-video, gets back a
 *           jobId, starts polling
 *
 *   startFolderAnalysis({images, zip}, refIndexOverride)
 *     IN:  a set of local image Files OR a single zip File, same
 *          threshold settings, refIndexOverride (which image is the
 *          reference face)
 *     DOES: POSTs multipart form to /api/ng/analyze-folder, same
 *           jobId + poll pattern
 *
 *   startImmichAnalysis(assetIds, refIndexOverride)  [see full body below
 *          for exact signature -- not reproduced in this doc block]
 *     IN:  a batch of Immich asset ids, same threshold settings
 *     DOES: POSTs to /api/ng/analyze-immich, same jobId + poll pattern
 *
 *   poll()  -- the shared tail for all three above
 *     Polls GET /api/ng/analysis-status/<jobId> every 800ms while
 *     status === "running". On the FIRST "done", calls buildRing(data.results)
 *     immediately (does not wait for another poll tick).
 *     RECEIVES per result row: { frame, origName, sim, pitch, yaw, roll,
 *     blur, bboxRatio, vertFillPct, bbox, frameId, passed, failReason }
 *     -- THIS is the actual faceID output contract: similarity, blur,
 *     pitch, yaw, roll, per analyzed frame, plus pass/fail against the
 *     configured thresholds.
 *
 *   buildRing(results)
 *     IN:  the raw results array from a completed poll
 *     OUT: sets this.ring = { anchorUrl, refFrameIdx, baseResults,
 *          sourceType }, where baseResults is the passed-only subset,
 *          re-shaped to { filename, frame, similarity, thumbUrl, pitch,
 *          yaw, roll, blur, bboxRatio, vertFillPct, bbox }, sorted by
 *          similarity descending.
 *     This is the actual boundary between "raw faceID output" and
 *     "ring-ready display data" -- everything downstream (RingVisualizer,
 *     pickersNG.js's setupPosePickerNG/setupScalePickerNG, the ranked
 *     sidebar list) reads from this.ring.baseResults, never from
 *     this.job.results directly.
 *
 * A genuinely separated "faceID interface" module would take (source,
 * settings) in and hand back a Promise/callback of the same per-frame
 * {sim, pitch, yaw, roll, blur, bbox} shape, with NO knowledge of
 * `this.job`/`this.ring`/ProjectManager -- CharacterProject would then
 * just be the thing that calls it and stores the result. Not done in
 * this pass; documented here so the seam is visible when it's time.
 *
 * ============================================================
 * COUPLING POINTS (everything this class reaches OUTSIDE itself)
 * ============================================================
 * None of these are passed in -- all are globals this file currently
 * assumes exist at call time:
 *
 *   - ProjectManager   -- .render(), .renderLeftRail(), .saveState(),
 *                          .nextId, .activeId, .uniquePlaceholderName()
 *                          Used constantly, all over the class -- every
 *                          state-mutating method calls back out to
 *                          ProjectManager to trigger a re-render or
 *                          persist state. This is the single biggest
 *                          reason CharacterProject can't be lifted out
 *                          without ProjectManager coming with it (or a
 *                          stub/interface standing in for it).
 *   - videoAudioEl     -- a single shared <audio> element across ALL
 *                          projects/tabs (not per-instance!) -- used for
 *                          play/pause/sync. Explains the "regression fix"
 *                          comment in stopPlayIfRunning() about tabs
 *                          fighting over shared playback state.
 *   - drawFrame(project, frameNo) -- canvas draw call, defined elsewhere
 *                          in appNG.js (see FrameScrubber in
 *                          appNG-module-contracts.md -- not yet extracted)
 *   - PlaybackModal    -- the video-popout player object (see
 *                          appNG-module-contracts.md's VideoPopout entry
 *                          -- not yet extracted)
 *   - applyResolutionSummaryNG(summary) -- called from poll() when a job
 *                          reports a resolutionSummary, only if
 *                          this.isActive
 *   - gatherExportParamsNG() -- called from exportFrames()/
 *                          exportSelectedImmichAssets() to merge in the
 *                          current Export Settings panel's values
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after ProjectManager, videoAudioEl, drawFrame, PlaybackModal,
 *      applyResolutionSummaryNG, and gatherExportParamsNG all exist.
 *   2. Remove the corresponding class definition from appNG.js.
 *   3. Decide whether/when to split out the faceID layer (see above) into
 *      its own module independent of ProjectManager.
 */

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
      this.immichAnalyzeRefIndex = 1; // 1-based, which asset (post alpha-sort by fetched filename) is the reference face for analyze-immich

      // analysis job + ring state
      this.job = null; // { jobId, status, error, sourceName, frameCount, passed, failedSim, failedBlur, results, sourceType, refFrame }
      this.ring = null; // { anchorUrl, refFrameIdx, baseResults, sourceType }
      this.playback = null; // { url, fps, jobId } -- rejected-frames-blanked reassembly, popped out in a modal
      this.playbackBuilding = false;
      this.rankedSortMetric = "sim";
      this.selectedFrames = new Set();
      this.staticPreviewFrame = null; // folder/image-set jobs only -- a chart-click result row, since there's no live decode to scrub (see renderFramePreview)

      // Pose Picker / Shot Scale Picker state -- ported from viz-render.js's
      // pose/scale picker pools. Per-project (not shared globals) so each
      // tab keeps its own dialed-in target pitch/yaw/scale and its own
      // sticky 3x3 grid slots when you switch tabs. _posePickerRingRef/
      // _scalePickerRingRef are identity markers (not persisted) used to
      // detect "this project just got a freshly-built ring" so the picker
      // resets sliders/pool exactly once per analysis run, not on every
      // render tick.
      this.posePickerPool = [];
      this.posePickerDisplayed = new Array(9).fill(null);
      this._posePickerRingRef = null;
      this.scalePickerPool = [];
      this.scalePickerDisplayed = new Array(9).fill(null);
      this._scalePickerRingRef = null;

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
      this.assetPoseCache = {}; // assetId -> {pitch,yaw,blur,vertFillPct} from on-demand "Detect pose" in the selected-modal's pose-scatter view; not persisted (cheap to re-fetch, matches original app's module-level cache)

      this._pollTimer = null;
      this._playTimer = null;
      this._lastFolderSource = null; // { images, zip } -- last folder/zip upload, kept for "Use as reference & re-analyze"; not persisted (Files don't survive JSON)
      this._lastImmichAssetIds = null; // last batch analyze-immich asset-id selection, kept for "Use as reference & re-analyze"; not persisted (re-tick if the tab was closed/reloaded)
    }

    get isActive() {
      return ProjectManager.activeId === this.id;
    }

    // stopPlayIfRunning() moved to videoNG.js (CharacterProject.prototype.stopPlayIfRunning)

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
        immichAnalyzeRefIndex: this.immichAnalyzeRefIndex,
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
      // previewId has the same problem as objectUrl: it only exists in the
      // server's in-memory _preview_jobs_ng dict (see routes/videoNG.py),
      // which is wiped on every server restart, not just every reload.
      // Leaving it in place caused a stale previewId to 404 against
      // /api/ng/preview-frame/ on a restored project after a restart.
      if (p.video) p.video.previewId = null;
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
      p.immichAnalyzeRefIndex = typeof data.immichAnalyzeRefIndex === "number" ? data.immichAnalyzeRefIndex : 1;
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
      // transient, rebuilt on first render from p.ring -- see the
      // _posePickerRingRef/_scalePickerRingRef comment in the constructor.
      p.posePickerPool = [];
      p.posePickerDisplayed = new Array(9).fill(null);
      p._posePickerRingRef = null;
      p.scalePickerPool = [];
      p.scalePickerDisplayed = new Array(9).fill(null);
      p._scalePickerRingRef = null;
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
      p._lastImmichAssetIds = null;
      return p;
    }

    // ---- video ingest + scrubbing: loadVideo, frameTime, frameFromTime,
    // syncAudioToFrame, stepTo, stepAndSyncAudio, togglePlay,
    // playFramesWithoutAudio -- all moved to videoNG.js (patched onto
    // CharacterProject.prototype there; see that file's header). ----

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

    // ---- analysis: batch-analyze a ticked selection of Immich assets ->
    // same pipeline as folder/zip, over assets fetched from Immich instead
    // of local uploads. Shares this.job/this.ring with video and
    // folder/zip (see the comment on those fields) -- running this
    // overwrites whichever analysis was last active, matching how
    // switching between the Video and Folder/Zip tasks already behaves.
    // Ported from routes/immich.py's analyze_immich via routes/immichNG.py. ----
    async startImmichAnalysis(assetIds, refIndexOverride) {
      if (!assetIds || !assetIds.length) return;
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null;
      this.staticPreviewFrame = null;
      const refIndex = refIndexOverride != null ? refIndexOverride : this.immichAnalyzeRefIndex;
      this._lastImmichAssetIds = assetIds.slice();
      this.job = {
        status: "running",
        sourceType: "immich",
        sourceName: `immich_selection_${assetIds.length}`,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0, passed: 0, failedSim: 0, failedBlur: 0,
        results: [],
        refFrame: refIndex,
      };
      ProjectManager.render();

      const body = {
        assetIds,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        refIndex,
        cacheFormat: this.cacheFormatPng ? "png" : "jpg",
      };

      try {
        const res = await fetch("/api/ng/analyze-immich", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          this.job = { status: "error", error: data.error || res.status, fetchErrors: data.fetchErrors };
          if (this.isActive) ProjectManager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.job.frameCount = data.imageCount;
        this.job.fetchErrors = data.fetchErrors;
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
        this.job.resolutionSummary = data.resolutionSummary;

        if (this.isActive && this.job.resolutionSummary) applyResolutionSummaryNG(this.job.resolutionSummary);

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
          bboxRatio: r.bboxRatio, vertFillPct: r.vertFillPct, bbox: r.bbox,
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
      const body = Object.assign(
        gatherExportParamsNG(),
        onlySelected ? { frames: Array.from(this.selectedFrames) } : {},
      );
      const res = await fetch(`/api/ng/export-job/${this.job.jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json();
    }

    // ---- Save selected Immich assets to disk (no analysis job involved --
    // exports straight from the Immich originals using the Export Settings
    // panel's crop/resize params). Ported from routes/export.py's
    // /api/export-immich-assets via routes/exportNG.py. ----
    async exportSelectedImmichAssets() {
      const assetIds = Array.from(this.selectedAssetIds);
      if (!assetIds.length) return null;
      const body = Object.assign(gatherExportParamsNG(), { assetIds });
      const res = await fetch(`/api/ng/export-immich-assets`, {
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

    async loadRandomImmichFace() {
      // Explicit button per John's call -- the original app used
      // random-face as a silent init() fallback when no ?assetId= was
      // in the URL; NG has no such default-landing concept, so this is
      // click-to-reroll only.
      this.immichError = null;
      this.immichSearching = true;
      if (this.isActive) ProjectManager.renderLeftRail();
      try {
        const res = await fetch(`/api/ng/random-face`);
        const data = await res.json();
        if (!res.ok || data.error) {
          this.immichError = data.error || res.status;
          return;
        }
        await this.loadImmichNeighbors(data.assetId, data.filename);
      } catch (e) {
        this.immichError = e.message;
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

// ---- exposed for appNG.js to construct instances of ----
window.CharacterProjectNG = CharacterProject;

