// projectNG.js
// CharacterProject and per-character state management.

(function () {
  "use strict";

  window.AppNG = window.AppNG || {};

  class CharacterProject {
    constructor(id, name) {
      this.id = id;
      this.name = name;

      this.task = null;
      this.video = null;
      this.videoFile = null;
      this.videoLoading = false;
      this.simThreshold = 0.1;
      this.blurThreshold = 1;
      this.cacheFormatPng = false;
      this.ringScale = 100;
      this.squeezeMinPct = 65;
      this.squeezeUserOverridden = false;
      this.ringSortMetric = "sim";
      this.sharpCutoffEnabled = false;
      this.sharpMinVal = 0;
      this.folderRefIndex = 1;
      this.immichAnalyzeRefIndex = 1;

      this.job = null;
      this.ring = null;
      this.playback = null;
      this.playbackBuilding = false;
      this.rankedSortMetric = "sim";
      this.selectedFrames = new Set();
      this.staticPreviewFrame = null;

      this.posePickerPool = [];
      this.posePickerDisplayed = new Array(9).fill(null);
      this._posePickerRingRef = null;
      this.scalePickerPool = [];
      this.scalePickerDisplayed = new Array(9).fill(null);
      this._scalePickerRingRef = null;

      this.immichSearchQuery = "";
      this.immichSearchResults = [];
      this.immichSearching = false;
      this.immichLoading = false;
      this.immichError = null;
      this.immichRing = null;
      this.immichRankedSortMetric = "sim";
      this.selectedAssetIds = new Set();
      this.assetPoseCache = {};

      this._pollTimer = null;
      this._playTimer = null;
      this._lastFolderSource = null;
      this._lastImmichAssetIds = null;
    }

    get isActive() {
      return (
        window.AppNG.manager && window.AppNG.manager.activeProjectId === this.id
      );
    }

    stopPlayIfRunning() {
      if (!this._playTimer) return;
      clearInterval(this._playTimer);
      this._playTimer = null;
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
      if (this.video && this.video.objectUrl) {
        URL.revokeObjectURL(this.video.objectUrl);
      }
    }

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
      const project = Object.create(CharacterProject.prototype);
      project.id = data.id;
      project.name = data.name;
      project.task = data.task || null;
      project.video = data.video || null;
      if (project.video) project.video.objectUrl = null;
      project.videoLoading = false;
      project.videoFile = null;
      project.simThreshold =
        typeof data.simThreshold === "number" ? data.simThreshold : 0.1;
      project.blurThreshold =
        typeof data.blurThreshold === "number" ? data.blurThreshold : 1;
      project.cacheFormatPng = !!data.cacheFormatPng;
      project.ringScale =
        typeof data.ringScale === "number" ? data.ringScale : 100;
      project.squeezeMinPct =
        typeof data.squeezeMinPct === "number" ? data.squeezeMinPct : 65;
      project.squeezeUserOverridden = !!data.squeezeUserOverridden;
      project.ringSortMetric = data.ringSortMetric || "sim";
      project.sharpCutoffEnabled = !!data.sharpCutoffEnabled;
      project.sharpMinVal =
        typeof data.sharpMinVal === "number" ? data.sharpMinVal : 0;
      project.folderRefIndex =
        typeof data.folderRefIndex === "number" ? data.folderRefIndex : 1;
      project.immichAnalyzeRefIndex =
        typeof data.immichAnalyzeRefIndex === "number"
          ? data.immichAnalyzeRefIndex
          : 1;
      project.job = data.job || null;
      project.playback = null;
      project.playbackBuilding = false;
      project.staticPreviewFrame = null;
      project.ring = data.ring || null;
      project.rankedSortMetric = data.rankedSortMetric || "sim";
      project.selectedFrames = new Set(
        Array.isArray(data.selectedFrames) ? data.selectedFrames : [],
      );
      project.posePickerPool = [];
      project.posePickerDisplayed = new Array(9).fill(null);
      project._posePickerRingRef = null;
      project.scalePickerPool = [];
      project.scalePickerDisplayed = new Array(9).fill(null);
      project._scalePickerRingRef = null;
      project.immichSearchQuery = "";
      project.immichSearchResults = [];
      project.immichSearching = false;
      project.immichLoading = false;
      project.immichError = null;
      project.immichRing = data.immichRing || null;
      project.immichRankedSortMetric = data.immichRankedSortMetric || "sim";
      project.selectedAssetIds = new Set(
        Array.isArray(data.selectedAssetIds) ? data.selectedAssetIds : [],
      );
      project.assetPoseCache = {};
      project._pollTimer = null;
      project._playTimer = null;
      project._lastFolderSource = null;
      project._lastImmichAssetIds = null;
      return project;
    }

    async loadVideo(file) {
      this.videoLoading = true;
      try {
        const formData = new FormData();
        formData.append("video", file);
        const res = await fetch("/api/ng/preview-video", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Could not load video");
        }
        if (this.video && this.video.objectUrl)
          URL.revokeObjectURL(this.video.objectUrl);
        const objectUrl = URL.createObjectURL(file);
        this.video = {
          previewId: data.previewId,
          fps: data.fps,
          totalFrames: data.totalFrames,
          duration: data.duration,
          currentFrame: 1,
          objectUrl,
          rangeStartSec: this.video ? this.video.rangeStartSec : null,
          rangeEndSec: this.video ? this.video.rangeEndSec : null,
        };
        this.videoFile = file;
        this.job = null;
        this.ring = null;
        this.playback = null;
        this.selectedFrames = new Set();
        this.staticPreviewFrame = null;
      } finally {
        this.videoLoading = false;
      }
    }

    frameTime(frameIdx) {
      const fps = this.video && this.video.fps > 0 ? this.video.fps : 24;
      return Math.max(0, (frameIdx - 1) / fps);
    }

    frameFromTime(timeSeconds) {
      const fps = this.video && this.video.fps > 0 ? this.video.fps : 24;
      return Math.min(
        this.video ? this.video.totalFrames : 1,
        Math.max(1, Math.floor(timeSeconds * fps) + 1),
      );
    }

    syncAudioToFrame(frameIdx) {
      if (
        typeof videoAudioEl === "undefined" ||
        !this.video ||
        !videoAudioEl.src ||
        videoAudioEl.dataset.objectUrl !== this.video.objectUrl
      ) {
        return;
      }
      try {
        videoAudioEl.currentTime = this.frameTime(frameIdx);
      } catch (e) {
        // seeking before metadata is ready can throw; ignore it
      }
    }

    stepTo(frameNo) {
      if (!this.video) return;
      const clamped = Math.max(1, Math.min(this.video.totalFrames, frameNo));
      this.video.currentFrame = clamped;
      if (this.isActive && typeof frameCounterEl !== "undefined") {
        frameCounterEl.textContent =
          "Frame: " + clamped + " / " + this.video.totalFrames;
        if (
          window.AppNG.render &&
          typeof window.AppNG.render.drawFrame === "function"
        ) {
          window.AppNG.render.drawFrame(this, clamped);
        }
      }
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.saveState === "function"
      ) {
        window.AppNG.manager.saveState();
      }
    }

    stepAndSyncAudio(frameNo) {
      this.stopPlayIfRunning();
      if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
        window.AppNG.render.setPlayingVisual(false);
      }
      this.stepTo(frameNo);
      this.syncAudioToFrame(this.video.currentFrame);
    }

    togglePlay() {
      if (this._playTimer) {
        this.stopPlayIfRunning();
        if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
          window.AppNG.render.setPlayingVisual(false);
        }
        return;
      }

      const hasAudio =
        !!this.video &&
        !!this.video.objectUrl &&
        typeof videoAudioEl !== "undefined" &&
        videoAudioEl.dataset.objectUrl === this.video.objectUrl;
      if (!hasAudio) {
        this.playFramesWithoutAudio();
        return;
      }

      if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
        window.AppNG.render.setPlayingVisual(true);
      }
      if (this.video.currentFrame >= this.video.totalFrames) this.stepTo(1);
      this.syncAudioToFrame(this.video.currentFrame);

      const playPromise = videoAudioEl.play();
      if (playPromise && playPromise.catch) {
        playPromise.catch(() => this.playFramesWithoutAudio());
      }

      const sampleMs =
        (this.video.fps > 0 ? 1000 / this.video.fps : 1000 / 24) / 2;
      this._playTimer = setInterval(() => {
        if (videoAudioEl.paused || videoAudioEl.ended) {
          this.stopPlayIfRunning();
          if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
            window.AppNG.render.setPlayingVisual(false);
          }
          return;
        }
        const target = this.frameFromTime(videoAudioEl.currentTime);
        if (target !== this.video.currentFrame) this.stepTo(target);
        if (target >= this.video.totalFrames) {
          this.stopPlayIfRunning();
          videoAudioEl.pause();
          if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
            window.AppNG.render.setPlayingVisual(false);
          }
        }
      }, sampleMs);
    }

    playFramesWithoutAudio() {
      if (!this.video) return;
      if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
        window.AppNG.render.setPlayingVisual(true);
      }
      const intervalMs = this.video.fps > 0 ? 1000 / this.video.fps : 1000 / 24;
      this._playTimer = setInterval(() => {
        const next = this.video.currentFrame + 1;
        if (next > this.video.totalFrames) {
          this.stopPlayIfRunning();
          if (this.isActive && window.AppNG.render && typeof window.AppNG.render.setPlayingVisual === "function") {
            window.AppNG.render.setPlayingVisual(false);
          }
          return;
        }
        this.stepTo(next);
      }, intervalMs);
    }

    sortedRanked() {
      if (!this.ring) return [];
      const key =
        this.rankedSortMetric === "sim" ? "similarity" : this.rankedSortMetric;
      const withMetric = this.ring.baseResults.filter(
        (r) => typeof r[key] === "number",
      );
      const withoutMetric = this.ring.baseResults.filter(
        (r) => typeof r[key] !== "number",
      );
      withMetric.sort((a, b) => b[key] - a[key]);
      return [...withMetric, ...withoutMetric];
    }

    squeezeFiltered(sorted) {
      const cutoff = this.squeezeMinPct / 100;
      const simKept = sorted.filter(
        (r) => (typeof r.similarity === "number" ? r.similarity : 1) >= cutoff,
      );

      if (!this.sharpCutoffEnabled) {
        return { kept: simKept, total: sorted.length };
      }

      const kept = simKept.filter((r) =>
        typeof r.blur !== "number" ? true : r.blur >= this.sharpMinVal,
      );
      return { kept, total: sorted.length };
    }

    // MIGRATED from appNG.js: analysis lifecycle + ring build + export/playback logic.
    async startAnalysis(refFrameOverride) {
      if (!this.video || !this.videoFile) return;
      const refFrame =
        refFrameOverride != null ? refFrameOverride : this.video.currentFrame;
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null;
      this.staticPreviewFrame = null;
      this.job = {
        status: "running",
        sourceType: "video",
        sourceName: this.videoFile.name,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0,
        passed: 0,
        failedSim: 0,
        failedBlur: 0,
        results: [],
        refFrame,
      };
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.render === "function"
      ) {
        window.AppNG.manager.render();
      }

      const form = new FormData();
      form.append("video", this.videoFile);
      form.append("simThreshold", this.simThreshold);
      form.append("blurThreshold", this.blurThreshold);
      form.append("refFrame", refFrame);
      form.append("cacheFormat", this.cacheFormatPng ? "png" : "jpg");
      if (this.video.rangeStartSec != null)
        form.append("startSec", this.video.rangeStartSec);
      if (this.video.rangeEndSec != null)
        form.append("endSec", this.video.rangeEndSec);

      try {
        const res = await fetch("/api/ng/analyze-video", {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          this.job = { status: "error", error: data.error || res.status };
          if (this.isActive && window.AppNG.manager)
            window.AppNG.manager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.poll();
      } catch (e) {
        this.job = { status: "error", error: e.message };
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
      }
    }

    async startFolderAnalysis({ images, zip }, refIndexOverride) {
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null;
      this.staticPreviewFrame = null;
      const refIndex =
        refIndexOverride != null ? refIndexOverride : this.folderRefIndex;
      const sourceName = zip ? zip.name.replace(/\.zip$/i, "") : "folder_set";
      this.job = {
        status: "running",
        sourceType: "folder",
        sourceName,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0,
        passed: 0,
        failedSim: 0,
        failedBlur: 0,
        results: [],
        refFrame: refIndex,
      };
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.render === "function"
      ) {
        window.AppNG.manager.render();
      }

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
      this._lastFolderSource = { images, zip };

      try {
        const res = await fetch("/api/ng/analyze-folder", {
          method: "POST",
          body: form,
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          this.job = { status: "error", error: data.error || res.status };
          if (this.isActive && window.AppNG.manager)
            window.AppNG.manager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.job.frameCount = data.imageCount;
        this.poll();
      } catch (e) {
        this.job = { status: "error", error: e.message };
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
      }
    }

    async startImmichAnalysis(assetIds, refIndexOverride) {
      if (!assetIds || !assetIds.length) return;
      this.stopPolling();
      this.selectedFrames = new Set();
      this.playback = null;
      this.staticPreviewFrame = null;
      const refIndex =
        refIndexOverride != null
          ? refIndexOverride
          : this.immichAnalyzeRefIndex;
      this._lastImmichAssetIds = assetIds.slice();
      this.job = {
        status: "running",
        sourceType: "immich",
        sourceName: `immich_selection_${assetIds.length}`,
        simThreshold: this.simThreshold,
        blurThreshold: this.blurThreshold,
        frameCount: 0,
        passed: 0,
        failedSim: 0,
        failedBlur: 0,
        results: [],
        refFrame: refIndex,
      };
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.render === "function"
      ) {
        window.AppNG.manager.render();
      }

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
          this.job = {
            status: "error",
            error: data.error || res.status,
            fetchErrors: data.fetchErrors,
          };
          if (this.isActive && window.AppNG.manager)
            window.AppNG.manager.render();
          return;
        }
        this.job.jobId = data.jobId;
        this.job.frameCount = data.imageCount;
        this.job.fetchErrors = data.fetchErrors;
        this.poll();
      } catch (e) {
        this.job = { status: "error", error: e.message };
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
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
          if (this.isActive && window.AppNG.manager)
            window.AppNG.manager.render();
          return;
        }

        this.job.status = data.status;
        this.job.error = data.error;
        this.job.frameCount = data.frameCount;
        this.job.results = data.results;
        this.job.passed = data.results.filter((r) => r.passed).length;
        this.job.failedSim = data.results.filter(
          (r) => !r.passed && r.failReason === "sim",
        ).length;
        this.job.failedBlur = data.results.filter(
          (r) => !r.passed && r.failReason === "blur",
        ).length;
        this.job.resolutionSummary = data.resolutionSummary;

        if (
          this.isActive &&
          this.job.resolutionSummary &&
          window.AppNG.export
        ) {
          window.AppNG.export.applyResolutionSummaryNG(
            this.job.resolutionSummary,
          );
        }

        if (data.status === "running") {
          this._pollTimer = setTimeout(() => this.poll(), 800);
          return;
        }

        if (data.status === "done") {
          this.buildRing(data.results);
        }
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
      } catch (e) {
        this.job.status = "error";
        this.job.error = e.message;
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
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
          pitch: r.pitch,
          yaw: r.yaw,
          roll: r.roll,
          blur: r.blur,
          bboxRatio: r.bboxRatio,
          vertFillPct: r.vertFillPct,
          bbox: r.bbox,
        }))
        .sort((a, b) => b.similarity - a.similarity);

      this.ring = {
        anchorUrl: `/api/ng/framefile/${this.job.jobId}_anchor`,
        refFrameIdx: this.job.refFrame,
        baseResults,
        sourceType,
      };
    }

    async exportFrames(onlySelected) {
      if (!this.job || !this.job.jobId) return null;
      const body = Object.assign(
        window.AppNG.export &&
          typeof window.AppNG.export.gatherExportParamsNG === "function"
          ? window.AppNG.export.gatherExportParamsNG()
          : {},
        onlySelected ? { frames: Array.from(this.selectedFrames) } : {},
      );
      const res = await fetch(`/api/ng/export-job/${this.job.jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json();
    }

    async exportSelectedImmichAssets() {
      const assetIds = Array.from(this.selectedAssetIds);
      if (!assetIds.length) return null;
      const body = Object.assign(
        window.AppNG.export &&
          typeof window.AppNG.export.gatherExportParamsNG === "function"
          ? window.AppNG.export.gatherExportParamsNG()
          : {},
        { assetIds },
      );
      const res = await fetch(`/api/ng/export-immich-assets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.json();
    }

    async buildPlayback() {
      if (!this.job || this.job.status !== "done" || this.playbackBuilding)
        return;
      this.playbackBuilding = true;
      if (this.isActive && window.AppNG.manager) window.AppNG.manager.render();

      try {
        const res = await fetch(`/api/ng/build-playback/${this.job.jobId}`, {
          method: "POST",
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          alert("Could not build playback: " + (data.error || res.status));
          return;
        }
        this.playback = {
          url: data.url,
          fps: data.fps || 24,
          jobId: this.job.jobId,
        };
      } catch (e) {
        alert("Could not build playback: " + e.message);
      } finally {
        this.playbackBuilding = false;
        if (this.isActive && window.AppNG.manager)
          window.AppNG.manager.render();
      }
    }

    sortedRanked() {
      if (!this.ring) return [];
      const key =
        this.rankedSortMetric === "sim" ? "similarity" : this.rankedSortMetric;
      const withMetric = this.ring.baseResults.filter(
        (r) => typeof r[key] === "number",
      );
      const withoutMetric = this.ring.baseResults.filter(
        (r) => typeof r[key] !== "number",
      );
      withMetric.sort((a, b) => b[key] - a[key]);
      return [...withMetric, ...withoutMetric];
    }

    toggleFrameSelection(frame) {
      if (this.selectedFrames.has(frame)) {
        this.selectedFrames.delete(frame);
      } else {
        this.selectedFrames.add(frame);
      }
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.saveState === "function"
      ) {
        window.AppNG.manager.saveState();
      }
    }

    async searchImmich(query) {
      this.immichSearchQuery = query;
      const trimmed = query.trim();
      if (!trimmed) {
        this.immichSearchResults = [];
        return;
      }

      this.immichSearching = true;
      try {
        const res = await fetch(
          `/api/ng/find-by-filename?name=${encodeURIComponent(trimmed)}`,
        );
        const data = await res.json();
        this.immichSearchResults = Array.isArray(data) ? data : [];
      } catch (e) {
        this.immichSearchResults = [];
      } finally {
        this.immichSearching = false;
      }
    }

    async loadRandomImmichFace() {
      this.immichError = null;
      this.immichSearching = true;
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
      }
    }

    async loadImmichNeighbors(assetId, filename) {
      this.immichLoading = true;
      this.immichError = null;

      try {
        const res = await fetch(
          `/api/ng/neighbors?assetId=${encodeURIComponent(assetId)}&limit=36`,
        );
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
        this.immichSearchResults = [];
      } catch (e) {
        this.immichError = e.message;
      } finally {
        this.immichLoading = false;
      }

      if (this.immichRing && this.immichRing.centerAssetId === assetId) {
        this.loadImmichCenterPose(assetId);
      }
    }

    async loadImmichCenterPose(assetId) {
      try {
        const res = await fetch(`/api/ng/asset-face-pose/${assetId}`);
        const data = await res.json();
        if (!res.ok || data.error) return;
        if (!this.immichRing || this.immichRing.centerAssetId !== assetId)
          return;
        this.immichRing.centerPose = data;
      } catch (e) {
        // pose is best-effort only
      }
    }

    recenterImmich(assetId, filename) {
      this.loadImmichNeighbors(assetId, filename);
    }

    sortedRankedImmich() {
      if (!this.immichRing) return [];
      const key =
        this.immichRankedSortMetric === "sim"
          ? "similarity"
          : this.immichRankedSortMetric;
      const withMetric = this.immichRing.baseResults.filter(
        (r) => typeof r[key] === "number",
      );
      const withoutMetric = this.immichRing.baseResults.filter(
        (r) => typeof r[key] !== "number",
      );
      withMetric.sort((a, b) => b[key] - a[key]);
      return [...withMetric, ...withoutMetric];
    }

    toggleAssetSelection(assetId) {
      if (this.selectedAssetIds.has(assetId)) {
        this.selectedAssetIds.delete(assetId);
      } else {
        this.selectedAssetIds.add(assetId);
      }
      if (
        window.AppNG.manager &&
        typeof window.AppNG.manager.saveState === "function"
      ) {
        window.AppNG.manager.saveState();
      }
    }
  }

  class ProjectManager {
    constructor() {
      this.projects = new Map();
      this.activeProjectId = null;
      this.storageKey = "immichRingNG:state";
    }

    addProject(project) {
      this.projects.set(project.id, project);
      if (!this.activeProjectId) {
        this.activeProjectId = project.id;
      }
      return project;
    }

    getActive() {
      return this.projects.get(this.activeProjectId) || null;
    }

    createProject(name) {
      const project = new CharacterProject(
        "project-" + (this.projects.size + 1),
        name || "Character " + (this.projects.size + 1),
      );
      this.addProject(project);
      this.activeProjectId = project.id;
      return project;
    }

    closeProject(id) {
      const project = this.projects.get(id);
      if (!project) return;
      project.destroy();
      this.projects.delete(id);
      if (this.activeProjectId === id) {
        const remaining = Array.from(this.projects.keys());
        this.activeProjectId = remaining.length
          ? remaining[remaining.length - 1]
          : null;
      }
    }

    setActive(projectId) {
      if (this.projects.has(projectId)) {
        this.activeProjectId = projectId;
      }
    }

    saveState() {
      try {
        const state = Array.from(this.projects.values()).map((p) =>
          p.toPlain(),
        );
        localStorage.setItem(this.storageKey, JSON.stringify(state));
      } catch (e) {
        // ignore storage errors in non-browser/test contexts
      }
    }

    loadState() {
      try {
        const raw = localStorage.getItem(this.storageKey);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.map((entry) => CharacterProject.fromPlain(entry));
      } catch (e) {
        return [];
      }
    }

    render() {
      if (
        typeof window.AppNG.render !== "undefined" &&
        typeof window.AppNG.render.renderNGChart === "function"
      ) {
        const active = this.getActive();
        if (active && active.job && active.job.results) {
          const wrap = document.getElementById("ng-ring-chart-wrap");
          if (wrap) window.AppNG.render.renderNGChart(active, wrap);
        }
      }
      this.renderTabs();
      this.renderBottomBar();
      this.renderLeftRail();
      this.renderMain();
      const active = this.getActive();
      if (
        active &&
        active.job &&
        active.job.resolutionSummary &&
        window.AppNG.export
      ) {
        window.AppNG.export.applyResolutionSummaryNG(
          active.job.resolutionSummary,
        );
      }
      this.saveState();
    }

    renderTabs() {
      if (typeof tabsEl === "undefined") return;
      tabsEl.innerHTML = "";
      this.projects.forEach((project) => {
        const tab = document.createElement("div");
        tab.className =
          "ng-tab" +
          (project.id === this.activeProjectId ? " ng-tab-active" : "");
        tab.setAttribute("role", "tab");
        tab.setAttribute(
          "aria-selected",
          project.id === this.activeProjectId ? "true" : "false",
        );
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
          if (this.dragSourceId) {
            this.reorderProject(this.dragSourceId, project.id);
          }
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
    }

    renderBottomBar() {
      if (typeof taskButtons === "undefined") return;
      const active = this.getActive();
      taskButtons.forEach((btn) => {
        btn.disabled = !active;
        btn.classList.toggle(
          "ng-task-active",
          !!active && active.task === btn.dataset.task,
        );
      });
    }

    renderMain() {
      if (
        typeof mainPlaceholderEl === "undefined" ||
        typeof stageWrapEl === "undefined" ||
        typeof sidebarEl === "undefined"
      )
        return;
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
        showPlaceholder(
          "Pick Video, Immich, or Folder / Zip below to get started with “" +
            active.name +
            "”.",
        );
        return;
      }
      if (active.task === "video") {
        if (!active.ring) {
          showPlaceholder(
            "No analysis yet for “" +
              active.name +
              "” — load a video, pick a frame, and press Run Analysis.",
          );
          return;
        }
      } else if (active.task === "folderzip") {
        if (!active.ring) {
          showPlaceholder(
            "No analysis yet for “" +
              active.name +
              "” — load a folder or .zip in the left rail to begin.",
          );
          return;
        }
      } else if (active.task === "immich") {
        const hasBatchRing = active.ring && active.ring.sourceType === "immich";
        if (!hasBatchRing) {
          if (active.immichLoading && !active.immichRing) {
            showPlaceholder("Loading Immich neighbors…");
            return;
          }
          if (active.immichError && !active.immichRing) {
            showPlaceholder("Immich error: " + active.immichError);
            return;
          }
          if (!active.immichRing) {
            showPlaceholder(
              "Search for a face by filename in the left rail’s Search section to begin, or tick assets and press Analyze.",
            );
            return;
          }
        }
      } else {
        showPlaceholder(
          "Pick Video, Immich, or Folder / Zip below to get started with “" +
            active.name +
            "”.",
        );
        return;
      }

      mainPlaceholderEl.style.display = "none";
      stageWrapEl.style.display = "";
      sidebarEl.style.display = "";
      this.renderStage(active);
    }

    renderStage(project) {
      if (typeof this.renderLeftRail !== "function") return;
      this.renderLeftRail();
      if (
        project.task === "immich" &&
        !(project.ring && project.ring.sourceType === "immich")
      ) {
        this.renderImmichStage(project);
      } else {
        this.renderVideoStage(project);
      }
    }

    renderVideoStage(project) {
      if (typeof immichSectionEl === "undefined") return;
      immichSectionEl.style.display = "none";
      const ring = project.ring;
      if (!ring) return;
      const anchorLabel = `Frame ${ring.refFrameIdx} (Anchor)`;
      const metricLabel = {
        sim: "Similarity",
        yaw: "Yaw",
        pitch: "Pitch",
        roll: "Roll",
        blur: "Sharpness",
      }[project.ringSortMetric];
      const modeLabel =
        ring.sourceType === "folder"
          ? "FOLDER / IMAGE-SET ANALYSIS (local, not in Immich)"
          : ring.sourceType === "immich"
            ? "IMMICH BATCH ANALYSIS (pose data for selected Immich assets)"
            : "VIDEO FRAME ANALYSIS (local, not in Immich)";
      if (typeof framesSectionTitleEl !== "undefined") {
        framesSectionTitleEl.textContent =
          ring.sourceType === "immich"
            ? "Immich batch results"
            : "Local frames";
      }
      if (typeof hudModeEl !== "undefined") hudModeEl.textContent = modeLabel;
      if (typeof hudFilenameEl !== "undefined")
        hudFilenameEl.textContent = `${anchorLabel} · sorted by ${metricLabel}`;
      if (typeof sidebarCurrentImgEl !== "undefined")
        sidebarCurrentImgEl.src = ring.anchorUrl;
      if (typeof sidebarCurrentFnameEl !== "undefined")
        sidebarCurrentFnameEl.textContent = anchorLabel;
      if (typeof sidebarCurrentModeEl !== "undefined")
        sidebarCurrentModeEl.textContent = modeLabel;
      if (typeof sidebarCurrentDetailEl !== "undefined")
        sidebarCurrentDetailEl.textContent = "match: 100.0%";

      const sorted = project.sortedRanked();
      const { kept, total } = project.squeezeFiltered(sorted);
      if (typeof squeezeVal !== "undefined")
        squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      if (typeof sharpVal !== "undefined")
        sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        if (typeof stageEl !== "undefined") stageEl.style.display = "none";
        if (typeof poseListViewEl !== "undefined")
          poseListViewEl.style.display = "flex";
        if (typeof poseListScrubberEl !== "undefined")
          poseListScrubberEl.style.display = "flex";
        this.renderRankedList(project, kept);
        return;
      }
      if (typeof stageEl !== "undefined") stageEl.style.display = "";
      if (typeof poseListViewEl !== "undefined")
        poseListViewEl.style.display = "none";
      if (typeof poseListScrubberEl !== "undefined")
        poseListScrubberEl.style.display = "none";
      if (typeof stageEl !== "undefined") {
        stageEl.innerHTML = "";
        const ringScale = project.ringScale / 100;
        [0.9, 0.7, 0.5, 0.35].forEach((band) => {
          const r = window.AppNG.shared.radiusForSim(band, ringScale);
          const ringEl = document.createElement("div");
          ringEl.className = "ng-ring";
          ringEl.style.width = r * 2 + "px";
          ringEl.style.height = r * 2 + "px";
          stageEl.appendChild(ringEl);
        });

        const center = document.createElement("div");
        center.className = "ng-node ng-node-center";
        center.style.width = window.AppNG.shared.CENTER_SIZE + "px";
        center.style.height = window.AppNG.shared.CENTER_SIZE + "px";
        center.style.transform = "translate(-50%, -50%)";
        center.dataset.baseX = 0;
        center.dataset.baseY = 0;
        center.innerHTML = `<img src="${ring.anchorUrl}">`;
        center.addEventListener("mouseenter", () =>
          window.AppNG.render.showHoverPreview({
            filename: "Reference (anchor)",
            thumbUrl: ring.anchorUrl,
            similarity: 1,
          }),
        );
        center.addEventListener("mouseleave", () =>
          window.AppNG.render.hideHoverPreview(),
        );
        stageEl.appendChild(center);

        const bands = Array.from(
          { length: window.AppNG.shared.BAND_COUNT },
          () => [],
        );
        kept.forEach((r) => {
          const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
          const bandIdx = Math.min(
            window.AppNG.shared.BAND_COUNT - 1,
            Math.floor((1 - t) * window.AppNG.shared.BAND_COUNT),
          );
          bands[bandIdx].push(r);
        });

        bands.forEach((bandResults, bandIdx) => {
          if (!bandResults.length) return;
          const t = 1 - bandIdx / (window.AppNG.shared.BAND_COUNT - 1);
          const avgSim = 0.25 + t * 0.75;
          const radius = window.AppNG.shared.radiusForSim(avgSim, ringScale);
          const size = window.AppNG.shared.sizeForSim(avgSim);
          const angleOffset = bandIdx * 0.6;

          bandResults.forEach((r, i) => {
            const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            const node = document.createElement("div");
            node.className =
              "ng-node" +
              (project.selectedFrames.has(r.frame) ? " ng-node-selected" : "");
            node.style.width = size + "px";
            node.style.height = size + "px";
            node.style.left = `calc(50% + ${x}px)`;
            node.style.top = `calc(50% + ${y}px)`;
            node.style.transform = "translate(-50%, -50%)";
            node.dataset.baseX = x;
            node.dataset.baseY = y;
            node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}%`;
            node.innerHTML = `<img src="${window.AppNG.shared.thumbUrlFor(r)}">`;
            node.addEventListener("mouseenter", () =>
              window.AppNG.render.showHoverPreview(r),
            );
            node.addEventListener("mouseleave", () =>
              window.AppNG.render.hideHoverPreview(),
            );
            node.ondblclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              project.toggleFrameSelection(r.frame);
              this.renderStage(project);
            };
            stageEl.appendChild(node);
          });
        });
      }
      this.renderRankedList(project, kept);
    }

    renderRankedList(project, kept) {
      if (typeof listBodyFramesEl === "undefined") return;
      listBodyFramesEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.frame = r.frame;
        const pct = (r.similarity * 100).toFixed(1);

        let poseHtml = "";
        if (
          r.pitch !== undefined &&
          r.yaw !== undefined &&
          r.roll !== undefined &&
          r.pitch !== null
        ) {
          let poseLine = `pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
          if (typeof r.blur === "number")
            poseLine += ` · sharp: ${r.blur.toFixed(0)}`;
          if (typeof r.vertFillPct === "number")
            poseLine += ` · ${(r.vertFillPct * 100).toFixed(0)}% frame ht`;
          poseHtml = `<div style="font-size:10px;color:var(--ng-text-dim);margin-top:2px;">${poseLine}</div>`;
        }

        row.innerHTML = `
          <input type="checkbox" class="ng-frame-select-cb" data-frame="${r.frame}" ${project.selectedFrames.has(r.frame) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${window.AppNG.shared.thumbUrlFor(r)}" loading="lazy">
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
        row.addEventListener("mouseenter", () =>
          window.AppNG.render.showHoverPreview(r),
        );
        row.addEventListener("mouseleave", () =>
          window.AppNG.render.hideHoverPreview(),
        );
        listBodyFramesEl.appendChild(row);
      });

      if (typeof framesSectionEl !== "undefined") {
        framesSectionEl.style.display = kept.length ? "" : "none";
      }
      if (typeof framesSectionCountEl !== "undefined") {
        framesSectionCountEl.textContent = kept.length
          ? `(${kept.length})`
          : "";
      }
    }

    renderImmichStage(project) {
      if (typeof framesSectionEl === "undefined") return;
      framesSectionEl.style.display = "none";
      const ring = project.immichRing;
      if (!ring) return;
      const anchorUrl = `/api/ng/preview/${ring.centerAssetId}`;
      const modeLabel =
        ring.mode === "clip"
          ? "CLIP image embedding (no face match found)"
          : "face embedding";
      const metricLabel = {
        sim: "Similarity",
        yaw: "Yaw",
        pitch: "Pitch",
        roll: "Roll",
        blur: "Sharpness",
      }[project.ringSortMetric];
      if (typeof hudModeEl !== "undefined")
        hudModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      if (typeof hudFilenameEl !== "undefined")
        hudFilenameEl.textContent = `${ring.centerFilename} · sorted by ${metricLabel}`;
      if (typeof sidebarCurrentImgEl !== "undefined")
        sidebarCurrentImgEl.src = anchorUrl;
      if (typeof sidebarCurrentFnameEl !== "undefined")
        sidebarCurrentFnameEl.textContent = ring.centerFilename;
      if (typeof sidebarCurrentModeEl !== "undefined")
        sidebarCurrentModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      if (typeof sidebarCurrentDetailEl !== "undefined") {
        if (ring.centerPose) {
          const p = ring.centerPose;
          sidebarCurrentDetailEl.textContent =
            `pitch: ${p.pitch.toFixed(1)} yaw: ${p.yaw.toFixed(1)} roll: ${p.roll.toFixed(1)}` +
            (typeof p.blur === "number"
              ? ` · sharpness: ${p.blur.toFixed(0)}`
              : "");
        } else {
          sidebarCurrentDetailEl.textContent = "match: 100.0%";
        }
      }

      const sorted = project.sortedRankedImmich();
      const { kept, total } = project.squeezeFiltered(sorted);
      if (typeof squeezeVal !== "undefined")
        squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      if (typeof sharpVal !== "undefined")
        sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        if (typeof stageEl !== "undefined") stageEl.style.display = "none";
        if (typeof poseListViewEl !== "undefined")
          poseListViewEl.style.display = "flex";
        if (typeof poseListScrubberEl !== "undefined")
          poseListScrubberEl.style.display = "flex";
        this.renderRankedListImmich(project, kept);
        return;
      }
      if (typeof stageEl !== "undefined") stageEl.style.display = "";
      if (typeof poseListViewEl !== "undefined")
        poseListViewEl.style.display = "none";
      if (typeof poseListScrubberEl !== "undefined")
        poseListScrubberEl.style.display = "none";
      if (typeof stageEl !== "undefined") {
        stageEl.innerHTML = "";
        const ringScale = project.ringScale / 100;
        [0.9, 0.7, 0.5, 0.35].forEach((band) => {
          const r = window.AppNG.shared.radiusForSim(band, ringScale);
          const ringEl = document.createElement("div");
          ringEl.className = "ng-ring";
          ringEl.style.width = r * 2 + "px";
          ringEl.style.height = r * 2 + "px";
          stageEl.appendChild(ringEl);
        });

        const center = document.createElement("div");
        center.className = "ng-node ng-node-center";
        center.style.width = window.AppNG.shared.CENTER_SIZE + "px";
        center.style.height = window.AppNG.shared.CENTER_SIZE + "px";
        center.style.transform = "translate(-50%, -50%)";
        center.dataset.baseX = 0;
        center.dataset.baseY = 0;
        center.innerHTML = `<img src="${anchorUrl}">`;
        center.addEventListener("mouseenter", () =>
          window.AppNG.render.showHoverPreview({
            filename: ring.centerFilename + " (centered)",
            thumbUrl: anchorUrl,
            similarity: 1,
          }),
        );
        center.addEventListener("mouseleave", () =>
          window.AppNG.render.hideHoverPreview(),
        );
        stageEl.appendChild(center);

        const bands = Array.from(
          { length: window.AppNG.shared.BAND_COUNT },
          () => [],
        );
        kept.forEach((r) => {
          const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
          const bandIdx = Math.min(
            window.AppNG.shared.BAND_COUNT - 1,
            Math.floor((1 - t) * window.AppNG.shared.BAND_COUNT),
          );
          bands[bandIdx].push(r);
        });

        bands.forEach((bandResults, bandIdx) => {
          if (!bandResults.length) return;
          const t = 1 - bandIdx / (window.AppNG.shared.BAND_COUNT - 1);
          const avgSim = 0.25 + t * 0.75;
          const radius = window.AppNG.shared.radiusForSim(avgSim, ringScale);
          const size = window.AppNG.shared.sizeForSim(avgSim);
          const angleOffset = bandIdx * 0.6;

          bandResults.forEach((r, i) => {
            const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            const node = document.createElement("div");
            node.className =
              "ng-node" +
              (project.selectedAssetIds.has(r.assetId)
                ? " ng-node-selected"
                : "");
            node.style.width = size + "px";
            node.style.height = size + "px";
            node.style.left = `calc(50% + ${x}px)`;
            node.style.top = `calc(50% + ${y}px)`;
            node.style.transform = "translate(-50%, -50%)";
            node.dataset.baseX = x;
            node.dataset.baseY = y;
            node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}%`;
            node.innerHTML = `<img src="${window.AppNG.shared.thumbUrlFor(r)}">`;
            node.addEventListener("mouseenter", () =>
              window.AppNG.render.showHoverPreview(r),
            );
            node.addEventListener("mouseleave", () =>
              window.AppNG.render.hideHoverPreview(),
            );
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
            stageEl.appendChild(node);
          });
        });
      }
      this.renderRankedListImmich(project, kept);
    }

    renderRankedListImmich(project, kept) {
      if (typeof listBodyImmichEl === "undefined") return;
      listBodyImmichEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.assetId = r.assetId;
        const pct = (r.similarity * 100).toFixed(1);
        row.innerHTML = `
          <input type="checkbox" class="ng-asset-select-cb" data-asset="${r.assetId}" ${project.selectedAssetIds.has(r.assetId) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${window.AppNG.shared.thumbUrlFor(r)}" loading="lazy">
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
        row
          .querySelector("img")
          .addEventListener("click", () =>
            project.recenterImmich(r.assetId, r.filename),
          );
        row.addEventListener("mouseenter", () =>
          window.AppNG.render.showHoverPreview(r),
        );
        row.addEventListener("mouseleave", () =>
          window.AppNG.render.hideHoverPreview(),
        );
        listBodyImmichEl.appendChild(row);
      });

      if (typeof immichSectionEl !== "undefined") {
        immichSectionEl.style.display = kept.length ? "" : "none";
      }
      if (typeof immichSectionCountEl !== "undefined") {
        immichSectionCountEl.textContent = kept.length
          ? `(${kept.length})`
          : "";
      }
    }

    renderLeftRail() {
      if (
        typeof leftRailEmptyEl === "undefined" ||
        typeof leftRailBodyEl === "undefined"
      )
        return;
      const active = this.getActive();
      if (!active) {
        leftRailEmptyEl.style.display = "";
        leftRailBodyEl.style.display = "none";
        return;
      }
      leftRailEmptyEl.style.display = "none";
      leftRailBodyEl.style.display = "flex";

      if (typeof videoAnalysisSectionEl !== "undefined") {
        videoAnalysisSectionEl.style.display =
          active.task === "video" || active.task === "folderzip" ? "" : "none";
      }
      if (typeof searchSectionEl !== "undefined") {
        searchSectionEl.style.display = active.task === "immich" ? "" : "none";
      }
      if (typeof analysisSectionTitleEl !== "undefined") {
        analysisSectionTitleEl.textContent =
          active.task === "folderzip"
            ? "Folder / Zip Analysis"
            : "Video Analysis";
      }
      if (typeof videoAnalysisBodyEl !== "undefined") {
        videoAnalysisBodyEl.style.display =
          active.task === "video" ? "" : "none";
      }
      if (typeof videoRangeRowEl !== "undefined")
        videoRangeRowEl.style.display = active.task === "video" ? "" : "none";
      if (typeof folderRowEl !== "undefined")
        folderRowEl.style.display = active.task === "folderzip" ? "" : "none";

      if (typeof ringScaleInput !== "undefined")
        ringScaleInput.value = active.ringScale;
      if (typeof ringScaleVal !== "undefined")
        ringScaleVal.textContent = active.ringScale + "%";
      if (typeof squeezeSlider !== "undefined")
        squeezeSlider.value = active.squeezeMinPct;
      if (typeof sharpEnableCb !== "undefined")
        sharpEnableCb.checked = active.sharpCutoffEnabled;
      if (typeof sharpControlsEl !== "undefined")
        sharpControlsEl.style.display = active.sharpCutoffEnabled
          ? "flex"
          : "none";
      if (typeof sharpSlider !== "undefined")
        sharpSlider.value = active.sharpMinVal;
      if (typeof folderRefIndexInput !== "undefined")
        folderRefIndexInput.value = active.folderRefIndex;
      if (typeof simThresholdInput !== "undefined")
        simThresholdInput.value = active.simThreshold;
      if (typeof blurThresholdInput !== "undefined")
        blurThresholdInput.value = active.blurThreshold;
      if (typeof cacheFormatPngCb !== "undefined")
        cacheFormatPngCb.checked = active.cacheFormatPng;

      if (typeof active.rang === "undefined") {
        // no-op placeholder to keep the migration safe in the split bootstrap stage
      }
      if (
        typeof immichSearchInput !== "undefined" &&
        document.activeElement !== immichSearchInput
      ) {
        immichSearchInput.value = active.immichSearchQuery;
      }
      if (window.AppNG.pickers && active.ring) {
        window.AppNG.pickers.setupPosePickerNG(active);
        window.AppNG.pickers.renderPosePickerGridNG(active);
        window.AppNG.pickers.setupScalePickerNG(active);
        window.AppNG.pickers.renderScalePickerGridNG(active);
      }
      if (typeof this.renderVideoAnalysisBody === "function")
        this.renderVideoAnalysisBody(active);
      if (typeof this.renderFramePreview === "function")
        this.renderFramePreview(active);
    }

    renameProject(id, name) {
      const project = this.projects.get(id);
      if (!project) return;
      const trimmed = (name || "").trim();
      if (!trimmed || trimmed === project.name) return;
      if (this.projects.size > 1) {
        for (const other of this.projects.values()) {
          if (
            other.id !== id &&
            other.name.trim().toLowerCase() === trimmed.toLowerCase()
          ) {
            alert(trimmed + " -- there can only be one.");
            return;
          }
        }
      }
      project.name = trimmed;
    }

    reorderProject(sourceId, targetId) {
      if (sourceId === targetId) return;
      const fromIdx = Array.from(this.projects.keys()).indexOf(sourceId);
      const toIdx = Array.from(this.projects.keys()).indexOf(targetId);
      if (fromIdx === -1 || toIdx === -1) return;
      const entries = Array.from(this.projects.entries());
      const [moved] = entries.splice(fromIdx, 1);
      entries.splice(toIdx, 0, moved);
      this.projects = new Map(entries);
      this.render();
    }

    setTask(task) {
      const project = this.getActive();
      if (!project) return;
      if (project.task === "video" && task !== "video") {
        project.stopPlayIfRunning();
      }
      project.task = task;
      this.render();
    }

    renderImmichSearch(project) {
      if (typeof immichSearchInput === "undefined") return;
      if (document.activeElement !== immichSearchInput) {
        immichSearchInput.value = project.immichSearchQuery;
      }
      if (typeof immichRandomFaceBtn !== "undefined") {
        immichRandomFaceBtn.disabled = project.immichSearching;
      }
      if (typeof immichSearchStatusEl !== "undefined") {
        immichSearchStatusEl.textContent = project.immichSearching
          ? "Searching…"
          : "";
      }
      if (typeof immichSearchResultsEl === "undefined") return;
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
        row.addEventListener("click", () =>
          project.loadImmichNeighbors(m.assetId, m.filename),
        );
        immichSearchResultsEl.appendChild(row);
      });
    }

    renderAnalysisStatus(project) {
      if (typeof analysisStatusEl === "undefined") return;
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
      analysisStatusEl.innerHTML = `
        Done — ${j.passed}/${j.frameCount} kept.
        <div id="ng-sim-sparkline-wrap" style="margin-top:8px;"></div>
        <button id="ng-btn-save-kept" class="ng-btn ng-btn-full ng-btn-accent">Save kept frames to disk</button>
        <button id="ng-btn-save-selected" class="ng-btn ng-btn-full ng-btn-accent" ${selCount ? "" : "disabled"}>Save ${selCount} selected frame${selCount === 1 ? "" : "s"} to disk</button>
        <button id="ng-btn-view-selected" class="ng-btn ng-btn-full">View selected</button>
        <div id="ng-export-result" class="ng-stub-note"></div>
      `;

      if (
        typeof window.AppNG.render !== "undefined" &&
        typeof window.AppNG.render.renderNGChart === "function"
      ) {
        const sparkWrap = analysisStatusEl.querySelector(
          "#ng-sim-sparkline-wrap",
        );
        if (sparkWrap) window.AppNG.render.renderNGChart(project, sparkWrap);
      }

      const exportResultEl =
        analysisStatusEl.querySelector("#ng-export-result");
      const saveKeptBtn = analysisStatusEl.querySelector("#ng-btn-save-kept");
      const saveSelectedBtn = analysisStatusEl.querySelector(
        "#ng-btn-save-selected",
      );
      const viewSelectedBtn = analysisStatusEl.querySelector(
        "#ng-btn-view-selected",
      );

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

      if (viewSelectedBtn) {
        viewSelectedBtn.onclick = () => this.openSelectedModal(project);
      }

      if (project.job.sourceType === "video") {
        const playbackBtn = document.createElement("button");
        playbackBtn.className = "ng-btn ng-btn-full ng-btn-accent";
        playbackBtn.style.marginTop = "6px";
        playbackBtn.disabled = project.playbackBuilding;
        playbackBtn.textContent = project.playbackBuilding
          ? "Building playback..."
          : project.playback
            ? "Reopen playback"
            : "Pop out playback (rejected frames blanked)";
        playbackBtn.addEventListener("click", () => {
          if (project.playback) {
            window.AppNG.render.PlaybackModal.openBuild(project);
          } else {
            project.buildPlayback();
          }
        });
        analysisStatusEl.appendChild(playbackBtn);
      }
    }

    renderImmichAnalysisStatus(project) {
      if (typeof immichAnalysisStatusEl === "undefined") return;
      const j = project.job;
      if (!j || j.sourceType !== "immich") {
        immichAnalysisStatusEl.innerHTML = "";
        return;
      }
      if (j.status === "error") {
        immichAnalysisStatusEl.innerHTML = `Error: ${j.error || "unknown error"}`;
        return;
      }
      if (j.status === "running") {
        immichAnalysisStatusEl.innerHTML = `
          Fetching &amp; analyzing… ${j.frameCount || 0} assets seen<br>
          ${j.passed || 0} kept, ${j.failedSim || 0} low sim, ${j.failedBlur || 0} blurry
          <div class="ng-progress-track"><div class="ng-progress-fill" style="width:${Math.min(100, ((j.frameCount || 0) / 200) * 100)}%"></div></div>
        `;
        return;
      }

      const selCount = project.selectedFrames.size;
      const fetchErrCount = (j.fetchErrors || []).length;
      immichAnalysisStatusEl.innerHTML = `
        Done — ${j.passed}/${j.frameCount} kept${fetchErrCount ? ` (${fetchErrCount} asset${fetchErrCount === 1 ? "" : "s"} failed to fetch)` : ""}.
        <div id="ng-immich-sim-sparkline-wrap" style="margin-top:8px;"></div>
        <button id="ng-immich-btn-save-kept" class="ng-btn ng-btn-full ng-btn-accent">Save kept frames to disk</button>
        <button id="ng-immich-btn-save-selected" class="ng-btn ng-btn-full ng-btn-accent" ${selCount ? "" : "disabled"}>Save ${selCount} selected frame${selCount === 1 ? "" : "s"} to disk</button>
        <button id="ng-immich-btn-view-selected" class="ng-btn ng-btn-full">View selected</button>
        <div id="ng-immich-export-result" class="ng-stub-note"></div>
      `;

      if (
        typeof window.AppNG.render !== "undefined" &&
        typeof window.AppNG.render.renderNGChart === "function"
      ) {
        const sparkWrap = immichAnalysisStatusEl.querySelector(
          "#ng-immich-sim-sparkline-wrap",
        );
        if (sparkWrap) window.AppNG.render.renderNGChart(project, sparkWrap);
      }

      const exportResultEl = immichAnalysisStatusEl.querySelector(
        "#ng-immich-export-result",
      );
      const saveKeptBtn = immichAnalysisStatusEl.querySelector(
        "#ng-immich-btn-save-kept",
      );
      const saveSelectedBtn = immichAnalysisStatusEl.querySelector(
        "#ng-immich-btn-save-selected",
      );
      const viewSelectedBtn = immichAnalysisStatusEl.querySelector(
        "#ng-immich-btn-view-selected",
      );

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

      if (viewSelectedBtn) {
        viewSelectedBtn.onclick = () => this.openSelectedModal(project);
      }
    }

    openSelectedModal(project) {
      if (typeof ngSelectedModal === "undefined") return;
      const overlay = document.getElementById("ng-selected-modal");
      const grid = document.getElementById("ng-selected-modal-grid");
      if (!overlay || !grid) return;
      const title = document.getElementById("ng-selected-modal-title");
      const poseLayoutCb = document.getElementById(
        "ng-selected-modal-pose-layout",
      );
      const spreadWrap = document.getElementById(
        "ng-selected-modal-spread-wrap",
      );
      const spreadSlider = document.getElementById("ng-selected-modal-spread");
      const realPreviewCb = document.getElementById(
        "ng-selected-modal-real-preview",
      );
      const deselectAllBtn = document.getElementById(
        "ng-selected-modal-deselect-all",
      );

      const items = [];
      (project.ring ? project.ring.baseResults : []).forEach((r) => {
        if (project.selectedFrames.has(r.frame))
          items.push({
            kind: "frame",
            frame: r.frame,
            filename: r.filename,
            thumb: window.AppNG.shared.thumbUrlFor(r),
            similarity: r.similarity,
            pitch: typeof r.pitch === "number" ? r.pitch : null,
            yaw: typeof r.yaw === "number" ? r.yaw : null,
          });
      });
      project.selectedAssetIds.forEach((assetId) => {
        const known = project.immichRing?.baseResults.find(
          (r) => r.assetId === assetId,
        );
        const cached = project.assetPoseCache[assetId];
        items.push({
          kind: "asset",
          assetId,
          filename: known?.filename || assetId,
          thumb: known
            ? window.AppNG.shared.thumbUrlFor(known)
            : `/api/ng/thumb/${assetId}`,
          similarity: known?.similarity,
          pitch: cached?.pitch ?? known?.pitch ?? null,
          yaw: cached?.yaw ?? known?.yaw ?? null,
        });
      });

      const removeItem = (item) => {
        if (item.kind === "asset") project.toggleAssetSelection(item.assetId);
        else project.toggleFrameSelection(item.frame);
        this.render();
        renderModal();
      };
      const imageSrc = (item) => {
        if (!realPreviewCb?.checked) return item.thumb;
        const params = new URLSearchParams(
          window.AppNG.export.gatherExportParamsNG(),
        );
        return item.kind === "frame" && project.job?.jobId
          ? `/api/ng/export-preview/${project.job.jobId}/${item.frame}?${params}`
          : item.kind === "asset"
            ? `/api/ng/export-preview-immich/${item.assetId}?${params}`
            : item.thumb;
      };
      const renderGrid = () => {
        grid.classList.remove("ng-pose-scatter-mode");
        grid.innerHTML = "";
        items.forEach((item) => {
          const cell = document.createElement("div");
          cell.className = "ng-selected-modal-cell";
          cell.innerHTML = `<img src="${imageSrc(item)}" loading="lazy"><div class="ng-selected-modal-cell-info">${item.filename}${typeof item.similarity === "number" ? ` — ${(item.similarity * 100).toFixed(1)}%` : ""}</div><button class="ng-selected-modal-remove" title="Remove from selection">&times;</button>`;
          cell.querySelector("img").addEventListener("click", () => {
            if (item.kind === "asset")
              project.recenterImmich(item.assetId, item.filename);
            else if (project.video && project.job?.sourceType === "video")
              project.stepAndSyncAudio(item.frame);
            else window.AppNG.render.showStaticFramePreviewNG(project, item);
            overlay.style.display = "none";
          });
          cell
            .querySelector(".ng-selected-modal-remove")
            .addEventListener("click", () => removeItem(item));
          grid.appendChild(cell);
        });
        if (!items.length)
          grid.appendChild(document.createTextNode("Nothing selected yet."));
      };
      const renderScatter = () => {
        grid.classList.add("ng-pose-scatter-mode");
        grid.innerHTML = "";
        const posed = items.filter(
          (item) => item.pitch !== null && item.yaw !== null,
        );
        if (!posed.length) {
          grid.textContent = "No pose data on the current selection.";
          return;
        }
        const stage = document.createElement("div");
        stage.className = "ng-pose-scatter-stage";
        const pitchMean =
          posed.reduce((sum, item) => sum + item.pitch, 0) / posed.length;
        const yawMean =
          posed.reduce((sum, item) => sum + item.yaw, 0) / posed.length;
        const spread = parseFloat(spreadSlider?.value) || 1;
        posed.forEach((item) => {
          const cell = document.createElement("div");
          cell.className = "ng-pose-scatter-item";
          cell.style.left = `${Math.max(4, Math.min(96, 50 + (item.yaw - yawMean) * spread))}%`;
          cell.style.top = `${Math.max(4, Math.min(96, 50 - (item.pitch - pitchMean) * spread))}%`;
          cell.innerHTML = `<img src="${imageSrc(item)}" loading="lazy"><div class="ng-pose-scatter-label">p${item.pitch.toFixed(0)} y${item.yaw.toFixed(0)}</div>`;
          cell.ondblclick = () => removeItem(item);
          stage.appendChild(cell);
        });
        grid.appendChild(stage);
      };
      const renderModal = () => {
        if (title) title.textContent = `Selected (${items.length})`;
        if (spreadWrap)
          spreadWrap.style.display = poseLayoutCb?.checked ? "flex" : "none";
        if (poseLayoutCb?.checked) renderScatter();
        else renderGrid();
      };
      if (poseLayoutCb) poseLayoutCb.onchange = renderModal;
      if (realPreviewCb) realPreviewCb.onchange = renderModal;
      if (spreadSlider) spreadSlider.oninput = renderModal;
      if (deselectAllBtn)
        deselectAllBtn.onclick = () => {
          project.selectedFrames.clear();
          project.selectedAssetIds.clear();
          this.render();
          renderModal();
        };
      renderModal();
      overlay.style.display = "flex";
    }

    findNeutralPose(project) {
      const hasBatchRing = project.ring && project.ring.sourceType === "immich";
      const baseResults =
        project.task === "immich" && !hasBatchRing
          ? project.immichRing
            ? project.immichRing.baseResults
            : null
          : project.ring
            ? project.ring.baseResults
            : null;
      if (!baseResults) return;

      const { kept } = project.squeezeFiltered(baseResults);
      const pool = kept.filter(
        (r) =>
          typeof r.yaw === "number" &&
          typeof r.pitch === "number" &&
          typeof r.roll === "number",
      );

      if (!pool.length) {
        if (typeof neutralPoseReadoutEl !== "undefined") {
          neutralPoseReadoutEl.style.display = "block";
          neutralPoseReadoutEl.textContent =
            "No frames with pose data in current working set.";
        }
        return;
      }

      let best = pool[0];
      let bestScore =
        Math.abs(best.yaw) + Math.abs(best.pitch) + Math.abs(best.roll);
      pool.forEach((r) => {
        const score = Math.abs(r.yaw) + Math.abs(r.pitch) + Math.abs(r.roll);
        if (score < bestScore) {
          best = r;
          bestScore = score;
        }
      });

      if (typeof neutralPoseReadoutEl !== "undefined") {
        neutralPoseReadoutEl.style.display = "block";
        neutralPoseReadoutEl.innerHTML = `Most neutral: <b>${best.filename}</b> — yaw ${best.yaw.toFixed(1)}° pitch ${best.pitch.toFixed(1)}° roll ${best.roll.toFixed(1)}° (sim ${(best.similarity * 100).toFixed(1)}%)<button type="button" id="ng-use-as-reference-btn" class="ng-btn ng-btn-full ng-btn-accent" style="margin-top:6px;">Use as reference and re-analyze</button>`;
        const useBtn = document.getElementById("ng-use-as-reference-btn");
        if (useBtn) {
          useBtn.addEventListener("click", () => {
            const sourceType = project.job ? project.job.sourceType : null;
            neutralPoseReadoutEl.style.display = "none";
            neutralPoseReadoutEl.innerHTML = "";
            if (sourceType === "immich") {
              if (!project._lastImmichAssetIds?.length) {
                neutralPoseReadoutEl.style.display = "block";
                neutralPoseReadoutEl.textContent =
                  "Original Immich selection is no longer available - re-tick assets and re-analyze.";
                return;
              }
              project.startImmichAnalysis(
                project._lastImmichAssetIds,
                best.frame,
              );
            } else if (project.task === "immich") {
              project.recenterImmich(best.assetId, best.filename);
            } else if (sourceType === "folder") {
              if (!project._lastFolderSource) {
                neutralPoseReadoutEl.style.display = "block";
                neutralPoseReadoutEl.textContent =
                  "Original folder or ZIP is no longer available - reload it first.";
                return;
              }
              project.startFolderAnalysis(
                project._lastFolderSource,
                best.frame,
              );
            } else if (project.videoFile) {
              project.startAnalysis(best.frame);
            } else {
              neutralPoseReadoutEl.style.display = "block";
              neutralPoseReadoutEl.textContent =
                "Original video is no longer available - reload it first.";
            }
          });
        }
        if (window.AppNG.render?.flashHighlightNG) {
          window.AppNG.render.flashHighlightNG(best.frame, best.assetId);
        }
      }
    }

    renderVideoAnalysisBody(project) {
      if (typeof videoAnalysisBodyEl === "undefined") return;
      if (project.task !== "video") {
        videoAnalysisBodyEl.innerHTML = "";
        return;
      }
      videoAnalysisBodyEl.innerHTML = "";
      if (project.videoLoading) {
        videoAnalysisBodyEl.appendChild(
          document.createTextNode("Loading video..."),
        );
      } else if (project.video) {
        const info = document.createElement("p");
        info.className = "ng-video-hint";
        info.textContent =
          project.video.fps.toFixed(2) +
          " fps, " +
          project.video.totalFrames +
          " frames, " +
          project.video.duration.toFixed(1) +
          "s";
        videoAnalysisBodyEl.appendChild(info);
      } else {
        videoAnalysisBodyEl.appendChild(
          document.createTextNode("No video loaded for this project yet."),
        );
      }
      if (typeof analysisStartInput !== "undefined") {
        analysisStartInput.value =
          project.video && project.video.rangeStartSec != null
            ? project.video.rangeStartSec
            : "";
      }
      if (typeof analysisEndInput !== "undefined") {
        analysisEndInput.value =
          project.video && project.video.rangeEndSec != null
            ? project.video.rangeEndSec
            : "";
      }
      if (typeof startAnalysisBtn !== "undefined") {
        startAnalysisBtn.disabled =
          !project.video ||
          project.videoLoading ||
          (project.job && project.job.status === "running");
      }
    }

    renderFramePreview(project) {
      if (
        typeof previewHintEl === "undefined" ||
        typeof previewCanvasEl === "undefined"
      )
        return;
      if (!project.video) {
        previewHintEl.textContent =
          project.task === "folderzip"
            ? "No folder/zip analyzed yet for this project."
            : "No video loaded for this project yet.";
        previewCanvasEl.style.display = "none";
        return;
      }
      previewHintEl.textContent =
        "Reference frame — use ← / → to step one actual video frame";
      previewCanvasEl.style.display = "";
      if (typeof frameCounterEl !== "undefined") {
        frameCounterEl.textContent =
          "Frame: " +
          project.video.currentFrame +
          " / " +
          project.video.totalFrames;
      }
    }
  }

  window.AppNG.project = {
    CharacterProject,
    ProjectManager,
  };
})();
