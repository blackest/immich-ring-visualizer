/**
 * videoNG.js -- the video player, consolidated.
 *
 * Pulled out of six files it was scattered across during the original
 * appNG.js decomposition (domRefsManifestNG.js, characterProjectNG.js,
 * poseListAndFrameHelpersNG.js, projectManagerNG.js, playbackModalNG.js,
 * bootstrapWiringNG.js) into one place, at John's request, after noticing
 * `videoPickerButton` -- the actual "Choose Video..." button -- was oddly
 * living inside a file called poseListAndFrameHelpersNG.js.
 *
 * This is a REHOMING, not a rewrite: every function/method body below is
 * unchanged from where it used to live. See git history on the five
 * source files for the exact before/after diffs.
 *
 * ============================================================
 * WHAT LIVES HERE
 * ============================================================
 *   - DOM refs for every video-exclusive element (Frame Preview controls,
 *     Video Analysis section, the playback popout modal). Elements SHARED
 *     with the folder/zip static-preview feature (previewCanvasEl,
 *     previewHintEl, previewControlsScrollEl) deliberately stayed in
 *     domRefsManifestNG.js -- chartAndStaticPreviewNG.js needs them too.
 *   - PlaybackModal -- the real <video>-element popout player (was
 *     playbackModalNG.js in full; that file no longer exists).
 *   - drawFrame(), setPlayingVisual(), videoPickerButton() -- were in
 *     poseListAndFrameHelpersNG.js; that file keeps its actual pose-list
 *     concern (renderPoseListNG, setupPoseListScrubberNG, flashHighlightNG,
 *     placeholder), none of which is video-specific.
 *   - CharacterProject.prototype patches: stopPlayIfRunning, loadVideo,
 *     frameTime, frameFromTime, syncAudioToFrame, stepTo,
 *     stepAndSyncAudio, togglePlay, playFramesWithoutAudio -- were method
 *     bodies inside the `class CharacterProject` literal in
 *     characterProjectNG.js. Patched onto the prototype here instead;
 *     requires characterProjectNG.js to have already defined the class
 *     (see load-order note below). buildPlayback() and the video-field
 *     lines inside toPlain()/fromPlain() were NOT moved -- they're bound
 *     up with job/ring/export/persistence concerns that touch far more
 *     than just video, and moving only half of a persistence function
 *     out to another file is more confusing than useful.
 *   - ProjectManager.renderVideoAnalysisBody / .renderFramePreview --
 *     were methods inside the ProjectManager object literal in
 *     projectManagerNG.js. Patched onto the object here the same way.
 *   - Wiring: every addEventListener() call for the Frame Preview
 *     controls, the Analyze-window range inputs, Start Analysis, the
 *     popout button, the playback modal itself, and the ←/→ keyboard
 *     shortcut -- were bare top-level statements inside
 *     bootstrapWiringNG.js's ~60-call wiring tail.
 *
 * ============================================================
 * LOAD ORDER (added to WIRING_ORDER.md)
 * ============================================================
 * Must load AFTER:
 *   - domRefsManifestNG.js         (shared previewCanvasEl etc.)
 *   - characterProjectNG.js        (patches CharacterProject.prototype)
 *   - poseListAndFrameHelpersNG.js (showHoverPreview et al, not required
 *                                   by this file, but keeps the "load
 *                                   near where the rest of the frame-
 *                                   preview family loads" convention)
 *   - projectManagerNG.js          (patches the ProjectManager object)
 * Must load BEFORE:
 *   - bootstrapWiringNG.js's tail, since that's where
 *     ProjectManager.loadState() + ProjectManager.render() actually run,
 *     and render() -> renderLeftRail() calls renderVideoAnalysisBody()/
 *     renderFramePreview(), which must already be attached by then.
 *
 * Concretely: loads right after projectManagerNG.js, right before
 * bootstrapWiringNG.js, in indexNG.html's script list.
 *
 * ============================================================
 * KNOWN, UNCHANGED LIMITATIONS (carried over, not fixed here)
 * ============================================================
 *   - The Frame Preview canvas + hidden <audio> element is a DIFFERENT
 *     player from PlaybackModal's real <video> element -- two systems for
 *     two jobs (frame-exact scrubbing for analysis vs. natural watching),
 *     not a bug.
 *   - Both depend on a blob: object URL that dies on page reload; the
 *     popout button disables itself after a refresh until the video is
 *     re-picked (see renderFramePreview() below for the exact check).
 */

  // =========================================================================
  // DOM refs -- video-exclusive elements only (see file header for what
  // deliberately stayed in domRefsManifestNG.js instead).
  // =========================================================================

  // ---- Video Analysis section (left rail) ----
  const videoAudioEl = document.getElementById("ng-video-audio");
  const videoAnalysisBodyEl = document.getElementById("ng-video-analysis-body");
  const analysisStartInput = document.getElementById("ng-analysis-start-sec");
  const analysisEndInput = document.getElementById("ng-analysis-end-sec");
  const videoRangeRowEl = document.querySelector(".ng-video-range");

  // ---- Frame Preview section (now lives in the main stage, not the
  // left rail -- see this file's header for the move rationale) ----
  const ngMainVideoPreviewEl = document.getElementById("ng-main-video-preview");
  // Once a ring exists, the main stage switches to showing it -- this is
  // where the SAME preview block (reparented, not duplicated) moves to
  // so there's still a way to re-scrub/re-pick a frame and re-run
  // analysis, or load a different video entirely, without the player
  // just vanishing the moment you've analyzed once.
  const ngRailVideoPreviewSlotEl = document.getElementById("ng-rail-video-preview-slot");
  const frameCounterEl = document.getElementById("ng-frame-counter");
  const rewindBtn = document.getElementById("ng-btn-rewind-frame");
  const prevFrameBtn = document.getElementById("ng-btn-prev-frame");
  const playBtn = document.getElementById("ng-btn-play-frames");
  const stopBtn = document.getElementById("ng-btn-stop-frames");
  const nextFrameBtn = document.getElementById("ng-btn-next-frame");
  const startAnalysisBtn = document.getElementById("ng-btn-start-analysis");
  const popoutVideoBtn = document.getElementById("ng-btn-popout-video");

  // ---- Playback modal (pop-out, real <video> element) ----
  const playbackModalEl = document.getElementById("ng-playback-modal");
  const playbackModalTitleEl = document.getElementById("ng-playback-modal-title");
  const playbackModalCloseBtn = document.getElementById("ng-playback-modal-close");
  const playbackVideoEl = document.getElementById("ng-playback-video");
  const playbackPrevFrameBtn = document.getElementById("ng-playback-prev-frame");
  const playbackNextFrameBtn = document.getElementById("ng-playback-next-frame");

  // =========================================================================
  // PlaybackModal -- real <video>-element popout player. Was
  // playbackModalNG.js in full.
  // =========================================================================

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

  // =========================================================================
  // Frame-scrubber draw/visual helpers -- were in poseListAndFrameHelpersNG.js.
  // =========================================================================

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

  // Moves the SAME preview block (not a duplicate) between its two
  // possible homes, so there's exactly one canvas/one set of controls/
  // one set of event listeners regardless of where it's currently shown.
  // "main": prominent, in the main stage, for watch-and-judge before
  //   there's a ring yet to show there instead.
  // "rail": compact, back in the left rail's Video Analysis section, so
  //   there's still a way to re-scrub/re-pick a frame or load a
  //   different video once a ring exists and the main stage has
  //   switched over to showing it.
  // "hidden": no video loaded for this project -- nothing to show.
  // appendChild() on a node already in that parent is a harmless no-op
  // reorder, so this is safe to call on every render().
  function placeVideoPreview(location) {
    if (location === "hidden") {
      ngMainVideoPreviewEl.style.display = "none";
      return;
    }
    if (location === "rail") {
      ngRailVideoPreviewSlotEl.appendChild(ngMainVideoPreviewEl);
      ngMainVideoPreviewEl.classList.add("ng-video-preview-compact");
    } else {
      mainEl.appendChild(ngMainVideoPreviewEl);
      ngMainVideoPreviewEl.classList.remove("ng-video-preview-compact");
    }
    ngMainVideoPreviewEl.style.display = "flex";
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

  // =========================================================================
  // CharacterProject.prototype patches -- were method bodies inside the
  // `class CharacterProject` literal in characterProjectNG.js. Requires
  // characterProjectNG.js to have run first (see load-order note above).
  // =========================================================================

  CharacterProject.prototype.stopPlayIfRunning = function () {
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
  };

  // ---- video ingest (ported from the previous NG pass, unchanged) ----
  CharacterProject.prototype.loadVideo = async function (file) {
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
  };

  CharacterProject.prototype.frameTime = function (frameIdx) {
    const fps = this.video.fps > 0 ? this.video.fps : 24;
    return Math.max(0, (frameIdx - 1) / fps);
  };

  CharacterProject.prototype.frameFromTime = function (timeSeconds) {
    const fps = this.video.fps > 0 ? this.video.fps : 24;
    return Math.min(this.video.totalFrames, Math.max(1, Math.floor(timeSeconds * fps) + 1));
  };

  CharacterProject.prototype.syncAudioToFrame = function (frameIdx) {
    if (!videoAudioEl.src || videoAudioEl.dataset.objectUrl !== this.video.objectUrl) return;
    try {
      videoAudioEl.currentTime = this.frameTime(frameIdx);
    } catch (e) {
      // seeking before metadata is ready can throw -- harmless, ignore
    }
  };

  CharacterProject.prototype.stepTo = function (frameNo) {
    const clamped = Math.max(1, Math.min(this.video.totalFrames, frameNo));
    this.video.currentFrame = clamped;
    if (this.isActive) {
      frameCounterEl.textContent = "Frame: " + clamped + " / " + this.video.totalFrames;
      drawFrame(this, clamped);
    }
    ProjectManager.saveState();
  };

  CharacterProject.prototype.stepAndSyncAudio = function (frameNo) {
    this.stopPlayIfRunning();
    if (this.isActive) setPlayingVisual(false);
    this.stepTo(frameNo);
    this.syncAudioToFrame(this.video.currentFrame);
  };

  CharacterProject.prototype.togglePlay = function () {
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
  };

  CharacterProject.prototype.playFramesWithoutAudio = function () {
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
  };

  // =========================================================================
  // ProjectManager patches -- were methods inside the ProjectManager object
  // literal in projectManagerNG.js. Requires projectManagerNG.js to have
  // run first (see load-order note above).
  // =========================================================================

  ProjectManager.renderVideoAnalysisBody = function (project) {
    if (project.task !== "video") {
      videoAnalysisBodyEl.innerHTML = "";
      return;
    }
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
  };

  ProjectManager.renderFramePreview = function (project) {
    if (!project.video) {
      if (project.staticPreviewFrame) {
        // folder/image-set job with a chart-click preview already shown --
        // redraw it rather than resetting to the "no video" placeholder,
        // since a full render() (selection toggles, tab switches, etc.)
        // must not silently wipe out what the person just clicked to view.
        showStaticFramePreviewNG(project, project.staticPreviewFrame);
        return;
      }
      previewHintEl.textContent = project.task === "folderzip"
        ? "No folder/zip analyzed yet for this project."
        : "No video loaded for this project yet.";
      previewCanvasEl.style.display = "none";
      previewControlsScrollEl.style.display = "none";
      return;
    }

    if (!project.video.previewId) {
      // previewId is server-memory-only (see _preview_jobs_ng in
      // routes/videoNG.py) and doesn't survive a server restart, same
      // underlying problem as objectUrl not surviving a page reload --
      // scrubbing/drawFrame() would just 404 against /api/ng/preview-frame/
      // if we tried. Show the same "needs re-loading" message the popout
      // button below already uses for the objectUrl case, rather than
      // silently failing frame fetches.
      previewHintEl.textContent = "Video needs to be re-loaded after a page reload before it can be scrubbed.";
      previewCanvasEl.style.display = "none";
      previewControlsScrollEl.style.display = "none";
      popoutVideoBtn.disabled = true;
      popoutVideoBtn.title = "Video needs to be re-loaded after a page reload before it can be popped out";
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
  };

  // =========================================================================
  // Wiring -- were bare top-level addEventListener() calls inside
  // bootstrapWiringNG.js's ~60-call wiring tail.
  // =========================================================================

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

// ---- exposed for other modules to call ----
window.VideoNG = { PlaybackModal, drawFrame, setPlayingVisual, videoPickerButton };
window.PlaybackModalNG = PlaybackModal; // kept for continuity with the old playbackModalNG.js export name
