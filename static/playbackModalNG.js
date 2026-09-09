/**
 * playbackModalNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 2555-2615.
 * This is the "VideoPopout" module from appNG-module-contracts.md (there
 * called "currently PlaybackModal").
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * A real <video> element with native audio -- mechanically DIFFERENT from
 * FrameScrubber (see drawFrame in poseListAndFrameHelpersNG.js): two
 * separate players for two separate jobs, confirmed by the source itself
 * (project.stopPlayIfRunning() is called from open() specifically to stop
 * the OTHER player before this one starts, so they don't both play audio
 * at once).
 *
 * Handles two sources:
 *   - open(project)      -- the raw source clip (project.video.objectUrl),
 *                            honoring the same start/end-sec range Analysis
 *                            Settings uses, so you can preview a bounded
 *                            range without running a full analysis first
 *   - openBuild(project) -- the reconstructed rejected-frames-blanked
 *                            build (project.playback.url)
 *
 * ============================================================
 * CONFIRMED COUPLING (exactly as flagged in appNG-module-contracts.md)
 * ============================================================
 *   - open()/openBuild() take a WHOLE `project` object and pull
 *     project.video.objectUrl, project.playback.url, project.name,
 *     project.video.fps directly out of it, rather than being handed
 *     plain values.
 *   - stepFrame() re-looks-up the project from ProjectManager.projects
 *     BY ID just to read its fps -- this.projectId is stored specifically
 *     so this lookup is possible later; the object itself is never
 *     retained.
 *   - Writes directly to module-level DOM element globals defined
 *     elsewhere in appNG.js: playbackModalTitleEl, playbackVideoEl,
 *     playbackModalEl -- not passed in, not owned by this object.
 *   - open() also calls setPlayingVisual(false) (defined in
 *     poseListAndFrameHelpersNG.js) and project.stopPlayIfRunning()
 *     (a CharacterProject method -- see characterProjectNG.js) directly.
 *
 * The contracts doc's suggested fix (not applied here, per "extract as-is"
 * instruction): a thin adapter at each of the two call sites (in
 * projectManagerNG.js / appNG.js) that extracts plain values (url, fps,
 * title, rangeStart/End) BEFORE handing off to this player, rather than
 * this player reaching into `project` itself. That would make the module's
 * real entry contract: `{ src, fps, title, rangeStart?, rangeEnd? }` in,
 * play-state-change events + stepFrame(delta)/play/pause/seek out -- as
 * already noted in appNG-module-contracts.md.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after playbackModalTitleEl/playbackVideoEl/playbackModalEl
 *      exist, after ProjectManager exists (for stepFrame's lookup), and
 *      after setPlayingVisual is defined.
 *   2. Remove the corresponding object literal from appNG.js.
 *   3. If/when doing the real fix: replace the `project` parameter on
 *      open()/openBuild() with the plain-value shape above, and move the
 *      ProjectManager.projects lookup in stepFrame() to the caller.
 */

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

// ---- exposed for appNG.js's event wiring to call ----
window.PlaybackModalNG = PlaybackModal;

