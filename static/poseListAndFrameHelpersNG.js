/**
 * poseListAndFrameHelpersNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no
 * logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 3031-3186.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * Two related but distinct concerns bundled in this extraction (kept
 * together here since they're adjacent in source and small; a future
 * pass could split them):
 *
 * 1. POSE-LIST VIEW -- the yaw/pitch/roll/blur ring-sort mode's horizontal
 *    scrubber-strip alternative to the radial ring, ported from
 *    viz-render.js's renderPoseList/setupPoseListScrubber. This is the
 *    "PoseListView" the contracts doc flagged as one of renderVideoStage's
 *    four unrelated jobs (mode branch: pose-list vs. ring view) -- THIS
 *    part is already correctly separated into its own function per the
 *    doc's note ("already semi-separate via renderPoseListNG, just needs
 *    the branch removed from renderVideoStage" -- that removal is NOT
 *    done here, renderVideoStage in projectManagerNG.js still has its own
 *    inline copy of the branching logic that calls out to this).
 *
 *      renderPoseListNG(project, anchorUrl, anchorLabel, combined,
 *        onToggle, isSelectedFn, onRecenter) -- note this one's contract
 *        is ALREADY mostly parameter-based rather than reaching into
 *        `project` for everything -- onToggle/isSelectedFn/onRecenter are
 *        callbacks, not direct project mutation. Closer to the toolkit-
 *        tier ideal than most of what's been extracted so far. Uses
 *        showHoverPreview/hideHoverPreview (see ringDisplayHelpersNG.js)
 *        on hover.
 *
 *      setupPoseListScrubberNG(metric, withMetric) -- the "dock-style
 *        lens/magnify" hover effect specific to the pose-list strip
 *        (distinct from attachNgLensEffect / wireNgPoseListLensEffect,
 *        which live in appNG.js's startup-wiring tail, NOT extracted in
 *        this pass -- see the note in ProjectManagerNG.js's header about
 *        the bootstrap-wiring tail being its own future extraction).
 *
 *      flashHighlightNG(frame, assetId) -- a brief visual flash/highlight,
 *        likely used for "jump to this frame" navigation feedback.
 *
 * 2. FRAME-SCRUBBER PIECES -- this is NOT the whole FrameScrubber module
 *    from appNG-module-contracts.md (renderFramePreview lives in
 *    projectManagerNG.js, already extracted, and is the actual owner of
 *    frame-by-frame stepping state) -- just its low-level draw/visual
 *    helpers that happened to be defined near the pose-list code:
 *
 *      drawFrame(project, frameNo) -- the actual canvas draw call.
 *        Referenced from BOTH characterProjectNG.js (playFramesWithoutAudio)
 *        AND projectManagerNG.js (renderFramePreview) -- this is a real
 *        shared dependency of both already-extracted files, confirming
 *        their own "TO MAKE THIS ACTUALLY RUN" notes about needing
 *        drawFrame defined first.
 *
 *      setPlayingVisual(isPlaying) -- toggles the play button's active
 *        visual state. Referenced from playbackModalNG.js's open()
 *        (confirming THAT file's coupling note too) and elsewhere.
 *
 *      placeholder(text) / videoPickerButton(project, label) -- small
 *        DOM-construction helpers, likely used by renderStage/
 *        renderVideoStage's empty-state rendering (projectManagerNG.js).
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 *   - showHoverPreview/hideHoverPreview -- ringDisplayHelpersNG.js
 *   - Module-level DOM globals: playBtn (setPlayingVisual), and whatever
 *     canvas/context drawFrame writes to (see full body below)
 *   - project.* fields read directly in drawFrame/renderPoseListNG (project
 *     passed in as a parameter, at least, so this part of the contract is
 *     explicit even if not fully generic)
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after ringDisplayHelpersNG.js (for hover preview) and after
 *      playBtn + drawFrame's canvas element exist.
 *   2. characterProjectNG.js and projectManagerNG.js both need this file
 *      loaded before them (they call drawFrame) -- update those two
 *      files' own "TO MAKE THIS ACTUALLY RUN" notes accordingly once
 *      real wiring begins.
 *   3. Remove the corresponding lines from appNG.js.
 */

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

  // videoPickerButton, drawFrame, and setPlayingVisual moved to videoNG.js
  // -- all three are video-exclusive, unlike placeholder() above which is
  // a generic DOM helper used by folder/immich empty-states too.

// ---- exposed for other modules to call ----
window.poseListAndFrameHelpersNG = {
  renderPoseListNG, setupPoseListScrubberNG, flashHighlightNG, placeholder,
};

