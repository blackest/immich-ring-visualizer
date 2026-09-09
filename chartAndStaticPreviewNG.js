/**
 * chartAndStaticPreviewNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no
 * logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 2617-2736.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 *   renderNGChart(project, wrap) -- the match-confidence sparkline chart
 *     with false-color shot-scale strip beneath it, ported from
 *     media-ingest.js's renderSimSparkline -- reads from THIS project's
 *     own job/results (passed in) rather than a module-level global,
 *     which is the correct pattern (contrast with pickersNG.js/
 *     playbackModalNG.js's module-level DOM-global habit -- this
 *     function's chart canvas is presumably built fresh into `wrap` each
 *     call rather than reusing a captured element).
 *
 *   showStaticFramePreviewNG(project, r) -- folder/image-set jobs only:
 *     a chart-click result row preview, since there's no live video
 *     decode to scrub for a folder job (see characterProjectNG.js's
 *     staticPreviewFrame field and renderFramePreview() in
 *     projectManagerNG.js, which branches between this and the live
 *     drawFrame() path depending on sourceType).
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 *   - Uses ringDisplayHelpersNG.js's shotScaleForNG()/SHOT_SCALE_BANDS
 *     for the shot-scale strip's false-color classification -- same
 *     classifier the scale picker uses independently (see
 *     ringDisplayHelpersNG.js's note on this).
 *   - showStaticFramePreviewNG likely writes to module-level DOM globals
 *     for the preview canvas/frame counter -- see full body below for
 *     exact element names.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after ringDisplayHelpersNG.js.
 *   2. Load after whatever DOM elements showStaticFramePreviewNG writes
 *      to exist.
 *   3. Remove the corresponding lines from appNG.js.
 */

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

  // ---- Pose Picker + Shot Scale Picker, ported from viz-render.js's

// ---- exposed for projectManagerNG.js's renderFramePreview/renderVideoStage to call ----
window.chartAndStaticPreviewNG = { renderNGChart, showStaticFramePreviewNG };

