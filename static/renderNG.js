// renderNG.js
// Rendering logic for charts, previews, and per-character UI.

(function () {
  "use strict";

  window.AppNG = window.AppNG || {};

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
    const padL = 4,
      padR = 4,
      padT = 8,
      padB = 4;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const minFrame = sorted[0].frame;
    const maxFrame = sorted[sorted.length - 1].frame;
    const frameSpan = Math.max(1, maxFrame - minFrame);

    const xFor = (frame) => padL + ((frame - minFrame) / frameSpan) * plotW;
    const yFor = (sim) => padT + (1 - Math.max(0, Math.min(1, sim))) * plotH;

    const blurVals = sorted.map((r) =>
      typeof r.blur === "number" ? r.blur : 0,
    );
    const blurMax = Math.max(1, blurThreshold * 1.4, ...blurVals) * 1.05;
    const yForBlur = (blur) => padT + (1 - Math.max(0, blur) / blurMax) * plotH;

    const linePoints = sorted
      .map((r) => `${xFor(r.frame).toFixed(1)},${yFor(r.sim).toFixed(1)}`)
      .join(" ");
    const thresholdY = yFor(threshold).toFixed(1);
    const blurThresholdY = yForBlur(blurThreshold).toFixed(1);

    const dots = sorted
      .map((r) => {
        const cx = xFor(r.frame).toFixed(1);
        const cy = yFor(r.sim).toFixed(1);
        const color = r.passed ? "#7cc4ff" : "#d9534f";
        return (
          `<circle cx="${cx}" cy="${cy}" r="7" fill="transparent" data-frame="${r.frame}" class="ng-spark-hit" style="cursor:pointer;"></circle>` +
          `<circle cx="${cx}" cy="${cy}" r="2" fill="${color}" style="pointer-events:none;"></circle>`
        );
      })
      .join("");

    const stripH = 14;
    const segW = Math.max(1, plotW / sorted.length);
    const stripSegs = sorted
      .map((r) => {
        const scale = window.AppNG.shared.shotScaleForNG(r);
        const x = xFor(r.frame).toFixed(1);
        const color = scale ? scale.color : "#2a2a32";
        const title = scale
          ? `${scale.label} (${(scale.pct * 100).toFixed(0)}%)`
          : "no scale data";
        return `<rect x="${(x - segW / 2).toFixed(1)}" y="0" width="${segW.toFixed(1)}" height="${stripH}" fill="${color}" data-frame="${r.frame}" class="ng-spark-hit ng-strip-seg" style="cursor:pointer;"><title>frame ${r.frame} — ${title}</title></rect>`;
      })
      .join("");

    const legend = window.AppNG.shared.SHOT_SCALE_BANDS.map(
      (b) =>
        `<span class="ng-shot-scale-legend-item"><span class="ng-shot-scale-swatch" style="background:${b.color};"></span>${b.label}</span>`,
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

  function showStaticFramePreviewNG(project, r) {
    if (!r) return;
    project.staticPreviewFrame = r;
    if (!project.isActive) return;
    if (typeof previewHintEl !== "undefined") {
      previewHintEl.textContent =
        r.origName || r.filename || `frame ${r.frame}`;
    }
    if (!r.frameId) {
      if (typeof previewCanvasEl !== "undefined")
        previewCanvasEl.style.display = "none";
      if (typeof previewControlsScrollEl !== "undefined")
        previewControlsScrollEl.style.display = "none";
      if (typeof previewHintEl !== "undefined") {
        previewHintEl.textContent = `Frame ${r.frame} was rejected (no face / didn't pass thresholds) — no stored image to show.`;
      }
      return;
    }
    const img = new Image();
    img.onload = () => {
      const active =
        window.AppNG.manager && window.AppNG.manager.getActive
          ? window.AppNG.manager.getActive()
          : null;
      if (!active || active.id !== project.id) return;
      if (typeof previewCanvasEl !== "undefined") {
        previewCanvasEl.width = img.naturalWidth;
        previewCanvasEl.height = img.naturalHeight;
        previewCanvasEl.getContext("2d").drawImage(img, 0, 0);
        previewCanvasEl.style.display = "";
      }
      if (typeof previewControlsScrollEl !== "undefined") {
        previewControlsScrollEl.style.display = "none";
      }
    };
    img.onerror = () => {
      if (typeof previewHintEl !== "undefined") {
        previewHintEl.textContent = `Could not load stored image for frame ${r.frame}`;
      }
    };
    img.src = `/api/ng/framefile/${r.frameId}?t=${Date.now()}`;
  }

  function showHoverPreview(r) {
    if (typeof hoverImg === "undefined" || typeof hoverPanel === "undefined")
      return;
    clearTimeout(window.AppNG.hoverTimer);
    window.AppNG.hoverTimer = setTimeout(() => {
      hoverImg.src = window.AppNG.shared.thumbUrlFor(r);
      const pctText =
        typeof r.similarity === "number"
          ? `${(r.similarity * 100).toFixed(1)}%`
          : "";
      let poseText = "";
      if (
        r.pitch !== undefined &&
        r.yaw !== undefined &&
        r.roll !== undefined &&
        r.pitch !== null
      ) {
        poseText = `<br>pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
        if (typeof r.blur === "number")
          poseText += ` &middot; sharpness: ${r.blur.toFixed(0)}`;
        if (typeof r.vertFillPct === "number")
          poseText += ` &middot; face: ${(r.vertFillPct * 100).toFixed(0)}% frame height`;
      }
      if (typeof hoverCaption !== "undefined") {
        hoverCaption.innerHTML = `${r.filename}${pctText ? ` &mdash; ${pctText}` : ""}${poseText}`;
      }
      hoverPanel.classList.add("active");
    }, 80);
  }

  function hideHoverPreview() {
    if (typeof hoverPanel === "undefined") return;
    clearTimeout(window.AppNG.hoverTimer);
    hoverPanel.classList.remove("active");
  }

  function renderPoseListNG(
    project,
    anchorUrl,
    anchorLabel,
    combined,
    onToggle,
    isSelectedFn,
    onRecenter,
  ) {
    if (typeof poseListViewEl === "undefined") return;
    poseListViewEl.innerHTML = "";

    const anchorWrap = document.createElement("div");
    anchorWrap.className = "ng-pose-list-anchor";
    anchorWrap.innerHTML = `<img src="${anchorUrl}"><div class="ng-plabel">${anchorLabel}</div>`;
    poseListViewEl.appendChild(anchorWrap);

    const metric = project.ringSortMetric;
    const withMetric = combined.filter((r) => typeof r[metric] === "number");
    const withoutMetric = combined.filter((r) => typeof r[metric] !== "number");
    withMetric.sort((a, b) => a[metric] - b[metric]);

    const PITCH_PX_PER_DEG = 1.6;
    const PITCH_CLAMP_DEG = 40;

    withMetric.forEach((r) => {
      const item = document.createElement("div");
      const selected = isSelectedFn(r);
      item.className =
        "ng-pose-list-item" + (selected ? " ng-pose-list-item-selected" : "");
      if (r.assetId) item.dataset.assetId = r.assetId;
      if (r.frame !== undefined) item.dataset.frame = r.frame;
      const unit = metric === "blur" ? "" : "°";
      const label = metric === "blur" ? "sharp" : metric;
      item.innerHTML = `<img src="${window.AppNG.shared.thumbUrlFor(r)}" loading="lazy"><div class="ng-plabel">${label}: ${r[metric].toFixed(1)}${unit}</div>`;

      if (typeof r.pitch === "number") {
        const clamped = Math.max(
          -PITCH_CLAMP_DEG,
          Math.min(PITCH_CLAMP_DEG, r.pitch),
        );
        const offsetPx = -clamped * PITCH_PX_PER_DEG;
        item.style.transform = `translateY(${offsetPx}px)`;
      }

      let clickTimer = null;
      item.onclick = () => {
        if (!onRecenter || !r.assetId) return;
        clearTimeout(clickTimer);
        clickTimer = setTimeout(() => onRecenter(r), 220);
      };
      item.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearTimeout(clickTimer);
        onToggle(r);
        item.classList.toggle("ng-pose-list-item-selected");
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
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
    if (typeof poseListViewEl === "undefined") return;
    if (typeof poseScrubLeftEl !== "undefined") {
      poseScrubLeftEl.textContent = withMetric.length
        ? `${metric}: ${withMetric[0][metric].toFixed(1)}°`
        : "";
    }
    if (typeof poseScrubRightEl !== "undefined") {
      poseScrubRightEl.textContent = withMetric.length
        ? `${metric}: ${withMetric[withMetric.length - 1][metric].toFixed(1)}°`
        : "";
    }

    const maxScroll = () =>
      Math.max(1, poseListViewEl.scrollWidth - poseListViewEl.clientWidth);
    let syncingFromScroll = false;
    if (typeof poseScrubSliderEl !== "undefined") {
      poseScrubSliderEl.value = 0;
      poseScrubSliderEl.oninput = () => {
        syncingFromScroll = true;
        poseListViewEl.scrollLeft =
          (parseFloat(poseScrubSliderEl.value) / 1000) * maxScroll();
        syncingFromScroll = false;
      };
    }
    poseListViewEl.onscroll = () => {
      if (syncingFromScroll || typeof poseScrubSliderEl === "undefined") return;
      poseScrubSliderEl.value = Math.round(
        (poseListViewEl.scrollLeft / maxScroll()) * 1000,
      );
    };
    poseListViewEl.onwheel = (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        poseListViewEl.scrollLeft += e.deltaY;
      }
    };
  }

  function flashHighlightNG(frame, assetId) {
    const selector =
      assetId != null
        ? `.ng-pose-list-item[data-asset-id="${CSS.escape(String(assetId))}"], .ng-list-row[data-asset-id="${CSS.escape(String(assetId))}"]`
        : `.ng-pose-list-item[data-frame="${CSS.escape(String(frame))}"], .ng-list-row[data-frame="${CSS.escape(String(frame))}"]`;
    const el = document.querySelector(selector);
    if (!el) return;
    el.scrollIntoView({
      behavior: "smooth",
      inline: "center",
      block: "nearest",
    });
    const prevShadow = el.style.boxShadow;
    let flashes = 0;
    const flashInterval = setInterval(() => {
      el.style.boxShadow = flashes % 2 === 0 ? "0 0 0 4px #7cc4ff" : prevShadow;
      flashes++;
      if (flashes > 5) {
        clearInterval(flashInterval);
        el.style.boxShadow = prevShadow;
      }
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
    if (!project || !project.video || typeof previewCanvasEl === "undefined")
      return;
    const img = new Image();
    img.onload = () => {
      const active =
        window.AppNG.manager && window.AppNG.manager.getActive
          ? window.AppNG.manager.getActive()
          : null;
      if (!active || active.id !== project.id) return;
      previewCanvasEl.width = img.naturalWidth;
      previewCanvasEl.height = img.naturalHeight;
      previewCanvasEl.getContext("2d").drawImage(img, 0, 0);
    };
    img.src =
      "/api/ng/preview-frame/" +
      project.video.previewId +
      "/" +
      frameNo +
      "?t=" +
      Date.now();
  }

  function setPlayingVisual(isPlaying) {
    if (typeof playBtn !== "undefined") {
      playBtn.classList.toggle("ng-video-play-active", isPlaying);
    }
  }

  function wireLeftRailChrome() {
    if (typeof leftRailEl === "undefined") return;
    if (typeof leftRailBodyEl !== "undefined") {
      const savedWidth = localStorage.getItem("immichRingNG:leftPanelWidth");
      if (savedWidth) leftRailEl.style.width = savedWidth + "px";
    }

    const savedCollapsedAll =
      localStorage.getItem("immichRingNG:leftPanelCollapsedAll") === "1";
    if (savedCollapsedAll && typeof collapseAllBtn !== "undefined") {
      leftRailEl.classList.add("collapsed-all");
      collapseAllBtn.textContent = "▸";
    }

    if (typeof controlsPaneEl !== "undefined") {
      const savedControlsPct = parseFloat(
        localStorage.getItem("immichRingNG:leftControlsPct"),
      );
      if (!Number.isNaN(savedControlsPct)) {
        controlsPaneEl.style.flexBasis =
          Math.max(24, Math.min(76, savedControlsPct)) + "%";
      }
    }

    document.querySelectorAll(".panel-section").forEach((sec) => {
      const key = "immichRingNG:section:" + sec.dataset.section;
      const saved = localStorage.getItem(key);
      if (saved === "1") sec.classList.add("expanded");
      if (saved === "0") sec.classList.remove("expanded");
      const header = sec.querySelector(".panel-section-header");
      if (header) {
        header.addEventListener("click", () => {
          sec.classList.toggle("expanded");
          localStorage.setItem(
            key,
            sec.classList.contains("expanded") ? "1" : "0",
          );
        });
      }
    });

    if (typeof collapseAllBtn !== "undefined") {
      collapseAllBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const collapsed = leftRailEl.classList.toggle("collapsed-all");
        collapseAllBtn.textContent = collapsed ? "▸" : "◄";
        localStorage.setItem(
          "immichRingNG:leftPanelCollapsedAll",
          collapsed ? "1" : "0",
        );
      });
    }

    let splitDragging = false;
    if (typeof splitterEl !== "undefined") {
      splitterEl.addEventListener("mousedown", (e) => {
        if (leftRailEl.classList.contains("collapsed-all")) return;
        splitDragging = true;
        splitterEl.classList.add("dragging");
        document.body.style.cursor = "ns-resize";
        e.preventDefault();
      });
    }
    document.addEventListener("mousemove", (e) => {
      if (!splitDragging) return;
      if (typeof leftRailBodyEl === "undefined") return;
      const rect = leftRailBodyEl.getBoundingClientRect();
      const pct = ((e.clientY - rect.top) / rect.height) * 100;
      if (typeof controlsPaneEl !== "undefined") {
        controlsPaneEl.style.flexBasis = Math.max(24, Math.min(76, pct)) + "%";
      }
    });
    document.addEventListener("mouseup", () => {
      if (!splitDragging) return;
      splitDragging = false;
      if (typeof splitterEl !== "undefined")
        splitterEl.classList.remove("dragging");
      document.body.style.cursor = "";
      if (
        typeof leftRailBodyEl !== "undefined" &&
        typeof controlsPaneEl !== "undefined"
      ) {
        const rect = leftRailBodyEl.getBoundingClientRect();
        const controlsRect = controlsPaneEl.getBoundingClientRect();
        localStorage.setItem(
          "immichRingNG:leftControlsPct",
          Math.round((controlsRect.height / rect.height) * 100),
        );
      }
    });

    let widthDragging = false;
    let startX = 0;
    let startWidth = 0;
    if (typeof resizeHandleEl !== "undefined") {
      resizeHandleEl.addEventListener("mousedown", (e) => {
        if (leftRailEl.classList.contains("collapsed-all")) return;
        widthDragging = true;
        resizeHandleEl.classList.add("dragging");
        startX = e.clientX;
        startWidth = leftRailEl.getBoundingClientRect().width;
        document.body.style.cursor = "ew-resize";
        e.preventDefault();
      });
    }
    document.addEventListener("mousemove", (e) => {
      if (!widthDragging) return;
      const newWidth = Math.max(
        240,
        Math.min(720, startWidth + (e.clientX - startX)),
      );
      leftRailEl.style.width = newWidth + "px";
    });
    document.addEventListener("mouseup", () => {
      if (!widthDragging) return;
      widthDragging = false;
      if (typeof resizeHandleEl !== "undefined")
        resizeHandleEl.classList.remove("dragging");
      document.body.style.cursor = "";
      localStorage.setItem(
        "immichRingNG:leftPanelWidth",
        Math.round(leftRailEl.getBoundingClientRect().width),
      );
    });
  }

  function wireRightSidebarListToggle() {
    if (
      typeof sidebarEl === "undefined" ||
      typeof toggleListBtn === "undefined"
    )
      return;

    function applyHidden(hidden) {
      sidebarEl.classList.toggle("ng-list-hidden", hidden);
      toggleListBtn.textContent = hidden ? "Show" : "Hide";
      toggleListBtn.title = hidden
        ? "Show ranked match list"
        : "Hide ranked match list";
    }

    applyHidden(localStorage.getItem("immichRingNG:listHidden") === "1");
    toggleListBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const hidden = !sidebarEl.classList.contains("ng-list-hidden");
      applyHidden(hidden);
      localStorage.setItem("immichRingNG:listHidden", hidden ? "1" : "0");
    });
  }

  const NG_FISHEYE_RADIUS = 160;
  const NG_FISHEYE_MAX_SCALE = 2.0;
  const NG_FISHEYE_MAX_PUSH = 46;
  let ngFisheyeRafPending = false;
  let ngFisheyeLastMouse = null;

  function applyNgFisheye(mx, my) {
    if (typeof stageEl === "undefined") return;
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
    if (typeof stageEl === "undefined") return;
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

  function attachNgLensEffect(
    container,
    itemSelector,
    { radius = 140, maxScale = 1.6 } = {},
  ) {
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

  function wireNgPoseListLensEffect() {
    if (typeof poseListViewEl !== "undefined") {
      attachNgLensEffect(poseListViewEl, ".ng-pose-list-item", {
        radius: 140,
        maxScale: 1.6,
      });
    }
  }

  function wirePersonClustersNG() {
    const loadBtn = document.getElementById("ng-pc-load-btn");
    const minFacesInput = document.getElementById("ng-pc-min-faces");
    const tightThresholdInput = document.getElementById(
      "ng-pc-tight-threshold",
    );
    const statusEl = document.getElementById("ng-pc-status");
    const listEl = document.getElementById("ng-pc-cluster-list");
    const gridEl = document.getElementById("ng-pc-thumb-grid");
    if (
      !loadBtn ||
      !minFacesInput ||
      !tightThresholdInput ||
      !statusEl ||
      !listEl ||
      !gridEl
    )
      return;

    let clusterRows = [];
    const renderRows = () => {
      const cutoff = parseFloat(tightThresholdInput.value) || 0.85;
      listEl.innerHTML = "";
      clusterRows.forEach((person) => {
        const row = document.createElement("div");
        row.className = "ng-pc-cluster-row";
        const pct = (person.avgSim * 100).toFixed(1);
        row.innerHTML = `<div class="ng-pc-cluster-info"><div class="ng-pc-cluster-fname">${person.name}${person.avgSim > cutoff ? ' <span class="ng-pc-tight-flag">● tight cluster</span>' : ""}</div><div class="ng-pc-simbar-track"><div class="ng-pc-simbar-fill" style="width:${pct}%"></div></div></div><div class="ng-pc-simpct">${pct}%</div><div class="ng-pc-facecount">${person.faceCount}</div>`;
        row.addEventListener("click", () =>
          loadPersonAssets(person.personId, row),
        );
        listEl.appendChild(row);
      });
    };
    const loadPersonAssets = async (personId, rowEl) => {
      listEl
        .querySelectorAll(".ng-pc-cluster-row")
        .forEach((row) => row.classList.remove("ng-pc-row-active"));
      rowEl.classList.add("ng-pc-row-active");
      gridEl.textContent = "Loading...";
      try {
        const response = await fetch(
          `/api/ng/person-assets/${personId}?limit=200`,
        );
        const assets = await response.json();
        if (!response.ok || assets.error)
          throw new Error(assets.error || response.status);
        gridEl.innerHTML = "";
        assets.forEach((asset) => {
          const cell = document.createElement("div");
          const img = document.createElement("img");
          img.src = `/api/ng/thumb/${asset.assetId}`;
          img.loading = "lazy";
          img.title = asset.filename;
          cell.appendChild(img);
          let clickTimer = null;
          cell.addEventListener("click", () => {
            clearTimeout(clickTimer);
            clickTimer = setTimeout(() => {
              const active = window.AppNG.manager?.getActive();
              if (!active) return;
              active.toggleAssetSelection(asset.assetId);
              window.AppNG.manager.render();
            }, 220);
          });
          cell.addEventListener("dblclick", (event) => {
            event.preventDefault();
            clearTimeout(clickTimer);
            const active = window.AppNG.manager?.getActive();
            if (!active) return;
            active.task = "immich";
            active.recenterImmich(asset.assetId, asset.filename);
            window.AppNG.manager.render();
          });
          gridEl.appendChild(cell);
        });
      } catch (error) {
        gridEl.textContent = `Failed to load assets: ${error.message}`;
      }
    };
    tightThresholdInput.addEventListener("change", renderRows);
    loadBtn.addEventListener("click", async () => {
      statusEl.textContent = "Loading...";
      listEl.innerHTML = "";
      gridEl.innerHTML = "";
      try {
        const minFaces = parseInt(minFacesInput.value, 10) || 5;
        const response = await fetch(
          `/api/ng/person-clusters?minFaces=${minFaces}&limit=40`,
        );
        clusterRows = await response.json();
        if (!response.ok || clusterRows.error)
          throw new Error(clusterRows.error || response.status);
        statusEl.textContent = `${clusterRows.length} person${clusterRows.length === 1 ? "" : "s"} (sorted tightest-cluster first)`;
        renderRows();
      } catch (error) {
        statusEl.textContent = `Error: ${error.message}`;
      }
    });
  }

  const PlaybackModal = {
    projectId: null,
    kind: null,
    rangeStartSec: null,
    rangeEndSec: null,

    open(project) {
      if (!project.video || !project.video.objectUrl) return;
      project.stopPlayIfRunning();
      if (project.isActive) setPlayingVisual(false);
      this.projectId = project.id;
      this.kind = "raw";
      this.rangeStartSec =
        project.video.rangeStartSec != null
          ? project.video.rangeStartSec
          : null;
      this.rangeEndSec =
        project.video.rangeEndSec != null ? project.video.rangeEndSec : null;
      if (typeof playbackModalTitleEl !== "undefined") {
        playbackModalTitleEl.textContent =
          this.rangeStartSec != null || this.rangeEndSec != null
            ? `${project.name} — source video (${this.rangeStartSec ?? 0}s–${this.rangeEndSec ?? "end"})`
            : `${project.name} — source video`;
      }
      if (typeof playbackVideoEl !== "undefined") {
        playbackVideoEl.src = project.video.objectUrl;
      }
      if (typeof playbackModalEl !== "undefined") {
        playbackModalEl.style.display = "flex";
      }
    },

    openBuild(project) {
      if (!project.playback) return;
      this.projectId = project.id;
      this.kind = "reconstructed";
      this.rangeStartSec = null;
      this.rangeEndSec = null;
      if (typeof playbackModalTitleEl !== "undefined") {
        playbackModalTitleEl.textContent = `${project.name} — playback (rejected frames blanked)`;
      }
      if (typeof playbackVideoEl !== "undefined") {
        playbackVideoEl.src = project.playback.url;
      }
      if (typeof playbackModalEl !== "undefined") {
        playbackModalEl.style.display = "flex";
      }
    },

    close() {
      this.projectId = null;
      this.kind = null;
      this.rangeStartSec = null;
      this.rangeEndSec = null;
      if (typeof playbackModalEl !== "undefined")
        playbackModalEl.style.display = "none";
      if (typeof playbackVideoEl !== "undefined") {
        playbackVideoEl.pause();
        playbackVideoEl.removeAttribute("src");
        playbackVideoEl.load();
      }
    },

    stepFrame(delta) {
      if (
        typeof playbackModalEl === "undefined" ||
        playbackModalEl.style.display === "none"
      )
        return;
      const project =
        window.AppNG.manager &&
        typeof window.AppNG.manager.getActive === "function"
          ? null
          : null;
      const activeProject =
        window.AppNG.manager && window.AppNG.manager.projects
          ? window.AppNG.manager.projects.get
            ? Array.from(window.AppNG.manager.projects.values()).find(
                (p) => p.id === this.projectId,
              )
            : window.AppNG.manager.projects.find((p) => p.id === this.projectId)
          : null;
      const fps =
        this.kind === "raw"
          ? (activeProject && activeProject.video && activeProject.video.fps) ||
            24
          : (activeProject &&
              activeProject.playback &&
              activeProject.playback.fps) ||
            24;
      if (typeof playbackVideoEl !== "undefined") {
        playbackVideoEl.pause();
        const step = delta / fps;
        playbackVideoEl.currentTime = Math.max(
          0,
          Math.min(
            playbackVideoEl.duration || Infinity,
            playbackVideoEl.currentTime + step,
          ),
        );
      }
    },
  };

  function wireNGGlobalControls() {
    if (typeof ringScaleInput !== "undefined") {
      ringScaleInput.addEventListener("input", (e) => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.ringScale = Number(e.target.value);
        if (typeof ringScaleVal !== "undefined")
          ringScaleVal.textContent = active.ringScale + "%";
        if (active.ring || active.immichRing) {
          if (
            window.AppNG.manager &&
            typeof window.AppNG.manager.renderStage === "function"
          ) {
            window.AppNG.manager.renderStage(active);
          }
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof squeezeSlider !== "undefined") {
      squeezeSlider.addEventListener("input", (e) => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.squeezeMinPct = Number(e.target.value);
        active.squeezeUserOverridden = true;
        if (active.ring || active.immichRing) {
          if (
            window.AppNG.manager &&
            typeof window.AppNG.manager.renderStage === "function"
          ) {
            window.AppNG.manager.renderStage(active);
          }
        } else if (typeof squeezeVal !== "undefined") {
          squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof ringSortCbs !== "undefined") {
      ringSortCbs.forEach((cb) => {
        cb.addEventListener("change", () => {
          const active =
            window.AppNG.manager && window.AppNG.manager.getActive
              ? window.AppNG.manager.getActive()
              : null;
          if (!active) return;
          if (cb.checked) {
            ringSortCbs.forEach((other) => {
              if (other !== cb) other.checked = false;
            });
            active.ringSortMetric = cb.dataset.metric;
            if (!active.squeezeUserOverridden) {
              active.squeezeMinPct =
                window.AppNG.shared && window.AppNG.shared.SQUEEZE_DEFAULTS
                  ? window.AppNG.shared.SQUEEZE_DEFAULTS[active.ringSortMetric]
                  : 0;
              if (typeof squeezeSlider !== "undefined")
                squeezeSlider.value = active.squeezeMinPct;
            }
          } else {
            cb.checked = true;
            return;
          }
          if (active.ring || active.immichRing) {
            if (
              window.AppNG.manager &&
              typeof window.AppNG.manager.renderStage === "function"
            ) {
              window.AppNG.manager.renderStage(active);
            }
          }
          if (
            window.AppNG.manager &&
            typeof window.AppNG.manager.saveState === "function"
          ) {
            window.AppNG.manager.saveState();
          }
        });
      });
    }

    if (typeof sharpEnableCb !== "undefined") {
      sharpEnableCb.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.sharpCutoffEnabled = sharpEnableCb.checked;
        if (typeof sharpControlsEl !== "undefined")
          sharpControlsEl.style.display = active.sharpCutoffEnabled
            ? "flex"
            : "none";
        if (active.ring || active.immichRing) {
          if (
            window.AppNG.manager &&
            typeof window.AppNG.manager.renderStage === "function"
          ) {
            window.AppNG.manager.renderStage(active);
          }
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof sharpSlider !== "undefined") {
      sharpSlider.addEventListener("input", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.sharpMinVal = Number(sharpSlider.value);
        if (active.ring || active.immichRing) {
          if (
            window.AppNG.manager &&
            typeof window.AppNG.manager.renderStage === "function"
          ) {
            window.AppNG.manager.renderStage(active);
          }
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof findNeutralBtn !== "undefined") {
      findNeutralBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (
          !active ||
          !window.AppNG.manager ||
          typeof window.AppNG.manager.findNeutralPose !== "function"
        )
          return;
        window.AppNG.manager.findNeutralPose(active);
      });
    }

    if (typeof simThresholdInput !== "undefined") {
      simThresholdInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.simThreshold = Number(simThresholdInput.value);
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.renderLeftRail === "function"
        ) {
          window.AppNG.manager.renderLeftRail();
        }
      });
    }

    if (typeof blurThresholdInput !== "undefined") {
      blurThresholdInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.blurThreshold = Number(blurThresholdInput.value);
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.renderLeftRail === "function"
        ) {
          window.AppNG.manager.renderLeftRail();
        }
      });
    }

    if (typeof cacheFormatPngCb !== "undefined") {
      cacheFormatPngCb.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.cacheFormatPng = cacheFormatPngCb.checked;
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof analysisStartInput !== "undefined") {
      analysisStartInput.addEventListener("change", () => {
        const project =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!project || !project.video) return;
        project.video.rangeStartSec =
          analysisStartInput.value === ""
            ? null
            : Number(analysisStartInput.value);
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof analysisEndInput !== "undefined") {
      analysisEndInput.addEventListener("change", () => {
        const project =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!project || !project.video) return;
        project.video.rangeEndSec =
          analysisEndInput.value === "" ? null : Number(analysisEndInput.value);
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof startAnalysisBtn !== "undefined") {
      startAnalysisBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video && active.videoFile) active.startAnalysis();
      });
    }

    if (typeof folderRefIndexInput !== "undefined") {
      folderRefIndexInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.folderRefIndex = Math.max(
          1,
          Number(folderRefIndexInput.value) || 1,
        );
        if (
          window.AppNG.manager &&
          typeof window.AppNG.manager.saveState === "function"
        ) {
          window.AppNG.manager.saveState();
        }
      });
    }

    if (typeof immichSimThresholdInput !== "undefined") {
      immichSimThresholdInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.simThreshold = Number(immichSimThresholdInput.value);
        if (window.AppNG.manager) window.AppNG.manager.saveState();
        window.AppNG.manager.renderLeftRail();
      });
    }

    if (typeof immichBlurThresholdInput !== "undefined") {
      immichBlurThresholdInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.blurThreshold = Number(immichBlurThresholdInput.value);
        if (window.AppNG.manager) window.AppNG.manager.saveState();
        window.AppNG.manager.renderLeftRail();
      });
    }

    if (typeof immichAnalyzeRefIndexInput !== "undefined") {
      immichAnalyzeRefIndexInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.immichAnalyzeRefIndex = Math.max(
          1,
          Number(immichAnalyzeRefIndexInput.value) || 1,
        );
        if (window.AppNG.manager) window.AppNG.manager.saveState();
      });
    }

    const startImmichAnalysis = () => {
      const active =
        window.AppNG.manager && window.AppNG.manager.getActive
          ? window.AppNG.manager.getActive()
          : null;
      if (!active || !active.selectedAssetIds.size) return;
      if (active.job && active.job.status === "running") return;
      active.startImmichAnalysis(Array.from(active.selectedAssetIds));
    };

    if (typeof immichAnalyzeBtn !== "undefined") {
      immichAnalyzeBtn.addEventListener("click", startImmichAnalysis);
    }
    if (typeof immichAnalyzeSelectedBtn !== "undefined") {
      immichAnalyzeSelectedBtn.addEventListener("click", startImmichAnalysis);
    }

    if (typeof immichExportSelectedBtn !== "undefined") {
      immichExportSelectedBtn.addEventListener("click", async () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active || !active.selectedAssetIds.size) return;
        const previousText = immichExportSelectedBtn.textContent;
        immichExportSelectedBtn.textContent = "Exporting...";
        immichExportSelectedBtn.disabled = true;
        try {
          const result = await active.exportSelectedImmichAssets();
          immichExportSelectedBtn.textContent = result.error
            ? `Error: ${result.error}`
            : `Saved ${result.exported} -> ${result.path}`;
        } catch (e) {
          immichExportSelectedBtn.textContent = `Error: ${e.message}`;
        }
        setTimeout(() => {
          immichExportSelectedBtn.textContent = previousText;
          immichExportSelectedBtn.disabled =
            !window.AppNG.manager.getActive()?.selectedAssetIds.size;
        }, 4000);
      });
    }

    if (typeof immichViewSelectedBtn !== "undefined") {
      immichViewSelectedBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && window.AppNG.manager) {
          window.AppNG.manager.openSelectedModal(active);
        }
      });
    }

    if (typeof immichClearSelectedBtn !== "undefined") {
      immichClearSelectedBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        active.selectedAssetIds.clear();
        if (active.immichRing) window.AppNG.manager.renderStage(active);
        window.AppNG.manager.saveState();
      });
    }

    if (typeof immichRandomFaceBtn !== "undefined") {
      immichRandomFaceBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && !active.immichSearching) active.loadRandomImmichFace();
      });
    }

    if (typeof loadFolderBtn !== "undefined") {
      const folderFilesInput = document.createElement("input");
      folderFilesInput.type = "file";
      folderFilesInput.accept = "image/*";
      folderFilesInput.multiple = true;
      folderFilesInput.style.display = "none";
      document.body.appendChild(folderFilesInput);
      folderFilesInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && folderFilesInput.files?.length) {
          active.startFolderAnalysis({
            images: Array.from(folderFilesInput.files),
          });
        }
        folderFilesInput.value = "";
      });
      loadFolderBtn.addEventListener("click", () => folderFilesInput.click());
    }

    if (typeof loadZipBtn !== "undefined") {
      const zipFileInput = document.createElement("input");
      zipFileInput.type = "file";
      zipFileInput.accept = ".zip";
      zipFileInput.style.display = "none";
      document.body.appendChild(zipFileInput);
      zipFileInput.addEventListener("change", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && zipFileInput.files?.[0]) {
          active.startFolderAnalysis({ zip: zipFileInput.files[0] });
        }
        zipFileInput.value = "";
      });
      loadZipBtn.addEventListener("click", () => zipFileInput.click());
    }

    if (typeof selectedModalOverlay !== "undefined") {
      if (typeof selectedModalCloseBtn !== "undefined") {
        selectedModalCloseBtn.addEventListener(
          "click",
          () => (selectedModalOverlay.style.display = "none"),
        );
      }
      selectedModalOverlay.addEventListener("click", (e) => {
        if (e.target === selectedModalOverlay) {
          selectedModalOverlay.style.display = "none";
        }
      });
    }

    if (typeof rankedSortRadios !== "undefined") {
      rankedSortRadios.forEach((radio) => {
        radio.addEventListener("change", () => {
          if (!radio.checked) return;
          const active =
            window.AppNG.manager && window.AppNG.manager.getActive
              ? window.AppNG.manager.getActive()
              : null;
          if (!active) return;
          if (active.task === "immich") {
            active.immichRankedSortMetric = radio.value;
            if (active.immichRing) window.AppNG.manager.renderStage(active);
          } else {
            active.rankedSortMetric = radio.value;
            if (active.ring) window.AppNG.manager.renderStage(active);
          }
          window.AppNG.manager.saveState();
        });
      });
    }

    if (typeof framesSelectAllBtn !== "undefined") {
      framesSelectAllBtn.addEventListener("click", () => {
        const active = window.AppNG.manager?.getActive();
        if (!active?.ring) return;
        active.ring.baseResults.forEach((r) =>
          active.selectedFrames.add(r.frame),
        );
        window.AppNG.manager.renderStage(active);
        window.AppNG.manager.saveState();
      });
    }
    if (typeof framesDeselectAllBtn !== "undefined") {
      framesDeselectAllBtn.addEventListener("click", () => {
        const active = window.AppNG.manager?.getActive();
        if (!active) return;
        active.selectedFrames.clear();
        if (active.ring) window.AppNG.manager.renderStage(active);
        window.AppNG.manager.saveState();
      });
    }
    if (typeof immichSelectAllBtn !== "undefined") {
      immichSelectAllBtn.addEventListener("click", () => {
        const active = window.AppNG.manager?.getActive();
        if (!active?.immichRing) return;
        active.immichRing.baseResults.forEach((r) =>
          active.selectedAssetIds.add(r.assetId),
        );
        window.AppNG.manager.renderStage(active);
        window.AppNG.manager.saveState();
      });
    }
    if (typeof immichDeselectAllBtn !== "undefined") {
      immichDeselectAllBtn.addEventListener("click", () => {
        const active = window.AppNG.manager?.getActive();
        if (!active) return;
        active.selectedAssetIds.clear();
        if (active.immichRing) window.AppNG.manager.renderStage(active);
        window.AppNG.manager.saveState();
      });
    }
    if (typeof immichRefreshBtn !== "undefined") {
      immichRefreshBtn.addEventListener("click", () => {
        const active = window.AppNG.manager?.getActive();
        if (!active?.immichRing || active.immichLoading) return;
        active.recenterImmich(
          active.immichRing.centerAssetId,
          active.immichRing.centerFilename,
        );
      });
    }
    if (typeof immichSaveSelectedBtn !== "undefined") {
      immichSaveSelectedBtn.addEventListener("click", async () => {
        const active = window.AppNG.manager?.getActive();
        if (!active?.selectedAssetIds.size) return;
        const previousText = immichSaveSelectedBtn.textContent;
        immichSaveSelectedBtn.textContent = "Saving...";
        immichSaveSelectedBtn.disabled = true;
        try {
          const result = await active.exportSelectedImmichAssets();
          if (typeof immichExportResultEl !== "undefined") {
            immichExportResultEl.textContent = result.error
              ? `Error: ${result.error}`
              : `Saved ${result.exported} images -> ${result.path}`;
          }
        } catch (e) {
          if (typeof immichExportResultEl !== "undefined") {
            immichExportResultEl.textContent = `Error: ${e.message}`;
          }
        }
        immichSaveSelectedBtn.textContent = previousText;
        immichSaveSelectedBtn.disabled = active.selectedAssetIds.size === 0;
      });
    }

    if (typeof immichSearchInput !== "undefined") {
      let immichSearchDebounce = null;
      immichSearchInput.addEventListener("input", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (!active) return;
        clearTimeout(immichSearchDebounce);
        const q = immichSearchInput.value;
        immichSearchDebounce = setTimeout(() => active.searchImmich(q), 250);
      });
    }

    if (typeof rewindBtn !== "undefined") {
      rewindBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video) active.stepAndSyncAudio(1);
      });
    }
    if (typeof prevFrameBtn !== "undefined") {
      prevFrameBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video)
          active.stepAndSyncAudio(active.video.currentFrame - 1);
      });
    }
    if (typeof nextFrameBtn !== "undefined") {
      nextFrameBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video)
          active.stepAndSyncAudio(active.video.currentFrame + 1);
      });
    }
    if (typeof playBtn !== "undefined") {
      playBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video && !active._playTimer) active.togglePlay();
      });
    }
    if (typeof stopBtn !== "undefined") {
      stopBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video && active._playTimer) active.togglePlay();
      });
    }
    if (typeof popoutVideoBtn !== "undefined") {
      popoutVideoBtn.addEventListener("click", () => {
        const active =
          window.AppNG.manager && window.AppNG.manager.getActive
            ? window.AppNG.manager.getActive()
            : null;
        if (active && active.video) PlaybackModal.open(active);
      });
    }

    if (typeof playbackModalCloseBtn !== "undefined") {
      playbackModalCloseBtn.addEventListener("click", () =>
        PlaybackModal.close(),
      );
    }
    if (typeof playbackModalEl !== "undefined") {
      const backdrop = playbackModalEl.querySelector(
        ".ng-playback-modal-backdrop",
      );
      if (backdrop)
        backdrop.addEventListener("click", () => PlaybackModal.close());
    }
    if (typeof playbackPrevFrameBtn !== "undefined") {
      playbackPrevFrameBtn.addEventListener("click", () =>
        PlaybackModal.stepFrame(-1),
      );
    }
    if (typeof playbackNextFrameBtn !== "undefined") {
      playbackNextFrameBtn.addEventListener("click", () =>
        PlaybackModal.stepFrame(1),
      );
    }
    if (typeof playbackVideoEl !== "undefined") {
      playbackVideoEl.addEventListener("loadedmetadata", () => {
        if (
          PlaybackModal.kind === "raw" &&
          PlaybackModal.rangeStartSec != null
        ) {
          playbackVideoEl.currentTime = PlaybackModal.rangeStartSec;
        }
      });
      playbackVideoEl.addEventListener("timeupdate", () => {
        if (
          PlaybackModal.kind === "raw" &&
          PlaybackModal.rangeEndSec != null &&
          playbackVideoEl.currentTime >= PlaybackModal.rangeEndSec
        ) {
          playbackVideoEl.pause();
          playbackVideoEl.currentTime = PlaybackModal.rangeEndSec;
        }
      });
    }

    document.addEventListener("keydown", (e) => {
      if (
        typeof playbackModalEl !== "undefined" &&
        playbackModalEl.style.display !== "none"
      ) {
        if (e.key === "Escape") PlaybackModal.close();
        return;
      }
      const active =
        window.AppNG.manager && window.AppNG.manager.getActive
          ? window.AppNG.manager.getActive()
          : null;
      if (!active || !active.video) return;
      if (
        document.activeElement &&
        ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)
      )
        return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        active.stepAndSyncAudio(active.video.currentFrame - 1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        active.stepAndSyncAudio(active.video.currentFrame + 1);
      }
    });
  }

  window.AppNG.render = {
    renderNGChart,
    showStaticFramePreviewNG,
    showHoverPreview,
    hideHoverPreview,
    renderPoseListNG,
    setupPoseListScrubberNG,
    flashHighlightNG,
    placeholder,
    videoPickerButton,
    drawFrame,
    setPlayingVisual,
    PlaybackModal,
    wireNGGlobalControls,
    wireLeftRailChrome,
    wireRightSidebarListToggle,
    wireNgFisheyeLensMouseMove,
    wireNgPoseListLensEffect,
    attachNgLensEffect,
    wirePersonClustersNG,
    wireAppShellNG,
  };

  function wireAppShellNG() {
    wireLeftRailChrome();
    wireRightSidebarListToggle();
    wireNgFisheyeLensMouseMove();
    wireNgPoseListLensEffect();
    if (window.AppNG.pickers) {
      if (!posePickerCellsNG.length && posePickerGridEl) {
        posePickerCellsNG =
          window.AppNG.pickers.buildPickerCellsNG(posePickerGridEl);
      }
      if (!scalePickerCellsNG.length && scalePickerGridEl) {
        scalePickerCellsNG =
          window.AppNG.pickers.buildPickerCellsNG(scalePickerGridEl);
      }
      window.AppNG.pickers.wirePosePickerControlsNG();
      window.AppNG.pickers.wireScalePickerControlsNG();
    }
    wirePersonClustersNG();
  }
})();
