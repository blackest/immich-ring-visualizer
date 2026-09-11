/**
 * pickersNG.js -- Pose Picker + Shot Scale Picker, as a MAIN-STAGE view.
 *
 * Was a pair of cramped 3x3 grids buried in the left rail. Now promoted to
 * a full-width workspace that takes over #ng-stage-wrap (the same slot the
 * ring and the pose-list strip live in), reached from two entry buttons in
 * the rail. The rail keeps only those buttons; every control (target
 * sliders, tolerance) lives in the stage.
 *
 * ONE stage, a Pose|Scale toggle inside it. `project.stageMode` drives
 * which of {"ring","posePicker","scalePicker"} the main area shows;
 * projectManagerNG.js's renderVideoStage() checks it and calls
 * PickerStageNG.render(project) (or .hide()).
 *
 * Both pickers search the current project's analyzed pool
 * (project.ring.baseResults -- rows key on `frame`, carry a pre-built
 * `thumbUrl`; there is no `frameId` on them, see buildRing() in
 * characterProjectNG.js) for candidates near a dialed-in target:
 *   - Pose  -- 2D {pitch, yaw} euclidean distance
 *   - Scale -- 1D face-height-% distance
 * Selecting a cell toggles project.selectedFrames; "Add all shown" adds
 * every visible (in-tolerance) frame. The ranked sidebar and, on the way
 * back to "ring", the ring itself reflect the selection.
 *
 * The thin-result case (tight tolerance + specific target -> 1-2 hits) is
 * the reason this exists: the stage renders those big with full readouts,
 * then a few just-outside-tolerance frames dimmed with their delta so you
 * can see what you're excluding.
 *
 * Classic script sharing page scope with the other *NG.js files. Loads
 * before projectManagerNG.js, but only touches ProjectManager from inside
 * functions (render/wire) that run later, so order is fine.
 * bootstrapWiringNG.js calls PickerStageNG.wire() once at startup.
 */
(function () {
  "use strict";

  // How many cells to show when NO tolerance filter is active (otherwise
  // every in-tolerance frame is shown, capped at IN_TOL_CAP).
  var TOP_N = 24;
  var IN_TOL_CAP = 60;
  // When a tolerance filter leaves fewer than this many hits, also show
  // this many of the nearest just-outside-tolerance frames, dimmed.
  var THIN_HITS = 6;
  var THIN_EXTRA = 8;

  var $ = function (id) { return document.getElementById(id); };

  var els = {};
  function refreshEls() {
    els.stage = $("ng-picker-stage");
    els.grid = $("ng-picker-stage-grid");
    els.empty = $("ng-picker-stage-empty");
    els.count = $("ng-picker-stage-count");
    els.target = $("ng-picker-stage-target");
    els.back = $("ng-picker-stage-back");
    els.addBtn = $("ng-picker-stage-add");
    els.tabs = Array.prototype.slice.call(document.querySelectorAll(".ng-picker-tab"));
    els.poseControls = $("ng-pose-picker-controls");
    els.scaleControls = $("ng-scale-picker-controls");
    els.posePitchRange = $("ng-pose-picker-pitch");
    els.posePitchNum = $("ng-pose-picker-pitch-num");
    els.poseYawRange = $("ng-pose-picker-yaw");
    els.poseYawNum = $("ng-pose-picker-yaw-num");
    els.poseTolOn = $("ng-pose-picker-tolerance-enable");
    els.poseTolVal = $("ng-pose-picker-tolerance-val");
    els.scaleRange = $("ng-scale-picker-slider");
    els.scaleNum = $("ng-scale-picker-num");
    els.scaleTolOn = $("ng-scale-picker-tolerance-enable");
    els.scaleTolVal = $("ng-scale-picker-tolerance-val");
    els.railPoseBtn = $("ng-open-pose-picker");
    els.railScaleBtn = $("ng-open-scale-picker");
  }

  // ---------------------------------------------------------------
  // pool building -- rebuilt only when the ring identity changes
  // ---------------------------------------------------------------
  function buildPools(project) {
    var base = (project.ring && project.ring.baseResults) || [];

    project._posePool = base
      .filter(function (r) {
        return r.frame != null && typeof r.pitch === "number" && typeof r.yaw === "number";
      })
      .map(function (r) {
        return {
          frame: r.frame, thumbUrl: r.thumbUrl,
          pitch: r.pitch, yaw: r.yaw,
          roll: typeof r.roll === "number" ? r.roll : null,
          blur: typeof r.blur === "number" ? r.blur : 0,
          vertFillPct: r.vertFillPct,
        };
      });

    var minFaceEl = $("ng-export-min-face");
    var minFacePx = parseFloat(minFaceEl ? minFaceEl.value : 0) || 0;
    project._scalePool = base
      .filter(function (r) {
        if (r.frame == null) return false;
        var sv = typeof r.vertFillPct === "number"
          ? r.vertFillPct
          : (typeof r.bboxRatio === "number" ? Math.sqrt(r.bboxRatio) : null);
        if (sv === null) return false;
        // honor Export Settings' "Min face" floor -- don't surface frames
        // the export pipeline would skip anyway (real cross-feature coupling).
        if (minFacePx > 0 && r.bbox) {
          if (r.bbox[3] - r.bbox[1] < minFacePx) return false;
        }
        return true;
      })
      .map(function (r) {
        return {
          frame: r.frame, thumbUrl: r.thumbUrl,
          scalePct: (typeof r.vertFillPct === "number" ? r.vertFillPct : Math.sqrt(r.bboxRatio)) * 100,
          blur: typeof r.blur === "number" ? r.blur : 0,
        };
      });

    // targets default to each pool's own centroid
    if (project._posePool.length) {
      project._posePitchTarget = Math.round(avg(project._posePool.map(function (it) { return it.pitch; })));
      project._poseYawTarget = Math.round(avg(project._posePool.map(function (it) { return it.yaw; })));
    }
    if (project._scalePool.length) {
      project._scaleTarget = Math.round(avg(project._scalePool.map(function (it) { return it.scalePct; })));
    }
  }
  function avg(xs) { return xs.reduce(function (s, v) { return s + v; }, 0) / xs.length; }

  function ensurePools(project) {
    if (project._pickerRingRef !== project.ring) {
      project._pickerRingRef = project.ring;
      buildPools(project);
    }
  }

  // ---------------------------------------------------------------
  // ranking
  // ---------------------------------------------------------------
  function rankedFor(project, mode) {
    if (mode === "scalePicker") {
      var st = numOr(project._scaleTarget, 30);
      var sTolOn = !!project._scaleTolOn;
      var sTol = numOr(project._scaleTol, 10);
      var sRanked = project._scalePool
        .map(function (it) { return { it: it, dist: Math.abs(it.scalePct - st) }; })
        .sort(function (a, b) { return a.dist - b.dist; });
      return sliceRanked(sRanked, sTolOn, sTol);
    }
    var pt = numOr(project._posePitchTarget, 0);
    var yt = numOr(project._poseYawTarget, 0);
    var pTolOn = !!project._poseTolOn;
    var pTol = numOr(project._poseTol, 5);
    var pRanked = project._posePool
      .map(function (it) { return { it: it, dist: Math.hypot(it.pitch - pt, it.yaw - yt) }; })
      .sort(function (a, b) { return a.dist - b.dist; });
    return sliceRanked(pRanked, pTolOn, pTol);
  }
  function numOr(v, d) { return typeof v === "number" && !isNaN(v) ? v : d; }

  // returns { shown: [{it,dist}], extra: [{it,dist}], poolSize, tolOn, tol, inTol }
  function sliceRanked(ranked, tolOn, tol) {
    var poolSize = ranked.length;
    if (!tolOn) {
      return { shown: ranked.slice(0, TOP_N), extra: [], poolSize: poolSize, tolOn: false, tol: tol, inTol: null };
    }
    var inTol = ranked.filter(function (r) { return r.dist <= tol; });
    var shown = inTol.slice(0, IN_TOL_CAP);
    var extra = [];
    if (inTol.length < THIN_HITS) {
      extra = ranked.slice(inTol.length, inTol.length + THIN_EXTRA);
    }
    return { shown: shown, extra: extra, poolSize: poolSize, tolOn: true, tol: tol, inTol: inTol.length };
  }

  // ---------------------------------------------------------------
  // render
  // ---------------------------------------------------------------
  function captionFor(it, dist, mode) {
    if (mode === "scalePicker") {
      return "face " + it.scalePct.toFixed(0) + "% · Δ" + dist.toFixed(1) +
        " · sharp " + it.blur.toFixed(0);
    }
    var rollTxt = it.roll != null ? " · roll " + it.roll.toFixed(0) + "°" : "";
    return "pitch " + it.pitch.toFixed(0) + "° yaw " + it.yaw.toFixed(0) + "°" +
      rollTxt + " · Δ" + dist.toFixed(1) + " · sharp " + it.blur.toFixed(0);
  }

  function makeCell(project, it, dist, mode, dimmed) {
    var cell = document.createElement("div");
    cell.className = "ng-picker-cell" + (dimmed ? " ng-picker-cell-dim" : "");
    if (project.selectedFrames.has(it.frame)) cell.classList.add("ng-picker-cell-sel");
    var img = document.createElement("img");
    img.loading = "lazy";
    img.src = it.thumbUrl;
    var cap = document.createElement("div");
    cap.className = "ng-picker-cell-cap";
    cap.textContent = (dimmed ? "outside — " : "") + captionFor(it, dist, mode);
    cell.appendChild(img);
    cell.appendChild(cap);
    cell.title = "frame " + it.frame + " — click to " +
      (project.selectedFrames.has(it.frame) ? "remove from" : "add to") + " selection";
    cell.addEventListener("click", function () {
      project.toggleFrameSelection(it.frame);
      ProjectManager.render();
    });
    return cell;
  }

  function render(project) {
    refreshEls();
    if (!els.stage) return;
    var mode = project.stageMode;
    if (mode !== "posePicker" && mode !== "scalePicker") { els.stage.style.display = "none"; return; }
    els.stage.style.display = "flex";
    ensurePools(project);

    // tab + controls visibility
    els.tabs.forEach(function (t) {
      t.classList.toggle("ng-picker-tab-active", t.dataset.picker === mode);
    });
    if (els.poseControls) els.poseControls.style.display = mode === "posePicker" ? "" : "none";
    if (els.scaleControls) els.scaleControls.style.display = mode === "scalePicker" ? "" : "none";

    var pool = mode === "scalePicker" ? project._scalePool : project._posePool;

    // push current target/tolerance values into the inputs
    if (mode === "posePicker") {
      setVal(els.posePitchRange, numOr(project._posePitchTarget, 0));
      setVal(els.posePitchNum, numOr(project._posePitchTarget, 0));
      setVal(els.poseYawRange, numOr(project._poseYawTarget, 0));
      setVal(els.poseYawNum, numOr(project._poseYawTarget, 0));
      if (els.poseTolOn) els.poseTolOn.checked = !!project._poseTolOn;
      if (els.poseTolVal) { els.poseTolVal.disabled = !project._poseTolOn; setVal(els.poseTolVal, numOr(project._poseTol, 5)); }
      els.target.textContent = pool.length
        ? "target  pitch " + numOr(project._posePitchTarget, 0) + "° · yaw " + numOr(project._poseYawTarget, 0) + "°"
        : "";
    } else {
      setVal(els.scaleRange, numOr(project._scaleTarget, 30));
      setVal(els.scaleNum, numOr(project._scaleTarget, 30));
      if (els.scaleTolOn) els.scaleTolOn.checked = !!project._scaleTolOn;
      if (els.scaleTolVal) { els.scaleTolVal.disabled = !project._scaleTolOn; setVal(els.scaleTolVal, numOr(project._scaleTol, 10)); }
      els.target.textContent = pool.length ? "target  face " + numOr(project._scaleTarget, 30) + "% of frame" : "";
    }

    els.grid.innerHTML = "";
    if (!pool.length) {
      els.empty.style.display = "";
      els.grid.style.display = "none";
      els.count.textContent = "";
      if (els.addBtn) els.addBtn.disabled = true;
      return;
    }
    els.empty.style.display = "none";
    els.grid.style.display = "";

    var r = rankedFor(project, mode);
    r.shown.forEach(function (row) { els.grid.appendChild(makeCell(project, row.it, row.dist, mode, false)); });
    r.extra.forEach(function (row) { els.grid.appendChild(makeCell(project, row.it, row.dist, mode, true)); });

    project._pickerShownFrames = r.shown.map(function (row) { return row.it.frame; });
    if (els.addBtn) els.addBtn.disabled = !project._pickerShownFrames.length;

    var unit = mode === "scalePicker" ? "%" : "°";
    els.count.textContent = r.tolOn
      ? r.inTol + " within " + r.tol + unit + " of target"
        + (r.extra.length ? " (+" + r.extra.length + " nearest outside)" : "")
        + " · " + r.poolSize + " in pool"
      : "nearest " + r.shown.length + " of " + r.poolSize + " in analyzed pool";
  }
  function setVal(el, v) { if (el) el.value = v; }

  function hide() {
    refreshEls();
    if (els.stage) els.stage.style.display = "none";
  }

  // ---------------------------------------------------------------
  // one-time wiring
  // ---------------------------------------------------------------
  function setMode(mode) {
    var p = ProjectManager.getActive();
    if (!p) return;
    p.stageMode = mode;
    ProjectManager.render();
  }

  function wire() {
    refreshEls();

    if (els.railPoseBtn) els.railPoseBtn.addEventListener("click", function () { setMode("posePicker"); });
    if (els.railScaleBtn) els.railScaleBtn.addEventListener("click", function () { setMode("scalePicker"); });
    els.tabs.forEach(function (t) {
      t.addEventListener("click", function () { setMode(t.dataset.picker); });
    });
    if (els.back) els.back.addEventListener("click", function () { setMode("ring"); });

    function poseChange() {
      var p = ProjectManager.getActive();
      if (!p) return;
      p._posePitchTarget = clamp(parseFloat(els.posePitchNum.value) || 0, -90, 90);
      p._poseYawTarget = clamp(parseFloat(els.poseYawNum.value) || 0, -90, 90);
      p._poseTolOn = els.poseTolOn.checked;
      p._poseTol = parseFloat(els.poseTolVal.value) || 5;
      render(p);
    }
    function scaleChange() {
      var p = ProjectManager.getActive();
      if (!p) return;
      p._scaleTarget = clamp(parseFloat(els.scaleNum.value) || 0, 0, 100);
      p._scaleTolOn = els.scaleTolOn.checked;
      p._scaleTol = parseFloat(els.scaleTolVal.value) || 10;
      render(p);
    }
    bindRangeNum(els.posePitchRange, els.posePitchNum, poseChange);
    bindRangeNum(els.poseYawRange, els.poseYawNum, poseChange);
    if (els.poseTolOn) els.poseTolOn.addEventListener("change", poseChange);
    if (els.poseTolVal) els.poseTolVal.addEventListener("input", poseChange);
    bindRangeNum(els.scaleRange, els.scaleNum, scaleChange);
    if (els.scaleTolOn) els.scaleTolOn.addEventListener("change", scaleChange);
    if (els.scaleTolVal) els.scaleTolVal.addEventListener("input", scaleChange);

    if (els.addBtn) els.addBtn.addEventListener("click", function () {
      var p = ProjectManager.getActive();
      if (!p || !p._pickerShownFrames) return;
      p._pickerShownFrames.forEach(function (f) { p.selectedFrames.add(f); });
      ProjectManager.render();
    });
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function bindRangeNum(range, num, cb) {
    if (range) range.addEventListener("input", function () { if (num) num.value = range.value; cb(); });
    if (num) num.addEventListener("input", function () { if (range) range.value = num.value; cb(); });
  }

  window.PickerStageNG = { render: render, hide: hide, wire: wire };
})();
