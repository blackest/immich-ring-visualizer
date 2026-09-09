// pickersNG.js
// Pose and scale picker logic for NG selection workflows.

(function () {
  "use strict";

  window.AppNG = window.AppNG || {};

  function buildPickerCellsNG(gridEl) {
    const cells = [];
    for (let i = 0; i < 9; i++) {
      const cell = document.createElement("div");
      cell.innerHTML = `<img loading="lazy">`;
      gridEl.appendChild(cell);
      cells.push(cell);
    }
    return cells;
  }

  function updatePickerCellVisualNG(project, cell, it, dist, onClick) {
    cell.style.visibility = "visible";
    const isSelected = project.selectedFrames.has(it.frame);
    const img = cell.querySelector("img");
    if (img.dataset.frame !== String(it.frame)) {
      img.src = it.thumbUrl;
      img.dataset.frame = it.frame;
    }
    img.title = it.titleText(dist, isSelected);
    img.style.borderColor = isSelected
      ? "var(--ng-accent)"
      : "var(--ng-border)";
    cell.onclick = onClick;
  }

  function setupPosePickerNG(project) {
    project.posePickerPool = (project.ring ? project.ring.baseResults : [])
      .filter(
        (r) =>
          r.frameId && typeof r.pitch === "number" && typeof r.yaw === "number",
      )
      .map((r) => ({
        frame: r.frame,
        thumbUrl: r.thumbUrl,
        pitch: r.pitch,
        yaw: r.yaw,
        blur: typeof r.blur === "number" ? r.blur : 0,
        vertFillPct: r.vertFillPct,
        titleText(dist, isSelected) {
          const scaleTxt =
            typeof this.vertFillPct === "number"
              ? `, face ${(this.vertFillPct * 100).toFixed(0)}% frame ht`
              : "";
          return `pitch ${this.pitch.toFixed(1)}, yaw ${this.yaw.toFixed(1)} (Δ${dist.toFixed(1)} from target), sharpness ${this.blur.toFixed(0)}${scaleTxt} — click to ${isSelected ? "remove from" : "add to"} selection`;
        },
      }));
    project.posePickerDisplayed = new Array(9).fill(null);

    if (!project.posePickerPool.length) {
      if (project.isActive) {
        posePickerEmptyEl.style.display = "block";
        posePickerControlsEl.style.display = "none";
      }
      return;
    }
    if (project.isActive) {
      posePickerEmptyEl.style.display = "none";
      posePickerControlsEl.style.display = "flex";
    }

    const pitchVals = project.posePickerPool.map((it) => it.pitch);
    const yawVals = project.posePickerPool.map((it) => it.yaw);
    project._posePickerPitchTarget = Math.round(
      pitchVals.reduce((s, v) => s + v, 0) / pitchVals.length,
    );
    project._posePickerYawTarget = Math.round(
      yawVals.reduce((s, v) => s + v, 0) / yawVals.length,
    );
  }

  function renderPosePickerGridNG(project) {
    if (!project.posePickerPool.length) {
      posePickerEmptyEl.style.display = "block";
      posePickerControlsEl.style.display = "none";
      return;
    }
    posePickerEmptyEl.style.display = "none";
    posePickerControlsEl.style.display = "flex";

    const pitchTarget =
      typeof project._posePickerPitchTarget === "number"
        ? project._posePickerPitchTarget
        : 0;
    const yawTarget =
      typeof project._posePickerYawTarget === "number"
        ? project._posePickerYawTarget
        : 0;
    posePickerPitchSlider.value = pitchTarget;
    posePickerYawSlider.value = yawTarget;
    posePickerPitchNum.value = pitchTarget;
    posePickerYawNum.value = yawTarget;
    posePickerToleranceEnable.checked = !!project._posePickerToleranceOn;
    posePickerToleranceVal.disabled = !project._posePickerToleranceOn;
    posePickerToleranceVal.value =
      typeof project._posePickerTolerance === "number"
        ? project._posePickerTolerance
        : 5;

    let ranked = project.posePickerPool
      .map((it) => ({
        it,
        dist: Math.hypot(it.pitch - pitchTarget, it.yaw - yawTarget),
      }))
      .sort((a, b) => a.dist - b.dist);
    if (project._posePickerToleranceOn) {
      ranked = ranked.filter((r) => r.dist <= posePickerToleranceVal.value);
    }
    ranked = ranked.slice(0, 9);
    const rankedFrames = ranked.map((r) => r.it.frame);

    const displayed = project.posePickerDisplayed;
    const keepSlot = displayed.map(
      (frame) => frame !== null && rankedFrames.includes(frame),
    );
    const toPlace = ranked.filter((r) => !displayed.includes(r.it.frame));
    let placeIdx = 0;

    for (let i = 0; i < 9; i++) {
      if (keepSlot[i]) {
        const match = ranked.find((r) => r.it.frame === displayed[i]);
        updatePickerCellVisualNG(
          project,
          posePickerCellsNG[i],
          match.it,
          match.dist,
          () => {
            project.toggleFrameSelection(match.it.frame);
            ProjectManager.render();
          },
        );
      } else if (placeIdx < toPlace.length) {
        const { it, dist } = toPlace[placeIdx++];
        displayed[i] = it.frame;
        updatePickerCellVisualNG(
          project,
          posePickerCellsNG[i],
          it,
          dist,
          () => {
            project.toggleFrameSelection(it.frame);
            ProjectManager.render();
          },
        );
      } else {
        displayed[i] = null;
        posePickerCellsNG[i].style.visibility = "hidden";
      }
    }

    posePickerCountEl.textContent = project._posePickerToleranceOn
      ? `${ranked.length} within ${posePickerToleranceVal.value}° of target (${project.posePickerPool.length} in pool)`
      : `${project.posePickerPool.length} in analyzed pool`;
  }

  function wirePosePickerControlsNG() {
    function onChange() {
      const active = ProjectManager.getActive();
      if (!active) return;
      active._posePickerPitchTarget = Math.max(
        -90,
        Math.min(90, parseFloat(posePickerPitchNum.value) || 0),
      );
      active._posePickerYawTarget = Math.max(
        -90,
        Math.min(90, parseFloat(posePickerYawNum.value) || 0),
      );
      active._posePickerToleranceOn = posePickerToleranceEnable.checked;
      active._posePickerTolerance =
        parseFloat(posePickerToleranceVal.value) || 5;
      renderPosePickerGridNG(active);
    }
    posePickerPitchSlider.addEventListener("input", () => {
      posePickerPitchNum.value = posePickerPitchSlider.value;
      onChange();
    });
    posePickerYawSlider.addEventListener("input", () => {
      posePickerYawNum.value = posePickerYawSlider.value;
      onChange();
    });
    posePickerPitchNum.addEventListener("input", onChange);
    posePickerYawNum.addEventListener("input", onChange);
    posePickerToleranceEnable.addEventListener("change", onChange);
    posePickerToleranceVal.addEventListener("input", onChange);
    posePickerSelectBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.posePickerDisplayed
        .filter((f) => f !== null)
        .forEach((frame) => active.selectedFrames.add(frame));
      ProjectManager.render();
    });
  }

  function setupScalePickerNG(project) {
    const minFacePxEl = document.getElementById("ng-export-min-face");
    const minFacePx = parseFloat(minFacePxEl ? minFacePxEl.value : 0) || 0;

    project.scalePickerPool = (project.ring ? project.ring.baseResults : [])
      .filter((r) => {
        if (!r.frameId) return false;
        const scaleVal =
          typeof r.vertFillPct === "number"
            ? r.vertFillPct
            : typeof r.bboxRatio === "number"
              ? Math.sqrt(r.bboxRatio)
              : null;
        if (scaleVal === null) return false;
        if (minFacePx > 0 && r.bbox) {
          const faceH = r.bbox[3] - r.bbox[1];
          if (faceH < minFacePx) return false;
        }
        return true;
      })
      .map((r) => ({
        frame: r.frame,
        thumbUrl: r.thumbUrl,
        scalePct:
          (typeof r.vertFillPct === "number"
            ? r.vertFillPct
            : Math.sqrt(r.bboxRatio)) * 100,
        blur: typeof r.blur === "number" ? r.blur : 0,
        titleText(dist, isSelected) {
          return `${this.scalePct.toFixed(0)}% frame ht (Δ${dist.toFixed(1)} from target), sharpness ${this.blur.toFixed(0)} — click to ${isSelected ? "remove from" : "add to"} selection`;
        },
      }));
    project.scalePickerDisplayed = new Array(9).fill(null);

    if (!project.scalePickerPool.length) {
      if (project.isActive) {
        scalePickerEmptyEl.style.display = "block";
        scalePickerControlsEl.style.display = "none";
      }
      return;
    }
    if (project.isActive) {
      scalePickerEmptyEl.style.display = "none";
      scalePickerControlsEl.style.display = "flex";
    }

    const scaleVals = project.scalePickerPool.map((it) => it.scalePct);
    project._scalePickerTarget = Math.round(
      scaleVals.reduce((s, v) => s + v, 0) / scaleVals.length,
    );
  }

  function renderScalePickerGridNG(project) {
    if (!project.scalePickerPool.length) {
      scalePickerEmptyEl.style.display = "block";
      scalePickerControlsEl.style.display = "none";
      return;
    }
    scalePickerEmptyEl.style.display = "none";
    scalePickerControlsEl.style.display = "flex";

    const target =
      typeof project._scalePickerTarget === "number"
        ? project._scalePickerTarget
        : 30;
    scalePickerSlider.value = target;
    scalePickerNum.value = target;
    scalePickerToleranceEnable.checked = !!project._scalePickerToleranceOn;
    scalePickerToleranceVal.disabled = !project._scalePickerToleranceOn;
    scalePickerToleranceVal.value =
      typeof project._scalePickerTolerance === "number"
        ? project._scalePickerTolerance
        : 10;

    let ranked = project.scalePickerPool
      .map((it) => ({ it, dist: Math.abs(it.scalePct - target) }))
      .sort((a, b) => a.dist - b.dist);
    if (project._scalePickerToleranceOn) {
      ranked = ranked.filter((r) => r.dist <= scalePickerToleranceVal.value);
    }
    ranked = ranked.slice(0, 9);
    const rankedFrames = ranked.map((r) => r.it.frame);

    const displayed = project.scalePickerDisplayed;
    const keepSlot = displayed.map(
      (frame) => frame !== null && rankedFrames.includes(frame),
    );
    const toPlace = ranked.filter((r) => !displayed.includes(r.it.frame));
    let placeIdx = 0;

    for (let i = 0; i < 9; i++) {
      if (keepSlot[i]) {
        const match = ranked.find((r) => r.it.frame === displayed[i]);
        updatePickerCellVisualNG(
          project,
          scalePickerCellsNG[i],
          match.it,
          match.dist,
          () => {
            project.toggleFrameSelection(match.it.frame);
            ProjectManager.render();
          },
        );
      } else if (placeIdx < toPlace.length) {
        const { it, dist } = toPlace[placeIdx++];
        displayed[i] = it.frame;
        updatePickerCellVisualNG(
          project,
          scalePickerCellsNG[i],
          it,
          dist,
          () => {
            project.toggleFrameSelection(it.frame);
            ProjectManager.render();
          },
        );
      } else {
        displayed[i] = null;
        scalePickerCellsNG[i].style.visibility = "hidden";
      }
    }

    scalePickerCountEl.textContent = project._scalePickerToleranceOn
      ? `${ranked.length} within ${scalePickerToleranceVal.value}% of target (${project.scalePickerPool.length} in pool)`
      : `${project.scalePickerPool.length} in analyzed pool`;
  }

  function wireScalePickerControlsNG() {
    function onChange() {
      const active = ProjectManager.getActive();
      if (!active) return;
      active._scalePickerTarget = Math.max(
        0,
        Math.min(100, parseFloat(scalePickerNum.value) || 0),
      );
      active._scalePickerToleranceOn = scalePickerToleranceEnable.checked;
      active._scalePickerTolerance =
        parseFloat(scalePickerToleranceVal.value) || 10;
      renderScalePickerGridNG(active);
    }
    scalePickerSlider.addEventListener("input", () => {
      scalePickerNum.value = scalePickerSlider.value;
      onChange();
    });
    scalePickerNum.addEventListener("input", onChange);
    scalePickerToleranceEnable.addEventListener("change", onChange);
    scalePickerToleranceVal.addEventListener("input", onChange);
    scalePickerSelectBtn.addEventListener("click", () => {
      const active = ProjectManager.getActive();
      if (!active) return;
      active.scalePickerDisplayed
        .filter((f) => f !== null)
        .forEach((frame) => active.selectedFrames.add(frame));
      ProjectManager.render();
    });
  }

  window.AppNG.pickers = {
    buildPickerCellsNG,
    updatePickerCellVisualNG,
    setupPosePickerNG,
    renderPosePickerGridNG,
    wirePosePickerControlsNG,
    setupScalePickerNG,
    renderScalePickerGridNG,
    wireScalePickerControlsNG,
  };
})();
