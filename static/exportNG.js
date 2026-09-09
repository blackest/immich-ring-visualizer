// exportNG.js
// Export settings and export parameter gathering logic.

(function () {
  "use strict";

  window.AppNG = window.AppNG || {};

  function gatherExportParamsNG() {
    const upscaleOn = document.getElementById("ng-export-upscale").checked;
    return {
      width:
        parseInt(document.getElementById("ng-export-width").value, 10) || 512,
      height:
        parseInt(document.getElementById("ng-export-height").value, 10) || 512,
      cropMode: document.getElementById("ng-export-crop-mode").value,
      minFacePx:
        parseFloat(document.getElementById("ng-export-min-face").value) || 0,
      margin:
        parseFloat(document.getElementById("ng-export-margin").value) || 2.2,
      interp: document.getElementById("ng-export-interp").value,
      upscale: upscaleOn,
      maxUpscale: upscaleOn
        ? parseFloat(document.getElementById("ng-export-max-upscale").value) ||
          null
        : null,
      padMode: upscaleOn
        ? document.getElementById("ng-export-pad-mode").value
        : "none",
      native: document.getElementById("ng-export-native").checked,
    };
  }

  let lastResolutionSummaryNG = null;
  function applyResolutionSummaryNG(summary) {
    if (summary !== undefined) lastResolutionSummaryNG = summary;
    summary = lastResolutionSummaryNG;
    const note = document.getElementById("ng-export-source-res-note");
    const matchBtn = document.getElementById("ng-export-match-source-btn");
    if (!note || !matchBtn) return;
    if (!summary) {
      note.style.display = "none";
      matchBtn.style.display = "none";
      return;
    }
    const nativeOn = document.getElementById("ng-export-native").checked;
    const exportW = parseInt(
      document.getElementById("ng-export-width").value,
      10,
    );
    const exportH = parseInt(
      document.getElementById("ng-export-height").value,
      10,
    );
    const downsampling =
      !nativeOn &&
      (summary.modeWidth > exportW || summary.modeHeight > exportH);

    let text;
    if (summary.uniform) {
      text = `Source: all ${summary.totalCount} images are ${summary.modeWidth}\u00d7${summary.modeHeight}`;
    } else {
      text = `Source: mostly ${summary.modeWidth}\u00d7${summary.modeHeight} (${summary.modeCount}/${summary.totalCount}), range ${summary.minWidth}\u2013${summary.maxWidth} \u00d7 ${summary.minHeight}\u2013${summary.maxHeight}`;
    }
    if (downsampling) {
      text += ` \u2014 <span style="color:#e0a94a;">exporting at ${exportW}\u00d7${exportH} throws away resolution \u2014 tick "native resolution" below to keep it</span>`;
    } else if (nativeOn) {
      text += ` \u2014 native resolution export is on, source pixels are kept`;
    }
    note.innerHTML = text;
    note.style.display = "block";
    matchBtn.style.display =
      !nativeOn &&
      (downsampling ||
        exportW !== summary.modeWidth ||
        exportH !== summary.modeHeight)
        ? "inline-block"
        : "none";
    matchBtn.dataset.w = summary.modeWidth;
    matchBtn.dataset.h = summary.modeHeight;
  }

  function wireExportSettingsNG() {
    const cropModeSel = document.getElementById("ng-export-crop-mode");
    const marginRow = document.getElementById("ng-export-margin-row");
    const maxUpscaleRow = document.getElementById("ng-export-max-upscale-row");
    const padRow = document.getElementById("ng-export-pad-row");
    const upscaleCb = document.getElementById("ng-export-upscale");
    function syncRows() {
      const isFace = cropModeSel.value === "face";
      const showCap = isFace && upscaleCb.checked;
      marginRow.style.display = isFace ? "flex" : "none";
      maxUpscaleRow.style.display = showCap ? "flex" : "none";
      padRow.style.display = showCap ? "flex" : "none";
    }
    cropModeSel.addEventListener("change", syncRows);
    upscaleCb.addEventListener("change", syncRows);
    syncRows();

    document.querySelectorAll(".ng-export-preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("ng-export-width").value = btn.dataset.w;
        document.getElementById("ng-export-height").value = btn.dataset.h;
        applyResolutionSummaryNG();
      });
    });

    const nativeCb = document.getElementById("ng-export-native");
    const widthInput = document.getElementById("ng-export-width");
    const heightInput = document.getElementById("ng-export-height");
    function syncNativeState() {
      const on = nativeCb.checked;
      [widthInput, heightInput].forEach((el) => {
        el.style.opacity = on ? "0.55" : "1";
      });
      document.querySelectorAll(".ng-export-preset-btn").forEach((btn) => {
        btn.style.opacity = on ? "0.4" : "1";
      });
    }
    nativeCb.addEventListener("change", () => {
      syncNativeState();
      applyResolutionSummaryNG();
    });
    syncNativeState();

    const matchBtn = document.getElementById("ng-export-match-source-btn");
    matchBtn.addEventListener("click", () => {
      widthInput.value = matchBtn.dataset.w;
      heightInput.value = matchBtn.dataset.h;
      matchBtn.style.display = "none";
    });

    widthInput.addEventListener("input", () => applyResolutionSummaryNG());
    heightInput.addEventListener("input", () => applyResolutionSummaryNG());
  }

  window.AppNG.export = {
    gatherExportParamsNG,
    applyResolutionSummaryNG,
    wireExportSettingsNG,
  };
})();
