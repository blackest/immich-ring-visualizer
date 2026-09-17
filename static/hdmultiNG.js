/**
 * hdmultiNG.js -- the "Hd-Multi" view (bottom-bar Hd-Multi task).
 *
 * One-off HiDream edit/multi-ref jobs: pick K (1-3) reference images,
 * write one prompt, get one result image back. Wired to the NG backend
 * at /api/ng/hdmulti/* (routes/hdmultiNG.py). Deliberately its own view
 * rather than folded into Generate's pose grid -- see routes/hdmultiNG.py's
 * module docstring for why (different inputs/outputs: 1-3 arbitrary
 * images + one prompt -> one result, not "one character -> a full sheet
 * of posed shots").
 *
 * No project/character concept at all -- a job here doesn't belong to
 * any tab's project, so nothing here reads or writes CharacterProject
 * state. Only owns its own pane/main DOM, same shape as Chat/Rachel/
 * Animate. Loaded after videogenNG.js and before bootstrapWiringNG.js
 * (which fires the first ProjectManager.render(), which calls
 * HdMultiNG.sync()).
 */
(function () {
  "use strict";

  var MAX_K = 3;

  // Same curated trained-resolution list as Generate's #ng-gen-size-preset
  // (this file's template, and HIDREAM_TRAINED_RESOLUTIONS in
  // hidream_engineNG.py has the full backend list this is drawn from).
  // Anything outside this list renders blank in this pipeline's
  // reference-conditioned mode -- which HD-Multi always is -- so unlike
  // refDims below, this is never fed a reference image's raw pixel size.
  var SIZE_PRESETS = [
    [2048, 2048], [2304, 1728], [1728, 2304], [2560, 1440], [1440, 2560],
  ];

  var inited = false;
  var els = {};
  var k = 1;
  var refImgEls = [null, null, null]; // decoded <img>, kept around to redraw at a new target size without re-fetching
  var refFiles = [null, null, null]; // the actual upload blob per slot -- always center-cropped+resized to exactly the current width/height
  var refUrls = [null, null, null]; // object URL of refFiles[i], for the slot preview (so the preview shows exactly what gets sent)
  var refDims = [null, null, null]; // original {w, h} per slot, read from the loaded image -- used only to auto-pick the closest-aspect preset, never sent as-is
  var sizeTouched = false; // true once the user picks a preset or edits width/height by hand -- stops auto-picking on future ref loads
  var currentJobId = null;
  var pollTimer = null;
  var busy = false;
  var lastResultSeed = null; // seed the most recent finished job actually used -- "Reuse this seed" copies this back into the seed field

  function $(id) {
    return document.getElementById(id);
  }

  function refreshEls() {
    els.pane = $("ng-hdmulti-pane");
    els.main = $("ng-hdmulti-main");
    els.controlsPane = $("ng-controls-pane");
    els.kRadios = Array.from(document.querySelectorAll('input[name="ng-hdmulti-k"]'));
    els.refSlots = $("ng-hdmulti-ref-slots");
    els.prompt = $("ng-hdmulti-prompt");
    els.generateBtn = $("ng-hdmulti-generate-btn");
    els.status = $("ng-hdmulti-status");
    els.log = $("ng-hdmulti-log");
    els.preview = $("ng-hdmulti-preview");
    els.sizePreset = $("ng-hdmulti-size-preset");
    els.width = $("ng-hdmulti-width");
    els.height = $("ng-hdmulti-height");
    els.steps = $("ng-hdmulti-steps");
    els.seed = $("ng-hdmulti-seed");
    els.resultWrap = $("ng-hdmulti-result-wrap");
    els.resultImg = $("ng-hdmulti-result-img");
    els.resultSeed = $("ng-hdmulti-result-seed");
    els.reuseSeedBtn = $("ng-hdmulti-reuse-seed");
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function revokeSlot(i) {
    if (refUrls[i]) {
      URL.revokeObjectURL(refUrls[i]);
      refUrls[i] = null;
    }
  }

  // Center-crop `img` to the target aspect ratio, then draw it at exactly
  // targetW x targetH. Cropping (not letterboxing) so the pipeline only
  // ever sees real pixels -- a black-bar pad would be a fake region the
  // instruction-edit model has to reconcile too, which tends to show up
  // as artifacts/bleed at the border instead.
  function cropAndResize(img, targetW, targetH) {
    return new Promise(function (resolve) {
      var srcRatio = img.naturalWidth / img.naturalHeight;
      var dstRatio = targetW / targetH;
      var sx, sy, sw, sh;
      if (srcRatio > dstRatio) {
        sh = img.naturalHeight;
        sw = sh * dstRatio;
        sx = (img.naturalWidth - sw) / 2;
        sy = 0;
      } else {
        sw = img.naturalWidth;
        sh = sw / dstRatio;
        sx = 0;
        sy = (img.naturalHeight - sh) / 2;
      }
      var canvas = document.createElement("canvas");
      canvas.width = targetW;
      canvas.height = targetH;
      canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, targetW, targetH);
      canvas.toBlob(resolve, "image/png");
    });
  }

  // Re-crops/resizes slot i's already-decoded image to whatever the
  // width/height fields currently say, and swaps in the resulting blob as
  // the thing that actually gets uploaded -- so the preview thumbnail and
  // the upload are always the same bytes, guaranteed to match the
  // resolution the request asks for (rather than leaving the pipeline to
  // reconcile a mismatch itself, which is what produced the blank/garbled
  // output before this).
  function reprocessSlot(i) {
    var img = refImgEls[i];
    if (!img) return;
    var targetW = (els.width && parseInt(els.width.value, 10)) || 2048;
    var targetH = (els.height && parseInt(els.height.value, 10)) || 2048;
    cropAndResize(img, targetW, targetH).then(function (blob) {
      if (refImgEls[i] !== img) return; // slot replaced/cleared while cropping
      revokeSlot(i);
      refFiles[i] = new File([blob], "ref" + i + ".png", { type: "image/png" });
      refUrls[i] = URL.createObjectURL(blob);
      renderRefSlots();
      updatePreview();
    });
  }

  function reprocessAllSlots() {
    for (var i = 0; i < MAX_K; i++) reprocessSlot(i);
  }

  function setRef(i, file) {
    if (!file) return;
    refImgEls[i] = null;
    refDims[i] = null;
    var rawUrl = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () {
      URL.revokeObjectURL(rawUrl); // decoded pixels stay usable on the <img> after this
      refImgEls[i] = img;
      refDims[i] = { w: img.naturalWidth, h: img.naturalHeight };
      if (i === firstFilledSlot() && !sizeTouched) applyPreset(nearestPreset(refDims[i]));
      reprocessSlot(i);
    };
    img.onerror = function () {
      URL.revokeObjectURL(rawUrl);
      setStatus("Couldn't read that image.");
    };
    img.src = rawUrl;
  }

  function clearRef(i) {
    revokeSlot(i);
    refImgEls[i] = null;
    refFiles[i] = null;
    refDims[i] = null;
    renderRefSlots();
    updatePreview();
  }

  function firstFilledSlot() {
    for (var i = 0; i < MAX_K; i++) {
      if (refImgEls[i]) return i;
    }
    return -1;
  }

  function nearestPreset(dims) {
    var ratio = dims.w / dims.h;
    var best = SIZE_PRESETS[0], minDiff = Infinity;
    SIZE_PRESETS.forEach(function (wh) {
      var diff = Math.abs(wh[0] / wh[1] - ratio);
      if (diff < minDiff) {
        minDiff = diff;
        best = wh;
      }
    });
    return best;
  }

  function applyPreset(wh) {
    if (els.sizePreset) els.sizePreset.value = wh[0] + "x" + wh[1];
    if (els.width) els.width.value = wh[0];
    if (els.height) els.height.value = wh[1];
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function updatePreview() {
    if (!els.preview) return;
    var prompt = (els.prompt && els.prompt.value || "").trim();
    var promptPreview = prompt
      ? escapeHtml(prompt.length > 160 ? prompt.slice(0, 160) + "…" : prompt)
      : "(none yet)";
    var refCount = refFiles.slice(0, k).filter(Boolean).length;
    var w = (els.width && parseInt(els.width.value, 10)) || 2048;
    var h = (els.height && parseInt(els.height.value, 10)) || 2048;
    var steps = (els.steps && parseInt(els.steps.value, 10)) || 28;
    var isPreset = els.sizePreset && els.sizePreset.value !== "custom";
    var seedRaw = (els.seed && els.seed.value || "").trim();
    var seedText = seedRaw ? seedRaw : "random";
    els.preview.innerHTML =
      "<strong>Will send:</strong> K=" + k + ", " + refCount + " reference image" + (refCount === 1 ? "" : "s") + "\n" +
      "<strong>Resolution:</strong> " + w + "×" + h + (isPreset ? "" : " (custom — may render blank)") + "\n" +
      "<strong>Steps:</strong> " + steps + " · <strong>Seed:</strong> " + seedText + "\n" +
      "<strong>Prompt:</strong> " + promptPreview;
  }

  function handleSlotPaste(e, i) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    for (var j = 0; j < items.length; j++) {
      if (items[j].kind === "file" && /^image\//.test(items[j].type)) {
        e.preventDefault();
        setRef(i, items[j].getAsFile());
        return;
      }
    }
  }

  function pickFromDisk(i) {
    var input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.addEventListener("change", function () {
      if (input.files && input.files[0]) setRef(i, input.files[0]);
    });
    input.click();
  }

  // Same merged preview/paste box as Generate's reference tray
  // (.ng-gen-ref-paste, see generateNG.js's renderRefPreview) -- one per
  // active K slot, each independently pasteable/pickable/removable
  // rather than a single shared active selection, since a compose job
  // needs all K images at once, not one-at-a-time picking.
  function renderRefSlots() {
    if (!els.refSlots) return;
    els.refSlots.innerHTML = "";
    for (var i = 0; i < k; i++) {
      (function (i) {
        var wrap = document.createElement("div");
        wrap.className = "ng-hdmulti-ref-slot-wrap";

        var label = document.createElement("div");
        label.className = "ng-gen-label";
        label.textContent = "Reference " + (i + 1);
        wrap.appendChild(label);

        var box = document.createElement("div");
        box.className = "ng-gen-ref-paste ng-hdmulti-ref-paste" + (refUrls[i] ? " ng-gen-ref-paste-filled" : "");
        box.tabIndex = 0;
        box.title = "Click here, then paste (Cmd+V) an image, or click the button below to choose from disk.";
        if (refUrls[i]) {
          box.innerHTML = '<img src="' + refUrls[i] + '" alt="">';
        } else {
          box.textContent = "Paste an image here (Ctrl+V)";
        }
        box.addEventListener("paste", function (e) {
          handleSlotPaste(e, i);
        });
        wrap.appendChild(box);

        var btnRow = document.createElement("div");
        btnRow.className = "ng-hdmulti-ref-slot-btns";

        var diskBtn = document.createElement("button");
        diskBtn.type = "button";
        diskBtn.className = "ng-gen-btn ng-gen-btn-quiet";
        diskBtn.textContent = "+ choose from disk…";
        diskBtn.addEventListener("click", function () {
          pickFromDisk(i);
        });
        btnRow.appendChild(diskBtn);

        if (refUrls[i]) {
          var clearBtn = document.createElement("button");
          clearBtn.type = "button";
          clearBtn.className = "ng-gen-btn ng-gen-btn-quiet";
          clearBtn.textContent = "Clear";
          clearBtn.addEventListener("click", function () {
            clearRef(i);
          });
          btnRow.appendChild(clearBtn);
        }
        wrap.appendChild(btnRow);

        els.refSlots.appendChild(wrap);
      })(i);
    }
  }

  function onKChange() {
    var checked = els.kRadios.filter(function (r) {
      return r.checked;
    })[0];
    var newK = checked ? parseInt(checked.value, 10) : 1;
    // Shrinking K drops the now-unused slots' files/previews rather than
    // just hiding them -- growing K back wouldn't otherwise know whether
    // a re-shown slot's stale file was intentional.
    for (var i = newK; i < MAX_K; i++) clearRef(i);
    k = newK;
    renderRefSlots();
    updatePreview();
  }

  function resetResult() {
    if (els.resultWrap) els.resultWrap.style.display = "none";
    if (els.resultImg) els.resultImg.src = "";
  }

  function renderLog(lines) {
    if (!els.log) return;
    if (!lines || !lines.length) {
      els.log.style.display = "none";
      els.log.textContent = "";
      return;
    }
    els.log.style.display = "";
    els.log.textContent = lines.join("\n");
    els.log.scrollTop = els.log.scrollHeight;
  }

  function setBusy(b) {
    busy = b;
    if (els.generateBtn) els.generateBtn.disabled = b;
    els.kRadios.forEach(function (r) {
      r.disabled = b;
    });
  }

  function startGenerate() {
    if (busy) return;
    var prompt = (els.prompt.value || "").trim();
    if (!prompt) {
      setStatus("Write a prompt first.");
      return;
    }
    var anyRef = refFiles.slice(0, k).some(function (f) {
      return !!f;
    });
    if (!anyRef) {
      setStatus("Add at least one reference image.");
      return;
    }

    var form = new FormData();
    form.append("prompt", prompt);
    for (var i = 0; i < k; i++) {
      if (refFiles[i]) form.append("ref" + i, refFiles[i]);
    }
    form.append("width", String((els.width && parseInt(els.width.value, 10)) || 2048));
    form.append("height", String((els.height && parseInt(els.height.value, 10)) || 2048));
    form.append("steps", String((els.steps && parseInt(els.steps.value, 10)) || 28));
    var seedRaw = (els.seed && els.seed.value || "").trim();
    if (seedRaw) form.append("seed", seedRaw);

    setBusy(true);
    resetResult();
    renderLog(null);
    setStatus("Starting…");

    fetch("/api/ng/hdmulti/generate", { method: "POST", body: form })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        if (data.error) {
          setStatus("Error: " + data.error);
          setBusy(false);
          return;
        }
        currentJobId = data.jobId;
        poll();
      })
      .catch(function (e) {
        setStatus("Error: " + e.message);
        setBusy(false);
      });
  }

  function poll() {
    if (!currentJobId) return;
    fetch("/api/ng/hdmulti/status/" + currentJobId)
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        if (data.error) {
          setStatus("Error: " + data.error);
          setBusy(false);
          return;
        }
        if (data.status === "running") {
          var lastLog = data.log && data.log.length ? data.log[data.log.length - 1] : "";
          setStatus("Generating… " + (lastLog || ""));
          renderLog(data.log);
          pollTimer = setTimeout(poll, 1500);
          return;
        }
        if (data.status === "error") {
          setStatus("Error: " + (data.error || "unknown error"));
          renderLog(data.log);
          setBusy(false);
          return;
        }
        // done
        setStatus(data.seed != null ? "Done — seed " + data.seed : "Done.");
        renderLog(data.log);
        setBusy(false);
        els.resultImg.src = "/api/ng/hdmulti/result/" + currentJobId + "?t=" + Date.now();
        if (els.resultSeed) els.resultSeed.textContent = data.seed != null ? "seed " + data.seed : "";
        lastResultSeed = data.seed != null ? String(data.seed) : null;
        if (els.reuseSeedBtn) els.reuseSeedBtn.style.display = lastResultSeed ? "" : "none";
        els.resultWrap.style.display = "";
      })
      .catch(function (e) {
        setStatus("Error: " + e.message);
        setBusy(false);
      });
  }

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane || !els.main) return;
    inited = true;

    els.kRadios.forEach(function (r) {
      r.addEventListener("change", onKChange);
    });
    els.generateBtn.addEventListener("click", startGenerate);
    if (els.prompt) els.prompt.addEventListener("input", updatePreview);
    if (els.sizePreset) {
      els.sizePreset.addEventListener("change", function () {
        sizeTouched = true;
        var v = els.sizePreset.value;
        if (v === "custom") {
          updatePreview();
          return;
        }
        var wh = v.split("x");
        els.width.value = wh[0];
        els.height.value = wh[1];
        updatePreview();
        reprocessAllSlots(); // re-crop already-loaded refs to the newly picked size
      });
    }
    [els.width, els.height].forEach(function (inp) {
      if (!inp) return;
      // "input" fires per keystroke -- just keep the preview text live.
      // Re-cropping every keystroke would thrash the canvas work for no
      // reason, so that only runs on "change" (blur/enter/spinner-click).
      inp.addEventListener("input", function () {
        sizeTouched = true;
        if (els.sizePreset) els.sizePreset.value = "custom";
        updatePreview();
      });
      inp.addEventListener("change", reprocessAllSlots);
    });
    if (els.steps) els.steps.addEventListener("input", updatePreview);
    if (els.seed) els.seed.addEventListener("input", updatePreview);
    if (els.reuseSeedBtn) {
      els.reuseSeedBtn.addEventListener("click", function () {
        if (lastResultSeed == null || !els.seed) return;
        els.seed.value = lastResultSeed;
        updatePreview();
      });
    }
    renderRefSlots();
    updatePreview();
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "hdmulti");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    // Runs last in ProjectManager.render()'s sync chain (see that file) --
    // same "unconditional hide-when-on wins" deal as VideoGenNG.sync,
    // just simpler since nothing needs to run after this one to restore
    // #ng-controls-pane when Hd-Multi isn't the active task.
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;
    init();
  }

  window.HdMultiNG = { sync: sync };
})();
