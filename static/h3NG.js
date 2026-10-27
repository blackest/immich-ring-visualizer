/**
 * h3NG.js -- the "H3" view (bottom-bar H3 task).
 *
 * MiniMax-H3 prompt/image-to-video, wired to the NG backend at
 * /api/ng/h3/* (routes/h3NG.py, h3_jobsNG.py, h3_engineNG.py). Twin of
 * videogenNG.js's queue-based shape -- same FIFO render queue, same
 * reference-image paste/drop/drag UX -- trimmed of what H3 doesn't
 * have server-side yet: no /enhance, /discuss, /chat (LTX's own
 * Gemma-side features), no
 * raw width/height/fps controls (steps default to h3_engineNG.H3_STEPS;
 * the optional Steps box overrides it). A `model` dropdown (h3 vs h3q8) replaces the model split
 * -- see h3_engineNG.py's docstring for why those are one engine, not
 * two -- and a `quality` dropdown (h3_engineNG.H3_QUALITIES, sent as
 * the named tier, resolved to width/height server-side) replaces
 * picking a canvas size directly.
 *
 * The durable Job Log panel lives in h3LogNG.js (calls loadRecipe below).
 *
 * Loaded after videogenNG.js and before bootstrapWiringNG.js (which
 * fires the first ProjectManager.render(), which calls H3NG.sync()).
 */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var API = "/api/ng/h3";

  var inited = false;
  var statusChecked = false;
  var generationDisabled = false;
  var currentRefBlob = null;
  var currentRefPreviewUrl = null;
  // Set only when currentRefBlob came from a live curation-session frame
  // -- see videogenNG.js's setReference for the full reasoning (this is
  // the same trick, just for H3's keyframe).
  var currentRefFrameId = null;
  var queue = []; // [{localId, jobId, status, refPreviewUrl, prompt, durationS, seed, model, quality, turbo, error, videoUrl, logTail}]
  var localSeq = 0;
  var pollTimer = null;
  var rowCache = {};

  // Duration bounds -- match h3_engineNG.py's H3_MIN/MAX_DURATION_S;
  // refreshed from /status once reachable so the two never drift apart.
  var durationBounds = { min: 3.0, max: 15.0 };

  var els = {};

  function refreshEls() {
    els.pane = document.getElementById("ng-h3-pane");
    els.main = document.getElementById("ng-h3-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.unavailable = document.getElementById("ng-h3-unavailable");
    els.unavailableText = document.getElementById("ng-h3-unavailable-text");
    els.licenseRow = document.getElementById("ng-h3-license-row");
    els.licenseCb = document.getElementById("ng-h3-license-cb");
    els.licenseText = document.getElementById("ng-h3-license-text");
    els.installBtn = document.getElementById("ng-h3-install-btn");
    els.installLog = document.getElementById("ng-h3-install-log");

    els.queueCount = document.getElementById("ng-h3-queue-count");
    els.queueEmpty = document.getElementById("ng-h3-queue-empty");
    els.queueClear = document.getElementById("ng-h3-queue-clear");
    els.queue = document.getElementById("ng-h3-queue");

    els.modeT2v = document.getElementById("ng-h3-mode-t2v");
    els.refCol = document.getElementById("ng-h3-ref-col");
    els.refPreview = document.getElementById("ng-h3-ref-preview");
    els.refEmpty = document.getElementById("ng-h3-ref-empty");
    els.refFile = document.getElementById("ng-h3-ref-file");
    els.refFileBtn = document.getElementById("ng-h3-ref-file-btn");
    els.refUseGen = document.getElementById("ng-h3-ref-use-gen");
    els.refPaste = document.getElementById("ng-h3-ref-paste");

    els.prompt = document.getElementById("ng-h3-prompt");
    els.duration = document.getElementById("ng-h3-duration");
    els.durationVal = document.getElementById("ng-h3-duration-val");
    els.seed = document.getElementById("ng-h3-seed");
    els.model = document.getElementById("ng-h3-model");
    els.quality = document.getElementById("ng-h3-quality");
    els.steps = document.getElementById("ng-h3-steps");
    els.stepsHint = document.getElementById("ng-h3-steps-hint");
    els.turboRow = document.getElementById("ng-h3-turbo-row");
    els.turbo = document.getElementById("ng-h3-turbo");
    els.v13Row = document.getElementById("ng-h3-v13-row");
    els.v13 = document.getElementById("ng-h3-v13");
    els.tristepRow = document.getElementById("ng-h3-tristep-row");
    els.tristep = document.getElementById("ng-h3-tristep");
    els.lorasBlock = document.getElementById("ng-h3-loras-block");
    els.lorasRows = document.getElementById("ng-h3-loras-rows");
    els.loraAdd = document.getElementById("ng-h3-lora-add");
    els.refmodsBlock = document.getElementById("ng-h3-refmods-block");
    els.refmodsNote = document.getElementById("ng-h3-refmods-note");
    els.refmodsRows = document.getElementById("ng-h3-refmods-rows");
    els.refmodAdd = document.getElementById("ng-h3-refmod-add");

    els.generateBtn = document.getElementById("ng-h3-generate-btn");
    els.status = document.getElementById("ng-h3-status");
    els.mainLog = document.getElementById("ng-h3-log");
  }

  // A speed adapter fixes the step count (Fast runs 3 steps on its own schedule, Turbo 4,
  // Faster Best 7), so while one is ticked the Steps box shows that number and is locked,
  // not the 16 default. The server forces the same value whatever is sent.
  var stepsDefaults = null;
  function syncStepsField() {
    if (!els.steps) return;
    if (stepsDefaults === null) {
      stepsDefaults = {
        placeholder: els.steps.placeholder,
        title: els.steps.title,
        hint: els.stepsHint ? els.stepsHint.textContent : "",
      };
    }
    var fixed = null;
    if (els.tristep && els.tristep.checked) fixed = { n: "3", why: "Fast runs 3 steps on its own schedule" };
    else if (els.turbo && els.turbo.checked) fixed = { n: "4", why: "Turbo fixes the step count at 4" };
    else if (els.v13 && els.v13.checked) fixed = { n: "7", why: "Faster Best fixes the step count at 7" };
    els.steps.disabled = !!fixed;
    if (fixed) els.steps.value = "";
    els.steps.placeholder = fixed ? fixed.n : stepsDefaults.placeholder;
    els.steps.title = fixed ? fixed.why : stepsDefaults.title;
    if (els.stepsHint) els.stepsHint.textContent = fixed ? fixed.why : stepsDefaults.hint;
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function setReference(blob, frameId) {
    if (currentRefPreviewUrl) URL.revokeObjectURL(currentRefPreviewUrl);
    currentRefBlob = blob;
    currentRefFrameId = frameId || null;
    currentRefPreviewUrl = URL.createObjectURL(blob);
    if (els.refPreview) {
      els.refPreview.src = currentRefPreviewUrl;
      els.refPreview.style.display = "";
    }
    if (els.refEmpty) els.refEmpty.style.display = "none";
  }

  // Same URL shape videogenNG.js's frameIdFromUrl relies on -- see that
  // file for the full reasoning.
  function frameIdFromUrl(url) {
    var m = /\/api\/ng\/framefile\/([^/?#]+)/.exec(url || "");
    return m ? m[1] : null;
  }

  function isT2vMode() {
    return !!(els.modeT2v && els.modeT2v.checked);
  }

  function onModeChange() {
    var t2v = isT2vMode();
    // Same "dim + disable in place, don't remove from the grid" reasoning
    // as videogenNG.js's onModeChange -- see its comment.
    if (els.refCol) els.refCol.classList.toggle("ng-vg-ref-col-disabled", t2v);
    setStatus("");
  }

  function onDiskFile() {
    var f = els.refFile.files && els.refFile.files[0];
    if (f) setReference(f);
  }

  function onRefPaste(e) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    var file = null;
    for (var i = 0; i < items.length; i++) {
      if (items[i].kind === "file" && /^image\//.test(items[i].type)) {
        file = items[i].getAsFile();
        break;
      }
    }
    if (!file) {
      setStatus("Clipboard has no image -- copy a frame first, then paste here.");
      return;
    }
    e.preventDefault();
    setReference(file);
    setStatus("Keyframe set from pasted frame.");
  }

  function onRefDragOver(e) {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    if (els.refCol) els.refCol.classList.add("ng-vg-ref-col-dragover");
  }

  function onRefDragLeave(e) {
    if (els.refCol && els.refCol.contains(e.relatedTarget)) return;
    if (els.refCol) els.refCol.classList.remove("ng-vg-ref-col-dragover");
  }

  function onRefDrop(e) {
    e.preventDefault();
    if (els.refCol) els.refCol.classList.remove("ng-vg-ref-col-dragover");
    var files = (e.dataTransfer && e.dataTransfer.files) || [];
    var file = null;
    for (var i = 0; i < files.length; i++) {
      if (/^image\//.test(files[i].type)) { file = files[i]; break; }
    }
    if (!file) {
      setStatus("That wasn't an image file.");
      return;
    }
    setReference(file);
    setStatus("Keyframe set from dropped image.");
  }

  function useGenerateReference() {
    var gen = window.GenerateNG && window.GenerateNG.getActiveReference
      ? window.GenerateNG.getActiveReference()
      : null;
    if (!gen || !gen.url) {
      setStatus("No reference set in Generate -- open a character or add a photo there first.");
      return;
    }
    setStatus("Pulling reference from Generate...");
    fetch(gen.url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.blob();
      })
      .then(function (blob) {
        setReference(blob, frameIdFromUrl(gen.url));
        setStatus("");
      })
      .catch(function (e) {
        setStatus("Could not pull Generate's reference: " + e.message);
      });
  }

  function setGenerationDisabled(disabled, health) {
    generationDisabled = disabled;
    if (els.generateBtn) els.generateBtn.disabled = disabled;
    if (els.unavailable) {
      els.unavailable.style.display = disabled ? "" : "none";
      if (disabled && els.unavailableText) {
        els.unavailableText.textContent =
          "H3 pipeline not reachable (venv/DiT/compact-pack missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
      var showInstall = !!(disabled && health && health.installable);
      if (els.licenseRow) {
        els.licenseRow.style.display = showInstall ? "" : "none";
        if (showInstall && els.licenseText) {
          els.licenseText.textContent =
            "I accept the " + (health.license_name || "engine license") +
            " (" + (health.license_note || "") + ") and want to download " +
            "~" + (health.install_size_gb || "?") + " GB.";
        }
      }
      if (els.installBtn) {
        els.installBtn.style.display = showInstall ? "" : "none";
        if (!showInstall) {
          els.installBtn.disabled = true;
          els.installBtn.textContent = "Install now";
          if (els.licenseCb) els.licenseCb.checked = false;
        } else {
          els.installBtn.disabled = !(els.licenseCb && els.licenseCb.checked);
        }
      }
    }
  }

  // ---- self-service install (see h3_installNG.py / engine_installNG.py) ----
  // Explicit opt-in: the button stays disabled until the license checkbox
  // is ticked (see setGenerationDisabled above and onLicenseCbChange below).
  var installPollTimer = null;

  function onLicenseCbChange() {
    if (els.installBtn) els.installBtn.disabled = !(els.licenseCb && els.licenseCb.checked);
  }

  function pollInstallJob(jobId) {
    fetch(API + "/install/" + jobId)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        if (!job || !job.status) return;
        if (els.installLog) {
          els.installLog.style.display = "";
          els.installLog.textContent = (job.log_tail || []).join("\n");
          els.installLog.scrollTop = els.installLog.scrollHeight;
        }
        if (job.status === "queued" || job.status === "installing") {
          if (els.installBtn) {
            els.installBtn.textContent = "Installing" + (job.step ? " (" + job.step + ")…" : "…");
          }
          return;
        }
        clearInterval(installPollTimer);
        installPollTimer = null;
        if (job.status === "completed") {
          if (els.installBtn) {
            els.installBtn.disabled = true;
            els.installBtn.textContent = "Installed — restart Ring Visualizer to use it";
          }
          setStatus("H3 installed. Restart the app to pick it up.");
        } else {
          if (els.installBtn) {
            els.installBtn.disabled = !(els.licenseCb && els.licenseCb.checked);
            els.installBtn.textContent = "Install now";
          }
          setStatus("Install failed: " + (job.error || "unknown error"));
        }
      })
      .catch(function () { /* keep polling -- a transient fetch error isn't fatal */ });
  }

  function startInstall() {
    if (!els.installBtn || !els.licenseCb || !els.licenseCb.checked) return;
    els.installBtn.disabled = true;
    els.installBtn.textContent = "Starting…";
    fetch(API + "/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_license: true }),
    })
      .then(function (res) {
        return res.json().then(function (payload) { return { ok: res.ok, payload: payload }; });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          if (installPollTimer) clearInterval(installPollTimer);
          installPollTimer = setInterval(function () { pollInstallJob(r.payload.job_id); }, 2500);
          pollInstallJob(r.payload.job_id);
        } else {
          els.installBtn.disabled = false;
          els.installBtn.textContent = "Install now";
          setStatus("Could not start install: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        els.installBtn.disabled = false;
        els.installBtn.textContent = "Install now";
        setStatus("Could not start install: " + e.message);
      });
  }

  // ---- User style/character LoRAs (GET /api/ng/h3/loras) ----
  // The library is whatever sits in the H3 loras folder; rows are built
  // here rather than in the HTML so the count can follow what the installed
  // runner can stack (max_stack, or max_stack_with_speed while turbo / v1.3
  // is ticked).
  var loraLib = { loras: [], max: 0, maxSpeed: 0, loaded: false };

  function loraLimit() {
    var speed = !!((els.turbo && els.turbo.checked) || (els.v13 && els.v13.checked)
      || (els.tristep && els.tristep.checked));
    return speed ? loraLib.maxSpeed : loraLib.max;
  }

  function loraEntry(name) {
    for (var i = 0; i < loraLib.loras.length; i++) {
      if (loraLib.loras[i].name === name) return loraLib.loras[i];
    }
    return null;
  }

  function updateLoraHint(row) {
    var hint = row.querySelector(".ng-h3-lora-hint");
    var entry = loraEntry(row.querySelector("select").value);
    var words = entry && entry.trigger_words && entry.trigger_words.length
      ? "Trigger: " + entry.trigger_words.join(" / ") : "";
    hint.textContent = words;
    hint.style.display = words ? "" : "none";
  }

  function syncLoraControls() {
    if (!els.lorasBlock) return;
    var limit = loraLimit();
    els.lorasBlock.hidden = !(loraLib.loras.length && (limit > 0 || els.lorasRows.children.length));
    var rows = els.lorasRows.children;
    // Over the limit (e.g. turbo got ticked on a single-slot runner): drop
    // the surplus rows rather than let a doomed request be sent.
    while (rows.length > limit) els.lorasRows.removeChild(rows[rows.length - 1]);
    if (els.loraAdd) els.loraAdd.disabled = rows.length >= limit;
  }

  function addLoraRow(name, strength) {
    if (!els.lorasRows) return;
    var row = document.createElement("div");
    row.className = "ng-h3-lora-row";
    row.style.cssText = "margin-bottom: 6px";
    var line = document.createElement("div");
    line.style.cssText = "display: flex; gap: 6px; align-items: center";
    var select = document.createElement("select");
    select.className = "ng-gen-input";
    select.style.cssText = "flex: 1; margin-bottom: 0";
    loraLib.loras.forEach(function (entry) {
      var opt = document.createElement("option");
      opt.value = entry.name;
      opt.textContent = (entry.ok ? "" : "(unusable) ") + (entry.title || entry.name);
      opt.disabled = !entry.ok;
      opt.title = entry.reason || "";
      select.appendChild(opt);
    });
    var strengthInput = document.createElement("input");
    strengthInput.type = "number";
    strengthInput.className = "ng-gen-input";
    strengthInput.step = "0.05";
    strengthInput.min = "0";
    strengthInput.max = "2";
    strengthInput.style.cssText = "width: 76px; margin-bottom: 0";
    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ng-gen-btn";
    remove.style.cssText = "width: auto";
    remove.textContent = "\u00d7";
    remove.title = "Remove this LoRA";
    var hint = document.createElement("div");
    hint.className = "ng-gen-label ng-h3-lora-hint";
    hint.style.cssText = "margin-top: 2px; text-transform: none; letter-spacing: 0; display: none";
    line.appendChild(select);
    line.appendChild(strengthInput);
    line.appendChild(remove);
    row.appendChild(line);
    row.appendChild(hint);
    els.lorasRows.appendChild(row);

    var first = loraLib.loras.filter(function (e) { return e.ok; })[0];
    if (name && loraEntry(name) && loraEntry(name).ok) select.value = name;
    else if (first) select.value = first.name;
    var chosen = loraEntry(select.value);
    strengthInput.value = strength != null ? strength
      : (chosen && chosen.recommended_strength) || 1;
    select.addEventListener("change", function () {
      var e = loraEntry(select.value);
      if (e && e.recommended_strength) strengthInput.value = e.recommended_strength;
      updateLoraHint(row);
    });
    remove.addEventListener("click", function () {
      els.lorasRows.removeChild(row);
      syncLoraControls();
    });
    updateLoraHint(row);
    syncLoraControls();
  }

  // [{name, strength}] for the rows currently on screen.
  function selectedLoras() {
    if (!els.lorasRows) return [];
    var out = [];
    Array.prototype.forEach.call(els.lorasRows.children, function (row) {
      var name = row.querySelector("select").value;
      var strength = parseFloat(row.querySelector("input").value);
      if (name) out.push({ name: name, strength: isNaN(strength) ? 1.0 : strength });
    });
    return out;
  }

  function ensureLorasLoaded() {
    if (loraLib.loaded) return;
    loraLib.loaded = true;
    fetch(API + "/loras")
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        loraLib.loras = (payload && payload.loras) || [];
        loraLib.max = (payload && payload.max_stack) || 0;
        loraLib.maxSpeed = (payload && payload.max_stack_with_speed) || 0;
        syncLoraControls();
      })
      .catch(function () {
        loraLib.loaded = false; // let a later sync retry
      });
  }

  // ---- RefMod references (GET /api/ng/h3/refmods) ----
  // Saved reference latents that the Ref2VA model reads as extra rows. Shown only
  // on the "H3 Ref2VA Q8" model; each row costs denoise time in proportion to its
  // `rows`, and a render is one 5 s window with no keyframe or speed adapter, so
  // the speed checkboxes are locked off while this model is picked.
  var refLib = { refmods: [], ready: false, ref2vaOk: false, supported: false,
                 maxVideos: 3, loaded: false };

  function refmodEntry(name) {
    for (var i = 0; i < refLib.refmods.length; i++) {
      if (refLib.refmods[i].name === name) return refLib.refmods[i];
    }
    return null;
  }

  function refmodsOn() { return !!(els.model && els.model.value === "h3ref"); }

  function refmodsNoteText() {
    if (refLib.ready) {
      return refLib.refmods.length
        ? "One 5 s window. Each reference adds denoise time (about 3x for a 2,800-row file)."
        : "No RefMod files yet -- put .safetensors RefMods in the H3 refmods folder.";
    }
    var missing = [];
    if (!refLib.ref2vaOk) missing.push("the Ref2VA Q8 weights");
    if (!refLib.supported) missing.push("a runner build that takes --refmod");
    return "Not available yet: this install is missing " + missing.join(" and ") + ".";
  }

  function updateRefmodHint(row) {
    var hint = row.querySelector(".ng-h3-refmod-hint");
    var entry = refmodEntry(row.querySelector("select").value);
    var bits = [];
    if (entry && entry.rows) bits.push(entry.rows.toLocaleString() + " reference rows");
    if (entry && entry.has_lora) bits.push("also carries a LoRA (not applied)");
    hint.textContent = bits.join(" \u00b7 ");
    hint.style.display = bits.length ? "" : "none";
  }

  function syncRefmodControls() {
    if (!els.refmodsBlock) return;
    var on = refmodsOn();
    els.refmodsBlock.hidden = !on;
    if (els.refmodsNote) els.refmodsNote.textContent = on ? refmodsNoteText() : "";
    var rows = els.refmodsRows.children;
    if (!on || !refLib.ready) {
      while (rows.length) els.refmodsRows.removeChild(rows[0]);
    }
    if (els.refmodAdd) {
      els.refmodAdd.disabled = !refLib.ready || !refLib.refmods.some(function (e) { return e.ok; })
        || rows.length >= refLib.maxVideos;
    }
    // Turbo / v1.3 were not made for the Ref2VA weights: lock them off here.
    // Fast (the TaoMate 3-step adapter) is the exception -- tested with a RefMod
    // on the Ref2VA Q8 -- so it is left alone.
    [els.turbo, els.v13].forEach(function (box) {
      if (!box) return;
      if (on) { box.checked = false; box.disabled = true; }
    });
  }

  function addRefmodRow(name, strength) {
    if (!els.refmodsRows) return;
    var row = document.createElement("div");
    row.className = "ng-h3-refmod-row";
    row.style.cssText = "margin-bottom: 6px";
    var line = document.createElement("div");
    line.style.cssText = "display: flex; gap: 6px; align-items: center";
    var select = document.createElement("select");
    select.className = "ng-gen-input";
    select.style.cssText = "flex: 1; margin-bottom: 0";
    // The shared library can hold ~1,500 files: a filter box narrows the list
    // (name or description), keeping the current pick if it still matches.
    var filter = document.createElement("input");
    filter.type = "search";
    filter.className = "ng-gen-input";
    filter.placeholder = "filter " + refLib.refmods.length + " RefMods…";
    filter.style.cssText = "width: 11em; margin-bottom: 0";
    function populate() {
      var keep = select.value;
      var q = filter.value.trim().toLowerCase();
      select.textContent = "";
      refLib.refmods.forEach(function (entry) {
        if (q && (entry.name + " " + (entry.description || "")).toLowerCase().indexOf(q) < 0) return;
        var opt = document.createElement("option");
        opt.value = entry.name;
        opt.textContent = (entry.ok ? "" : "(unusable) ") + entry.name;
        opt.disabled = !entry.ok;
        opt.title = entry.reason || "";
        select.appendChild(opt);
      });
      var opts = Array.prototype.slice.call(select.options);
      var pick = opts.filter(function (o) { return o.value === keep; })[0]
        || opts.filter(function (o) { return !o.disabled; })[0];
      if (pick) select.value = pick.value;
      if (row.parentNode) updateRefmodHint(row);
    }
    populate();
    var strengthInput = document.createElement("input");
    strengthInput.type = "number";
    strengthInput.className = "ng-gen-input";
    strengthInput.step = "0.05";
    strengthInput.min = "0";
    strengthInput.max = "1";
    strengthInput.style.cssText = "width: 76px; margin-bottom: 0";
    var remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ng-gen-btn";
    remove.style.cssText = "width: auto";
    remove.textContent = "\u00d7";
    remove.title = "Remove this RefMod";
    var hint = document.createElement("div");
    hint.className = "ng-gen-label ng-h3-refmod-hint";
    hint.style.cssText = "margin-top: 2px; text-transform: none; letter-spacing: 0; display: none";
    line.appendChild(filter);
    line.appendChild(select);
    line.appendChild(strengthInput);
    line.appendChild(remove);
    row.appendChild(line);
    row.appendChild(hint);
    els.refmodsRows.appendChild(row);

    var first = refLib.refmods.filter(function (e) { return e.ok; })[0];
    if (name && refmodEntry(name) && refmodEntry(name).ok) select.value = name;
    else if (first) select.value = first.name;
    strengthInput.value = strength != null ? strength : 1;
    select.addEventListener("change", function () { updateRefmodHint(row); });
    filter.addEventListener("input", populate);
    remove.addEventListener("click", function () {
      els.refmodsRows.removeChild(row);
      syncRefmodControls();
    });
    updateRefmodHint(row);
    syncRefmodControls();
  }

  // [{name, strength}] for the rows on screen -- only meaningful on h3ref.
  function selectedRefmods() {
    if (!els.refmodsRows || !refmodsOn()) return [];
    var out = [];
    Array.prototype.forEach.call(els.refmodsRows.children, function (row) {
      var name = row.querySelector("select").value;
      var strength = parseFloat(row.querySelector("input").value);
      if (name) out.push({ name: name, strength: isNaN(strength) ? 1.0 : strength });
    });
    return out;
  }

  function ensureRefmodsLoaded() {
    if (refLib.loaded) return;
    refLib.loaded = true;
    fetch(API + "/refmods")
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        refLib.refmods = (payload && payload.refmods) || [];
        refLib.ready = !!(payload && payload.ready);
        refLib.ref2vaOk = !!(payload && payload.ref2va_ok);
        refLib.supported = !!(payload && payload.refmod_supported);
        refLib.maxVideos = (payload && payload.max_videos) || 3;
        syncRefmodControls();
      })
      .catch(function () {
        refLib.loaded = false; // let a later sync retry
      });
  }

  function ensureStatusChecked() {
    if (statusChecked) return;
    statusChecked = true;
    var model = els.model ? els.model.value : "h3q8";
    fetch(API + "/status?model=" + encodeURIComponent(model))
      .then(function (r) { return r.json(); })
      .then(function (h) {
        setGenerationDisabled(!!(h && h.reachable === false), h);
        if (h && h.min_duration_s != null) {
          durationBounds = { min: h.min_duration_s, max: h.max_duration_s };
          if (els.duration) {
            els.duration.min = durationBounds.min;
            els.duration.max = durationBounds.max;
          }
        }
        var turboOk = !!(h && h.turbo_ok);
        if (els.turbo) els.turbo.disabled = !turboOk;
        if (els.turboRow) {
          els.turboRow.title = turboOk
            ? ""
            : "turbo LoRA files not found on this machine";
          els.turboRow.classList.toggle("ng-vg-ref-col-disabled", !turboOk);
        }
        var v13Ok = !!(h && h.v13_ok);
        if (els.v13) els.v13.disabled = !v13Ok;
        if (els.v13Row) {
          els.v13Row.title = v13Ok
            ? ""
            : "lightx2v v1.3 adapter not found -- build it from Phosphene's Speed row";
          els.v13Row.classList.toggle("ng-vg-ref-col-disabled", !v13Ok);
        }
        // Fast needs the adapter file AND a runner that takes --sigma-subset.
        var tristepOk = !!(h && h.tristep_ok && h.tristep_supported);
        if (els.tristep) {
          els.tristep.disabled = !tristepOk;
          if (!tristepOk) els.tristep.checked = false;
        }
        if (els.tristepRow) {
          els.tristepRow.title = tristepOk
            ? (h.lazy_adaln ? "Low-memory AdaLN load is on for this runner" : "")
            : (h && h.tristep_ok === false
              ? "TaoMate 3-step adapter not found -- Phosphene's Speed row downloads it"
              : "this H3 runner has no --sigma-subset -- update the minimax-h3-mlx runner");
          els.tristepRow.classList.toggle("ng-vg-ref-col-disabled", !tristepOk);
        }
        syncStepsField();
        // The status answer must not unlock what the Ref2VA model locks off.
        if (refmodsOn()) {
          [els.turbo, els.v13].forEach(function (box) {
            if (box) { box.checked = false; box.disabled = true; }
          });
        }
      })
      .catch(function () {
        statusChecked = false; // let a later sync retry
      });
  }

  function generateVideo() {
    var prompt = (els.prompt.value || "").trim();
    var t2v = isT2vMode();
    if (!t2v && !currentRefBlob) {
      setStatus("Pick a keyframe image first, or check text → video.");
      return;
    }
    if (!prompt) {
      setStatus("Describe the shot first.");
      els.prompt.focus();
      return;
    }
    var durationS = parseFloat(els.duration.value) || 5.0;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    var model = els.model ? els.model.value : "h3q8";
    var quality = els.quality ? els.quality.value : "high";
    var turbo = !!(els.turbo && !els.turbo.disabled && els.turbo.checked);
    var v13 = !!(els.v13 && !els.v13.disabled && els.v13.checked);
    var tristep = !!(els.tristep && !els.tristep.disabled && els.tristep.checked);

    var localId = "h" + ++localSeq;
    var item = {
      localId: localId,
      jobId: null,
      status: "submitting",
      refPreviewUrl: t2v ? null : currentRefPreviewUrl,
      prompt: prompt,
      durationS: durationS,
      seed: seed,
      model: model,
      quality: quality,
      turbo: turbo,
      v13: v13,
      tristep: tristep,
      error: null,
      videoUrl: null,
      logTail: [],
    };
    queue.unshift(item);
    renderQueue();

    var form = new FormData();
    if (!t2v) {
      if (currentRefFrameId) {
        form.append("ref_frame_id", currentRefFrameId);
      } else {
        var ext = currentRefBlob.type === "image/png" ? ".png" : ".jpg";
        form.append("file", currentRefBlob, "keyframe" + ext);
      }
    }
    form.append("prompt", prompt);
    form.append("duration_s", String(durationS));
    form.append("model", model);
    form.append("quality", quality);
    var steps = els.steps ? parseInt(els.steps.value, 10) : NaN;
    if (!isNaN(steps) && steps >= 4 && steps <= 40 && !els.steps.disabled) form.append("steps", String(steps));
    if (turbo) form.append("turbo", "true");
    if (v13) form.append("v13", "true");
    if (tristep) form.append("tristep", "true");
    var loras = selectedLoras();
    if (loras.length) form.append("loras", JSON.stringify(loras));
    var refmods = selectedRefmods();
    if (refmods.length) form.append("refmods", JSON.stringify(refmods));
    if (seed !== null && !isNaN(seed)) form.append("seed", String(seed));

    fetch(API + "/generate", { method: "POST", body: form })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          item.jobId = r.payload.job_id;
          item.status = "queued";
          setStatus("Queued.");
          ensurePolling();
        } else {
          item.status = "failed";
          item.error = (r.payload && r.payload.error) || "could not start job";
        }
        renderQueue();
      })
      .catch(function (e) {
        item.status = "failed";
        item.error = e.message;
        renderQueue();
      });
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(function (q) {
      return q.jobId && (q.status === "queued" || q.status === "rendering");
    });
    if (!active.length) {
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    active.forEach(function (item) {
      fetch(API + "/jobs/" + item.jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (!job || !job.status) return;
          if (job.log_tail) item.logTail = job.log_tail;
          if (job.status === "completed") {
            item.status = "done";
            item.videoUrl = job.video_url + "?v=" + Date.now();
          } else if (job.status === "failed") {
            item.status = "failed";
            item.error = job.error || "generation failed";
          } else {
            item.status = job.status; // "queued" | "rendering"
          }
          renderQueue();
        })
        .catch(function () {
          /* transient -- try again next tick */
        });
    });
  }

  function removeItem(localId) {
    var item = queue.filter(function (q) { return q.localId === localId; })[0];
    if (!item) return;
    queue = queue.filter(function (q) { return q.localId !== localId; });
    renderQueue();
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  function rowSignature(item) {
    return JSON.stringify([
      item.status, item.prompt, item.durationS, item.seed, item.model,
      item.quality, item.error, item.videoUrl, item.jobId, item.refPreviewUrl,
    ]);
  }

  function buildQueueRow(item) {
    var row = document.createElement("div");
    row.className = "ng-gen-queue-row st-" + item.status;

    var thumbs = document.createElement("div");
    thumbs.className = "thumbs";
    if (item.refPreviewUrl) {
      var img = document.createElement("img");
      img.src = item.refPreviewUrl;
      thumbs.appendChild(img);
    }
    row.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "meta";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = item.prompt;
    meta.appendChild(title);
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = item.durationS + "s, " + item.model +
      (item.quality ? ", " + item.quality : "") +
      (item.seed !== null ? ", seed " + item.seed : "");
    meta.appendChild(sub);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
    var st = document.createElement("span");
    st.className = "st";
    st.textContent =
      item.status === "done" ? "done" :
      item.status === "failed" ? "failed" :
      item.status === "rendering" ? "rendering..." :
      item.status === "submitting" ? "submitting..." : "queued";
    statusRow.appendChild(st);

    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pose-remove";
    // removeItem() also DELETEs the job server-side, which kills the
    // subprocess while it's rendering -- so this doubles as the abort
    // button; the title/label just makes that explicit in that state.
    rm.title = item.status === "rendering" ? "cancel render" : "remove from queue";
    rm.textContent = "✕";
    rm.addEventListener("click", function () { removeItem(item.localId); });
    statusRow.appendChild(rm);

    row.appendChild(statusRow);

    if (item.status === "failed" && item.error) {
      var err = document.createElement("div");
      err.className = "log";
      err.style.color = "var(--ng-danger)";
      err.textContent = item.error;
      row.appendChild(err);
    } else if (item.status === "done" && item.videoUrl) {
      var video = document.createElement("video");
      video.src = item.videoUrl;
      video.controls = true;
      video.style.gridColumn = "1 / -1";
      video.style.width = "100%";
      video.style.borderRadius = "4px";
      video.style.marginTop = "4px";
      row.appendChild(video);

      var dl = document.createElement("a");
      dl.className = "ng-btn ng-vg-download";
      dl.href = API + "/jobs/" + item.jobId + "/download";
      dl.download = "h3-" + item.jobId + ".mp4";
      dl.textContent = "Download video";
      dl.style.gridColumn = "1 / -1";
      dl.style.marginTop = "4px";
      row.appendChild(dl);
    }

    return row;
  }

  function renderQueue() {
    if (!els.queue) return;
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";

    els.queue.innerHTML = "";
    var newCache = {};
    queue.forEach(function (item) {
      var sig = rowSignature(item);
      var cached = rowCache[item.localId];
      var row = (cached && cached.sig === sig) ? cached.el : buildQueueRow(item);
      newCache[item.localId] = { sig: sig, el: row };
      els.queue.appendChild(row);
    });
    rowCache = newCache;

    updateMainLog();
  }

  function updateMainLog() {
    if (!els.mainLog) return;
    var active = queue.filter(function (q) { return q.status === "rendering"; })[0];
    if (!active || !active.logTail || !active.logTail.length) {
      els.mainLog.style.display = "none";
      els.mainLog.textContent = "";
      return;
    }
    els.mainLog.style.display = "";
    els.mainLog.textContent = active.logTail.join("\n");
    els.mainLog.scrollTop = els.mainLog.scrollHeight;
  }

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane) return;
    inited = true;

    if (els.installBtn) els.installBtn.addEventListener("click", startInstall);
    if (els.licenseCb) els.licenseCb.addEventListener("change", onLicenseCbChange);

    els.refFileBtn.addEventListener("click", function () { els.refFile.click(); });
    els.refFile.addEventListener("change", onDiskFile);
    els.refUseGen.addEventListener("click", useGenerateReference);
    if (els.refPaste) els.refPaste.addEventListener("paste", onRefPaste);
    if (els.refCol) {
      els.refCol.addEventListener("dragover", onRefDragOver);
      els.refCol.addEventListener("dragleave", onRefDragLeave);
      els.refCol.addEventListener("drop", onRefDrop);
    }
    if (els.modeT2v) els.modeT2v.addEventListener("change", onModeChange);

    els.duration.addEventListener("input", function () {
      els.durationVal.textContent = parseFloat(els.duration.value).toFixed(1) + "s";
    });
    if (els.model) {
      els.model.addEventListener("change", function () {
        statusChecked = false; // re-check reachability for the newly picked model
        ensureStatusChecked();
        ensureRefmodsLoaded();
        syncRefmodControls();
      });
    }
    if (els.turbo && els.model) {
      // Turbo's and v1.3's LoRAs only apply correctly against the bf16 DiT
      // -- against h3q8 (quantized) every non-adaLN module silently fails
      // its shape check and the adapter does nothing (see
      // h3_engineNG.py's generate_h3_video_ng). Force+lock model to "h3"
      // while either is on rather than let that combination be picked and
      // quietly do the wrong thing. The two are mutually exclusive (one
      // speed adapter per run); v1.3 is also only validated at "high".
      var modelBeforeSpeed = null;
      var qualityBeforeV13 = null;
      var syncSpeedLocks = function () {
        var anySpeed = els.turbo.checked || (els.v13 && els.v13.checked);
        if (anySpeed && modelBeforeSpeed === null) {
          modelBeforeSpeed = els.model.value;
          els.model.value = "h3";
          els.model.disabled = true;
        } else if (!anySpeed && modelBeforeSpeed !== null) {
          els.model.disabled = false;
          els.model.value = modelBeforeSpeed;
          modelBeforeSpeed = null;
        }
        if (els.quality) {
          var v13On = !!(els.v13 && els.v13.checked);
          if (v13On && qualityBeforeV13 === null) {
            qualityBeforeV13 = els.quality.value;
            els.quality.value = "high";
            els.quality.disabled = true;
          } else if (!v13On && qualityBeforeV13 !== null) {
            els.quality.disabled = false;
            els.quality.value = qualityBeforeV13;
            qualityBeforeV13 = null;
          }
        }
        statusChecked = false;
        ensureStatusChecked();
      };
      els.turbo.addEventListener("change", function () {
        if (els.turbo.checked && els.v13) els.v13.checked = false;
        if (els.turbo.checked && els.tristep) els.tristep.checked = false;
        syncSpeedLocks();
      });
      if (els.v13) {
        els.v13.addEventListener("change", function () {
          if (els.v13.checked) els.turbo.checked = false;
          if (els.v13.checked && els.tristep) els.tristep.checked = false;
          syncSpeedLocks();
        });
      }
      // Fast never locks the model (it works on the Q8 DiTs); ticking it only has
      // to release a lock turbo / v1.3 may have been holding, and clear them.
      if (els.tristep) {
        els.tristep.addEventListener("change", function () {
          if (els.tristep.checked) {
            els.turbo.checked = false;
            if (els.v13) els.v13.checked = false;
          }
          syncSpeedLocks();
        });
      }
    }

    if (els.loraAdd) {
      els.loraAdd.addEventListener("click", function () {
        if (els.lorasRows.children.length < loraLimit()) addLoraRow();
      });
    }
    [els.turbo, els.v13, els.tristep].forEach(function (box) {
      if (box) box.addEventListener("change", syncLoraControls);
      if (box) box.addEventListener("change", syncStepsField);
    });
    if (els.refmodAdd) {
      els.refmodAdd.addEventListener("click", function () {
        if (els.refmodsRows.children.length < refLib.maxVideos) addRefmodRow();
      });
    }

    els.generateBtn.addEventListener("click", generateVideo);
    els.queueClear.addEventListener("click", function () {
      queue = queue.filter(function (q) {
        return q.status !== "done" && q.status !== "failed";
      });
      renderQueue();
    });
  }

  // Quality tiers by canvas size -- mirrors h3_engineNG.H3_QUALITIES so a
  // Job Log entry (which stores width/height, not the tier name) can
  // re-select the right dropdown option.
  var QUALITY_BY_DIMS = {
    "640x384": "draft", "768x448": "standard",
    "1024x576": "high", "1344x768": "native",
  };

  // Called by h3LogNG.js's "Use this": refills the compose form from a
  // durable Job Log entry (h3_job_logsNG.py). refUrl is the entry's saved
  // keyframe (null = it was a text-to-video job).
  function loadRecipe(entry, refUrl) {
    refreshEls();
    if (!els.prompt) return;
    els.prompt.value = entry.prompt || "";
    if (els.seed) els.seed.value = entry.seed != null ? entry.seed : "";
    if (typeof entry.duration_s === "number" && els.duration) {
      els.duration.value = entry.duration_s;
      if (els.durationVal) els.durationVal.textContent = parseFloat(els.duration.value).toFixed(1) + "s";
    }
    var tier = QUALITY_BY_DIMS[entry.width + "x" + entry.height];
    if (tier && els.quality) els.quality.value = tier;
    // Model before turbo: ticking turbo locks the model dropdown to "h3".
    if (els.turbo && els.turbo.checked && !entry.turbo) {
      els.turbo.checked = false;
      els.turbo.dispatchEvent(new Event("change"));
    }
    if (els.model && entry.model && !entry.turbo) els.model.value = entry.model;
    if (entry.turbo && els.turbo && !els.turbo.disabled && !els.turbo.checked) {
      els.turbo.checked = true;
      els.turbo.dispatchEvent(new Event("change"));
    }
    if (els.tristep && !els.tristep.disabled && !!els.tristep.checked !== !!entry.tristep) {
      els.tristep.checked = !!entry.tristep;
      els.tristep.dispatchEvent(new Event("change"));
    }
    if (els.lorasRows) {
      els.lorasRows.innerHTML = "";
      (entry.loras || []).forEach(function (l) {
        if (els.lorasRows.children.length < loraLimit() && loraEntry(l.name)) {
          addLoraRow(l.name, l.strength);
        }
      });
      syncLoraControls();
    }
    if (els.modeT2v) {
      els.modeT2v.checked = !refUrl;
      onModeChange();
    }
    setStatus("Loaded prompt/seed/settings from the job log.");
    if (!refUrl) return;
    fetch(refUrl)
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.blob(); })
      .then(function (blob) { setReference(blob); })
      .catch(function (e) {
        if (els.modeT2v) { els.modeT2v.checked = true; onModeChange(); }
        setStatus("Loaded settings, but the saved keyframe wouldn't load: " + e.message);
      });
  }

  // ---- called from ProjectManager.render() every tick ----
  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "h3");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;

    init();
    ensureStatusChecked();
    ensureLorasLoaded();
    ensureRefmodsLoaded();
    syncRefmodControls();
    renderQueue();
  }

  window.H3NG = { sync: sync, loadRecipe: loadRecipe };
})();
