/**
 * generateNG.js -- the "Generate" view (bottom-bar Generate task).
 *
 * Self-contained character-sheet generation, wired to the NG backend at
 * /api/ng/generate/* (routes/generateNG.py). This file deliberately does
 * NOT touch or reuse phosphene-sheet.js -- that stays the original page's
 * working code. The only thing this view reads from the rest of the app
 * is the active project's reference image(s): the ring anchor plus any
 * ring candidates the user has selected, surfaced as a small tray. A
 * photo picked from disk works too, with no project at all.
 *
 * Model: a flat FIFO "render queue" (a shared printer). Each queue entry
 * is one (reference image, pose) pairing. "Add to queue" fans the ticked
 * poses out into one job per pose -- POST /sheet-from-upload with
 * views=["<key>"] -- so the backend's single worker renders them one at a
 * time, in add order, and different references/characters interleave
 * naturally (petra pose A, then petra pose C from another photo, ...).
 *
 * Distinct reference images can't share a character id -- the backend's
 * create_draft_character_ng refuses to overwrite an existing avatar and
 * the upload route silently swallows that -- so each reference gets its
 * own trigger: "petra", then "petra-2", "petra-3", ... (auto, invisible;
 * the queue row shows the ref thumbnail so they're still tellable apart).
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after projectManagerNG.js and before bootstrapWiringNG.js (which fires
 * the first ProjectManager.render(), which calls GenerateNG.sync()).
 * In-memory only -- the queue is lost on refresh, same as the backend's
 * own job store.
 */
(function () {
  "use strict";

  var POLL_MS = 2000;
  var API = "/api/ng/generate";
  var POSE_STORAGE_KEY = "ringviz-ng-generate-poses";
  var CUSTOM_POSE_STORAGE_KEY = "ringviz-ng-generate-custom-poses";

  // ---- module state ----
  var inited = false;
  var presetsLoaded = false;
  var poseListSeeded = false; // seed ticks from localStorage on first render only
  var poseCatalogue = []; // [{key, pose, preset}]
  var styleNames = ["none"];
  var sceneChoices = []; // [{key, label, phrase}] -- per-shot "scene" dropdown options
  var customPoses = loadCustomPoses(); // [{key, pose}] user-added poses, persisted
  // Per-shot state for the pose grid's "edit prompt" / scene override --
  // key -> {scene, promptOverride, open}. In-memory only (unlike ticked
  // poses/custom poses, not persisted -- these are meant as one-off
  // tweaks before queuing, not standing preferences).
  var shotOverrides = {};
  var refs = []; // [{id, label, kind: "proj"|"disk", url, file?}]
  var activeRefId = null;
  var queue = []; // [{localId, character, key, refId, refUrl, settings, jobId, status, thumbUrl, error}]
  var localSeq = 0;
  var pollTimer = null;
  var pumping = false;
  var generationDisabled = false;
  var refTriggers = {}; // "base|refId" -> trigger
  var baseCounts = {}; // base -> distinct-ref count so far

  var ST_LABEL = {
    pending: "waiting",
    submitting: "submitting…",
    queued: "queued",
    rendering: "rendering…",
    done: "done",
    failed: "failed",
  };

  // ---- DOM (owned here; queried lazily so script load order can't bite) ----
  var els = {};
  function $(id) {
    return document.getElementById(id);
  }
  function refreshEls() {
    els.pane = $("ng-generate-pane");
    els.main = $("ng-generate-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-gen-unavailable");
    els.name = $("ng-gen-name");
    els.refTray = $("ng-gen-ref-tray");
    els.refFile = $("ng-gen-ref-file");
    els.refFileBtn = $("ng-gen-ref-file-btn");
    els.style = $("ng-gen-style");
    els.sizePreset = $("ng-gen-size-preset");
    els.width = $("ng-gen-width");
    els.height = $("ng-gen-height");
    els.steps = $("ng-gen-steps");
    els.wardrobe = $("ng-gen-wardrobe");
    els.hair = $("ng-gen-hair");
    els.seed = $("ng-gen-seed");
    els.identityLock = $("ng-gen-identity-lock");
    els.poseAll = $("ng-gen-pose-all");
    els.poseNone = $("ng-gen-pose-none");
    els.poseAddCustom = $("ng-gen-pose-add-custom");
    els.customPoseForm = $("ng-gen-custom-pose-form");
    els.customPoseName = $("ng-gen-custom-pose-name");
    els.customPoseText = $("ng-gen-custom-pose-text");
    els.customPoseCancel = $("ng-gen-custom-pose-cancel");
    els.customPoseSave = $("ng-gen-custom-pose-save");
    els.poseList = $("ng-gen-pose-list");
    els.addBtn = $("ng-gen-add-btn");
    els.status = $("ng-gen-status");
    els.queue = $("ng-gen-queue");
    els.queueEmpty = $("ng-gen-queue-empty");
    els.queueCount = $("ng-gen-queue-count");
    els.queueClear = $("ng-gen-queue-clear");
    els.ringActions = $("ng-gen-ring-actions");
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // Trigger / character-id: the backend's _safe_id_ng allows only
  // [A-Za-z0-9 _-], so strip everything else.
  function slugName(s) {
    var v = String(s || "")
      .replace(/[^A-Za-z0-9 _-]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    return v || "character";
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  // ---- one-time wiring ----
  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane) return;
    inited = true;

    els.refFileBtn.addEventListener("click", function () {
      els.refFile.click();
    });
    els.refFile.addEventListener("change", onDiskFile);

    els.sizePreset.addEventListener("change", function () {
      var v = els.sizePreset.value;
      if (v === "custom") return;
      var wh = v.split("x");
      els.width.value = wh[0];
      els.height.value = wh[1];
    });
    [els.width, els.height].forEach(function (inp) {
      inp.addEventListener("input", function () {
        els.sizePreset.value = "custom";
      });
    });

    els.poseAll.addEventListener("click", function (e) {
      e.preventDefault();
      togglePoses(true);
    });
    els.poseNone.addEventListener("click", function (e) {
      e.preventDefault();
      togglePoses(false);
    });
    // Delegated: fires for every checkbox tick/untick since the rows are
    // rebuilt wholesale by renderPoseList(). Persist so a page refresh
    // doesn't wipe the picks (they used to be pure DOM state -- gone the
    // moment you reloaded).
    els.poseList.addEventListener("change", function (e) {
      if (e.target && e.target.type === "checkbox") savePoseKeys();
    });

    els.poseAddCustom.addEventListener("click", function (e) {
      e.preventDefault();
      els.customPoseForm.style.display = "";
      els.customPoseName.value = "";
      els.customPoseText.value = "";
      els.customPoseName.focus();
    });
    els.customPoseCancel.addEventListener("click", function () {
      els.customPoseForm.style.display = "none";
    });
    els.customPoseSave.addEventListener("click", function () {
      var pose = (els.customPoseText.value || "").trim();
      if (!pose) {
        els.customPoseText.focus();
        return;
      }
      var name = (els.customPoseName.value || "").trim();
      var base = "custom_" + slugName(name || pose).toLowerCase().replace(/\s+/g, "_");
      var key = base;
      var n = 2;
      while (customPoses.some(function (c) { return c.key === key; })) {
        key = base + "_" + n++;
      }
      customPoses.push({ key: key, pose: pose, name: name || pose });
      saveCustomPoses();
      els.customPoseForm.style.display = "none";
      renderPoseList();
      // Auto-tick the pose that was just added -- otherwise it's easy to
      // add one and forget to actually select it before "Add to queue".
      var cb = els.poseList.querySelector('input[value="' + key.replace(/"/g, "") + '"]');
      if (cb) {
        cb.checked = true;
        savePoseKeys();
      }
    });

    els.addBtn.addEventListener("click", addToQueue);
    els.queueClear.addEventListener("click", function () {
      queue = queue.filter(function (q) {
        return q.status !== "done" && q.status !== "failed";
      });
      renderQueue();
    });
  }

  // ---- called from ProjectManager.render() every tick ----
  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "generate");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (els.controlsPane) els.controlsPane.style.display = on ? "none" : "";
    if (!on) return;

    init();
    ensurePresetsAndHealth();

    if (els.name && !els.name.value && active && active.name) {
      els.name.placeholder = "e.g. " + slugName(active.name);
    }
    rebuildRefTray(active);
    renderQueue();
  }

  // ---- presets + engine-health, fetched once ----
  function ensurePresetsAndHealth() {
    if (presetsLoaded) return;
    presetsLoaded = true;

    fetch(API + "/presets")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        styleNames =
          data && data.styles && data.styles.length ? data.styles : ["none"];
        sceneChoices = (data && data.scenes) || [];
        var seen = {};
        poseCatalogue = [];
        var presets = (data && data.presets) || {};
        Object.keys(presets).forEach(function (pname) {
          (presets[pname] || []).forEach(function (s) {
            if (seen[s.key]) return;
            seen[s.key] = true;
            poseCatalogue.push({
              key: s.key,
              pose: s.pose_phrase || "",
              preset: pname,
            });
          });
        });
        renderStyleOptions();
        renderPoseList();
      })
      .catch(function () {
        presetsLoaded = false; // let a later sync retry
      });

    fetch(API + "/status")
      .then(function (r) {
        return r.json();
      })
      .then(function (h) {
        setGenerationDisabled(!!(h && h.reachable === false), h);
      })
      .catch(function () {
        /* fail open */
      });
  }

  function renderStyleOptions() {
    if (!els.style) return;
    var cur = els.style.value;
    els.style.innerHTML = "";
    styleNames.forEach(function (name) {
      var o = document.createElement("option");
      o.value = name;
      o.textContent = name === "none" ? "No style" : name.replace(/_/g, " ");
      els.style.appendChild(o);
    });
    if (styleNames.indexOf(cur) >= 0) els.style.value = cur;
  }

  function loadSavedPoseKeys() {
    try {
      var raw = localStorage.getItem(POSE_STORAGE_KEY);
      var arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return []; // storage unavailable (private mode, etc.) -- just skip
    }
  }

  function savePoseKeys() {
    if (!els.poseList) return;
    try {
      var keys = Array.prototype.map.call(
        els.poseList.querySelectorAll("input:checked"),
        function (cb) {
          return cb.value;
        }
      );
      localStorage.setItem(POSE_STORAGE_KEY, JSON.stringify(keys));
    } catch (e) {
      /* storage unavailable -- fine, ticks just won't survive a refresh */
    }
  }

  function loadCustomPoses() {
    try {
      var raw = localStorage.getItem(CUSTOM_POSE_STORAGE_KEY);
      var arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveCustomPoses() {
    try {
      localStorage.setItem(CUSTOM_POSE_STORAGE_KEY, JSON.stringify(customPoses));
    } catch (e) {
      /* storage unavailable -- custom poses just won't survive a refresh */
    }
  }

  // Fetches the prompt this shot would currently render with (given the
  // job-level settings + this shot's own scene override, if any) and
  // fills the textarea with it -- called when a pose card's "edit
  // prompt" is first opened, and by its "reset" action.
  function fetchPromptPreview(pose, ta) {
    ta.value = "";
    ta.placeholder = "loading preview…";
    var s = readSettings();
    var body = {
      wardrobe: s.wardrobe,
      hair_color: s.hair_color,
      identity_lock: s.identity_lock,
      style: s.style,
      scene: (shotOverrides[pose.key] && shotOverrides[pose.key].scene) || "",
    };
    if (pose.isCustom) {
      body.custom_pose = pose.pose;
      body.custom_key = pose.key;
    } else {
      body.views = [pose.key];
    }
    fetch(API + "/preview-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        ta.placeholder = "";
        if (data && data.prompt) ta.value = data.prompt;
      })
      .catch(function () {
        ta.placeholder = "preview failed -- type a prompt manually";
      });
  }

  function renderPoseList() {
    if (!els.poseList) return;
    var checked = {};
    var existing = els.poseList.querySelectorAll("input:checked");
    if (existing.length) {
      existing.forEach(function (cb) {
        checked[cb.value] = true;
      });
    } else if (!poseListSeeded) {
      // First render this page load: nothing is ticked yet in the DOM
      // (there's no DOM), so seed from what was ticked last session
      // instead of always starting from a blank slate.
      loadSavedPoseKeys().forEach(function (k) {
        checked[k] = true;
      });
    }
    poseListSeeded = true;
    els.poseList.innerHTML = "";

    var all = poseCatalogue.concat(
      customPoses.map(function (c) {
        return { key: c.key, pose: c.pose, isCustom: true };
      })
    );

    all.forEach(function (p) {
      var row = document.createElement("label");
      row.className = "ng-gen-pose-row" + (p.isCustom ? " ng-gen-pose-row-custom" : "");

      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = p.key;
      if (checked[p.key]) cb.checked = true;
      row.appendChild(cb);

      var body = document.createElement("div");
      body.className = "pose-body";

      var head = document.createElement("div");
      head.className = "pose-head";
      var kSpan = document.createElement("span");
      kSpan.className = "k";
      kSpan.textContent = p.key;
      head.appendChild(kSpan);

      var sceneSel = document.createElement("select");
      sceneSel.className = "pose-scene";
      var defOpt = document.createElement("option");
      defOpt.value = "";
      defOpt.textContent = "Default scene";
      sceneSel.appendChild(defOpt);
      sceneChoices.forEach(function (s) {
        var o = document.createElement("option");
        o.value = s.key;
        o.textContent = s.label;
        sceneSel.appendChild(o);
      });
      var st0 = shotOverrides[p.key];
      sceneSel.value = (st0 && st0.scene) || "";
      // Selects/textareas/buttons inside a <label> don't forward clicks
      // to the checkbox in any modern browser, but stop propagation
      // anyway -- cheap insurance against ticking/unticking the pose by
      // accident while just picking a scene.
      sceneSel.addEventListener("click", function (e) {
        e.stopPropagation();
      });
      sceneSel.addEventListener("change", function () {
        var s = (shotOverrides[p.key] = shotOverrides[p.key] || {});
        s.scene = sceneSel.value;
      });
      head.appendChild(sceneSel);
      body.appendChild(head);

      var pSpan = document.createElement("span");
      pSpan.className = "p";
      pSpan.textContent = p.pose;
      body.appendChild(pSpan);

      // ---- editable prompt (pre-submission preview + override) ----
      var promptToggle = document.createElement("div");
      promptToggle.className = "pose-prompt-toggle";
      var isOpen = !!(st0 && st0.open);
      promptToggle.textContent = (isOpen ? "▾ " : "▸ ") + "edit prompt";
      body.appendChild(promptToggle);

      var editorWrap = document.createElement("div");
      editorWrap.className = "pose-prompt-editor";
      editorWrap.style.display = isOpen ? "" : "none";
      var ta = document.createElement("textarea");
      ta.rows = 4;
      if (st0 && typeof st0.promptOverride === "string") ta.value = st0.promptOverride;
      editorWrap.appendChild(ta);

      var actions = document.createElement("div");
      actions.className = "pose-prompt-actions";
      var resetBtn = document.createElement("button");
      resetBtn.type = "button";
      resetBtn.className = "ng-gen-reroll";
      resetBtn.textContent = "reset to default";
      actions.appendChild(resetBtn);
      editorWrap.appendChild(actions);
      body.appendChild(editorWrap);

      ta.addEventListener("click", function (e) {
        e.stopPropagation();
      });
      ta.addEventListener("input", function () {
        var s = (shotOverrides[p.key] = shotOverrides[p.key] || {});
        s.promptOverride = ta.value;
      });
      resetBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        var s = (shotOverrides[p.key] = shotOverrides[p.key] || {});
        s.promptOverride = null;
        fetchPromptPreview(p, ta);
      });
      promptToggle.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        var s = (shotOverrides[p.key] = shotOverrides[p.key] || {});
        s.open = !s.open;
        promptToggle.textContent = (s.open ? "▾ " : "▸ ") + "edit prompt";
        editorWrap.style.display = s.open ? "" : "none";
        if (s.open && typeof s.promptOverride !== "string") {
          fetchPromptPreview(p, ta);
        }
      });

      row.appendChild(body);

      if (p.isCustom) {
        var rm = document.createElement("button");
        rm.type = "button";
        rm.className = "pose-remove";
        rm.title = "remove this custom pose";
        rm.textContent = "×";
        rm.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();
          customPoses = customPoses.filter(function (c) {
            return c.key !== p.key;
          });
          delete shotOverrides[p.key];
          saveCustomPoses();
          renderPoseList();
        });
        row.appendChild(rm);
      }

      els.poseList.appendChild(row);
    });
  }

  function togglePoses(on) {
    els.poseList.querySelectorAll("input").forEach(function (cb) {
      cb.checked = on;
    });
    savePoseKeys();
  }

  // ---- reference tray ----
  function rebuildRefTray(active) {
    var disk = refs.filter(function (r) {
      return r.kind === "disk";
    });
    var proj = [];
    var ring = active && active.ring;
    if (ring) {
      if (ring.anchorUrl) {
        proj.push({
          id: "anchor",
          label: "anchor",
          kind: "proj",
          url: ring.anchorUrl,
        });
      }
      if (active.selectedFrames && active.selectedFrames.size && active.sortedRanked) {
        active
          .sortedRanked()
          .filter(function (r) {
            return active.selectedFrames.has(r.frame);
          })
          .forEach(function (r) {
            if (r.thumbUrl) {
              proj.push({
                id: "f" + r.frame,
                label: "#" + r.frame,
                kind: "proj",
                url: r.thumbUrl,
              });
            }
          });
      }
    }
    refs = proj.concat(disk);
    var stillThere = refs.some(function (r) {
      return r.id === activeRefId;
    });
    if (!stillThere) activeRefId = refs.length ? refs[0].id : null;
    renderRefTray();
  }

  function renderRefTray() {
    if (!els.refTray) return;
    els.refTray.innerHTML = "";
    refs.forEach(function (ref) {
      var d = document.createElement("div");
      d.className =
        "ng-gen-ref" + (ref.id === activeRefId ? " ng-gen-ref-active" : "");
      d.innerHTML =
        '<img src="' +
        escapeHtml(ref.url) +
        '" alt=""><span class="ng-gen-ref-badge">' +
        escapeHtml(ref.label) +
        "</span>";
      d.addEventListener("click", function () {
        activeRefId = ref.id;
        renderRefTray();
      });
      els.refTray.appendChild(d);
    });
  }

  function onDiskFile() {
    var f = els.refFile.files && els.refFile.files[0];
    if (!f) return;
    var n = ++localSeq;
    var diskCount = refs.filter(function (r) {
      return r.kind === "disk";
    }).length;
    refs.push({
      id: "disk" + n,
      label: "disk " + (diskCount + 1),
      kind: "disk",
      url: URL.createObjectURL(f),
      file: f,
    });
    activeRefId = "disk" + n;
    els.refFile.value = "";
    renderRefTray();
  }

  // ---- settings ----
  function readSettings() {
    var seedRaw = (els.seed.value || "").trim();
    return {
      style: els.style.value || "none",
      wardrobe: (els.wardrobe.value || "").trim(),
      hair_color: (els.hair.value || "").trim(),
      seed: seedRaw === "" ? -1 : parseInt(seedRaw, 10),
      identity_lock: !!els.identityLock.checked,
      width: parseInt(els.width.value, 10) || 2048,
      height: parseInt(els.height.value, 10) || 2048,
      steps: parseInt(els.steps.value, 10) || 28,
    };
  }

  function triggerForRef(refId, base) {
    var k = base + "|" + refId;
    if (refTriggers[k]) return refTriggers[k];
    var n = (baseCounts[base] || 0) + 1;
    baseCounts[base] = n;
    var trig = n === 1 ? base : base + "-" + n;
    refTriggers[k] = trig;
    return trig;
  }

  // ---- add ticked poses to the queue ----
  function addToQueue() {
    if (generationDisabled) return;
    var ref = refs.find(function (r) {
      return r.id === activeRefId;
    });
    if (!ref) {
      setStatus("Pick a reference image first (a ring candidate or a photo from disk).");
      return;
    }
    var keys = Array.prototype.slice
      .call(els.poseList.querySelectorAll("input:checked"))
      .map(function (cb) {
        return cb.value;
      });
    if (!keys.length) {
      setStatus("Tick at least one pose.");
      return;
    }

    var activeProj = window.ProjectManager && window.ProjectManager.getActive();
    var base = slugName(
      els.name.value || (activeProj && activeProj.name) || "character"
    );
    var character = triggerForRef(ref.id, base);
    var settings = readSettings();

    keys.forEach(function (key) {
      var custom = customPoses.find(function (c) {
        return c.key === key;
      });
      var ov = shotOverrides[key] || {};
      queue.push({
        localId: ++localSeq,
        character: character,
        base: base, // for correcting refTriggers below if the server suffixes
        key: key,
        // Snapshotted at queue time (like `settings` above) so editing a
        // pose card's scene/prompt afterward doesn't retroactively change
        // an already-queued item.
        customPose: custom ? custom.pose : null,
        scene: ov.scene || "",
        promptOverride: typeof ov.promptOverride === "string" ? ov.promptOverride : "",
        refId: ref.id,
        refUrl: ref.url,
        settings: settings,
        jobId: null,
        status: "pending",
        thumbUrl: null,
        error: null,
        logTail: null,
        prompt: null, // the full text sent to the engine, from job_status_ng()
      });
    });
    setStatus(
      "Queued " +
        keys.length +
        " pose" +
        (keys.length === 1 ? "" : "s") +
        " for “" +
        character +
        "”."
    );
    renderQueue();
    pump();
  }

  function fetchRefBlob(ref) {
    if (ref && ref.file) return Promise.resolve(ref.file);
    return fetch(ref.url).then(function (r) {
      if (!r.ok)
        throw new Error("could not fetch the reference image (HTTP " + r.status + ")");
      return r.blob();
    });
  }

  // Submit pending entries one at a time so the backend's FIFO queue
  // ends up in the same order they were added.
  function pump() {
    if (pumping) return;
    pumping = true;
    (function next() {
      var item = queue.find(function (q) {
        return q.status === "pending";
      });
      if (!item) {
        pumping = false;
        ensurePolling();
        return;
      }
      item.status = "submitting";
      item.error = null;
      renderQueue();

      var ref = refs.find(function (r) {
        return r.id === item.refId;
      }) || { url: item.refUrl };

      fetchRefBlob(ref)
        .then(function (blob) {
          var form = new FormData();
          var ext = blob.type === "image/png" ? ".png" : ".jpg";
          form.append("file", blob, "reference" + ext);
          form.append("trigger", item.character);
          if (item.customPose) {
            form.append("custom_pose", item.customPose);
            form.append("custom_key", item.key);
          } else {
            form.append("views", JSON.stringify([item.key]));
          }
          if (item.scene) form.append("scene", item.scene);
          if (item.promptOverride) form.append("prompt_override", item.promptOverride);
          form.append("style", item.settings.style);
          form.append("wardrobe", item.settings.wardrobe);
          form.append("hair_color", item.settings.hair_color);
          form.append("seed", String(item.settings.seed));
          form.append(
            "identity_lock",
            item.settings.identity_lock ? "true" : "false"
          );
          form.append("width", String(item.settings.width));
          form.append("height", String(item.settings.height));
          form.append("steps", String(item.settings.steps));
          return fetch(API + "/sheet-from-upload", {
            method: "POST",
            body: form,
          });
        })
        .then(function (res) {
          return res.json().then(function (payload) {
            return { ok: res.ok, status: res.status, payload: payload };
          });
        })
        .then(function (r) {
          if (!r.ok || !r.payload || !r.payload.ok || !r.payload.job_id) {
            item.status = "failed";
            item.error =
              (r.payload && r.payload.error) || "HTTP " + r.status;
          } else {
            item.jobId = r.payload.job_id;
            item.status = "queued";
            // The backend auto-suffixes the trigger (default, default-2, ...)
            // when this name already belongs to a different photo, so it
            // can differ from our client-side guess -- without this,
            // thumbUrl/reroll() below would keep pointing at the wrong
            // (or nonexistent) character.
            if (r.payload.trigger && r.payload.trigger !== item.character) {
              refTriggers[item.base + "|" + item.refId] = r.payload.trigger;
              item.character = r.payload.trigger;
            }
          }
        })
        .catch(function (e) {
          item.status = "failed";
          item.error = String(e);
        })
        .then(function () {
          renderQueue();
          next();
        });
    })();
  }

  // ---- polling ----
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
      fetch(API + "/sheet-jobs/" + item.jobId)
        .then(function (r) {
          return r.json();
        })
        .then(function (job) {
          if (!job || !job.status) return;
          var shot = (job.shots || [])[0];
          if (shot && shot.prompt) item.prompt = shot.prompt;
          if (job.log_tail && job.log_tail.length) item.logTail = job.log_tail;
          if (job.status === "completed") {
            item.status = "done";
            item.thumbUrl =
              API +
              "/characters/" +
              encodeURIComponent(item.character) +
              "/shots/" +
              encodeURIComponent(item.key) +
              "?v=" +
              Date.now();
          } else if (job.status === "failed") {
            item.status = "failed";
            item.error =
              job.error || (shot && shot.status) || "generation failed";
          } else if (shot && shot.status === "rendering") {
            item.status = "rendering";
          } else {
            item.status = "queued";
          }
          renderQueue();
        })
        .catch(function () {
          /* transient -- try again next tick */
        });
    });
  }

  function reroll(item) {
    fetch(
      API +
        "/characters/" +
        encodeURIComponent(item.character) +
        "/sheet/reroll",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shot_key: item.key,
          width: item.settings.width,
          height: item.settings.height,
          steps: item.settings.steps,
        }),
      }
    )
      .then(function (r) {
        return r.json();
      })
      .then(function (payload) {
        if (payload && payload.ok && payload.job_id) {
          item.jobId = payload.job_id;
          item.status = "queued";
          item.thumbUrl = null;
          item.error = null;
          renderQueue();
          ensurePolling();
        } else {
          setStatus(
            "Reroll failed: " + ((payload && payload.error) || "unknown error")
          );
        }
      })
      .catch(function (e) {
        setStatus("Reroll failed: " + e);
      });
  }

  function removeItem(localId) {
    var item = queue.find(function (q) { return q.localId === localId; });
    if (!item) return;
    queue = queue.filter(function (q) { return q.localId !== localId; });
    renderQueue();
    // Best-effort: tell the backend to drop/kill the job too. Fire and
    // forget -- the row is already gone from view either way, and a
    // job that's pending/submitting/done/failed has no backend job (or
    // nothing left) to cancel.
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/sheet-jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  // ---- queue view ----
  function renderQueue() {
    if (!els.queue) return;
    els.queue.innerHTML = "";
    queue.forEach(function (item) {
      var row = document.createElement("div");
      row.className = "ng-gen-queue-row st-" + item.status;

      var thumbs = document.createElement("div");
      thumbs.className = "thumbs";
      if (item.refUrl) {
        var ri = document.createElement("img");
        ri.src = item.refUrl;
        ri.title = "reference";
        thumbs.appendChild(ri);
      }
      if (item.thumbUrl) {
        var oi = document.createElement("img");
        oi.src = item.thumbUrl;
        oi.title = "result";
        thumbs.appendChild(oi);
      }

      var meta = document.createElement("div");
      meta.className = "meta";
      var sub =
        item.settings.style === "none" ? "no style" : item.settings.style;
      if (item.error) sub += " · " + item.error;
      meta.innerHTML =
        '<div class="title">' +
        escapeHtml(item.character) +
        " · " +
        escapeHtml(item.key) +
        "</div>" +
        '<div class="sub">' +
        escapeHtml(sub) +
        "</div>";

      var right = document.createElement("div");
      right.className = "status-row";
      var st = document.createElement("div");
      st.className = "st";
      st.textContent = ST_LABEL[item.status] || item.status;
      right.appendChild(st);

      var actions = document.createElement("div");
      actions.className = "actions";
      if (item.status === "failed") {
        var retry = document.createElement("button");
        retry.className = "ng-gen-reroll";
        retry.textContent = "retry";
        retry.addEventListener("click", function () {
          item.status = "pending";
          item.jobId = null;
          item.error = null;
          renderQueue();
          pump();
        });
        actions.appendChild(retry);
      } else if (item.status === "done") {
        var rb = document.createElement("button");
        rb.className = "ng-gen-reroll";
        rb.textContent = "reroll";
        rb.addEventListener("click", function () {
          reroll(item);
        });
        actions.appendChild(rb);
      }

      var rm = document.createElement("button");
      rm.type = "button";
      rm.className = "pose-remove";
      rm.title = "remove from queue";
      rm.textContent = "✕";
      rm.addEventListener("click", function () {
        removeItem(item.localId);
      });
      actions.appendChild(rm);
      right.appendChild(actions);

      row.appendChild(thumbs);
      row.appendChild(meta);
      row.appendChild(right);

      // Live progress -- the HiDream subprocess's own stdout (step
      // counter etc.), surfaced by job_status_ng()'s log_tail. Shown
      // while a job is in flight, and kept on a failed row for context.
      var logEl = null;
      if (item.logTail && item.logTail.length && item.status !== "done") {
        logEl = document.createElement("div");
        logEl.className = "log";
        logEl.textContent = item.logTail.slice(-10).join("\n");
        row.appendChild(logEl);
      }

      // The exact prompt sent to the engine (from job_status_ng). Handy
      // for judging a bad render and as a starting point for a custom
      // prompt -- collapsed by default, auto-open on a failed row.
      if (item.prompt) {
        var pWrap = document.createElement("div");
        pWrap.className = "prompt";
        var open = item._promptOpen || item.status === "failed";

        var pHead = document.createElement("div");
        pHead.className = "prompt-head";
        var tog = document.createElement("span");
        tog.className = "prompt-tog";
        tog.textContent = (open ? "▾ " : "▸ ") + "prompt";
        tog.addEventListener("click", function () {
          item._promptOpen = !open;
          renderQueue();
        });
        var cp = document.createElement("button");
        cp.className = "ng-gen-reroll prompt-copy";
        cp.textContent = "copy";
        cp.addEventListener("click", function () {
          if (navigator.clipboard) navigator.clipboard.writeText(item.prompt);
          cp.textContent = "copied";
          setTimeout(function () { cp.textContent = "copy"; }, 1200);
        });
        pHead.appendChild(tog);
        pHead.appendChild(cp);
        pWrap.appendChild(pHead);

        if (open) {
          var pBody = document.createElement("div");
          pBody.className = "prompt-body";
          pBody.textContent = item.prompt;
          pWrap.appendChild(pBody);
        }
        row.appendChild(pWrap);
      }

      els.queue.appendChild(row);
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    });

    var n = queue.length;
    if (els.queueCount) els.queueCount.textContent = n ? "(" + n + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = n ? "none" : "";

    renderRingActions();
  }

  // ---- "add finished shots to ring" actions, one row per distinct
  // character with at least one done queue entry. Always targets
  // whichever project is currently active (same scoping the ref tray
  // already uses) -- not necessarily the project that was active when
  // the shot was queued, matching how the rest of this view treats the
  // active project as the implicit destination. ----
  function distinctDoneCharacters() {
    var seen = {};
    var list = [];
    queue.forEach(function (q) {
      if (q.status !== "done" || seen[q.character]) return;
      seen[q.character] = true;
      list.push(q.character);
    });
    return list;
  }

  function addSheetToRing(character, btn, titleEl, baseTitle) {
    var activeProj = window.ProjectManager && window.ProjectManager.getActive();
    if (!activeProj) {
      setStatus("No active project to add the ring into.");
      return;
    }
    btn.disabled = true;
    btn.textContent = "Adding…";
    activeProj
      .startSheetAnalysis(character)
      .then(function () {
        btn.disabled = false;
        var job = activeProj.job;
        if (job && job.status === "error") {
          btn.textContent = "Retry";
          titleEl.textContent = baseTitle + " · " + (job.error || "add-to-ring failed");
        } else {
          // The job itself may still be "running" at this point (poll()
          // isn't awaited) -- ProjectManager's own render loop picks up
          // the rest as it progresses, same as folder/video/Immich
          // analysis. This just confirms the request was accepted.
          btn.textContent = "Add to ring again";
        }
      })
      .catch(function (e) {
        btn.disabled = false;
        btn.textContent = "Retry";
        titleEl.textContent = baseTitle + " · request failed: " + e;
      });
  }

  function renderRingActions() {
    if (!els.ringActions) return;
    els.ringActions.innerHTML = "";
    distinctDoneCharacters().forEach(function (character) {
      var doneCount = queue.filter(function (q) {
        return q.character === character && q.status === "done";
      }).length;

      var row = document.createElement("div");
      row.className = "ng-gen-ring-row";

      var title = document.createElement("div");
      title.className = "title";
      var baseTitle =
        character + " — " + doneCount + " shot" + (doneCount === 1 ? "" : "s") + " ready";
      title.textContent = baseTitle;
      row.appendChild(title);

      var btn = document.createElement("button");
      btn.className = "ng-gen-btn ng-gen-btn-quiet";
      btn.type = "button";
      btn.textContent = "Add to ring";
      btn.addEventListener("click", function () {
        addSheetToRing(character, btn, title, baseTitle);
      });
      row.appendChild(btn);

      els.ringActions.appendChild(row);
    });
  }

  function setGenerationDisabled(disabled, health) {
    generationDisabled = !!disabled;
    if (els.addBtn) els.addBtn.disabled = generationDisabled;
    if (!els.unavailable) return;
    if (generationDisabled) {
      var missing = [];
      if (health && !health.python_ok) missing.push("HiDream venv");
      if (health && !health.model_ok) missing.push("HiDream model");
      if (health && !health.script_ok) missing.push("generator script");
      els.unavailable.textContent =
        "Character-sheet generation needs the local HiDream engine, which " +
        "isn't set up on this machine" +
        (missing.length ? " (missing: " + missing.join(", ") + ")" : "") +
        ". This view only works where the HiDream-O1 MLX lab is installed.";
      els.unavailable.style.display = "block";
    } else {
      els.unavailable.style.display = "none";
    }
  }

  // ---- hooks for characterIONG.js (Save/Load) ----

  // The currently-picked reference, for Save to embed. {url, origin} or null.
  function getActiveReference() {
    var r = refs.find(function (x) {
      return x.id === activeRefId;
    });
    if (!r) return null;
    var origin =
      r.kind === "disk"
        ? "disk"
        : r.id === "anchor"
        ? "anchor"
        : "selected-frame:" + r.label.replace(/^#/, "");
    return { url: r.url, origin: origin };
  }

  // Load calls this with a saved reference's data URI -- lands it in the
  // tray as a disk-kind ref (survives rebuildRefTray) and selects it.
  function addExternalReference(dataUri, label) {
    if (!dataUri) return;
    var blob;
    try {
      var parts = String(dataUri).split(",");
      var mime = (parts[0].match(/data:([^;]+)/) || [])[1] || "image/jpeg";
      var bin = atob(parts[1] || "");
      var arr = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      blob = new Blob([arr], { type: mime });
    } catch (e) {
      return;
    }
    var id = "ext" + ++localSeq;
    refs.push({
      id: id,
      label: label || "saved",
      kind: "disk",
      url: URL.createObjectURL(blob),
      file: blob,
    });
    activeRefId = id;
    refreshEls();
    renderRefTray();
  }

  // Drops an already-hosted image (e.g. a character's avatar route) into
  // the tray as a disk-kind ref and makes it active -- no fetch/decode
  // needed up front, fetchRefBlob() pulls the bytes lazily at submit time.
  // Used by characterPickerNG.js so opening a character from the picker
  // grid lands with its reference photo ready to render immediately.
  function addReferenceFromUrl(url, label) {
    if (!url) return;
    var id = "ext" + ++localSeq;
    refs.push({ id: id, label: label || "reference", kind: "disk", url: url });
    activeRefId = id;
    refreshEls();
    renderRefTray();
  }

  window.GenerateNG = {
    sync: sync,
    getActiveReference: getActiveReference,
    addExternalReference: addExternalReference,
    addReferenceFromUrl: addReferenceFromUrl,
  };
})();
