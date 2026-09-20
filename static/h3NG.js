/**
 * h3NG.js -- the "H3" view (bottom-bar H3 task).
 *
 * MiniMax-H3 prompt/image-to-video, wired to the NG backend at
 * /api/ng/h3/* (routes/h3NG.py, h3_jobsNG.py, h3_engineNG.py). Twin of
 * videogenNG.js's queue-based shape -- same FIFO render queue, same
 * reference-image paste/drop/drag UX -- trimmed of what H3 doesn't
 * have server-side yet: no /enhance, /discuss, /chat (LTX's own
 * Gemma-side features), no LoRA dropdown, no durable Job Log view, no
 * width/height/fps controls (fixed at h3_engineNG.H3_WIDTH/HEIGHT/
 * STEPS for now). A `model` dropdown (h3 vs h3q8) replaces all of
 * that -- see h3_engineNG.py's docstring for why those are one engine,
 * not two.
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
  var queue = []; // [{localId, jobId, status, refPreviewUrl, prompt, durationS, seed, model, error, videoUrl, logTail}]
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

    els.generateBtn = document.getElementById("ng-h3-generate-btn");
    els.status = document.getElementById("ng-h3-status");
    els.mainLog = document.getElementById("ng-h3-log");
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
      if (disabled) {
        els.unavailable.textContent =
          "H3 pipeline not reachable (venv/DiT/compact-pack missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
    }
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
      item.error, item.videoUrl, item.jobId, item.refPreviewUrl,
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
    rm.title = "remove from queue";
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
    renderQueue();
  }

  window.H3NG = { sync: sync };
})();
