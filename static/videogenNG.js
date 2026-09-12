/**
 * videogenNG.js -- the "Animate" view (bottom-bar Animate task).
 *
 * Self-contained image-to-video generation, wired to the NG backend at
 * /api/ng/videogen/* (routes/videogenNG.py, video_jobsNG.py, LTX-2.5 via
 * ltx_engineNG.py). One reference image + a motion prompt + a duration in,
 * one mp4 out -- deliberately no pose grid, no character registration,
 * no draft-character bookkeeping like Generate has; a video render isn't
 * part of a character's sheet.
 *
 * The ONLY external input read from the rest of the app is a convenience
 * button that pulls whatever image is currently the active reference in
 * GenerateNG's ref tray (window.GenerateNG.getActiveReference()) -- handy
 * since opening a character from the picker grid already lands its
 * avatar there (see characterPickerNG.js). Otherwise this view is fully
 * self-contained: pick a photo from disk, type a prompt, go.
 *
 * Model: a flat FIFO render queue, same shape as Generate's -- each entry
 * is one (reference, prompt, duration) job. The backend's single LTX
 * worker (video_jobsNG.py) renders one at a time; queue rows here just
 * reflect that. In-memory only, lost on refresh, matching the backend's
 * own job store.
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after generateNG.js and before bootstrapWiringNG.js.
 */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var API = "/api/ng/videogen";

  // ---- module state ----
  var inited = false;
  var statusChecked = false;
  var generationDisabled = false;
  var currentRefBlob = null; // File/Blob picked from disk or pulled from Generate
  var currentRefPreviewUrl = null; // object URL for the <img> preview
  var queue = []; // [{localId, jobId, status, refPreviewUrl, prompt, durationS, seed, error, videoUrl, logTail}]
  var localSeq = 0;
  var pollTimer = null;

  var els = {};

  function refreshEls() {
    els.pane = document.getElementById("ng-videogen-pane");
    els.main = document.getElementById("ng-videogen-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.unavailable = document.getElementById("ng-vg-unavailable");

    els.queueHead = document.getElementById("ng-vg-queue-head");
    els.queueCount = document.getElementById("ng-vg-queue-count");
    els.queueEmpty = document.getElementById("ng-vg-queue-empty");
    els.queueClear = document.getElementById("ng-vg-queue-clear");
    els.queue = document.getElementById("ng-vg-queue");

    els.refPreview = document.getElementById("ng-vg-ref-preview");
    els.refEmpty = document.getElementById("ng-vg-ref-empty");
    els.refFile = document.getElementById("ng-vg-ref-file");
    els.refFileBtn = document.getElementById("ng-vg-ref-file-btn");
    els.refUseGen = document.getElementById("ng-vg-ref-use-gen");

    els.prompt = document.getElementById("ng-vg-prompt");
    els.enhanceBtn = document.getElementById("ng-vg-enhance-btn");
    els.duration = document.getElementById("ng-vg-duration");
    els.durationVal = document.getElementById("ng-vg-duration-val");
    els.seed = document.getElementById("ng-vg-seed");

    els.generateBtn = document.getElementById("ng-vg-generate-btn");
    els.status = document.getElementById("ng-vg-status");
    els.mainLog = document.getElementById("ng-vg-log");
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function setReference(blob) {
    if (currentRefPreviewUrl) URL.revokeObjectURL(currentRefPreviewUrl);
    currentRefBlob = blob;
    currentRefPreviewUrl = URL.createObjectURL(blob);
    if (els.refPreview) {
      els.refPreview.src = currentRefPreviewUrl;
      els.refPreview.style.display = "";
    }
    if (els.refEmpty) els.refEmpty.style.display = "none";
  }

  function onDiskFile() {
    var f = els.refFile.files && els.refFile.files[0];
    if (f) setReference(f);
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
        setReference(blob);
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
          "LTX pipeline not reachable (binary/model/gemma missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
    }
  }

  function ensureStatusChecked() {
    if (statusChecked) return;
    statusChecked = true;
    fetch(API + "/status")
      .then(function (r) { return r.json(); })
      .then(function (h) {
        setGenerationDisabled(!!(h && h.reachable === false), h);
        if (els.enhanceBtn) {
          var enhanceOk = !!(h && h.enhance_gemma_ok);
          els.enhanceBtn.disabled = !enhanceOk;
          els.enhanceBtn.title = enhanceOk
            ? ""
            : "Chat Gemma-3 checkpoint not found at " + ((h && h.enhance_gemma_path) || "?");
        }
      })
      .catch(function () {
        statusChecked = false; // let a later sync retry
      });
  }

  function enhancePrompt() {
    var prompt = (els.prompt.value || "").trim();
    if (!prompt) {
      setStatus("Describe the motion first.");
      els.prompt.focus();
      return;
    }
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);

    els.enhanceBtn.disabled = true;
    setStatus("Enhancing prompt with Gemma...");

    var body = { prompt: prompt };
    if (seed !== null && !isNaN(seed)) body.seed = seed;

    fetch(API + "/enhance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.prompt) {
          els.prompt.value = r.payload.prompt; // replaces the given prompt
          setStatus("Prompt enhanced.");
        } else {
          setStatus("Enhance failed: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        setStatus("Enhance failed: " + e.message);
      })
      .then(function () {
        els.enhanceBtn.disabled = false;
      });
  }

  function generateVideo() {
    var prompt = (els.prompt.value || "").trim();
    if (!currentRefBlob) {
      setStatus("Pick a reference image first.");
      return;
    }
    if (!prompt) {
      setStatus("Describe the motion first.");
      els.prompt.focus();
      return;
    }
    var durationS = parseFloat(els.duration.value) || 3.0;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);

    var localId = "v" + ++localSeq;
    var item = {
      localId: localId,
      jobId: null,
      status: "submitting",
      refPreviewUrl: currentRefPreviewUrl,
      prompt: prompt,
      durationS: durationS,
      seed: seed,
      error: null,
      videoUrl: null,
      logTail: [],
    };
    queue.unshift(item);
    renderQueue();

    var form = new FormData();
    var ext = currentRefBlob.type === "image/png" ? ".png" : ".jpg";
    form.append("file", currentRefBlob, "reference" + ext);
    form.append("prompt", prompt);
    form.append("duration_s", String(durationS));
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
    // Best-effort: tell the backend to drop/kill the job too. Fire and
    // forget -- the row is already gone from view either way, and a
    // job that's already done/failed has nothing left to cancel.
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  function renderQueue() {
    if (!els.queue) return;
    els.queue.innerHTML = "";
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";

    queue.forEach(function (item) {
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
      sub.textContent = item.durationS + "s" + (item.seed !== null ? ", seed " + item.seed : "");
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

        // Explicit download link, forced as an attachment server-side --
        // more reliable than relying on the <video> controls' own save
        // option (which iPad Safari in particular often lacks for
        // inline playback).
        var dl = document.createElement("a");
        dl.className = "ng-btn ng-vg-download";
        dl.href = API + "/jobs/" + item.jobId + "/download";
        dl.download = "animate-" + item.jobId + ".mp4";
        dl.textContent = "Download video";
        dl.style.gridColumn = "1 / -1";
        dl.style.marginTop = "4px";
        row.appendChild(dl);
      }

      els.queue.appendChild(row);
    });

    updateMainLog();
  }

  // Only one job ever renders at a time (single backend worker), so the
  // main-stage log just tails whichever queue item is currently
  // "rendering" -- plenty of room there to actually read it, unlike the
  // narrow rail queue row it used to live in.
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

    els.duration.addEventListener("input", function () {
      els.durationVal.textContent = parseFloat(els.duration.value).toFixed(1) + "s";
    });

    els.enhanceBtn.addEventListener("click", enhancePrompt);
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
    var on = !!(active && active.task === "videogen");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (els.controlsPane && (!active || (active.task !== "generate" && active.task !== "chat" && active.task !== "rachel"))) {
      els.controlsPane.style.display = on ? "none" : "";
    }
    if (!on) return;

    init();
    ensureStatusChecked();
    renderQueue();
  }

  window.VideoGenNG = { sync: sync };
})();
