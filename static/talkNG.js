/* Talk view (task === "talk") -- a character (a LoRA and/or a face picture), a voice and a line in, a
 * lip-synced clip out. Rail pane = the talk queue, main = the form. Backend: routes/talkNG.py (the voice is
 * made by the Speech engine, then an H3 audio-drive render makes the picture).
 *
 * The queue is rebuilt from the server (GET /api/ng/talk/jobs) when the view opens and every few seconds, so a
 * reload or a lost submit can never hide a job; a waiting job can be cancelled and a running one stopped from
 * its row. Exposes window.TalkNG = { sync }. Called from ProjectManager.render().
 */
(function () {
  "use strict";

  var API = "/api/ng/talk";
  var SPEECH_API = "/api/ng/speech";
  var POLL_MS = 3000;
  var SYNC_MS = 8000;
  var SUBMIT_GIVE_UP_MS = 60000;

  var els = {};
  var inited = false;
  var queue = [];
  var localSeq = 0;
  var pollTimer = null;
  var syncTimer = null;
  var dismissed = {};
  var clipFile = null;
  var faceFile = null;
  var optionsLoaded = false;
  var maxChars = 600;
  var finishOk = false;

  function $(id) { return document.getElementById(id); }

  function refreshEls() {
    els.pane = $("ng-talk-pane");
    els.main = $("ng-talk-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-talk-unavailable");
    els.unavailableText = $("ng-talk-unavailable-text");
    els.queueCount = $("ng-talk-queue-count");
    els.queueEmpty = $("ng-talk-queue-empty");
    els.queueClear = $("ng-talk-queue-clear");
    els.queue = $("ng-talk-queue");
    els.lora = $("ng-talk-lora");
    els.loraNote = $("ng-talk-lora-note");
    els.loraStrength = $("ng-talk-lora-strength");
    els.faceWrap = $("ng-talk-face-wrap");
    els.faceEmpty = $("ng-talk-face-empty");
    els.faceName = $("ng-talk-face-name");
    els.faceFile = $("ng-talk-face-file");
    els.faceClear = $("ng-talk-face-clear");
    els.subject = $("ng-talk-subject");
    els.scene = $("ng-talk-scene");
    els.voice = $("ng-talk-voice");
    els.style = $("ng-talk-style");
    els.presetWrap = $("ng-talk-preset-wrap");
    els.charWrap = $("ng-talk-character-wrap");
    els.character = $("ng-talk-character");
    els.charAudio = $("ng-talk-character-audio");
    els.charEmpty = $("ng-talk-character-empty");
    els.clipWrap = $("ng-talk-clip-wrap");
    els.clipEmpty = $("ng-talk-clip-empty");
    els.clipName = $("ng-talk-clip-name");
    els.clipFile = $("ng-talk-clip-file");
    els.cloneExtra = $("ng-talk-clone-extra");
    els.refText = $("ng-talk-reftext");
    els.consent = $("ng-talk-consent");
    els.text = $("ng-talk-text");
    els.selectAll = $("ng-talk-selectall");
    els.count = $("ng-talk-count");
    els.quality = $("ng-talk-quality");
    els.fast = $("ng-talk-fast");
    els.seed = $("ng-talk-seed");
    els.finish = $("ng-talk-finish");
    els.finishNote = $("ng-talk-finish-note");
    els.finishOpts = $("ng-talk-finish-opts");
    els.finishClarity = $("ng-talk-finish-clarity");
    els.finishTone = $("ng-talk-finish-tone");
    els.finishSat = $("ng-talk-finish-sat");
    els.finishHold = $("ng-talk-finish-hold");
    els.generate = $("ng-talk-generate");
    els.status = $("ng-talk-status");
  }

  function setStatus(msg, isError) {
    if (!els.status) return;
    els.status.textContent = msg || "";
    els.status.style.color = isError ? "var(--ng-danger)" : "";
  }

  function kind() {
    var r = document.querySelector('input[name="ng-talk-kind"]:checked');
    return r ? r.value : "preset";
  }

  function fillSelect(sel, items, valueKey, labelFn) {
    sel.innerHTML = "";
    items.forEach(function (it) {
      var o = document.createElement("option");
      o.value = it[valueKey];
      o.textContent = labelFn(it);
      sel.appendChild(o);
    });
  }

  // ---- options: the character LoRAs (the H3 library) and the voices (the Speech view's lists) ----

  function loadOptions() {
    if (optionsLoaded) return;
    Promise.all([
      fetch("/api/ng/h3/loras").then(function (r) { return r.json(); }),
      fetch(SPEECH_API + "/voices").then(function (r) { return r.json(); }),
    ]).then(function (res) {
      var loras = ((res[0] && res[0].loras) || []).filter(function (l) { return l.ok; });
      var d = res[1];
      optionsLoaded = true;
      var none = [{ name: "", label: "(none -- use a face picture)" }];
      fillSelect(els.lora, none.concat(loras.map(function (l) {
        return { name: l.name, label: l.name + (l.size_mb ? " (" + l.size_mb + " MB)" : "") };
      })), "name", function (l) { return l.label; });
      maxChars = Math.min(d.max_chars || 600, 600);
      fillSelect(els.voice, d.presets, "id", function (p) {
        return p.label + " -- " + p.sub + " (" + p.native + ")";
      });
      var def = d.presets.filter(function (p) { return p.id === "aiden"; })[0];
      if (def) els.voice.value = def.id;
      fillSelect(els.style, d.styles, "id", function (s) { return s.label; });
      fillSelect(els.character, d.characters, "id", function (c) { return c.label; });
      els.charEmpty.style.display = d.characters.length ? "none" : "";
      els.character.style.display = d.characters.length ? "" : "none";
      els.charAudio.style.display = d.characters.length ? "" : "none";
      onCharacterChange();
      onLoraChange();
      updateCount();
    }).catch(function () { /* retried next time the view is opened */ });
  }

  function checkStatus() {
    fetch(API + "/status")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        finishOk = !!d.finish_ok;
        if (els.finish) {
          els.finish.disabled = !finishOk;
          if (!finishOk) els.finish.checked = false;
          els.finishNote.style.display = finishOk ? "none" : "";
          els.finishNote.textContent = finishOk ? "" : "Soft finish is not available: " + (d.finish_problems || []).join("; ");
          els.finishOpts.style.display = finishOk && els.finish.checked ? "" : "none";
        }
        if (!els.unavailable) return;
        if (d.ok) { els.unavailable.style.display = "none"; return; }
        els.unavailable.style.display = "";
        els.unavailableText.textContent = (d.problems || []).join(" ");
      })
      .catch(function () { /* transient */ });
  }

  function onLoraChange() {
    if (!els.lora || !els.loraNote) return;
    var has = !!els.lora.value;
    els.loraNote.textContent = has ? "" : "Without a LoRA the face comes from the picture alone.";
    els.loraStrength.disabled = !has;
  }

  function onCharacterChange() {
    if (!els.character || !els.charAudio) return;
    var name = els.character.value;
    els.charAudio.src = name ? SPEECH_API + "/voices/" + encodeURIComponent(name) + "/sample" : "";
  }

  function onKindChange() {
    var k = kind();
    els.presetWrap.style.display = k === "preset" ? "" : "none";
    els.charWrap.style.display = k === "character" ? "" : "none";
    els.clipWrap.style.display = k === "clip" ? "" : "none";
    els.cloneExtra.style.display = k === "preset" ? "none" : "";
    // The confirmation is per request: it clears whenever the voice changes.
    if (els.consent) els.consent.checked = false;
  }

  function setClip(file) {
    clipFile = file || null;
    els.clipName.textContent = clipFile ? clipFile.name : "";
    els.clipName.style.display = clipFile ? "" : "none";
    els.clipEmpty.style.display = clipFile ? "none" : "";
    if (els.consent) els.consent.checked = false;
  }

  function setFace(file) {
    faceFile = file || null;
    els.faceName.textContent = faceFile ? faceFile.name : "";
    els.faceName.style.display = faceFile ? "" : "none";
    els.faceEmpty.style.display = faceFile ? "none" : "";
    els.faceClear.style.display = faceFile ? "" : "none";
  }

  function updateCount() {
    if (!els.count || !els.text) return;
    var n = els.text.value.length;
    var words = (els.text.value.trim().match(/\S+/g) || []).length;
    els.count.textContent = words + " words, " + n + " / " + maxChars;
    els.count.style.color = n > maxChars ? "var(--ng-danger)" : words > 16 ? "var(--ng-warn, #c80)" : "";
  }

  // ---- queue ---------------------------------------------------------------------------------

  function snippet(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "..." : s; }

  function isActive(item) { return item.status === "queued" || item.status === "rendering"; }

  // A finished clip is still worth polling while its soft finish runs.
  function finishing(item) { return item.status === "done" && (item.finishStatus === "queued" || item.finishStatus === "rendering"); }
  function needsPolling(item) { return item.jobId && (isActive(item) || finishing(item)); }

  function finishForm(form) {
    if (els.finishClarity.value !== "") form.append("clarity_strength", els.finishClarity.value);
    if (els.finishTone.value !== "") form.append("tone_strength", els.finishTone.value);
    if (els.finishSat.value !== "") form.append("saturation", els.finishSat.value);
    if (els.finishHold.value !== "") form.append("hold", els.finishHold.value);
  }

  function itemFromJob(job) {
    var item = {
      localId: "t" + ++localSeq, jobId: job.job_id, status: "queued", text: job.text || "", voice: "",
      lora: "", stage: "queued", progress: {}, error: null, audioUrl: null, videoUrl: null, downloadUrl: null,
      originalUrl: null, finishStatus: null, finishError: null, finished: false,
      createdAt: Date.now(), jobIdAt: 0, removed: false,
    };
    applyJob(item, job);
    return item;
  }

  function applyJob(item, job) {
    item.voice = job.voice || item.voice;
    item.lora = job.lora || "";
    item.stage = job.stage || item.stage;
    item.progress = job.progress || {};
    if (job.audio_url && !item.audioUrl) item.audioUrl = job.audio_url + "?v=" + Date.now();
    item.finishStatus = job.finish_status || null;
    item.finishError = job.finish_error || null;
    if (job.status === "completed") {
      item.status = "done";
      // The shown clip switches to the finished one when it lands (the url changes, so the player reloads).
      if (job.video_url && (item.finished !== !!job.finished || !item.videoUrl)) {
        item.videoUrl = job.video_url + "?v=" + Date.now();
      }
      item.finished = !!job.finished;
      item.originalUrl = job.original_video_url || item.originalUrl;
      item.downloadUrl = job.download_url || item.downloadUrl;
    } else if (job.status === "failed") {
      item.status = "failed";
      item.error = job.cancelled ? "cancelled" : (job.error || "it failed");
    } else {
      item.status = job.status;
    }
  }

  function stageLabel(item) {
    if (item.status === "done") return "done";
    if (item.status === "failed") return "failed";
    if (item.status === "submitting") return "submitting...";
    if (item.status === "queued") return "queued";
    if (item.stage === "speaking") return "making the voice...";
    if (item.stage === "finishing" || finishing(item)) return "finishing (colour pass)...";
    var p = item.progress || {};
    if (p.step && p.steps) return "rendering, step " + p.step + "/" + p.steps;
    return p.phase ? "rendering (" + p.phase.replace(/_/g, " ") + ")" : "rendering...";
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(needsPolling);
    if (!active.length) { clearInterval(pollTimer); pollTimer = null; return; }
    active.forEach(function (item) {
      fetch(API + "/jobs/" + item.jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (!job) return;
          if (!job.status) {
            if (job.error) {
              item.status = "failed";
              item.error = "The server no longer has this job (it restarted).";
              renderQueue();
            }
            return;
          }
          applyJob(item, job);
          renderQueue();
        })
        .catch(function () { /* transient */ });
    });
  }

  function syncFromServer() {
    var startedAt = Date.now();
    fetch(API + "/jobs")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var jobs = (data && data.jobs) || [];
        var byId = {};
        jobs.forEach(function (j) { byId[j.job_id] = j; });
        var known = {};
        queue.forEach(function (q) { if (q.jobId) known[q.jobId] = true; });
        jobs.slice().reverse().forEach(function (j) {
          if (!known[j.job_id] && !dismissed[j.job_id]) queue.push(itemFromJob(j));
        });
        queue.forEach(function (q) {
          if (q.jobId && byId[q.jobId]) {
            applyJob(q, byId[q.jobId]);
          } else if (q.jobId && isActive(q) && q.jobIdAt && q.jobIdAt < startedAt - 1000) {
            q.status = "failed";
            q.error = "The server no longer has this job (it restarted).";
          } else if (!q.jobId && q.status === "submitting" && Date.now() - q.createdAt > SUBMIT_GIVE_UP_MS) {
            q.status = "failed";
            q.error = "The submit never completed.";
          }
        });
        renderQueue();
        if (queue.some(needsPolling)) ensurePolling();
      })
      .catch(function () { /* server busy or restarting */ });
  }

  function ensureSyncing() {
    if (syncTimer) return;
    syncTimer = setInterval(function () {
      if (els.pane && els.pane.style.display !== "none") syncFromServer();
    }, SYNC_MS);
  }

  function removeItem(localId) {
    var item = queue.filter(function (q) { return q.localId === localId; })[0];
    if (!item) return;
    item.removed = true;
    if (item.jobId) dismissed[item.jobId] = true;
    queue = queue.filter(function (q) { return q.localId !== localId; });
    renderQueue();
    if (item.jobId && isActive(item)) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  function buildRow(item) {
    var row = document.createElement("div");
    row.className = "ng-gen-queue-row st-" + item.status;
    var thumbs = document.createElement("div");
    thumbs.className = "thumbs";
    thumbs.textContent = "🗣";
    row.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "meta";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = snippet(item.text, 90);
    meta.appendChild(title);
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = [item.lora, item.voice].filter(Boolean).join(", ");
    meta.appendChild(sub);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
    var st = document.createElement("span");
    st.className = "st";
    st.textContent = stageLabel(item);
    statusRow.appendChild(st);

    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pose-remove";
    var running = item.status === "rendering";
    rm.title = running ? "stop this job" : item.status === "queued" ? "cancel this queued job" : "remove from the list";
    rm.textContent = running ? "stop" : item.status === "queued" ? "cancel" : "✕";
    if (isActive(item)) rm.style.cssText = "width:auto; padding:2px 8px; font-size:0.7857rem";
    rm.addEventListener("click", function () { removeItem(item.localId); });
    statusRow.appendChild(rm);
    row.appendChild(statusRow);

    if (item.status === "failed" && item.error) {
      var err = document.createElement("div");
      err.className = "log";
      err.style.color = "var(--ng-danger)";
      err.textContent = item.error;
      row.appendChild(err);
    }
    if (item.status === "done" && item.videoUrl) {
      var video = document.createElement("video");
      video.src = item.videoUrl;
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      row.appendChild(video);
      if (item.downloadUrl) {
        var dl = document.createElement("a");
        dl.className = "ng-btn";
        dl.href = item.downloadUrl;
        dl.textContent = item.finished ? "Download finished mp4" : "Download mp4";
        dl.style.gridColumn = "1 / -1";
        dl.style.marginTop = "4px";
        row.appendChild(dl);
      }
      if (item.finished && item.originalUrl) {
        var orig = document.createElement("a");
        orig.className = "ng-btn";
        orig.href = item.originalUrl;
        orig.target = "_blank";
        orig.textContent = "Original (unfinished)";
        orig.style.gridColumn = "1 / -1";
        orig.style.marginTop = "4px";
        row.appendChild(orig);
      }
      if (!item.finished && !finishing(item) && finishOk) {
        var fb = document.createElement("button");
        fb.type = "button";
        fb.className = "ng-btn";
        fb.textContent = "Soft finish";
        fb.title = "A gentle colour pass: lifts crushed shadows, softens skin. The original is kept.";
        fb.style.gridColumn = "1 / -1";
        fb.style.marginTop = "4px";
        fb.addEventListener("click", function () { finishItem(item); });
        row.appendChild(fb);
      }
      if (item.finishError) {
        var fe = document.createElement("div");
        fe.className = "log";
        fe.style.color = "var(--ng-danger)";
        fe.textContent = "Soft finish did not happen: " + item.finishError;
        row.appendChild(fe);
      }
    }
    return row;
  }

  function renderQueue() {
    if (!els.queue) return;
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";
    els.queue.innerHTML = "";
    queue.forEach(function (item) { els.queue.appendChild(buildRow(item)); });
  }

  function finishItem(item) {
    var form = new FormData();
    finishForm(form);
    item.finishStatus = "queued";
    item.finishError = null;
    renderQueue();
    fetch(API + "/jobs/" + item.jobId + "/finish", { method: "POST", body: form })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, payload: p }; }); })
      .then(function (r) {
        if (!r.ok) {
          item.finishStatus = null;
          item.finishError = (r.payload && r.payload.error) || "could not start the finish";
        } else {
          ensurePolling();
        }
        renderQueue();
      })
      .catch(function (e) { item.finishStatus = null; item.finishError = e.message; renderQueue(); });
  }

  // ---- submit --------------------------------------------------------------------------------

  function makeItTalk() {
    var k = kind();
    var text = (els.text.value || "").trim();
    if (!text) { setStatus("Write what they say first.", true); return; }
    if (text.length > maxChars) { setStatus("That is over " + maxChars + " characters; the clip is 5 seconds.", true); return; }
    if (!els.lora.value && !faceFile) { setStatus("Pick a character LoRA or add a face picture.", true); return; }
    if (k === "clip" && !clipFile) { setStatus("Add a voice clip first.", true); return; }
    if (k !== "preset" && !els.consent.checked) {
      setStatus("Tick the box to confirm the voice is yours or that you have permission to clone it.", true);
      return;
    }

    var form = new FormData();
    form.append("kind", k);
    form.append("text", text);
    form.append("quality", els.quality.value || "draft");
    if (els.fast && els.fast.value) form.append("fast", "true");
    if (els.seed.value !== "") form.append("seed", els.seed.value);
    if (els.finish && els.finish.checked) {
      form.append("finish", "true");
      finishForm(form);
    }
    if (els.lora.value) {
      form.append("lora", els.lora.value);
      form.append("lora_strength", els.loraStrength.value || "1");
    }
    if (faceFile) form.append("face", faceFile, faceFile.name);
    if (els.subject.value.trim()) form.append("subject", els.subject.value.trim());
    if (els.scene.value.trim()) form.append("scene", els.scene.value.trim());
    if (k === "preset") {
      form.append("voice", els.voice.value);
      form.append("style", els.style.value || "natural");
    } else {
      form.append("consent", "true");
      if (els.refText.value.trim()) form.append("ref_text", els.refText.value.trim());
      if (k === "character") form.append("character_voice", els.character.value);
      if (k === "clip") form.append("file", clipFile, clipFile.name);
    }

    var voiceName = k === "preset" ? (els.voice.options[els.voice.selectedIndex] || {}).text || ""
      : k === "character" ? els.character.value : "uploaded clip";
    var item = {
      localId: "t" + ++localSeq, jobId: null, status: "submitting", text: text, voice: voiceName.split(" -- ")[0],
      lora: els.lora.value || "", stage: "queued", progress: {}, error: null, audioUrl: null, videoUrl: null,
      downloadUrl: null, originalUrl: null, finishStatus: null, finishError: null, finished: false,
      createdAt: Date.now(), jobIdAt: 0, removed: false,
    };
    queue.unshift(item);
    renderQueue();
    setStatus("Sending...");

    fetch(API + "/generate", { method: "POST", body: form })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, payload: p }; }); })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          item.jobId = r.payload.job_id;
          item.jobIdAt = Date.now();
          if (item.removed) {
            fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
            return;
          }
          item.status = "queued";
          setStatus("Queued.");
          ensurePolling();
        } else {
          item.status = "failed";
          item.error = (r.payload && r.payload.error) || "could not start the job";
          setStatus(item.error, true);
        }
        renderQueue();
      })
      .catch(function (e) {
        item.status = "failed";
        item.error = e.message;
        setStatus(e.message, true);
        renderQueue();
      });
  }

  // ---- wiring --------------------------------------------------------------------------------

  function wireDrop(box, onFile) {
    ["dragenter", "dragover"].forEach(function (ev) {
      box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.remove("drag"); });
    });
    box.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) onFile(f);
    });
  }

  function init() {
    if (inited) return;
    if (!els.main || !els.generate) return;
    inited = true;
    Array.prototype.forEach.call(document.querySelectorAll('input[name="ng-talk-kind"]'), function (r) {
      r.addEventListener("change", onKindChange);
    });
    els.lora.addEventListener("change", onLoraChange);
    els.finish.addEventListener("change", function () {
      els.finishOpts.style.display = finishOk && els.finish.checked ? "" : "none";
    });
    els.character.addEventListener("change", onCharacterChange);
    els.text.addEventListener("input", updateCount);
    els.selectAll.addEventListener("click", function () { els.text.focus(); els.text.select(); });
    els.generate.addEventListener("click", makeItTalk);
    els.queueClear.addEventListener("click", function () {
      queue.forEach(function (q) {
        if (q.jobId && (q.status === "done" || q.status === "failed")) dismissed[q.jobId] = true;
      });
      queue = queue.filter(function (q) { return q.status !== "done" && q.status !== "failed"; });
      renderQueue();
    });
    els.clipFile.addEventListener("change", function () { setClip(els.clipFile.files[0]); });
    els.faceFile.addEventListener("change", function () { setFace(els.faceFile.files[0]); });
    els.faceClear.addEventListener("click", function () { els.faceFile.value = ""; setFace(null); });
    // Drag and drop (the iPad has no file-picker habit): drop a voice clip or a picture on its box.
    wireDrop(els.clipWrap, setClip);
    wireDrop(els.faceWrap, setFace);
    onKindChange();
    updateCount();
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "talk");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;
    init();
    loadOptions();
    checkStatus();
    renderQueue();
    syncFromServer();
    ensureSyncing();
  }

  window.TalkNG = { sync: sync };
})();
