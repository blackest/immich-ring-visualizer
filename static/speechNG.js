/* Speech view (task === "speech") -- text to speech in a preset voice, a character's saved voice, or a
 * clip. Rail pane = the speech queue, main = the form. Backend: routes/speechNG.py.
 *
 * The queue is rebuilt from the server (GET /api/ng/speech/jobs) when the view opens and every few seconds,
 * so a reload or a lost submit can never hide a job; a waiting job can be cancelled and a running one
 * stopped from its row. Exposes window.SpeechNG = { sync }. Called from ProjectManager.render().
 */
(function () {
  "use strict";

  var API = "/api/ng/speech";
  var POLL_MS = 2000;
  var SYNC_MS = 6000;
  var SUBMIT_GIVE_UP_MS = 60000;

  var els = {};
  var inited = false;
  var queue = [];
  var localSeq = 0;
  var pollTimer = null;
  var syncTimer = null;
  var dismissed = {};
  var clipFile = null;
  var voicesLoaded = false;
  var maxChars = 2000;

  function $(id) { return document.getElementById(id); }

  function refreshEls() {
    els.pane = $("ng-speech-pane");
    els.main = $("ng-speech-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-speech-unavailable");
    els.unavailableText = $("ng-speech-unavailable-text");
    els.queueCount = $("ng-speech-queue-count");
    els.queueEmpty = $("ng-speech-queue-empty");
    els.queueClear = $("ng-speech-queue-clear");
    els.queue = $("ng-speech-queue");
    els.voice = $("ng-speech-voice");
    els.style = $("ng-speech-style");
    els.styleText = $("ng-speech-style-text");
    els.presetWrap = $("ng-speech-preset-wrap");
    els.charWrap = $("ng-speech-character-wrap");
    els.character = $("ng-speech-character");
    els.charAudio = $("ng-speech-character-audio");
    els.charEmpty = $("ng-speech-character-empty");
    els.clipWrap = $("ng-speech-clip-wrap");
    els.clipEmpty = $("ng-speech-clip-empty");
    els.clipName = $("ng-speech-clip-name");
    els.clipFile = $("ng-speech-clip-file");
    els.cloneExtra = $("ng-speech-clone-extra");
    els.designWrap = $("ng-speech-design-wrap");
    els.design = $("ng-speech-design");
    els.designRadio = document.querySelector('input[name="ng-speech-kind"][value="design"]');
    els.refText = $("ng-speech-reftext");
    els.consent = $("ng-speech-consent");
    els.text = $("ng-speech-text");
    els.selectAll = $("ng-speech-selectall");
    els.count = $("ng-speech-count");
    els.language = $("ng-speech-language");
    els.speed = $("ng-speech-speed");
    els.seed = $("ng-speech-seed");
    els.generate = $("ng-speech-generate");
    els.status = $("ng-speech-status");
  }

  function setStatus(msg, isError) {
    if (!els.status) return;
    els.status.textContent = msg || "";
    els.status.style.color = isError ? "var(--ng-danger)" : "";
  }

  function kind() {
    var r = document.querySelector('input[name="ng-speech-kind"]:checked');
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

  function loadVoices() {
    if (voicesLoaded) return;
    fetch(API + "/voices")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        voicesLoaded = true;
        maxChars = d.max_chars || maxChars;
        fillSelect(els.voice, d.presets, "id", function (p) {
          return p.label + " -- " + p.sub + " (" + p.native + ")";
        });
        var def = d.presets.filter(function (p) { return p.id === "aiden"; })[0];
        if (def) els.voice.value = def.id;
        fillSelect(els.style, d.styles, "id", function (s) { return s.label; });
        fillSelect(els.language, d.languages.map(function (l) { return { id: l }; }), "id", function (l) {
          return l.id.charAt(0).toUpperCase() + l.id.slice(1);
        });
        els.language.value = "english";
        fillSelect(els.character, d.characters, "id", function (c) { return c.label; });
        els.charEmpty.style.display = d.characters.length ? "none" : "";
        els.character.style.display = d.characters.length ? "" : "none";
        els.charAudio.style.display = d.characters.length ? "" : "none";
        onCharacterChange();
        updateCount();
      })
      .catch(function () { /* retried next time the view is opened */ });
  }

  function checkStatus() {
    fetch(API + "/status")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (els.designRadio) {            // the option only shows when the voice-design model is installed
          var lab = els.designRadio.closest("label");
          if (lab) lab.style.display = d.design_ok ? "" : "none";
        }
        if (!els.unavailable) return;
        if (d.ok) { els.unavailable.style.display = "none"; return; }
        els.unavailable.style.display = "";
        els.unavailableText.textContent = "The speech engine is not ready: " + (d.missing || []).join("; ");
      })
      .catch(function () { /* transient */ });
  }

  function onCharacterChange() {
    if (!els.character || !els.charAudio) return;
    var name = els.character.value;
    els.charAudio.src = name ? API + "/voices/" + encodeURIComponent(name) + "/sample" : "";
  }

  function onKindChange() {
    var k = kind();
    els.presetWrap.style.display = k === "preset" ? "" : "none";
    els.charWrap.style.display = k === "character" ? "" : "none";
    els.clipWrap.style.display = k === "clip" ? "" : "none";
    els.designWrap.style.display = k === "design" ? "" : "none";
    els.cloneExtra.style.display = (k === "preset" || k === "design") ? "none" : "";
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

  function updateCount() {
    if (!els.count || !els.text) return;
    var n = els.text.value.length;
    els.count.textContent = n + " / " + maxChars;
    els.count.style.color = n > maxChars ? "var(--ng-danger)" : "";
  }

  // ---- queue ---------------------------------------------------------------------------------

  function snippet(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "..." : s; }

  function itemFromJob(job) {
    var item = {
      localId: "s" + ++localSeq, jobId: job.job_id, status: "queued", text: job.text || "",
      voice: job.voice || "", style: job.style || "", kind: job.kind || "", audioSeconds: null, error: null, audioUrl: null,
      createdAt: Date.now(), jobIdAt: 0, removed: false,
    };
    applyJob(item, job);
    return item;
  }

  function applyJob(item, job) {
    item.voice = job.voice || item.voice;
    item.kind = job.kind || item.kind;
    item.audioSeconds = job.audio_seconds || item.audioSeconds;
    if (job.cancelled && job.status === "queued") {
      item.status = "failed";
      item.error = "cancelled";
    } else if (job.status === "completed") {
      item.status = "done";
      if (!item.audioUrl) item.audioUrl = job.audio_url + "?v=" + Date.now();
    } else if (job.status === "failed") {
      item.status = "failed";
      item.error = job.cancelled ? "cancelled" : (job.error || "generation failed");
    } else {
      item.status = job.status;
    }
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(function (q) { return q.jobId && (q.status === "queued" || q.status === "rendering"); });
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
          var active = q.status === "queued" || q.status === "rendering";
          if (q.jobId && byId[q.jobId]) {
            applyJob(q, byId[q.jobId]);
          } else if (q.jobId && active && q.jobIdAt && q.jobIdAt < startedAt - 1000) {
            q.status = "failed";
            q.error = "The server no longer has this job (it restarted).";
          } else if (!q.jobId && q.status === "submitting" && Date.now() - q.createdAt > SUBMIT_GIVE_UP_MS) {
            q.status = "failed";
            q.error = "The submit never completed.";
          }
        });
        renderQueue();
        if (queue.some(function (q) { return q.jobId && (q.status === "queued" || q.status === "rendering"); })) {
          ensurePolling();
        }
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
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  function buildRow(item) {
    var row = document.createElement("div");
    row.className = "ng-gen-queue-row st-" + item.status;
    var thumbs = document.createElement("div");
    thumbs.className = "thumbs";
    thumbs.textContent = "🔊";
    row.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "meta";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = snippet(item.text, 90);
    meta.appendChild(title);
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = (item.voice || "") + (item.audioSeconds ? ", " + item.audioSeconds + "s" : "");
    meta.appendChild(sub);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
    var st = document.createElement("span");
    st.className = "st";
    st.textContent = item.status === "done" ? "done" : item.status === "failed" ? "failed"
      : item.status === "rendering" ? "speaking..." : item.status === "submitting" ? "submitting..." : "queued";
    statusRow.appendChild(st);

    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pose-remove";
    var isActive = item.status === "queued" || item.status === "rendering";
    rm.title = item.status === "rendering" ? "stop this job" : item.status === "queued" ? "cancel this queued job" : "remove from the list";
    rm.textContent = item.status === "rendering" ? "stop" : item.status === "queued" ? "cancel" : "✕";
    if (isActive) rm.style.cssText = "width:auto; padding:2px 8px; font-size:0.7857rem";
    rm.addEventListener("click", function () { removeItem(item.localId); });
    statusRow.appendChild(rm);
    row.appendChild(statusRow);

    if (item.status === "failed" && item.error) {
      var err = document.createElement("div");
      err.className = "log";
      err.style.color = "var(--ng-danger)";
      err.textContent = item.error;
      row.appendChild(err);
    } else if (item.status === "done" && item.audioUrl) {
      var audio = document.createElement("audio");
      audio.src = item.audioUrl;
      audio.controls = true;
      audio.style.gridColumn = "1 / -1";
      audio.style.width = "100%";
      audio.style.marginTop = "4px";
      row.appendChild(audio);
      var dl = document.createElement("a");
      dl.className = "ng-btn";
      dl.href = API + "/jobs/" + item.jobId + "/download";
      dl.textContent = "Download wav";
      dl.style.gridColumn = "1 / -1";
      dl.style.marginTop = "4px";
      row.appendChild(dl);
      if (item.kind === "design") row.appendChild(buildSaveVoice(item));
    }
    return row;
  }

  // A described voice differs on every generation. Keeping a take as a saved voice makes it a clone reference.
  function buildSaveVoice(item) {
    var box = document.createElement("div");
    box.style.cssText = "grid-column: 1 / -1; display: flex; gap: 6px; margin-top: 4px; align-items: center; flex-wrap: wrap";
    var input = document.createElement("input");
    input.type = "text"; input.maxLength = 64; input.className = "ng-gen-input";
    input.placeholder = "Name this voice, e.g. mara"; input.style.flex = "1 1 120px";
    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "ng-gen-btn ng-gen-btn-quiet"; btn.style.width = "auto";
    btn.textContent = "save this voice";
    var msg = document.createElement("span");
    msg.className = "ng-gen-status";
    btn.addEventListener("click", function () {
      var name = input.value.trim();
      if (!name) { msg.textContent = "Give it a name first."; return; }
      var form = new FormData(); form.append("name", name);
      fetch(API + "/jobs/" + item.jobId + "/save-voice", { method: "POST", body: form })
        .then(function (r) { return r.json().then(function (p) { return { ok: r.ok, p: p }; }); })
        .then(function (r) {
          if (!r.ok) { msg.textContent = r.p.error || "could not save"; return; }
          msg.textContent = "Saved as " + r.p.name + ". It is now a saved voice (reload the voice lists to see it).";
          voicesLoaded = false; loadVoices();
        })
        .catch(function (e) { msg.textContent = e.message; });
    });
    box.appendChild(input); box.appendChild(btn); box.appendChild(msg);
    return box;
  }

  function renderQueue() {
    if (!els.queue) return;
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";
    els.queue.innerHTML = "";
    queue.forEach(function (item) { els.queue.appendChild(buildRow(item)); });
  }

  // ---- submit --------------------------------------------------------------------------------

  function speak() {
    var k = kind();
    var text = (els.text.value || "").trim();
    if (!text) { setStatus("Write a script first.", true); return; }
    if (text.length > maxChars) { setStatus("The script is over " + maxChars + " characters.", true); return; }
    if (k === "clip" && !clipFile) { setStatus("Add a voice clip first.", true); return; }
    var design = (els.design.value || "").replace(/\s+/g, " ").trim();
    if (k === "design" && (design.length < 8 || design.length > 300)) {
      setStatus("Describe the voice in 8 to 300 characters, e.g. a warm, slow, older male voice.", true);
      return;
    }
    if ((k === "character" || k === "clip") && !els.consent.checked) {
      setStatus("Tick the box to confirm the voice is yours or that you have permission to clone it.", true);
      return;
    }

    var form = new FormData();
    form.append("kind", k);
    form.append("text", text);
    form.append("language", els.language.value || "english");
    form.append("speed", els.speed.value || "1");
    if (els.seed.value !== "") form.append("seed", els.seed.value);
    if (k === "preset") {
      form.append("voice", els.voice.value);
      form.append("style", els.style.value || "natural");
      if (els.styleText.value.trim()) form.append("style_text", els.styleText.value.trim());
    } else if (k === "design") {
      form.append("design_text", design);
    } else {
      form.append("consent", "true");
      if (els.refText.value.trim()) form.append("ref_text", els.refText.value.trim());
      if (k === "character") form.append("character", els.character.value);
      if (k === "clip") form.append("file", clipFile, clipFile.name);
    }

    var voiceName = k === "preset" ? (els.voice.options[els.voice.selectedIndex] || {}).text || ""
      : k === "character" ? els.character.value : k === "design" ? "described voice" : "uploaded clip";
    var item = {
      localId: "s" + ++localSeq, jobId: null, status: "submitting", text: text,
      voice: voiceName.split(" -- ")[0], style: "", kind: k, audioSeconds: null, error: null, audioUrl: null,
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

  function init() {
    if (inited) return;
    if (!els.main || !els.generate) return;
    inited = true;
    Array.prototype.forEach.call(document.querySelectorAll('input[name="ng-speech-kind"]'), function (r) {
      r.addEventListener("change", onKindChange);
    });
    els.character.addEventListener("change", onCharacterChange);
    els.text.addEventListener("input", updateCount);
    els.selectAll.addEventListener("click", function () { els.text.focus(); els.text.select(); });
    els.generate.addEventListener("click", speak);
    els.queueClear.addEventListener("click", function () {
      queue.forEach(function (q) {
        if (q.jobId && (q.status === "done" || q.status === "failed")) dismissed[q.jobId] = true;
      });
      queue = queue.filter(function (q) { return q.status !== "done" && q.status !== "failed"; });
      renderQueue();
    });
    els.clipFile.addEventListener("change", function () { setClip(els.clipFile.files[0]); });
    // Drag and drop (the iPad has no file-picker habit): drop an audio file anywhere on the clip box.
    ["dragenter", "dragover"].forEach(function (ev) {
      els.clipWrap.addEventListener(ev, function (e) { e.preventDefault(); els.clipWrap.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      els.clipWrap.addEventListener(ev, function (e) { e.preventDefault(); els.clipWrap.classList.remove("drag"); });
    });
    els.clipWrap.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) setClip(f);
    });
    onKindChange();
    updateCount();
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "speech");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;
    init();
    loadVoices();
    checkStatus();
    renderQueue();
    syncFromServer();
    ensureSyncing();
  }

  window.SpeechNG = { sync: sync };
})();
