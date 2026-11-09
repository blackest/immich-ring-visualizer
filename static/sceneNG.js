/* Scene view (task === "scene") -- a script of lines said by different characters, rendered as one joined clip.
 * Rail pane = the scene queue, main = characters + script + options. Backend: routes/sceneNG.py (one speech pass per
 * character, one H3 lip-sync clip per line, then an exposure-matched join).
 *
 * Script format:   KNIGHT [on the battlements]: The wall has stood a long time.
 *                  WOMAN [in the great hall]: Come in out of the wind.
 * The optional [setting] says where THAT line happens; a character's picture ("look") with the same name is used as
 * that clip's first frame. A line starting with "&" puts its speaker in the SAME frame as the line before (a phone call,
 * a split-screen): both characters need a picture, and the two lines together are about 12 words.
 *
 * The queue is rebuilt from the server (GET /api/ng/scene/jobs) when the view opens and every few seconds, so a reload
 * or a lost submit can never hide a job. Exposes window.SceneNG = { sync }. Called from ProjectManager.render().
 */
(function () {
  "use strict";

  var API = "/api/ng/scene";
  var SPEECH_API = "/api/ng/speech";
  var POLL_MS = 4000;
  var SYNC_MS = 9000;
  var SUBMIT_GIVE_UP_MS = 60000;
  var MAX_LINES = 12;

  var els = {};
  var inited = false;
  var queue = [];
  var localSeq = 0;
  var pollTimer = null;
  var syncTimer = null;
  var dismissed = {};
  var optionsLoaded = false;
  var options = { refmods: [], loras: [], presets: [], styles: [], voices: [] };
  var cards = [];                       // one object per character card: { root, el:{...}, looks:[{name,file}], clip }
  var finishOk = false;

  function $(id) { return document.getElementById(id); }

  function refreshEls() {
    els.pane = $("ng-scene-pane");
    els.main = $("ng-scene-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-scene-unavailable");
    els.unavailableText = $("ng-scene-unavailable-text");
    els.queueCount = $("ng-scene-queue-count");
    els.queueEmpty = $("ng-scene-queue-empty");
    els.queueClear = $("ng-scene-queue-clear");
    els.queue = $("ng-scene-queue");
    els.cards = $("ng-scene-cards");
    els.addChar = $("ng-scene-add-char");
    els.script = $("ng-scene-script");
    els.selectAll = $("ng-scene-selectall");
    els.count = $("ng-scene-count");
    els.quality = $("ng-scene-quality");
    els.fast = $("ng-scene-fast");
    els.match = $("ng-scene-match");
    els.ambience = $("ng-scene-ambience");
    els.finish = $("ng-scene-finish");
    els.finishNote = $("ng-scene-finish-note");
    els.finishOpts = $("ng-scene-finish-opts");
    els.fClarity = $("ng-scene-finish-clarity");
    els.fTone = $("ng-scene-finish-tone");
    els.fSat = $("ng-scene-finish-sat");
    els.seed = $("ng-scene-seed");
    els.generate = $("ng-scene-generate");
    els.status = $("ng-scene-status");
  }

  function setStatus(msg, isError) {
    if (!els.status) return;
    els.status.textContent = msg || "";
    els.status.style.color = isError ? "var(--ng-danger)" : "";
  }

  function mk(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
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

  // ---- options (LoRAs, voices) and readiness ------------------------------------------------------------------

  function loadOptions() {
    if (optionsLoaded) return;
    Promise.all([
      fetch("/api/ng/h3/loras").then(function (r) { return r.json(); }),
      fetch(SPEECH_API + "/voices").then(function (r) { return r.json(); }),
      fetch("/api/ng/h3/refmods").then(function (r) { return r.json(); }).catch(function () { return {}; }),
    ]).then(function (res) {
      options.refmods = ((res[2] && res[2].refmods) || []).filter(function (m) { return m.ok !== false; });
      options.loras = ((res[0] && res[0].loras) || []).filter(function (l) { return l.ok; });
      options.presets = res[1].presets || [];
      options.styles = res[1].styles || [];
      options.voices = res[1].characters || [];
      optionsLoaded = true;
      cards.forEach(populateCard);
    }).catch(function () { /* retried next time the view is opened */ });
  }

  function checkStatus() {
    fetch(API + "/status")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        finishOk = !!d.finish_ok;
        MAX_LINES = d.max_lines || MAX_LINES;
        if (els.finish) {
          els.finish.disabled = !finishOk;
          els.match.disabled = !finishOk;
          if (!finishOk) { els.finish.checked = false; els.match.checked = false; }
          els.finishNote.style.display = finishOk ? "none" : "";
          els.finishNote.textContent = finishOk ? "" : "Exposure matching and the soft finish are not available: " + (d.finish_problems || []).join("; ");
          els.finishOpts.style.display = finishOk && els.finish.checked ? "" : "none";
        }
        if (!els.unavailable) return;
        if (d.ok) { els.unavailable.style.display = "none"; return; }
        els.unavailable.style.display = "";
        els.unavailableText.textContent = (d.problems || []).join(" ");
      })
      .catch(function () { /* transient */ });
  }

  // ---- character cards ----------------------------------------------------------------------------------------

  function addCharacter(name) {
    var c = { looks: [], clip: null, el: {} };
    var root = mk("div", "ng-scene-card");
    c.root = root;
    var head = mk("div", "ng-scene-card-head");
    c.el.name = mk("input", "ng-gen-input");
    c.el.name.type = "text";
    c.el.name.maxLength = 30;
    c.el.name.placeholder = "Name, as used in the script (e.g. KNIGHT)";
    c.el.name.value = name || "";
    var rm = mk("button", "ng-gen-btn ng-gen-btn-quiet", "remove");
    rm.type = "button";
    rm.style.width = "auto";
    rm.addEventListener("click", function () {
      cards = cards.filter(function (x) { return x !== c; });
      root.remove();
    });
    head.appendChild(c.el.name);
    head.appendChild(rm);
    root.appendChild(head);

    c.el.subject = mk("input", "ng-gen-input");
    c.el.subject.type = "text";
    c.el.subject.maxLength = 200;
    c.el.subject.placeholder = "Who is on screen, e.g. an older man with a bald head and a grey beard";
    root.appendChild(c.el.subject);

    root.appendChild(mk("div", "ng-gen-label", "Face"));
    var faceRow = mk("div", "ng-gen-identity-row");
    var loraWrap = mk("div");
    loraWrap.appendChild(mk("label", "ng-gen-label", "LoRA (optional)"));
    c.el.lora = mk("select", "ng-gen-input");
    loraWrap.appendChild(c.el.lora);
    var rmWrap = mk("div");
    rmWrap.appendChild(mk("label", "ng-gen-label", "RefMod (a whole character, one shot each, no shared frames)"));
    c.el.refmod = mk("select", "ng-gen-input");
    rmWrap.appendChild(c.el.refmod);
    var strWrap = mk("div");
    strWrap.appendChild(mk("label", "ng-gen-label", "Strength"));
    c.el.strength = mk("input", "ng-gen-input");
    c.el.strength.type = "number"; c.el.strength.min = "0"; c.el.strength.max = "2"; c.el.strength.step = "0.05"; c.el.strength.value = "1";
    strWrap.appendChild(c.el.strength);
    faceRow.appendChild(loraWrap);
    faceRow.appendChild(rmWrap);
    faceRow.appendChild(strWrap);
    root.appendChild(faceRow);

    c.el.looksBox = mk("div", "ng-scene-looks");
    root.appendChild(c.el.looksBox);
    var addLook = mk("button", "ng-gen-btn ng-gen-btn-quiet", "add a picture (a look)");
    addLook.type = "button";
    addLook.style.width = "auto";
    addLook.addEventListener("click", function () { addLookRow(c); });
    root.appendChild(addLook);
    addMaker(c, root);

    root.appendChild(mk("div", "ng-gen-label", "Voice"));
    var kinds = mk("div");
    ["preset", "character", "clip"].forEach(function (k, i) {
      var lab = mk("label", "ng-checkbox-row");
      var r = document.createElement("input");
      r.type = "radio"; r.name = "ng-scene-kind-" + (++localSeq); r.value = k; r.checked = i === 0;
      c.el["kind_" + k] = r;
      lab.appendChild(r);
      lab.appendChild(document.createTextNode(" " + (k === "preset" ? "Preset voice" : k === "character" ? "A saved voice" : "A voice clip")));
      kinds.appendChild(lab);
      r.addEventListener("change", function () { updateVoiceKind(c); });
    });
    root.appendChild(kinds);

    c.el.presetWrap = mk("div");
    c.el.voice = mk("select", "ng-gen-input");
    c.el.style = mk("select", "ng-gen-input");
    c.el.presetWrap.appendChild(c.el.voice);
    c.el.presetWrap.appendChild(c.el.style);
    root.appendChild(c.el.presetWrap);

    c.el.savedWrap = mk("div");
    c.el.saved = mk("select", "ng-gen-input");
    c.el.savedWrap.appendChild(c.el.saved);
    c.el.savedWrap.style.display = "none";
    root.appendChild(c.el.savedWrap);

    c.el.clipWrap = mk("div", "ng-dropzone");
    c.el.clipWrap.style.display = "none";
    c.el.clipName = mk("p", "ng-placeholder", "No clip yet. Drag an audio file here, or:");
    c.el.clipFile = document.createElement("input");
    c.el.clipFile.type = "file"; c.el.clipFile.accept = "audio/*,video/*";
    c.el.clipFile.addEventListener("change", function () { setClip(c, c.el.clipFile.files[0]); });
    c.el.clipWrap.appendChild(c.el.clipName);
    c.el.clipWrap.appendChild(c.el.clipFile);
    wireDrop(c.el.clipWrap, function (f) { setClip(c, f); });
    root.appendChild(c.el.clipWrap);

    c.el.cloneExtra = mk("div");
    c.el.cloneExtra.style.display = "none";
    c.el.refText = mk("textarea", "ng-gen-input");
    c.el.refText.rows = 2; c.el.refText.maxLength = 600;
    c.el.refText.placeholder = "What the sample says (optional; a closer match)";
    var consentLab = mk("label", "ng-checkbox-row");
    c.el.consent = document.createElement("input");
    c.el.consent.type = "checkbox";
    consentLab.appendChild(c.el.consent);
    consentLab.appendChild(document.createTextNode(" This voice is my own, or I have the speaker's permission to clone it."));
    c.el.cloneExtra.appendChild(c.el.refText);
    c.el.cloneExtra.appendChild(consentLab);
    root.appendChild(c.el.cloneExtra);

    cards.push(c);
    els.cards.appendChild(root);
    populateCard(c);
    return c;
  }

  // ---- make pictures from a picture (Klein): a look in a new place, or the eight-shot identity set -------------------

  function kleinPoll(jobId, onDone, onUpdate) {
    var tries = 0;
    (function tick() {
      fetch("/api/ng/klein/status/" + jobId).then(function (r) { return r.json(); }).then(function (d) {
        var j = d.job || d;
        if (onUpdate) onUpdate(j);
        if (j.status === "done") return onDone(j);
        if (j.status === "error" || j.status === "cancelled") return onDone(j);
        if (++tries < 400) setTimeout(tick, 4000);
      }).catch(function () { if (++tries < 400) setTimeout(tick, 6000); });
    })();
  }

  function sourcePicture(c) {
    var f = c.el.srcFile.files[0];
    if (f) return f;
    var withFile = c.looks.filter(function (l) { return l.file; })[0];
    return withFile ? withFile.file : null;
  }

  function addMaker(c, root) {
    var box = document.createElement("details");
    box.className = "ng-scene-maker";
    box.appendChild(mk("summary", null, "Make pictures from a picture (Klein)"));
    box.appendChild(mk("p", "ng-placeholder",
      "Start from one picture of the character (choose it here, or the first look above is used)."));
    c.el.srcFile = document.createElement("input");
    c.el.srcFile.type = "file"; c.el.srcFile.accept = "image/*";
    box.appendChild(c.el.srcFile);

    box.appendChild(mk("div", "ng-gen-label", "A look: the same person somewhere else"));
    c.el.lookSetting = mk("input", "ng-gen-input");
    c.el.lookSetting.type = "text"; c.el.lookSetting.maxLength = 160;
    c.el.lookSetting.placeholder = "Where, e.g. in a torch-lit great hall, wearing a dark cloak";
    box.appendChild(c.el.lookSetting);
    var lookBtn = mk("button", "ng-gen-btn ng-gen-btn-quiet", "make 2 looks");
    lookBtn.type = "button"; lookBtn.style.width = "auto";
    c.el.lookMsg = mk("div", "ng-gen-status");
    c.el.lookPics = mk("div", "ng-scene-pics");
    box.appendChild(lookBtn); box.appendChild(c.el.lookMsg); box.appendChild(c.el.lookPics);

    box.appendChild(mk("div", "ng-gen-label", "An identity RefMod: eight shots of the same person"));
    box.appendChild(mk("p", "ng-placeholder",
      "Four faces at different angles and expressions and four body shots, all with the same face, hair, skin and " +
      "outfit, then a RefMod built from them and your picture. About 25 minutes."));
    c.el.kitName = mk("input", "ng-gen-input");
    c.el.kitName.type = "text"; c.el.kitName.maxLength = 63;
    c.el.kitName.placeholder = "RefMod name, e.g. knight_v1 (leave empty for the pictures only)";
    box.appendChild(c.el.kitName);
    var kitBtn = mk("button", "ng-gen-btn ng-gen-btn-quiet", "build the eight-shot set");
    kitBtn.type = "button"; kitBtn.style.width = "auto";
    c.el.kitMsg = mk("div", "ng-gen-status");
    c.el.kitPics = mk("div", "ng-scene-pics");
    box.appendChild(kitBtn); box.appendChild(c.el.kitMsg); box.appendChild(c.el.kitPics);
    root.appendChild(box);

    lookBtn.addEventListener("click", function () { makeLooks(c, lookBtn); });
    kitBtn.addEventListener("click", function () { buildKit(c, kitBtn); });
  }

  function who(c) {
    return (c.el.subject.value || "").trim().replace(/^(an?|the)\s+/i, "");
  }

  function makeLooks(c, btn) {
    var pic = sourcePicture(c), setting = c.el.lookSetting.value.trim();
    if (!pic) { c.el.lookMsg.textContent = "Choose a picture first."; return; }
    if (!setting) { c.el.lookMsg.textContent = "Say where."; return; }
    var form = new FormData();
    form.append("picture", pic, pic.name);
    form.append("setting", setting);
    form.append("who", who(c));
    form.append("n", "2");
    btn.disabled = true;
    c.el.lookMsg.textContent = "Sending...";
    c.el.lookPics.innerHTML = "";
    fetch(API + "/looks", { method: "POST", body: form })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, p: p }; }); })
      .then(function (r) {
        if (!r.ok) { btn.disabled = false; c.el.lookMsg.textContent = r.p.error || "could not start"; return; }
        c.el.lookMsg.textContent = "Making the pictures (about 3 minutes each)...";
        kleinPoll(r.p.klein_job, function (j) {
          btn.disabled = false;
          if (j.status !== "done") { c.el.lookMsg.textContent = "Klein failed: " + (j.error || j.status); return; }
          c.el.lookMsg.textContent = "Click a picture to use it as the look \"" + setting.slice(0, 40) + "\".";
          (j.results || []).forEach(function (res) { showLookPick(c, r.p.klein_job, res.file, setting); });
        }, function (j) {
          if (j.status === "running" || j.status === "waiting") {
            c.el.lookMsg.textContent = (j.status === "waiting" ? "Waiting for the video render to finish... " : "Making the pictures (about 3 minutes each)... ") +
              (j.results ? j.results.length : 0) + " of 2";
          }
        });
      })
      .catch(function (e) { btn.disabled = false; c.el.lookMsg.textContent = e.message; });
  }

  function showLookPick(c, jobId, file, setting) {
    var img = document.createElement("img");
    img.src = "/api/ng/klein/image/" + jobId + "/" + file;
    img.className = "ng-scene-pick";
    img.title = "use as the look \"" + setting.slice(0, 40) + "\"";
    img.addEventListener("click", function () {
      fetch(img.src).then(function (r) { return r.blob(); }).then(function (blob) {
        var f = new File([blob], file, { type: blob.type || "image/png" });
        addLookRow(c, setting.slice(0, 40), f);
        c.el.lookMsg.textContent = "Added as a look. Tag script lines [" + setting.slice(0, 40) + "] to start from it.";
      });
    });
    c.el.lookPics.appendChild(img);
  }

  function buildKit(c, btn) {
    var pic = sourcePicture(c);
    if (!pic) { c.el.kitMsg.textContent = "Choose a picture first."; return; }
    var form = new FormData();
    form.append("picture", pic, pic.name);
    if (c.el.kitName.value.trim()) form.append("name", c.el.kitName.value.trim());
    btn.disabled = true;
    c.el.kitMsg.textContent = "Sending...";
    c.el.kitPics.innerHTML = "";
    fetch(API + "/kit", { method: "POST", body: form })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, p: p }; }); })
      .then(function (r) {
        if (!r.ok) { btn.disabled = false; c.el.kitMsg.textContent = r.p.error || "could not start"; return; }
        var tries = 0;
        (function tick() {
          fetch(API + "/kit/" + r.p.job_id).then(function (x) { return x.json(); }).then(function (d) {
            c.el.kitPics.innerHTML = "";
            (d.shots || []).forEach(function (sh) {
              if (!sh.url) return;
              var im = document.createElement("img"); im.src = sh.url; im.className = "ng-scene-pick"; im.title = sh.label;
              c.el.kitPics.appendChild(im);
            });
            var label = d.stage === "refmod" ? "Building the RefMod..." : d.stage === "done" ? "" : "Making shot " + Math.min(d.done + 1, d.total) + " of " + d.total + "...";
            if (d.stage === "done") {
              btn.disabled = false;
              c.el.kitMsg.textContent = d.name ? "Done: the RefMod " + d.name + " is in the H3 RefMod list." : "Done: the eight pictures are above.";
              return;
            }
            if (d.stage === "failed" || d.stage === "cancelled") { btn.disabled = false; c.el.kitMsg.textContent = "Stopped: " + (d.error || d.stage); return; }
            c.el.kitMsg.textContent = label;
            if (++tries < 600) setTimeout(tick, 5000);
          }).catch(function () { if (++tries < 600) setTimeout(tick, 8000); });
        })();
      })
      .catch(function (e) { btn.disabled = false; c.el.kitMsg.textContent = e.message; });
  }

  function populateCard(c) {
    if (!c.el.lora) return;
    var prev = c.el.lora.value;
    fillSelect(c.el.lora, [{ name: "", label: "(none, use pictures)" }].concat(options.loras.map(function (l) {
      return { name: l.name, label: l.name + (l.size_mb ? " (" + l.size_mb + " MB)" : "") };
    })), "name", function (l) { return l.label; });
    c.el.lora.value = prev || "";
    var prevRm = c.el.refmod.value;
    fillSelect(c.el.refmod, [{ name: "", label: "(none)" }].concat(options.refmods.map(function (m) {
      return { name: m.name, label: m.name };
    })), "name", function (m) { return m.label; });
    c.el.refmod.value = prevRm || "";
    if (options.presets.length && !c.el.voice.options.length) {
      fillSelect(c.el.voice, options.presets, "id", function (p) { return p.label + " -- " + p.sub; });
      var def = options.presets.filter(function (p) { return p.id === "aiden"; })[0];
      if (def) c.el.voice.value = def.id;
      fillSelect(c.el.style, options.styles, "id", function (s) { return s.label; });
    }
    var prevSaved = c.el.saved.value;            // refilled whenever the lists arrive, keeping the choice
    fillSelect(c.el.saved, options.voices.length ? options.voices : [{ id: "", label: "(no saved voices yet)" }],
      "id", function (v) { return v.label; });
    if (prevSaved) c.el.saved.value = prevSaved;
  }

  function updateVoiceKind(c) {
    var k = c.el.kind_character.checked ? "character" : c.el.kind_clip.checked ? "clip" : "preset";
    c.el.presetWrap.style.display = k === "preset" ? "" : "none";
    c.el.savedWrap.style.display = k === "character" ? "" : "none";
    c.el.clipWrap.style.display = k === "clip" ? "" : "none";
    c.el.cloneExtra.style.display = k === "preset" ? "none" : "";
    c.el.consent.checked = false;          // the confirmation is per request: it clears whenever the voice changes
  }

  function voiceKind(c) {
    return c.el.kind_character.checked ? "character" : c.el.kind_clip.checked ? "clip" : "preset";
  }

  function setClip(c, file) {
    c.clip = file || null;
    c.el.clipName.textContent = c.clip ? c.clip.name : "No clip yet. Drag an audio file here, or:";
    c.el.consent.checked = false;
  }

  function addLookRow(c, initialName, initialFile) {
    var row = mk("div", "ng-scene-look-row");
    var look = { name: null, file: null, row: row };
    look.name = mk("input", "ng-gen-input");
    look.name.type = "text";
    look.name.maxLength = 40;
    look.name.placeholder = "Setting name, e.g. battlements";
    var file = document.createElement("input");
    file.type = "file"; file.accept = "image/*";
    file.addEventListener("change", function () { look.file = file.files[0] || null; });
    if (initialName) look.name.value = initialName;
    if (initialFile) {                                 // a picture we made: put it in the file box as if it had been chosen
      look.file = initialFile;
      try { var dt = new DataTransfer(); dt.items.add(initialFile); file.files = dt.files; } catch (e) { /* shown by name below */ }
    }
    var rm = mk("button", "ng-gen-btn ng-gen-btn-quiet", "remove");
    rm.type = "button"; rm.style.width = "auto";
    rm.addEventListener("click", function () {
      c.looks = c.looks.filter(function (x) { return x !== look; });
      row.remove();
    });
    row.appendChild(look.name);
    row.appendChild(file);
    row.appendChild(rm);
    wireDrop(row, function (f) { look.file = f; });
    c.looks.push(look);
    c.el.looksBox.appendChild(row);
  }

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

  // ---- script helpers ------------------------------------------------------------------------------------------

  function scriptLines() {
    var lines = [];
    (els.script.value || "").split(/\r?\n/).forEach(function (raw0) {
      if (!raw0.trim()) return;
      var raw = raw0.replace(/^\s*&\s*/, "");                    // "& NAME: ..." shares the previous line's shot
      var m = raw.match(/^\s*([A-Za-z0-9][A-Za-z0-9 _-]{0,29}?)\s*(?:\[[^\]\n]{1,120}\])?\s*:\s*(.*\S)\s*$/);
      if (m) lines.push({ speaker: m[1].trim(), words: (m[2].match(/\S+/g) || []).length });
      else if (lines.length) lines[lines.length - 1].words += (raw.match(/\S+/g) || []).length;
    });
    return lines;
  }

  function updateCount() {
    if (!els.count) return;
    var lines = scriptLines();
    var long = lines.filter(function (l) { return l.words > 18; }).length;
    els.count.textContent = lines.length + " line" + (lines.length === 1 ? "" : "s") + " (about 6 minutes of rendering each with Fast)" +
      (long ? ", " + long + " too long (over 18 words)" : "");
    els.count.style.color = (lines.length > MAX_LINES || long) ? "var(--ng-danger)" : "";
  }

  // ---- queue ---------------------------------------------------------------------------------------------------

  function snippet(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "..." : s; }
  function isActive(item) { return item.status === "queued" || item.status === "rendering"; }

  function itemFromJob(job) {
    var item = { localId: "sc" + ++localSeq, jobId: job.job_id, status: "queued", title: "", stage: "queued", clips: [],
      error: null, videoUrl: null, downloadUrl: null, createdAt: Date.now(), jobIdAt: 0, removed: false };
    applyJob(item, job);
    return item;
  }

  function applyJob(item, job) {
    item.stage = job.stage || item.stage;
    item.clips = job.clips || item.clips;
    item.bedErrors = job.bed_errors || {};
    if (item.clips.length) {
      item.title = item.clips[0].speaker + ": " + item.clips[0].text +
        (item.clips.length > 1 ? "  (+" + (item.clips.length - 1) + " more)" : "");
    }
    if (job.status === "completed") {
      item.status = "done";
      if (!item.videoUrl && job.video_url) item.videoUrl = job.video_url + "?v=" + Date.now();
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
    if (item.stage === "voicing") return "making the voices...";
    if (item.stage === "ambience") return "making the ambience...";
    if (item.stage === "joining") return "joining the clips...";
    var n = item.clips.length;
    var cur = item.clips.filter(function (c) { return c.status === "rendering"; })[0];
    var done = item.clips.filter(function (c) { return c.status === "done"; }).length;
    if (cur) {
      var p = cur.progress || {};
      return "clip " + (cur.index + 1) + " of " + n + (p.step && p.steps ? ", step " + p.step + "/" + p.steps : " rendering...");
    }
    return done + " of " + n + " clips done";
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(function (q) { return q.jobId && isActive(q); });
    if (!active.length) { clearInterval(pollTimer); pollTimer = null; return; }
    active.forEach(function (item) {
      fetch(API + "/jobs/" + item.jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (!job) return;
          if (!job.status) {
            if (job.error) { item.status = "failed"; item.error = "The server no longer has this job (it restarted)."; renderQueue(); }
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
        var byId = {}; jobs.forEach(function (j) { byId[j.job_id] = j; });
        var known = {}; queue.forEach(function (q) { if (q.jobId) known[q.jobId] = true; });
        jobs.slice().reverse().forEach(function (j) {
          if (!known[j.job_id] && !dismissed[j.job_id]) queue.push(itemFromJob(j));
        });
        queue.forEach(function (q) {
          if (q.jobId && byId[q.jobId]) applyJob(q, byId[q.jobId]);
          else if (q.jobId && isActive(q) && q.jobIdAt && q.jobIdAt < startedAt - 1000) {
            q.status = "failed"; q.error = "The server no longer has this job (it restarted).";
          } else if (!q.jobId && q.status === "submitting" && Date.now() - q.createdAt > SUBMIT_GIVE_UP_MS) {
            q.status = "failed"; q.error = "The submit never completed.";
          }
        });
        renderQueue();
        if (queue.some(function (q) { return q.jobId && isActive(q); })) ensurePolling();
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
    if (item.jobId && isActive(item)) fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
  }

  function buildRow(item) {
    var row = mk("div", "ng-gen-queue-row st-" + item.status);
    row.appendChild(mk("div", "thumbs", "🎬"));
    var meta = mk("div", "meta");
    meta.appendChild(mk("div", "title", snippet(item.title, 90)));
    meta.appendChild(mk("div", "sub", item.clips.length + " clip" + (item.clips.length === 1 ? "" : "s")));
    row.appendChild(meta);

    var statusRow = mk("div", "status-row");
    statusRow.appendChild(mk("span", "st", stageLabel(item)));
    var rm = mk("button", "pose-remove");
    rm.type = "button";
    var running = item.status === "rendering";
    rm.title = running ? "stop this scene" : item.status === "queued" ? "cancel this queued scene" : "remove from the list";
    rm.textContent = running ? "stop" : item.status === "queued" ? "cancel" : "✕";
    if (isActive(item)) rm.style.cssText = "width:auto; padding:2px 8px; font-size:0.7857rem";
    rm.addEventListener("click", function () { removeItem(item.localId); });
    statusRow.appendChild(rm);
    row.appendChild(statusRow);

    if (item.status === "failed" && item.error) {
      var err = mk("div", "log", item.error);
      err.style.color = "var(--ng-danger)";
      row.appendChild(err);
    }
    var finishedClips = item.clips.filter(function (c) { return c.status === "done"; });
    if (item.status === "failed" && finishedClips.length) {
      row.appendChild(mk("div", "log", finishedClips.length + " clip(s) finished before it stopped; they are in the H3 job list."));
    }
    if (item.status === "done" && item.videoUrl) {
      var video = document.createElement("video");
      video.src = item.videoUrl; video.controls = true; video.playsInline = true; video.preload = "metadata";
      row.appendChild(video);
      if (item.downloadUrl) {
        var dl = mk("a", "ng-btn", "Download mp4");
        dl.href = item.downloadUrl; dl.style.gridColumn = "1 / -1"; dl.style.marginTop = "4px";
        row.appendChild(dl);
      }
    }
    if (item.status === "done" && item.bedErrors && Object.keys(item.bedErrors).length) {
      var bw = mk("div", "log", "No ambience for: " + Object.keys(item.bedErrors).join(", "));
      bw.style.color = "var(--ng-danger)";
      row.appendChild(bw);
    }
    var warn = item.clips.filter(function (c) { return c.finish_error; });
    if (item.status === "done" && warn.length) {
      var w = mk("div", "log", "The exposure step did not run on " + warn.length + " clip(s): " + warn[0].finish_error);
      w.style.color = "var(--ng-danger)";
      row.appendChild(w);
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

  // ---- submit --------------------------------------------------------------------------------------------------

  function buildRequest() {
    var script = (els.script.value || "").trim();
    if (!script) throw new Error("Write the script first.");
    if (!cards.length) throw new Error("Add at least one character.");
    var lines = scriptLines();
    if (lines.length > MAX_LINES) throw new Error("A scene is at most " + MAX_LINES + " lines for now.");
    var form = new FormData();
    var chars = [];
    var seen = {};
    cards.forEach(function (c, ci) {
      var name = c.el.name.value.trim();
      if (!name) throw new Error("Every character needs a name.");
      if (seen[name.toLowerCase()]) throw new Error(name + " is listed twice.");
      seen[name.toLowerCase()] = true;
      var ch = { name: name, subject: c.el.subject.value.trim(), looks: [] };
      if (c.el.refmod.value) {
        if (c.el.lora.value || c.looks.length) throw new Error(name + ": a RefMod is the whole character; clear the LoRA and the pictures.");
        ch.refmod = c.el.refmod.value;
      }
      if (c.el.lora.value) { ch.lora = c.el.lora.value; ch.lora_strength = parseFloat(c.el.strength.value || "1"); }
      c.looks.forEach(function (look, li) {
        var lname = look.name.value.trim();
        if (!lname || !look.file) throw new Error(name + ": each picture needs a setting name and a file.");
        var field = "look_" + ci + "_" + li;
        form.append(field, look.file, look.file.name);
        ch.looks.push({ name: lname, file: field });
      });
      if (ch.looks.length) ch.default_look = ch.looks[0].name;
      if (!ch.lora && !ch.refmod && !ch.looks.length) throw new Error(name + " needs a RefMod, a LoRA or at least one picture.");
      var k = voiceKind(c);
      var v = { kind: k, language: "english", speed: 1.0 };
      if (k === "preset") { v.voice = c.el.voice.value; v.style = c.el.style.value || "natural"; }
      else {
        if (!c.el.consent.checked) throw new Error(name + ": tick the box to confirm the voice is yours or that you have permission to clone it.");
        v.consent = true;
        if (c.el.refText.value.trim()) v.ref_text = c.el.refText.value.trim();
        if (k === "character") {
          if (!c.el.saved.value) throw new Error(name + ": there is no saved voice to use.");
          v.character_voice = c.el.saved.value;
        } else {
          if (!c.clip) throw new Error(name + ": add a voice clip.");
          var vf = "voice_" + ci;
          form.append(vf, c.clip, c.clip.name);
          v.clip_file = vf;
        }
      }
      ch.voice = v;
      chars.push(ch);
    });
    lines.forEach(function (l) {
      if (!seen[l.speaker.toLowerCase()]) throw new Error(l.speaker + " speaks in the script but is not a character above.");
      if (l.words > 18) throw new Error("A line of " + l.speaker + "'s is over 18 words; a clip holds about 12. Split it into two lines.");
    });
    form.append("script", script);
    form.append("characters", JSON.stringify(chars));
    form.append("quality", els.quality.value || "draft");
    form.append("fast", els.fast.value === "1" ? "true" : "false");           // sent explicitly: empty means "default on"
    form.append("match_exposure", els.match.checked ? "true" : "false");
    if (els.ambience.checked) form.append("ambience", "true");
    if (els.finish.checked) {
      form.append("finish", "true");
      if (els.fClarity.value !== "") form.append("clarity_strength", els.fClarity.value);
      if (els.fTone.value !== "") form.append("tone_strength", els.fTone.value);
      if (els.fSat.value !== "") form.append("saturation", els.fSat.value);
    }
    if (els.seed.value !== "") form.append("seed", els.seed.value);
    return { form: form, title: lines[0].speaker + ": " + (script.split(/\r?\n/)[0] || "") };
  }

  function makeScene() {
    var req;
    try { req = buildRequest(); } catch (e) { setStatus(e.message, true); return; }
    var item = { localId: "sc" + ++localSeq, jobId: null, status: "submitting", title: req.title, stage: "queued", clips: [],
      error: null, videoUrl: null, downloadUrl: null, createdAt: Date.now(), jobIdAt: 0, removed: false };
    queue.unshift(item);
    renderQueue();
    setStatus("Sending...");
    fetch(API + "/generate", { method: "POST", body: req.form })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, payload: p }; }); })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          item.jobId = r.payload.job_id; item.jobIdAt = Date.now();
          if (item.removed) { fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {}); return; }
          item.status = "queued";
          setStatus("Queued.");
          ensurePolling();
        } else {
          item.status = "failed";
          item.error = (r.payload && r.payload.error) || "could not start the scene";
          setStatus(item.error, true);
        }
        renderQueue();
      })
      .catch(function (e) { item.status = "failed"; item.error = e.message; setStatus(e.message, true); renderQueue(); });
  }

  // ---- wiring --------------------------------------------------------------------------------------------------

  function init() {
    if (inited) return;
    if (!els.main || !els.generate) return;
    inited = true;
    els.addChar.addEventListener("click", function () { addCharacter(""); });
    els.script.addEventListener("input", updateCount);
    els.selectAll.addEventListener("click", function () { els.script.focus(); els.script.select(); });
    els.generate.addEventListener("click", makeScene);
    els.finish.addEventListener("change", function () {
      els.finishOpts.style.display = finishOk && els.finish.checked ? "" : "none";
    });
    els.queueClear.addEventListener("click", function () {
      queue.forEach(function (q) { if (q.jobId && (q.status === "done" || q.status === "failed")) dismissed[q.jobId] = true; });
      queue = queue.filter(function (q) { return q.status !== "done" && q.status !== "failed"; });
      renderQueue();
    });
    if (!cards.length) { addCharacter(""); addCharacter(""); }
    updateCount();
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "scene");
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

  window.SceneNG = { sync: sync };
})();
