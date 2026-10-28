// Enhanced H3 view (task === "enhancedh3"): a sentence plus up to 9 pictures,
// 3 clips and 3 sounds go to OpenH3-IR (routes/enhancedh3NG.py), which writes
// the brief; it comes back as one editable prompt per 5 s chained window.
// Render posts the (edited) prompts to the H3 view's POST /api/ng/h3/generate
// and polls its job; see ENHANCED_H3_PLAN.md for what comes after.
(function () {
  "use strict";

  var API = "/api/ng/enhanced-h3";
  var els = {};
  var inited = false;
  var options = null;
  var items = { image: [], video: [], audio: [] };
  var lastBrief = null;
  var busy = false;
  var lastDone = null;    // job id of the latest finished render: what Extend continues
  var extendFrom = null;  // job id being continued; null = a normal brief and render

  var LABEL = { image: "Picture", video: "Clip", audio: "Sound" };
  var ACCEPT = { image: "image/", video: "video/", audio: "audio/" };

  function $(id) { return document.getElementById(id); }

  function refreshEls() {
    ["pane", "main", "status", "intent", "tray", "file", "refnote", "seconds", "shots", "aspect",
      "creativity", "director", "effort", "seed", "line", "speaker", "language", "delivery",
      "silent", "write", "msg", "result", "summary", "diag", "windows", "change", "refine",
      "quality", "fast", "render", "renderstatus", "progress", "video", "extend", "extendnote"
    ].forEach(function (k) { els[k] = $("ng-enhancedh3-" + k); });
  }

  function setMsg(text) { if (els.msg) els.msg.textContent = text || ""; }

  function limit(kind) {
    var m = (options && options.limits && options.limits.max_assets) || {};
    return { image: m.images || 9, video: m.videos || 3, audio: m.audios || 3 }[kind];
  }

  function fillSelect(sel, values, pick) {
    var keep = sel.value;
    values.forEach(function (v) {
      var o = document.createElement("option");
      o.value = typeof v === "string" ? v : v.id;
      o.textContent = typeof v === "string" ? v : v.name;
      sel.appendChild(o);
    });
    sel.value = keep || pick || "";
  }

  function loadOptions() {
    fetch(API + "/status").then(function (r) { return r.json(); }).then(function (s) {
      els.status.textContent = !s.ok ? s.error || "Compiler not running."
        : s.llm ? "Compiler ready (" + s.profile + ")." : "Compiler is up but can't reach its language model.";
    });
    fetch(API + "/options").then(function (r) { return r.json(); }).then(function (o) {
      if (o.error) return;
      options = o;
      if (!els.aspect.options.length) {
        fillSelect(els.aspect, o.limits.aspects, "16:9");
        fillSelect(els.creativity, o.limits.creativity, "balanced");
        fillSelect(els.effort, o.limits.effort, "standard");
        fillSelect(els.director, o.directors);
        fillSelect(els.language, o.limits.dialogue_languages, "English");
        restoreFields();
      }
      renderTray();
    }).catch(function () {});
  }

  // ---- keep the work across a page refresh ----
  // Text, settings, the brief (with edited window prompts) and a running
  // render's job id. Files can't be kept here: re-add the pictures after a refresh.

  var STORE = "enhancedh3NG.state";
  var FIELDS = ["intent", "seconds", "shots", "aspect", "creativity", "director", "effort", "seed",
    "line", "speaker", "language", "delivery", "quality"];
  var saved = null;
  var renderJob = null;
  var briefNeedsFirstFrame = false;
  var saveTimer = null;

  function loadSaved() {
    try { saved = JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { saved = null; }
  }

  function saveNow() {
    var state = { fields: {}, silent: els.silent.checked, fast: els.fast.checked,
      brief: lastBrief, needsFirstFrame: briefNeedsFirstFrame, job: renderJob,
      lastDone: lastDone, extendFrom: extendFrom };
    FIELDS.forEach(function (k) { state.fields[k] = els[k].value; });
    if (lastBrief && lastBrief.windows) {
      state.prompts = [].map.call(els.windows.querySelectorAll("textarea"), function (ta) { return ta.value; });
    }
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { /* private window or full */ }
  }

  function saveSoon() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveNow, 400);
  }

  function restoreFields() {
    if (!saved) return;
    FIELDS.forEach(function (k) {
      var v = saved.fields && saved.fields[k];
      if (v !== undefined && v !== null) els[k].value = v;
    });
    els.silent.checked = !!saved.silent;
    if (saved.fast !== undefined) els.fast.checked = !!saved.fast;
  }

  function restoreBrief() {
    if (!saved || !saved.brief) return;
    briefNeedsFirstFrame = !!saved.needsFirstFrame;
    showBrief(saved.brief);
    var boxes = els.windows.querySelectorAll("textarea");
    (saved.prompts || []).forEach(function (text, i) { if (boxes[i]) boxes[i].value = text; });
    setMsg("Restored your last brief. Re-add any pictures before rendering.");
  }

  // ---- media tray ----

  function addFiles(kind, files) {
    Array.prototype.forEach.call(files, function (f) {
      if (f.type.indexOf(ACCEPT[kind]) !== 0) return;
      if (items[kind].length >= limit(kind)) return;
      var defaultRole = kind === "image" ? (items.image.length ? "subject" : "frame_anchor_first")
        : kind === "video" ? "subject" : "bgm";
      items[kind].push({ file: f, url: URL.createObjectURL(f), role: defaultRole, note: "" });
    });
    renderTray();
  }

  function kindOf(file) {
    return ["image", "video", "audio"].filter(function (k) { return file.type.indexOf(ACCEPT[k]) === 0; })[0];
  }

  function renderTray() {
    ["image", "video", "audio"].forEach(function (kind) {
      var group = els.tray.querySelector('.ng-eh3-group[data-kind="' + kind + '"]');
      group.querySelector(".ng-eh3-count").textContent = "(" + items[kind].length + "/" + limit(kind) + ")";
      var box = group.querySelector(".ng-eh3-items");
      box.textContent = "";
      var roles = (options && options.roles && options.roles[kind]) || [];
      items[kind].forEach(function (it, i) {
        var row = document.createElement("div");
        row.className = "ng-eh3-item";
        var thumb = document.createElement(kind === "image" ? "img" : kind);
        thumb.src = it.url;
        if (kind !== "image") thumb.controls = kind === "audio";
        row.appendChild(thumb);
        var fields = document.createElement("div");
        fields.className = "ng-eh3-fields";
        var sel = document.createElement("select");
        sel.className = "ng-gen-input";
        roles.forEach(function (r) {
          var o = document.createElement("option");
          o.value = r; o.textContent = LABEL[kind] + " " + (i + 1) + ": " + r.replace(/_/g, " ");
          sel.appendChild(o);
        });
        sel.value = it.role;
        sel.onchange = function () { it.role = sel.value; updateRefNote(); };
        var at = null;
        if (kind === "image") {
          at = document.createElement("input");
          at.type = "text"; at.className = "ng-gen-input"; at.placeholder = "keyframe at, e.g. 2.5s (optional)";
          at.value = it.at || "";
          at.oninput = function () { it.at = at.value.trim(); updateRefNote(); };
        }
        var note = document.createElement("input");
        note.type = "text"; note.className = "ng-gen-input"; note.placeholder = "note (optional)";
        note.value = it.note;
        note.oninput = function () { it.note = note.value; };
        fields.appendChild(sel); fields.appendChild(note);
        if (at) fields.appendChild(at);
        row.appendChild(fields);
        var rm = document.createElement("button");
        rm.type = "button"; rm.className = "ng-gen-btn"; rm.style.width = "auto"; rm.textContent = "×";
        rm.onclick = function () { URL.revokeObjectURL(it.url); items[kind].splice(i, 1); renderTray(); };
        row.appendChild(rm);
        box.appendChild(row);
      });
    });
    updateRefNote();
  }

  // The renderer takes first/last frames only today; the rest shape the brief.
  function updateRefNote() {
    var extra = items.image.filter(function (it) { return it.role !== "frame_anchor_first" && it.role !== "frame_anchor_last" && !it.at; }).length +
      items.video.length + items.audio.length;
    els.refnote.textContent = extra
      ? "Only the first-, last-frame and timed keyframe pictures reach the video model for now; the rest shape the brief's wording."
      : "";
  }

  function wireTray() {
    ["image", "video", "audio"].forEach(function (kind) {
      var group = els.tray.querySelector('.ng-eh3-group[data-kind="' + kind + '"]');
      var drop = group.querySelector(".ng-eh3-drop");
      group.querySelector(".ng-eh3-add").onclick = function () {
        els.file.accept = kind + "/*";
        els.file.onchange = function () { addFiles(kind, els.file.files); els.file.value = ""; };
        els.file.click();
      };
      drop.ondragover = function (e) { e.preventDefault(); drop.classList.add("over"); };
      drop.ondragleave = function () { drop.classList.remove("over"); };
      drop.ondrop = function (e) {
        e.preventDefault(); drop.classList.remove("over");
        addFiles(kind, e.dataTransfer.files);
      };
    });
    // Paste anywhere in the view (not inside a text box that wants text).
    document.addEventListener("paste", function (e) {
      if (!els.main || els.main.style.display === "none") return;
      var files = e.clipboardData ? e.clipboardData.files : [];
      if (!files.length) return;
      Array.prototype.forEach.call(files, function (f) {
        var kind = kindOf(f);
        if (kind) addFiles(kind, [f]);
      });
      e.preventDefault();
    });
  }

  // ---- brief ----

  function dialogue() {
    var text = els.line.value.trim();
    if (!text || els.delivery.value === "sung") return null;
    return [{ text: text, language: els.language.value || "English", speaker: els.speaker.value.trim() || undefined }];
  }

  function writeBrief() {
    if (busy) return;
    var intent = els.intent.value.trim();
    if (!intent) { setMsg("Describe what happens first."); els.intent.focus(); return; }
    var line = els.line.value.trim();
    if (line && els.delivery.value === "sung") {
      intent += ' The ' + (els.speaker.value.trim() || "character") + ' sings: "' + line + '".';
    }

    var form = new FormData();
    form.append("intent", intent);
    form.append("seconds", els.seconds.value || "10");
    ["shots", "aspect", "creativity", "director", "effort", "seed"].forEach(function (k) {
      if (els[k].value) form.append(k, els[k].value);
    });
    if (els.silent.checked) form.append("silent", "true");
    var d = dialogue();
    if (d) {
      form.append("dialogue", JSON.stringify(d));
    }
    briefNeedsFirstFrame = !extendFrom && items.image.some(function (it) { return it.role === "frame_anchor_first"; });
    var assets = [], n = 0;
    // An extension is written as plain text: the earlier clip goes to the renderer
    // only, and a tray picture would add a <Picture 1> header the extend path lacks.
    ["image", "video", "audio"].forEach(function (kind) {
      if (extendFrom) return;
      items[kind].forEach(function (it) {
        var field = "f" + n++;
        form.append(field, it.file, it.file.name);
        assets.push({ field: field, kind: kind, role: it.role, note: it.note || undefined });
      });
    });
    if (assets.length) form.append("assets", JSON.stringify(assets));

    run("Writing the brief (this can take a minute)...", fetch(API + "/brief", { method: "POST", body: form }));
  }

  function refine() {
    var change = els.change.value.trim();
    if (!lastBrief || !change || busy) return;
    run("Rewriting...", fetch(API + "/refine", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: lastBrief.id, change: change, seconds: parseFloat(els.seconds.value) || 10 }),
    }));
  }

  function run(msg, request) {
    busy = true; els.write.disabled = true; els.refine.disabled = true;
    setMsg(msg);
    request.then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (r) {
        if (!r.ok) { setMsg(r.body.error || "Failed."); return; }
        showBrief(r.body);
        setMsg(r.body.windows ? "Brief ready: " + r.body.windows.length + " window(s). Edit below, then Render." + (r.body.note ? " " + r.body.note : "") : "");
        els.result.scrollIntoView({ behavior: "smooth", block: "start" });
        saveNow();
      })
      .catch(function (e) { setMsg(e.message); })
      .then(function () { busy = false; els.write.disabled = false; els.refine.disabled = false; });
  }

  function showBrief(b) {
    lastBrief = b;
    if (b.status && b.status !== "ready" && b.question) {
      els.result.style.display = "";
      els.summary.textContent = "The compiler asks: " + b.question +
        (b.default_if_unanswered ? " (default: " + b.default_if_unanswered + ")" : "");
      els.diag.textContent = ""; els.windows.textContent = "";
      return;
    }
    els.result.style.display = "";
    var t = b.target || {};
    els.summary.textContent = (b.mode || "") + " · " + (t.effective_seconds || "?") + " s · " +
      (b.windows || []).length + " window(s) · " + (b.prompt_tokens || "?") + " tokens";
    els.diag.textContent = "";
    (b.diagnostics || []).forEach(function (d) {
      var p = document.createElement("div");
      p.className = "ng-eh3-warn";
      p.textContent = typeof d === "string" ? d : (d.severity ? d.severity + ": " : "") + (d.message || JSON.stringify(d));
      els.diag.appendChild(p);
    });
    els.windows.textContent = "";
    (b.windows || []).forEach(function (w, i) {
      var box = document.createElement("div");
      box.className = "ng-eh3-window";
      var label = document.createElement("label");
      label.className = "ng-gen-label";
      label.textContent = "Window " + (i + 1) + (w.continues ? " (carries on the previous shot)" : " (shot " + w.shots.join(", ") + ")");
      var ta = document.createElement("textarea");
      ta.className = "ng-gen-input";
      ta.value = w.prompt;
      ta.dataset.window = String(i);
      box.appendChild(label); box.appendChild(ta);
      els.windows.appendChild(box);
    });
  }

  // ---- render ----

  var pollTimer = null;

  function setRender(text) { els.renderstatus.textContent = text || ""; }

  function updateExtend() {
    els.extend.style.display = lastDone || extendFrom ? "" : "none";
    els.extend.textContent = extendFrom ? "Stop extending" : "Extend this video";
    els.extendnote.textContent = extendFrom
      ? "Extending the last render. Describe the same people and place and what happens next; Seconds is the time to add. Tray items are ignored."
      : "";
  }

  function toggleExtend() {
    extendFrom = extendFrom ? null : lastDone;
    updateExtend();
    saveNow();
  }

  function render() {
    if (!lastBrief || !lastBrief.windows) return;
    var prompts = [].map.call(els.windows.querySelectorAll("textarea"), function (ta) { return ta.value.trim(); });
    if (prompts.some(function (p) { return !p; })) { setRender("A window prompt is empty."); return; }
    var form = new FormData();
    var seconds = (lastBrief.target && lastBrief.target.effective_seconds) || parseFloat(els.seconds.value) || 5;
    form.append("duration_s", String(seconds));
    if (prompts.length > 1) form.append("chain_prompts", JSON.stringify(prompts));
    else form.append("prompt", prompts[0]);
    form.append("quality", els.quality.value);
    if (els.fast.checked) form.append("tristep", "true");
    if (els.seed.value) form.append("seed", els.seed.value);
    var first = extendFrom ? null : items.image.filter(function (it) { return it.role === "frame_anchor_first"; })[0];
    if (!first && briefNeedsFirstFrame) {
      setRender("This brief used a first-frame picture. Add it again (role: frame anchor first), or it would render as text-only.");
      return;
    }
    if (first) form.append("file", first.file, first.file.name);
    var last = items.image.filter(function (it) { return it.role === "frame_anchor_last"; })[0];
    if (last) form.append("last_file", last.file, last.file.name);
    if (!extendFrom) items.image.filter(function (it) { return it.at; }).forEach(function (it) {
      form.append("keyframe_files", it.file, it.file.name);
      form.append("keyframe_times", it.at);
    });

    els.render.disabled = true;
    els.video.style.display = "none";
    setRender("Submitting...");
    var source = extendFrom
      ? fetch("/api/ng/h3/jobs/" + extendFrom + "/video").then(function (r) {
        if (!r.ok) throw new Error("The clip to extend is no longer on the server.");
        return r.blob();
      }).then(function (blob) { form.append("extend_file", blob, "extend.mp4"); })
      : Promise.resolve();
    source.then(function () { return fetch("/api/ng/h3/generate", { method: "POST", body: form }); })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (r) {
        if (!r.ok || !r.body.job_id) { setRender(r.body.error || "Could not start the render."); els.render.disabled = false; return; }
        renderJob = r.body.job_id;
        saveNow();
        pollJob(r.body.job_id);
      })
      .catch(function (e) { setRender(e.message); els.render.disabled = false; });
  }

  // Where a running job is, from the runner's log lines: the current window
  // (it prints "### window k/n ###" from window 2 on), stage and denoise step.
  var STAGES = [["text_encode", 0.00, "encoding the prompt"], ["keyframe_encode", 0.12, "encoding the start frame"],
    ["dit_load", 0.14, "loading the model"], ["joint_denoise", 0.20, "denoising"],
    ["video_vae_decode", 0.90, "decoding the video"], ["audio_vae_decode", 0.98, "decoding the sound"]];

  function progressOf(job) {
    var log = job.log_tail || [];
    var windows = (lastBrief && lastBrief.windows && lastBrief.windows.length) || 1;
    var start = 0, win = 1;
    log.forEach(function (line, i) {
      var m = /### window (\d+)\/(\d+) ###/.exec(line);
      if (m) { start = i; win = parseInt(m[1], 10); windows = parseInt(m[2], 10); }
    });
    var stage = null, step = null;
    log.slice(start).forEach(function (line) {
      var m = /== w\d+_(\w+?) ==/.exec(line);
      if (m) { stage = m[1].replace(/_q8$/, ""); step = null; }
      var s2 = /step (\d+)\/(\d+):/.exec(line);
      if (s2) step = [parseInt(s2[1], 10), parseInt(s2[2], 10)];
    });
    var inWindow = 0, what = "starting";
    STAGES.forEach(function (st) {
      if (stage && stage.indexOf(st[0]) === 0) { inWindow = st[1]; what = st[2]; }
    });
    if (stage && stage.indexOf("joint_denoise") === 0 && step) inWindow = 0.20 + 0.70 * (step[0] / step[1]);
    var frac = Math.min(0.99, (win - 1 + inWindow) / windows);
    var elapsed = job.dispatched_at ? Date.now() / 1000 - job.dispatched_at : 0;
    var text = "Window " + win + " of " + windows + " · " + what +
      (step ? ", step " + step[0] + " of " + step[1] : "");
    if (elapsed) {
      text += " · " + Math.floor(elapsed / 60) + ":" + ("0" + Math.floor(elapsed % 60)).slice(-2) + " elapsed";
      if (frac > 0.15) text += " · about " + Math.max(1, Math.round(elapsed * (1 - frac) / frac / 60)) + " min left";
    }
    return { frac: frac, text: text };
  }

  function pollJob(jobId) {
    clearInterval(pollTimer);
    var tick = function () {
      fetch("/api/ng/h3/jobs/" + jobId).then(function (r) { return r.json(); }).then(function (job) {
        if (!job.status) {
          clearInterval(pollTimer);
          renderJob = null; saveNow();
          setRender("That render is no longer on the server (it restarted?). Check the H3 view's job log.");
          els.render.disabled = false;
        } else if (job.status === "completed") {
          clearInterval(pollTimer);
          renderJob = null; saveNow();
          els.progress.style.display = "none";
          setRender("Done.");
          lastDone = jobId; extendFrom = null; updateExtend(); saveNow();
          els.video.src = job.video_url + "?v=" + Date.now();
          els.video.style.display = "";
          els.render.disabled = false;
        } else if (job.status === "failed") {
          clearInterval(pollTimer);
          renderJob = null; saveNow();
          els.progress.style.display = "none";
          setRender("Failed: " + (job.error || "unknown error"));
          els.render.disabled = false;
        } else {
          if (job.status === "rendering") {
            var pr = progressOf(job);
            setRender(pr.text);
            els.progress.style.display = "";
            els.progress.value = Math.round(pr.frac * 100);
          } else {
            setRender(job.status || "queued");
          }
        }
      }).catch(function () {});
    };
    pollTimer = setInterval(tick, 3000);
    tick();
  }

  function init() {
    if (inited || !els.main) return;
    inited = true;
    els.write.onclick = writeBrief;
    els.refine.onclick = refine;
    els.render.onclick = render;
    els.extend.onclick = toggleExtend;
    loadSaved();
    restoreFields();
    restoreBrief();
    els.main.addEventListener("input", saveSoon);
    els.main.addEventListener("change", saveSoon);
    if (saved && saved.job) {
      renderJob = saved.job;
      els.render.disabled = true;
      pollJob(saved.job);
    }
    if (saved) { lastDone = saved.lastDone || null; extendFrom = saved.extendFrom || null; }
    updateExtend();
    wireTray();
    loadOptions();
    renderTray();
  }

  function sync(active) {
    refreshEls();
    var on = !!(active && active.task === "enhancedh3");
    if (els.pane) els.pane.style.display = on ? "" : "none";
    if (els.main) els.main.style.display = on ? "" : "none";
    if (!on) return;
    var controls = document.getElementById("ng-controls-pane");
    if (controls) controls.style.display = "none";
    init();
  }

  window.EnhancedH3NG = { sync: sync };
})();
