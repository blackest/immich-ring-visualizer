/**
 * kleinNG.js -- the "Klein" view (bottom-bar Klein task).
 *
 * FLUX 3 Klein pictures: a prompt alone makes a picture from text; adding references edits or
 * combines them. Three labelled slots like Phosphene's (subject, location, extra), several
 * variations per job, and a gallery of earlier jobs whose pictures can be sent on: into a reference
 * slot, or to the H3 page as its first frame. Backend: routes/kleinNG.py (/api/ng/klein/*).
 * Builds its own DOM into #ng-klein-compose; no project or character state is read or written.
 */
(function () {
  "use strict";

  var API = "/api/ng/klein/";
  var SLOTS = [
    { key: "subject", label: "Subject", hint: "who or what" },
    { key: "location", label: "Location", hint: "where" },
    { key: "extra", label: "Extra", hint: "a prop, a style, a pose" },
  ];
  var SIZES = [["1024x576", "16:9 wide"], ["1024x1024", "square"], ["1024x768", "4:3"], ["768x1024", "portrait 3:4"],
    ["576x1024", "tall 9:16"], ["768x432", "16:9 small (fast)"]];
  var VARIATIONS = [1, 2, 4, 6, 8, 10];

  var built = false;
  var els = {};
  var slots = SLOTS.map(function () { return { file: null, from: null, url: null }; });
  var pollTimer = null;

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function button(text, onclick, title) {
    var b = el("button", "ng-gen-btn", text);
    b.type = "button"; b.style.width = "auto";
    if (title) b.title = title;
    b.onclick = onclick;
    return b;
  }

  function imgUrl(job, file) { return API + "image/" + job + "/" + file; }

  // ---- reference slots ------------------------------------------------------------------------
  function setSlot(i, source) {
    var s = slots[i];
    if (s.url && s.file) URL.revokeObjectURL(s.url);
    s.file = source && source.file || null;
    s.from = source && source.from || null;
    s.url = source ? (source.file ? URL.createObjectURL(source.file) : source.url) : null;
    drawSlots();
    updateHint();
  }

  function drawSlots() {
    els.slots.textContent = "";
    SLOTS.forEach(function (def, i) {
      var s = slots[i];
      var box = el("div", "ng-klein-slot");
      box.appendChild(el("div", "ng-gen-label", def.label + " "));
      box.firstChild.appendChild(el("span", "ng-gen-status", "(" + def.hint + ")"));
      var drop = el("div", "ng-gen-ref-paste" + (s.url ? " ng-gen-ref-paste-filled" : ""));
      drop.tabIndex = 0;
      if (s.url) {
        var img = el("img"); img.src = s.url; img.alt = def.label;
        drop.appendChild(img);
      } else {
        drop.appendChild(el("span", "", "Drop, paste or click"));
      }
      var input = el("input"); input.type = "file"; input.accept = "image/*"; input.style.display = "none";
      input.onchange = function () { if (input.files[0]) setSlot(i, { file: input.files[0] }); };
      drop.onclick = function () { input.click(); };
      drop.ondragover = function (e) { e.preventDefault(); };
      drop.ondrop = function (e) {
        e.preventDefault();
        var f = e.dataTransfer && e.dataTransfer.files[0];
        if (f && /^image\//.test(f.type)) setSlot(i, { file: f });
      };
      drop.onpaste = function (e) {
        var items = (e.clipboardData && e.clipboardData.items) || [];
        for (var k = 0; k < items.length; k++) {
          if (items[k].kind === "file" && /^image\//.test(items[k].type)) { setSlot(i, { file: items[k].getAsFile() }); e.preventDefault(); return; }
        }
      };
      box.appendChild(drop); box.appendChild(input);
      if (s.url) box.appendChild(button("Clear", function () { setSlot(i, null); }));
      els.slots.appendChild(box);
    });
  }

  // The pictures named in order, skipping empty slots: the numbers the prompt wording uses.
  function filledSlots() {
    return SLOTS.map(function (d, i) { return slots[i].url ? d.key : null; }).filter(Boolean);
  }

  function wording() {
    var f = filledSlots(), n = function (k) { return "reference " + (f.indexOf(k) + 1); };
    if (!f.length) return "";
    var part = [];
    if (f.indexOf("subject") >= 0) part.push("the person from " + n("subject"));
    if (f.indexOf("location") >= 0) part.push((part.length ? "in" : "the place from") + " the location from " + n("location"));
    if (f.indexOf("extra") >= 0) part.push((part.length ? "with" : "using") + " the " + "style, prop or pose from " + n("extra"));
    return part.join(", ") + ", ";
  }

  function updateHint() {
    var w = wording();
    els.hint.textContent = w
      ? "Name each picture or the model blends them. Starting words: \"" + w + "...\""
      : "No references: Klein makes the picture from your words alone.";
    els.insert.style.display = w ? "" : "none";
  }

  // ---- generate -------------------------------------------------------------------------------
  function setStatus(msg) { els.status.textContent = msg || ""; }

  function generate() {
    var form = new FormData();
    form.append("prompt", els.prompt.value);
    var wh = els.size.value.split("x");
    form.append("width", wh[0]); form.append("height", wh[1]);
    form.append("n", els.count.value);
    if (els.seed.value) form.append("seed", els.seed.value);
    if (els.model.value) form.append("model", els.model.value);
    slots.forEach(function (s, i) {
      if (s.file) form.append("ref" + i, s.file, s.file.name || "ref.png");
      else if (s.from) form.append("ref_from" + i, s.from);
    });
    els.go.disabled = true;
    setStatus("Sending...");
    fetch(API + "generate", { method: "POST", body: form }).then(function (r) {
      return r.json().then(function (b) { return { ok: r.ok, body: b }; });
    }).then(function (r) {
      els.go.disabled = false;
      if (!r.ok) { setStatus(r.body.error || "Could not start."); return; }
      setStatus("Queued.");
      loadJobs();
    }).catch(function (e) { els.go.disabled = false; setStatus(e.message); });
  }

  // ---- gallery --------------------------------------------------------------------------------
  function sendToVideo(job, file) {
    fetch(imgUrl(job, file)).then(function (r) { return r.blob(); }).then(function (blob) {
      if (!window.H3NG || !window.H3NG.useAsFirstFrame) { setStatus("The H3 view is not loaded."); return; }
      window.H3NG.useAsFirstFrame(blob);
      var btn = document.querySelector('.ng-task-btn[data-task="h3"]');
      if (btn) btn.click();
    });
  }

  function pictureCell(job, r) {
    var cell = el("div", "ng-klein-pic");
    var a = el("a"); a.href = imgUrl(job.id, r.file); a.target = "_blank";
    var img = el("img"); img.src = imgUrl(job.id, r.file); img.loading = "lazy"; img.alt = job.prompt;
    a.appendChild(img); cell.appendChild(a);
    cell.appendChild(el("div", "ng-gen-status", "seed " + r.seed + " · " + r.width + "x" + r.height + " · " + r.seconds + "s"));
    var row = el("div", "ng-klein-pic-btns");
    SLOTS.forEach(function (def, i) {
      row.appendChild(button("→ " + def.label, function () {
        setSlot(i, { from: job.id + "/" + r.file, url: imgUrl(job.id, r.file) });
        els.main.scrollTo({ top: 0, behavior: "smooth" });
      }, "Use this picture as the " + def.label.toLowerCase() + " reference"));
    });
    row.appendChild(button("→ Video first frame", function () { sendToVideo(job.id, r.file); },
      "Open the H3 view with this picture as its first frame"));
    cell.appendChild(row);
    return cell;
  }

  function jobCard(job) {
    var card = el("div", "ng-klein-job");
    var head = el("div", "ng-gen-label", job.prompt.length > 160 ? job.prompt.slice(0, 160) + "..." : job.prompt);
    card.appendChild(head);
    var info = job.status + (job.refs.length ? " · refs: " + job.refs.join(", ") : " · text only") + " · " + job.model;
    if (job.status === "running" || job.status === "queued" || job.status === "waiting") {
      info += " · " + job.results.length + " of " + job.n;
      if (job.status === "waiting") info += " (waiting for the video render to finish)";
    }
    if (job.error) info += " · " + job.error;
    card.appendChild(el("div", "ng-gen-status", info));
    var grid = el("div", "ng-klein-grid");
    job.results.forEach(function (r) { grid.appendChild(pictureCell(job, r)); });
    card.appendChild(grid);
    var bar = el("div", "ng-klein-pic-btns");
    bar.appendChild(button("Use this prompt", function () { els.prompt.value = job.prompt; els.main.scrollTo({ top: 0, behavior: "smooth" }); }));
    if (job.status === "running" || job.status === "queued" || job.status === "waiting") {
      bar.appendChild(button("Stop", function () { fetch(API + "cancel/" + job.id, { method: "POST" }).then(loadJobs); }));
    }
    card.appendChild(bar);
    return card;
  }

  function loadJobs() {
    fetch(API + "jobs").then(function (r) { return r.json(); }).then(function (b) {
      els.gallery.textContent = "";
      var running = false;
      (b.jobs || []).forEach(function (job) {
        els.gallery.appendChild(jobCard(job));
        if (["running", "queued", "waiting"].indexOf(job.status) >= 0) { running = true; els.log.textContent = (job.log || []).slice(-3).join("\n"); }
      });
      els.log.style.display = running ? "" : "none";
      clearTimeout(pollTimer);
      if (running && els.main && els.main.style.display !== "none") pollTimer = setTimeout(loadJobs, 3000);
    }).catch(function () {});
  }

  // ---- build ----------------------------------------------------------------------------------
  function build() {
    var root = document.getElementById("ng-klein-compose");
    root.appendChild(el("div", "ng-gen-label", "References (optional)"));
    els.slots = el("div", "ng-klein-slots");
    root.appendChild(els.slots);
    els.hint = el("div", "ng-gen-status");
    root.appendChild(els.hint);
    root.appendChild(el("label", "ng-gen-label", "Prompt"));
    els.prompt = el("textarea", "ng-gen-pose-modal-textarea");
    els.prompt.rows = 5;
    els.prompt.placeholder = "Describe the whole picture you want, including the change. e.g. the woman from reference 1 standing in profile at the window of the kitchen from reference 2, dim amber lamp light, evening";
    root.appendChild(els.prompt);
    els.insert = button("Put the reference wording at the start", function () {
      var w = wording();
      if (w && els.prompt.value.indexOf("from reference") < 0) els.prompt.value = w + els.prompt.value;
      els.prompt.focus();
    });
    root.appendChild(els.insert);

    var opts = el("div", "ng-klein-opts");
    var field = function (label, node) { var w = el("label", "ng-klein-field"); w.appendChild(el("span", "ng-gen-status", label)); w.appendChild(node); opts.appendChild(w); };
    els.size = el("select", "ng-gen-input");
    SIZES.forEach(function (s) { var o = el("option", "", s[0].replace("x", " × ") + " · " + s[1]); o.value = s[0]; els.size.appendChild(o); });
    field("Size", els.size);
    els.count = el("select", "ng-gen-input");
    VARIATIONS.forEach(function (n) { var o = el("option", "", n + (n === 1 ? " picture" : " variations")); o.value = n; els.count.appendChild(o); });
    els.count.value = "4";
    field("How many", els.count);
    els.seed = el("input", "ng-gen-input"); els.seed.type = "number"; els.seed.placeholder = "random";
    field("Seed", els.seed);
    els.model = el("select", "ng-gen-input");
    field("Model", els.model);
    root.appendChild(opts);

    var bar = el("div", "ng-gen-action-bar");
    els.go = button("Generate", generate);
    els.status = el("span", "ng-gen-status");
    bar.appendChild(els.go); bar.appendChild(els.status);
    root.appendChild(bar);
    els.log = el("div", "ng-vg-mainlog"); els.log.style.display = "none";
    root.appendChild(els.log);
    root.appendChild(el("div", "ng-gen-label", "Pictures"));
    els.gallery = el("div", "ng-klein-gallery");
    root.appendChild(els.gallery);
    drawSlots();
    updateHint();
    built = true;
  }

  function loadHealth() {
    fetch(API + "health").then(function (r) { return r.json(); }).then(function (h) {
      els.model.textContent = "";
      (h.models || []).forEach(function (m) { var o = el("option", "", m); o.value = m; els.model.appendChild(o); });
      if (h.default) els.model.value = h.default;
      els.go.disabled = !h.ready;
      if (!h.ready) setStatus(h.worker_found ? "The Klein weights are not downloaded (do that in Phosphene)." : "The FLUX 3 worker is not installed (do that in Phosphene).");
    }).catch(function () { setStatus("Could not reach the server."); });
  }

  function sync(active) {
    var pane = document.getElementById("ng-klein-pane");
    els.main = document.getElementById("ng-klein-main");
    var controls = document.getElementById("ng-controls-pane");
    if (!pane || !els.main) return;
    var on = !!(active && active.task === "klein");
    pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && controls) controls.style.display = "none";
    if (!on) { clearTimeout(pollTimer); return; }
    if (!built) { build(); loadHealth(); }
    loadJobs();
  }

  window.KleinNG = { sync: sync };
})();
