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
  for (var n = 4; n <= 10; n++) SLOTS.push({ key: "ref" + n, label: "Reference " + n, hint: "anything else to include", more: true });
  var shown = 3; // how many slots are on screen: the named three, plus any added or filled
  var SIZES = [["1024x576", "16:9 wide"], ["1024x1024", "square"], ["1024x768", "4:3"], ["768x1024", "portrait 3:4"],
    ["576x1024", "tall 9:16"], ["768x432", "16:9 small (fast)"],
    ["1536x864", "16:9 larger (about 3.5 min)"], ["2048x1152", "16:9 large (about 6 min)"],
    ["3840x2160", "16:9 4K (about 30 min)"]];
  var LARGE_NOTE = "Pictures over 2048 × 1152 are decoded in tiles, which can leave faint colour patches. 4K takes about 30 minutes.";
  var VARIATIONS = [1, 2, 4, 6, 8, 10];
  // MiniMax-H3-Image is one more "model" value beside Klein's own (klein_jobsNG.H3_IMAGE_MODEL): a text-to-picture or an
  // edit of up to 6 references, Picture 1 being the one that is edited. It has its own Shape and Quality instead of Size.
  var H3 = "h3image", H3_LABEL = "MiniMax-H3-Image";
  var H3_SHAPES = [["match", "Same as the first reference"], ["16:9", "16:9 wide"], ["3:2", "3:2"], ["4:3", "4:3"],
    ["1:1", "square"], ["3:4", "3:4 portrait"], ["2:3", "2:3 portrait"], ["9:16", "9:16 tall"]];
  var H3_TIERS = [["draft", "Draft · 4 MP (about 8 minutes with a picture)"], ["standard", "Standard · 8 MP (about twice as long)"],
    ["high", "High · 16 MP (much longer)"]];
  var kleinOk = true, kleinMsg = "", h3ok = false, h3problems = [], lastH3 = false;
  function sizeNote() {
    var wh = els.size.value.split("x").map(Number);
    els.sizeNote.hidden = !(wh[0] * wh[1] > 2048 * 1152);
  }

  var built = false;
  var els = {};
  var slots = SLOTS.map(function () { return { file: null, from: null, url: null }; });
  var pollTimer = null;
  var loraList = [];
  var histLimit = 30, histQuery = "", searchTimer = null;
  var out = { file: null, from: null, url: null, img: null };

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
    if (els.steps) updateAdvanced();
  }

  function drawSlots() {
    els.slots.textContent = "";
    slots.forEach(function (x, k) { if (x.url && k + 1 > shown) shown = k + 1; });
    SLOTS.forEach(function (def, i) {
      if (i >= shown) return;
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
    if (shown < SLOTS.length) {
      els.slots.appendChild(button("+ Add another reference (" + shown + " of " + SLOTS.length + " shown)", function () { shown++; drawSlots(); },
        "Klein takes up to ten reference pictures"));
    }
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
    SLOTS.forEach(function (d) {
      if (d.more && f.indexOf(d.key) >= 0) part.push((part.length ? "with" : "using") + " the picture from " + n(d.key));
    });
    return part.join(", ") + ", ";
  }

  function updateHint() {
    var w = wording();
    if (isH3()) {
      els.hint.textContent = filledSlots().length
        ? "Picture 1 (the first reference you filled) is the one that is edited. Name the others in the words, like \"the hat from Picture 2 on the person in Picture 1\"."
        : "No references: MiniMax-H3-Image makes the picture from your words alone. Add a reference to edit it.";
      els.insert.style.display = "none";
      return;
    }
    els.hint.textContent = w
      ? "Name each picture or the model blends them. Starting words: \"" + w + "...\""
      : "No references: Klein makes the picture from your words alone.";
    els.insert.style.display = w ? "" : "none";
  }

  function isBase() { return /^base-/.test(els.model.value); }
  function isH3() { return !!els.model && els.model.value === H3; }

  // Fields that do nothing in the current state are greyed out, not hidden, so they stay discoverable.
  function updateAdvanced() {
    var h3 = isH3(), base = isBase(), refs = filledSlots().length > 0;
    [els.sizeField, els.adv, els.layoutHost].forEach(function (x) { if (x) x.style.display = h3 ? "none" : ""; });
    [els.shapeField, els.tierField].forEach(function (x) { if (x) x.style.display = h3 ? "" : "none"; });
    if (h3) {
      var matchOpt = [].filter.call(els.shape.options, function (o) { return o.value === "match"; })[0];
      matchOpt.disabled = !refs;
      if (!els.shape.dataset.touched) els.shape.value = refs ? "match" : "16:9";   // the default follows the references
      else if (!refs && els.shape.value === "match") els.shape.value = "16:9";
    }
    if (h3 !== lastH3) {                      // switching engine: a picture takes minutes, so one is the default for H3
      lastH3 = h3;
      if (!els.count.dataset.touched) els.count.value = h3 ? "1" : "4";
    }
    var ready = h3 ? h3ok : kleinOk;
    els.go.disabled = !ready;
    if (!ready) setStatus(h3 ? "MiniMax-H3-Image is not ready: " + (h3problems.join("; ") || "check Phosphene") : kleinMsg);
    updateHint();
    els.steps.placeholder = base ? "50 (default)" : "16 (default)";
    els.guidance.disabled = !base;
    els.guidance.placeholder = base ? "3.5 (default)" : "base model only";
    els.matchRef.disabled = !refs;
    if (!refs) els.matchRef.checked = false;
    els.size.disabled = els.matchRef.checked;
    els.tile.disabled = els.memory.value !== "fast";
    loraList.forEach(function (l) {
      if (!l.row) return;
      var off = l.distilled_only && base;
      l.row.box.disabled = off;
      l.row.box.parentNode.title = off ? l.title + " only works with the distilled model" : (l.notes || l.title);
    });
  }

  // ---- LoRAs (FLUX 3 ones from the shared folder; the outpaint one has its own button) ---------------
  function chosenLoras() {
    return loraList.filter(function (l) { return l.row && l.row.box.checked && !l.row.box.disabled; })
      .map(function (l) { return { name: l.name, scale: parseFloat(l.row.scale.value) || l.recommended_strength }; });
  }

  function drawLoras() {
    els.loras.textContent = "";
    var usable = loraList.filter(function (l) { return l.kind !== "outpaint"; });
    if (!usable.length) {
      els.loras.appendChild(el("div", "ng-gen-status", "No style LoRAs yet. Put FLUX 3 LoRAs in the shared loras folder, in a folder called flux3. The Outpaint button has its own."));
      return;
    }
    usable.forEach(function (l) {
      var row = el("label", "ng-klein-lora");
      l.row = { box: el("input"), scale: el("input", "ng-gen-input") };
      l.row.box.type = "checkbox";
      l.row.scale.type = "number"; l.row.scale.step = 0.1; l.row.scale.min = 0; l.row.scale.max = 3;
      l.row.scale.value = l.recommended_strength; l.row.scale.style.width = "5em";
      row.title = l.notes || l.title;
      row.appendChild(l.row.box);
      row.appendChild(el("span", "", " " + l.title + " (" + l.size_mb + " MB) strength "));
      row.appendChild(l.row.scale);
      els.loras.appendChild(row);
    });
    updateAdvanced();
  }

  // ---- outpaint -------------------------------------------------------------------------------------
  function setOutSource(source) {
    if (out.url && out.file) URL.revokeObjectURL(out.url);
    out.file = source && source.file || null;
    out.from = source && source.from || null;
    out.url = source ? (source.file ? URL.createObjectURL(source.file) : source.url) : null;
    out.img = null;
    if (out.url) {
      var im = new Image();
      im.onload = function () { out.img = im; drawOutPreview(); };
      im.src = out.url;
    }
    drawOutPreview();
  }

  function pads() {
    return ["left", "top", "right", "bottom"].map(function (k) { return Math.max(0, parseInt(els["pad_" + k].value, 10) || 0); });
  }

  // The same sums as klein_outpaintNG.plan_ng, so the preview matches what the server will make.
  function outPlan() {
    if (!out.img) return null;
    var p = pads(), w = out.img.naturalWidth, h = out.img.naturalHeight;
    var fw = w + p[0] + p[2], fh = h + p[1] + p[3];
    var sc = Math.min(1, Math.sqrt(1048576 / (fw * fh)));
    var sw = Math.max(1, Math.round(w * sc)), sh = Math.max(1, Math.round(h * sc));
    var ox = Math.round(p[0] * sc), oy = Math.round(p[1] * sc);
    var cw = Math.max(256, Math.ceil((ox + sw + Math.round(p[2] * sc)) / 16) * 16);
    var ch = Math.max(256, Math.ceil((oy + sh + Math.round(p[3] * sc)) / 16) * 16);
    return { sc: sc, sw: sw, sh: sh, ox: ox, oy: oy, cw: cw, ch: ch, none: !(p[0] || p[1] || p[2] || p[3]) };
  }

  function drawOutPreview() {
    var c = els.outCanvas, plan = outPlan(), ctx = c.getContext("2d");
    if (!plan) {
      c.width = 320; c.height = 180;
      ctx.fillStyle = "#808080"; ctx.fillRect(0, 0, c.width, c.height);
      els.outInfo.textContent = "Choose a picture to extend.";
      return;
    }
    var k = Math.min(1, 480 / plan.cw, 360 / plan.ch);
    c.width = Math.round(plan.cw * k); c.height = Math.round(plan.ch * k);
    ctx.fillStyle = "#808080"; ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(out.img, plan.ox * k, plan.oy * k, plan.sw * k, plan.sh * k);
    els.outInfo.textContent = plan.none ? "Add padding on at least one side."
      : "The new picture will be " + plan.cw + " × " + plan.ch + (plan.sc < 1 ? " (your picture is shrunk to " + Math.round(plan.sc * 100) + "% to fit one megapixel)" : "") + ".";
  }

  function outPreset(l, t, r, b) {
    var w = out.img ? out.img.naturalWidth : 0, h = out.img ? out.img.naturalHeight : 0;
    [l, t, r, b].forEach(function (f, i) { els["pad_" + ["left", "top", "right", "bottom"][i]].value = Math.round(f(w, h)); });
    drawOutPreview();
  }

  function outSetStatus(msg) { els.outStatus.textContent = msg || ""; }

  function outpaint() {
    var form = new FormData();
    if (out.file) form.append("image", out.file, out.file.name || "in.png");
    else if (out.from) form.append("image_from", out.from);
    pads().forEach(function (v, i) { form.append(["left", "top", "right", "bottom"][i], v); });
    form.append("scene", els.outScene.value);
    form.append("strength", els.outStrength.value || "1.5");
    form.append("keep_original", els.outKeep.checked ? "1" : "0");
    form.append("n", els.outCount.value);
    if (els.outSeed.value) form.append("seed", els.outSeed.value);
    if (els.model.value && !isH3() && !/^base-/.test(els.model.value)) form.append("model", els.model.value);
    els.outGo.disabled = true;
    outSetStatus("Sending...");
    fetch(API + "outpaint", { method: "POST", body: form }).then(function (r) {
      return r.json().then(function (b) { return { ok: r.ok, body: b }; });
    }).then(function (r) {
      els.outGo.disabled = false;
      if (!r.ok) { outSetStatus(r.body.error || "Could not start."); return; }
      outSetStatus("Queued: " + r.body.plan.canvas[0] + " × " + r.body.plan.canvas[1] + ".");
      loadJobs();
    }).catch(function (e) { els.outGo.disabled = false; outSetStatus(e.message); });
  }

  function showMode(mode) {
    els.makeBox.style.display = mode === "make" ? "" : "none";
    els.outBox.style.display = mode === "outpaint" ? "" : "none";
    els.modeMake.style.fontWeight = mode === "make" ? "700" : "400";
    els.modeOut.style.fontWeight = mode === "outpaint" ? "700" : "400";
    if (mode === "outpaint") drawOutPreview();
  }

  function buildOutpaint(box) {
    box.appendChild(el("div", "ng-gen-label", "Picture to extend"));
    var drop = el("div", "ng-gen-ref-paste");
    drop.tabIndex = 0;
    drop.appendChild(el("span", "", "Drop, paste or click, or use → Outpaint on any picture below"));
    var input = el("input"); input.type = "file"; input.accept = "image/*"; input.style.display = "none";
    input.onchange = function () { if (input.files[0]) setOutSource({ file: input.files[0] }); };
    drop.onclick = function () { input.click(); };
    drop.ondragover = function (e) { e.preventDefault(); };
    drop.ondrop = function (e) { e.preventDefault(); var f = e.dataTransfer && e.dataTransfer.files[0]; if (f && /^image\//.test(f.type)) setOutSource({ file: f }); };
    drop.onpaste = function (e) {
      var items = (e.clipboardData && e.clipboardData.items) || [];
      for (var k = 0; k < items.length; k++) if (items[k].kind === "file" && /^image\//.test(items[k].type)) { setOutSource({ file: items[k].getAsFile() }); e.preventDefault(); return; }
    };
    box.appendChild(drop); box.appendChild(input);
    els.outCanvas = el("canvas"); els.outCanvas.style.maxWidth = "100%";
    box.appendChild(els.outCanvas);
    els.outInfo = el("div", "ng-gen-status");
    box.appendChild(els.outInfo);

    var padBox = el("div", "ng-klein-opts");
    ["left", "top", "right", "bottom"].forEach(function (k) {
      var w = el("label", "ng-klein-field"); w.appendChild(el("span", "ng-gen-status", "Add " + k + " (px)"));
      els["pad_" + k] = el("input", "ng-gen-input"); els["pad_" + k].type = "number"; els["pad_" + k].min = 0; els["pad_" + k].step = 16; els["pad_" + k].value = 0;
      els["pad_" + k].oninput = drawOutPreview;
      w.appendChild(els["pad_" + k]); padBox.appendChild(w);
    });
    box.appendChild(padBox);
    var presets = el("div", "ng-klein-pic-btns");
    presets.appendChild(button("Wider", function () { outPreset(function (w) { return w * 0.25; }, function () { return 0; }, function (w) { return w * 0.25; }, function () { return 0; }); }, "A quarter of the width on the left and right"));
    presets.appendChild(button("Taller", function () { outPreset(function () { return 0; }, function (w, h) { return h * 0.25; }, function () { return 0; }, function (w, h) { return h * 0.25; }); }, "A quarter of the height above and below"));
    presets.appendChild(button("All round", function () { outPreset(function (w) { return w * 0.15; }, function (w, h) { return h * 0.15; }, function (w) { return w * 0.15; }, function (w, h) { return h * 0.15; }); }, "15% on every side"));
    presets.appendChild(button("Reset", function () { outPreset(function () { return 0; }, function () { return 0; }, function () { return 0; }, function () { return 0; }); }));
    box.appendChild(presets);

    box.appendChild(el("label", "ng-gen-label", "What is in the picture (helps the new parts match)"));
    els.outScene = el("textarea", "ng-gen-pose-modal-textarea"); els.outScene.rows = 3;
    els.outScene.placeholder = "Describe the whole scene, not just the new bit. e.g. a woman in a green coat at a rainy bus stop on a Sheffield street, early evening";
    box.appendChild(els.outScene);

    var opts = el("div", "ng-klein-opts");
    var field = function (label, node, title) { var w = el("label", "ng-klein-field"); w.appendChild(el("span", "ng-gen-status", label)); w.appendChild(node); if (title) w.title = title; opts.appendChild(w); };
    els.outStrength = el("input", "ng-gen-input"); els.outStrength.type = "number"; els.outStrength.step = 0.1; els.outStrength.min = 0; els.outStrength.max = 3; els.outStrength.value = 1.5;
    field("LoRA strength", els.outStrength, "1.5 is the author's recommendation; 1.0 slips a little more when you add a lot");
    els.outCount = el("select", "ng-gen-input");
    [1, 2, 4].forEach(function (n) { var o = el("option", "", n + (n === 1 ? " picture" : " variations")); o.value = n; els.outCount.appendChild(o); });
    field("How many", els.outCount);
    els.outSeed = el("input", "ng-gen-input"); els.outSeed.type = "number"; els.outSeed.placeholder = "random";
    field("Seed", els.outSeed);
    els.outKeep = el("input"); els.outKeep.type = "checkbox"; els.outKeep.checked = true;
    field("Put the original back", els.outKeep, "The model keeps your picture to within a pixel or two; this pastes the exact original over it with a soft edge");
    box.appendChild(opts);

    var bar = el("div", "ng-gen-action-bar");
    els.outGo = button("Extend", outpaint);
    els.outStatus = el("span", "ng-gen-status");
    bar.appendChild(els.outGo); bar.appendChild(els.outStatus);
    box.appendChild(bar);
    drawOutPreview();
  }

  // ---- generate -------------------------------------------------------------------------------
  function setStatus(msg) { els.status.textContent = msg || ""; }

  function generate() {
    var form = new FormData();
    var h3 = isH3();
    form.append("prompt", els.layout && !h3 ? els.layout.apply(els.prompt.value) : els.prompt.value);
    form.append("n", els.count.value);
    if (els.seed.value) form.append("seed", els.seed.value);
    if (els.model.value) form.append("model", els.model.value);
    if (h3) {
      form.append("shape", els.shape.value); form.append("tier", els.tier.value);
    } else {
      var wh = els.size.value.split("x");
      form.append("width", wh[0]); form.append("height", wh[1]);
      if (els.steps.value) form.append("steps", els.steps.value);
      if (els.guidance.value && isBase()) form.append("guidance", els.guidance.value);
      if (els.matchRef.checked && filledSlots().length) form.append("match_ref", "1");
      var picked = chosenLoras();
      if (picked.length) form.append("loras", JSON.stringify(picked));
      if (els.memory.value === "fast") {
        form.append("low_ram", "0");
        if (els.tile.value) form.append("vae_tile", els.tile.value);
      }
    }
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

  // What made this picture: every setting the job recorded, so a good one can be remade or varied.
  function settingsPanel(job, r) {
    var o = job.options || {}, det = el("details", "ng-klein-adv");
    det.appendChild(el("summary", "ng-gen-status", "Settings"));
    var rows = job.model === H3 ? [
      ["Seed", r.seed], ["Size", r.width + " × " + r.height], ["Model", H3_LABEL],
      ["Shape", o.shape === "match" ? "same as the first reference" : o.shape], ["Quality", o.tier], ["Made in", r.seconds + " s"],
    ] : [
      ["Seed", r.seed], ["Size", r.width + " × " + r.height], ["Model", job.model],
      ["Steps", o.steps || "default"], ["Guidance", job.model.indexOf("base-") === 0 ? (o.guidance || "3.5 (default)") : "not used"],
      ["Memory", o.low_ram === false ? "faster, tile " + (o.vae_tile || "default") : "low"],
      ["Sized like the first reference", o.match_ref ? "yes" : "no"],
      ["LoRAs", (o.loras || []).map(function (l) { return l.name + " at " + l.scale; }).join(", ") || "none"],
      ["Made in", r.seconds + " s"],
    ];
    if (job.outpaint) rows.push(["Outpaint", "pads " + ["left", "top", "right", "bottom"].map(function (k) { return k + " " + (job.outpaint[k] || 0); }).join(", ")]);
    var table = el("div", "ng-klein-settings");
    rows.forEach(function (x) { table.appendChild(el("span", "ng-gen-status", x[0])); table.appendChild(el("span", "", String(x[1]))); });
    det.appendChild(table);
    det.appendChild(el("div", "ng-gen-status", "Prompt"));
    det.appendChild(el("div", "ng-klein-promptbox", job.prompt));
    if ((job.ref_files || []).length) {
      det.appendChild(el("div", "ng-gen-status", "References used"));
      var refs = el("div", "ng-klein-refthumbs");
      job.ref_files.forEach(function (f, i) {
        var img = el("img"); img.src = API + "ref/" + job.id + "/" + f; img.title = job.refs[i]; img.alt = job.refs[i];
        refs.appendChild(img);
      });
      det.appendChild(refs);
    }
    var bar = el("div", "ng-klein-pic-btns");
    if (!job.outpaint) bar.appendChild(button("Remake with these settings", function () { reuse(job, r.seed); },
      "Copy the prompt, seed, model, LoRAs and references into the form"));
    bar.appendChild(button("Copy prompt", function () { navigator.clipboard && navigator.clipboard.writeText(job.prompt); }));
    det.appendChild(bar);
    return det;
  }

  function pictureCell(job, r) {
    var cell = el("div", "ng-klein-pic");
    var a = el("a"); a.href = imgUrl(job.id, r.file); a.target = "_blank";
    var img = el("img"); img.src = imgUrl(job.id, r.file); img.loading = "lazy"; img.alt = job.prompt;
    a.appendChild(img); cell.appendChild(a);
    cell.appendChild(el("div", "ng-gen-status", "seed " + r.seed + " · " + r.width + "x" + r.height + " · " + r.seconds + "s"));
    cell.appendChild(settingsPanel(job, r));
    var row = el("div", "ng-klein-pic-btns");
    SLOTS.slice(0, 3).concat([{ label: "Another", more: true }]).forEach(function (def, k) {
      row.appendChild(button("→ " + def.label, function () {
        var i = def.more ? slots.findIndex(function (x, idx) { return idx >= 3 && !x.url; }) : k;
        if (i < 0) { setStatus("All ten reference slots are full."); return; }
        setSlot(i, { from: job.id + "/" + r.file, url: imgUrl(job.id, r.file) });
        els.main.scrollTo({ top: 0, behavior: "smooth" });
      }, def.more ? "Put this picture in the next free reference slot (up to ten)" : "Use this picture as the " + def.label.toLowerCase() + " reference"));
    });
    row.appendChild(button("→ Outpaint", function () {
      setOutSource({ from: job.id + "/" + r.file, url: imgUrl(job.id, r.file) });
      showMode("outpaint");
      els.main.scrollTo({ top: 0, behavior: "smooth" });
    }, "Extend this picture's edges"));
    row.appendChild(button("→ Video first frame", function () { sendToVideo(job.id, r.file); },
      "Open the H3 view with this picture as its first frame"));
    cell.appendChild(row);
    return cell;
  }

  function reuse(job, seed) {
    var o = job.options || {};
    els.prompt.value = job.prompt;
    if (job.width && [].some.call(els.size.options, function (x) { return x.value === job.width + "x" + job.height; })) { els.size.value = job.width + "x" + job.height; sizeNote(); }
    if ([].some.call(els.model.options, function (x) { return x.value === job.model; })) els.model.value = job.model;
    if (job.model === H3) {
      if (o.tier) els.tier.value = o.tier;
      if (o.shape) { els.shape.value = o.shape; els.shape.dataset.touched = "1"; }
    }
    els.seed.value = seed != null ? seed : job.seed;
    els.steps.value = o.steps || "";
    els.guidance.value = o.guidance || "";
    els.memory.value = o.low_ram === false ? "fast" : "low";
    els.tile.value = o.vae_tile || "";
    // references go back in the slots they came from (straight from the old job's folder on the server)
    slots.forEach(function (x, i) { if (x.url) setSlot(i, null); });
    var missing = 0;
    (job.refs || []).forEach(function (name, k) {
      var i = SLOTS.findIndex(function (d) { return d.key === name; }), f = (job.ref_files || [])[k];
      if (i < 0 || !f) { missing++; return; }
      setSlot(i, { from: job.id + "/refs/" + f, url: API + "ref/" + job.id + "/" + f });
    });
    els.matchRef.checked = !!o.match_ref && filledSlots().length > 0;
    loraList.forEach(function (l) {
      if (!l.row) return;
      var used = (o.loras || []).filter(function (x) { return x.name === l.name; })[0];
      l.row.box.checked = !!used;
      if (used) l.row.scale.value = used.scale;
    });
    updateAdvanced();
    els.main.scrollTo({ top: 0, behavior: "smooth" });
    setStatus("Settings copied" + (job.refs && job.refs.length ? ", references put back" : "") + (missing ? " (" + missing + " could not be)" : "") + ".");
  }

  function jobCard(job) {
    var card = el("div", "ng-klein-job");
    var title = job.outpaint ? (job.prompt.replace(/^Outpaint the image:[^.]*\.\s*/, "").replace(/^Scene:\s*/, "") || "(no scene description)") : job.prompt;
    var head = el("div", "ng-gen-label", title.length > 160 ? title.slice(0, 160) + "..." : title);
    card.appendChild(head);
    var info = job.status + (job.outpaint ? " · outpaint" : job.refs.length ? " · refs: " + job.refs.join(", ") : " · text only") + " · " + (job.model === H3 ? H3_LABEL : job.model);
    if (job.status === "running" || job.status === "queued" || job.status === "waiting") {
      info += " · " + job.results.length + " of " + job.n;
      if (job.status === "waiting") info += " (waiting for the video render to finish)";
    }
    if (job.error) info += " · " + job.error;
    if (job.ref_notes && job.ref_notes.length) info += " · " + job.ref_notes.join("; ");
    card.appendChild(el("div", "ng-gen-status", info));
    var grid = el("div", "ng-klein-grid");
    job.results.forEach(function (r) { grid.appendChild(pictureCell(job, r)); });
    card.appendChild(grid);
    var bar = el("div", "ng-klein-pic-btns");
    bar.appendChild(button("Use this prompt", function () { els.prompt.value = job.prompt; els.main.scrollTo({ top: 0, behavior: "smooth" }); }));
    if (!job.outpaint) bar.appendChild(button("Reuse settings", function () { reuse(job); },
      "Copy this job's prompt, size, seed, model, LoRAs and references into the form"));
    if (job.status === "running" || job.status === "queued" || job.status === "waiting") {
      bar.appendChild(button("Stop", function () { fetch(API + "cancel/" + job.id, { method: "POST" }).then(loadJobs); }));
    }
    if (["running", "queued", "waiting"].indexOf(job.status) < 0) {
      var del = button("Delete", function () {
        if (!del.dataset.sure) { del.dataset.sure = "1"; del.textContent = "Really delete? Click again"; setTimeout(function () { del.dataset.sure = ""; del.textContent = "Delete"; }, 4000); return; }
        fetch(API + "delete/" + job.id, { method: "POST" }).then(loadJobs);
      }, "Remove this job and its pictures from the disk");
      bar.appendChild(del);
    }
    card.appendChild(bar);
    return card;
  }

  function loadJobs() {
    fetch(API + "jobs?limit=" + histLimit + "&q=" + encodeURIComponent(histQuery)).then(function (r) { return r.json(); }).then(function (b) {
      els.gallery.textContent = "";
      var shownN = (b.jobs || []).length;
      els.histInfo.textContent = b.total ? "Showing " + shownN + " of " + b.total + (histQuery ? " matching" : " jobs") : (histQuery ? "Nothing matches." : "No pictures yet.");
      els.histMore.style.display = b.total > shownN ? "" : "none";
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
  // ---- improve prompt: a local Ollama model rewrites the prompt; the user accepts or discards ------
  function buildImprove(root) {
    var box = el("div", "ng-klein-improve");
    var row = el("div", "ng-klein-pic-btns");
    els.improveModel = el("select", "ng-gen-input"); els.improveModel.style.maxWidth = "16em";
    els.improveBtn = button("Improve prompt", improvePrompt, "A local model writes a fuller description from your words. You review it before it is used");
    row.appendChild(els.improveBtn); row.appendChild(els.improveModel);
    box.appendChild(row);
    var det = el("details", "ng-klein-adv");
    det.appendChild(el("summary", "ng-gen-status", "How the rewrite is instructed"));
    els.improveSystem = el("textarea", "ng-gen-pose-modal-textarea"); els.improveSystem.rows = 8;
    det.appendChild(els.improveSystem);
    box.appendChild(det);
    els.improveStatus = el("div", "ng-gen-status"); box.appendChild(els.improveStatus);
    els.improveOut = el("textarea", "ng-gen-pose-modal-textarea"); els.improveOut.rows = 5; els.improveOut.style.display = "none";
    box.appendChild(els.improveOut);
    els.improveAct = el("div", "ng-klein-pic-btns"); els.improveAct.style.display = "none";
    els.improveAct.appendChild(button("Use this", function () {
      els.prompt.value = els.improveOut.value; hideImprove(); els.prompt.focus();
    }));
    els.improveAct.appendChild(button("Discard", hideImprove));
    box.appendChild(els.improveAct);
    root.appendChild(box);
    fetch(API + "improve").then(function (r) { return r.json(); }).then(function (b) { els.improveSystem.value = b.system || ""; }).catch(function () {});
    fetch("/api/ng/chat/models").then(function (r) { return r.json(); }).then(function (b) {
      var loaded = b.loaded || [];
      (b.models || []).forEach(function (m) { var o = el("option", "", m.name); o.value = m.name; els.improveModel.appendChild(o); });
      if (loaded.length) els.improveModel.value = loaded[0];
      if (!els.improveModel.options.length) els.improveStatus.textContent = "No local model found. Is Ollama running?";
    }).catch(function () { els.improveStatus.textContent = "Could not reach Ollama."; });
  }

  function hideImprove() { els.improveOut.style.display = "none"; els.improveAct.style.display = "none"; }

  function improvePrompt() {
    els.improveBtn.disabled = true; hideImprove();
    els.improveStatus.textContent = "Rewriting...";
    fetch(API + "improve", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: els.prompt.value, model: els.improveModel.value, system: els.improveSystem.value }) })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (r) {
        els.improveBtn.disabled = false;
        if (!r.ok) { els.improveStatus.textContent = r.body.error || "Could not rewrite."; return; }
        els.improveStatus.textContent = "Edit it if you like, then use it.";
        els.improveOut.value = r.body.prompt; els.improveOut.style.display = ""; els.improveAct.style.display = "";
      }).catch(function (e) { els.improveBtn.disabled = false; els.improveStatus.textContent = e.message; });
  }

  function build() {
    var host = document.getElementById("ng-klein-compose");
    var modes = el("div", "ng-klein-pic-btns");
    els.modeMake = button("Make a picture", function () { showMode("make"); });
    els.modeOut = button("Outpaint", function () { showMode("outpaint"); }, "Extend a picture past its edges");
    modes.appendChild(els.modeMake); modes.appendChild(els.modeOut);
    host.appendChild(modes);
    els.makeBox = el("div"); els.outBox = el("div");
    host.appendChild(els.makeBox); host.appendChild(els.outBox);
    var root = els.makeBox;
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
    buildImprove(root);
    var layoutHost = el("div");
    els.layoutHost = layoutHost;
    root.appendChild(layoutHost);
    els.layout = window.KleinLayoutNG ? window.KleinLayoutNG.build(layoutHost, function () { return els.size.value.split("x").map(Number); }) : null;

    var opts = el("div", "ng-klein-opts");
    var field = function (label, node) { var w = el("label", "ng-klein-field"); w.appendChild(el("span", "ng-gen-status", label)); w.appendChild(node); opts.appendChild(w); return w; };
    els.size = el("select", "ng-gen-input");
    SIZES.forEach(function (s) { var o = el("option", "", s[0].replace("x", " × ") + " · " + s[1]); o.value = s[0]; els.size.appendChild(o); });
    els.sizeField = field("Size", els.size);
    els.sizeNote = el("span", "ng-gen-status", LARGE_NOTE);
    els.size.parentNode.appendChild(els.sizeNote);
    els.size.addEventListener("change", function () { sizeNote(); if (els.layout) els.layout.refresh(); });
    sizeNote();
    els.count = el("select", "ng-gen-input");
    VARIATIONS.forEach(function (n) { var o = el("option", "", n + (n === 1 ? " picture" : " variations")); o.value = n; els.count.appendChild(o); });
    els.count.value = "4";
    els.count.addEventListener("change", function () { els.count.dataset.touched = "1"; });
    field("How many", els.count);
    els.seed = el("input", "ng-gen-input"); els.seed.type = "number"; els.seed.placeholder = "random";
    field("Seed", els.seed);
    els.model = el("select", "ng-gen-input");
    field("Model", els.model);
    els.shape = el("select", "ng-gen-input");
    H3_SHAPES.forEach(function (x) { var o = el("option", "", x[1]); o.value = x[0]; els.shape.appendChild(o); });
    els.shape.addEventListener("change", function () { els.shape.dataset.touched = "1"; });
    els.shapeField = field("Shape", els.shape);
    els.tier = el("select", "ng-gen-input");
    H3_TIERS.forEach(function (x) { var o = el("option", "", x[1]); o.value = x[0]; els.tier.appendChild(o); });
    els.tierField = field("Quality", els.tier);
    root.appendChild(opts);

    var adv = el("details", "ng-klein-adv");
    els.adv = adv;
    adv.appendChild(el("summary", "ng-gen-status", "Advanced"));
    var advBox = el("div", "ng-klein-opts");
    var advField = function (label, node, title) {
      var w = el("label", "ng-klein-field"); w.appendChild(el("span", "ng-gen-status", label)); w.appendChild(node);
      if (title) w.title = title;
      advBox.appendChild(w);
    };
    els.steps = el("input", "ng-gen-input"); els.steps.type = "number"; els.steps.min = 1; els.steps.max = 100;
    advField("Steps", els.steps, "More steps is slower; the default suits each model");
    els.guidance = el("input", "ng-gen-input"); els.guidance.type = "number"; els.guidance.step = 0.5; els.guidance.min = 0; els.guidance.max = 20;
    advField("Guidance", els.guidance, "How hard the picture follows the words. Only the base model uses it");
    els.matchRef = el("input"); els.matchRef.type = "checkbox";
    advField("Edit at the size of the first reference", els.matchRef, "Ignores the Size choice and keeps the first reference's shape");
    els.memory = el("select", "ng-gen-input");
    [["low", "Low memory (default)"], ["fast", "Faster, uses more memory"]].forEach(function (m) { var o = el("option", "", m[1]); o.value = m[0]; els.memory.appendChild(o); });
    advField("Memory", els.memory, "Low memory leaves room for a video render running alongside");
    els.tile = el("input", "ng-gen-input"); els.tile.type = "number"; els.tile.step = 16; els.tile.min = 128; els.tile.placeholder = "512 (default)";
    advField("Decode tile (px)", els.tile, "Smaller tiles use less memory while the picture is decoded");
    adv.appendChild(el("div", "ng-gen-label", "LoRAs"));
    els.loras = el("div", "ng-klein-loras");
    adv.appendChild(els.loras);
    ["input", "change"].forEach(function (ev) { advBox.addEventListener(ev, updateAdvanced); els.model.addEventListener(ev, updateAdvanced); });
    adv.appendChild(advBox);
    root.appendChild(adv);

    var bar = el("div", "ng-gen-action-bar");
    els.go = button("Generate", generate);
    els.status = el("span", "ng-gen-status");
    bar.appendChild(els.go); bar.appendChild(els.status);
    root.appendChild(bar);
    buildOutpaint(els.outBox);
    els.log = el("div", "ng-vg-mainlog"); els.log.style.display = "none";
    host.appendChild(els.log);
    host.appendChild(el("div", "ng-gen-label", "Pictures"));
    var hist = el("div", "ng-klein-pic-btns");
    els.histSearch = el("input", "ng-gen-input"); els.histSearch.type = "search"; els.histSearch.placeholder = "Search prompts, models and settings";
    els.histSearch.addEventListener("input", function () {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(function () { histQuery = els.histSearch.value; histLimit = 30; loadJobs(); }, 300);
    });
    els.histInfo = el("span", "ng-gen-status");
    hist.appendChild(els.histSearch); hist.appendChild(els.histInfo);
    host.appendChild(hist);
    els.gallery = el("div", "ng-klein-gallery");
    host.appendChild(els.gallery);
    els.histMore = button("Show more", function () { histLimit += 30; loadJobs(); });
    host.appendChild(els.histMore);
    drawSlots();
    updateHint();
    updateAdvanced();
    showMode("make");
    built = true;
  }

  function loadHealth() {
    fetch(API + "health").then(function (r) { return r.json(); }).then(function (h) {
      els.model.textContent = "";
      (h.models || []).forEach(function (m) { var o = el("option", "", m); o.value = m; els.model.appendChild(o); });
      if (h.h3image) { var ho = el("option", "", H3_LABEL); ho.value = H3; els.model.appendChild(ho); }
      if (h.default) els.model.value = h.default;
      else if (!(h.models || []).length && h.h3image) els.model.value = H3;
      h3ok = !!(h.h3image && h.h3image.ok); h3problems = (h.h3image && h.h3image.problems) || [];
      kleinOk = !!h.ready;
      kleinMsg = h.ready ? "" : (h.worker_found ? "The Klein weights are not downloaded (do that in Phosphene)." : "The FLUX 3 worker is not installed (do that in Phosphene).");
      loraList = h.loras || [];
      drawLoras();
      els.outGo.disabled = !h.ready || !loraList.some(function (l) { return l.name === "flux-3-image-klein-outpaint"; });
      if (els.outGo.disabled && h.ready) outSetStatus("The Outpaint LoRA is not installed in the shared loras folder (flux3).");
      updateAdvanced();
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
