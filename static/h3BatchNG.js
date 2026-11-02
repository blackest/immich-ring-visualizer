/**
 * h3BatchNG.js -- reference pictures for each window of a scene, in the H3 view.
 *
 * Reference pictures work on one 5 s window, so a scene of several prompt windows (written with the
 * Enhanced prompt pop-out, static/h3EnhancedPopoutNG.js) is rendered as one 5 s render per window and
 * joined at the end (h3_batchNG.py, POST /api/ng/h3/batch). This file adds the picture trays under the
 * windows list, the "carry the last frame" tick, and a panel that follows the scene to its film.
 *
 *   H3BatchNG.mount(box, count)   draw the trays for `count` windows into `box`
 *   H3BatchNG.hasBatch()          true when any picture is attached, or the carry tick is on
 *   H3BatchNG.run(opts)           opts: {prompts, quality, steps, tristep, flex}; sends and follows the scene
 */
(function () {
  "use strict";

  var API = "/api/ng/h3/batch";
  var shared = [];       // File[] for every window
  var perWindow = [];    // File[][] one list per window
  var carry = false;
  var count = 0;
  var panel = null, box = null, current = null, timer = null;

  function node(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  function tray(label, files, hint) {
    var row = node("div", "ng-h3b-row");
    row.appendChild(node("div", "ng-gen-status", label));
    var thumbs = node("div", "ng-h3b-thumbs");
    files.forEach(function (f, i) {
      var cell = node("span", "ng-h3b-thumb");
      var isVideo = /^video\//.test(f.type);
      var el = node(isVideo ? "video" : "img");
      el.src = URL.createObjectURL(f);
      el.title = f.name;
      var x = node("button", "ng-h3b-x", "×");
      x.type = "button"; x.title = "Remove";
      x.addEventListener("click", function () { files.splice(i, 1); redraw(); });
      cell.appendChild(el); cell.appendChild(x); thumbs.appendChild(cell);
    });
    var input = node("input"); input.type = "file"; input.multiple = true; input.accept = "image/*,video/*"; input.hidden = true;
    input.addEventListener("change", function () {
      Array.prototype.forEach.call(input.files, function (f) { files.push(f); });
      redraw();
    });
    var add = node("button", "ng-gen-btn", "Add pictures");
    add.type = "button"; add.style.width = "auto";
    add.addEventListener("click", function () { input.click(); });
    thumbs.appendChild(add);
    row.appendChild(thumbs); row.appendChild(input);
    if (hint) row.appendChild(node("div", "ng-gen-status", hint));
    return row;
  }

  function redraw() { if (box) mount(box, count); }

  function mount(host, n) {
    box = host; count = n;
    while (perWindow.length < n) perWindow.push([]);
    perWindow.length = n;
    var old = host.querySelector(".ng-h3b"); if (old) old.remove();
    var wrap = node("div", "ng-h3b");
    wrap.appendChild(node("div", "ng-gen-label", "Reference pictures"));
    wrap.appendChild(node("div", "ng-gen-status",
      "Each window is rendered as its own 5 s clip with these pictures, then the clips are joined. In a prompt, " +
      "<Picture 1> is the first picture a window gets: the shared ones come first, then that window's own, then the carried frame."));
    wrap.appendChild(tray("Every window", shared));
    for (var i = 0; i < n; i++) {
      var own = shared.length;
      wrap.appendChild(tray("Window " + (i + 1) + " (its own pictures start at <Picture " + (own + 1) + ">)", perWindow[i]));
    }
    var label = node("label", "ng-klein-lora");
    var tick = node("input"); tick.type = "checkbox"; tick.checked = carry;
    tick.addEventListener("change", function () { carry = tick.checked; });
    label.appendChild(tick);
    label.appendChild(document.createTextNode(" Also give each later window the last frame of the one before it as a picture (numbered last)"));
    wrap.appendChild(label);
    if (!panel) panel = node("div", "ng-h3b-panel");
    wrap.appendChild(panel);
    host.appendChild(wrap);
  }

  function hasBatch() {
    return count > 1 && (carry || shared.length > 0 || perWindow.some(function (l) { return l.length > 0; }));
  }

  function show(lines, filmUrl, canStop) {
    panel.textContent = "";
    lines.forEach(function (l) { panel.appendChild(node("div", "ng-gen-status", l)); });
    if (filmUrl) {
      var v = node("video"); v.controls = true; v.src = filmUrl; v.style.maxWidth = "100%";
      panel.appendChild(v);
    }
    if (canStop && current) {
      var stop = node("button", "ng-gen-btn", "Stop the scene");
      stop.type = "button"; stop.style.width = "auto";
      stop.addEventListener("click", function () { fetch(API + "/" + current, { method: "DELETE" }); });
      panel.appendChild(stop);
    }
  }

  function poll() {
    clearTimeout(timer);
    if (!current) return;
    fetch(API + "/" + current).then(function (r) { return r.json(); }).then(function (s) {
      if (s.error && !s.windows) { show([s.error]); current = null; return; }
      var lines = s.windows.map(function (w) { return "Window " + w.n + ": " + w.status; });
      if (s.status === "completed") {
        lines.push("Done. The film is in the Job Log.");
        show(lines, s.film && s.film.video_url); current = null; return;
      }
      if (s.status === "failed" || s.status === "cancelled") {
        lines.push(s.status === "failed" ? "Stopped: " + (s.error || "a window failed") : "Stopped.");
        show(lines); current = null; return;
      }
      show(lines, null, true);
      timer = setTimeout(poll, 5000);
    }).catch(function () { timer = setTimeout(poll, 8000); });
  }

  function run(opts) {
    var form = new FormData();
    form.append("prompts", JSON.stringify(opts.prompts));
    form.append("carry", carry ? "true" : "false");
    form.append("quality", opts.quality || "high");
    if (opts.steps) form.append("steps", String(opts.steps));
    if (opts.tristep) form.append("tristep", "true");
    if (opts.flex) form.append("flex", "true");
    shared.forEach(function (f) { form.append("shared_files", f, f.name); });
    perWindow.forEach(function (list, i) { list.forEach(function (f) { form.append("window_" + (i + 1) + "_files", f, f.name); }); });
    show(["Sending the scene..."]);
    fetch(API, { method: "POST", body: form }).then(function (r) {
      return r.json().then(function (p) { return { ok: r.ok, p: p }; });
    }).then(function (r) {
      if (!r.ok) { show([r.p.error || "Could not start the scene."]); return; }
      current = r.p.batch_id;
      show(["Queued."], null, true);
      poll();
    }).catch(function (e) { show([e.message]); });
  }

  window.H3BatchNG = { mount: mount, hasBatch: hasBatch, run: run };
})();
