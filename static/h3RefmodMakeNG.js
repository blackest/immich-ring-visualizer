// H3 view: make a RefMod from pictures. A small form inside the RefMod block: a name, the pictures
// (drop them or pick them), the canvas and an optional picture limit. The server side is
// routes/h3_toolsNG.py (POST /api/ng/h3/refmods/create), which queues a task job in the H3 queue; this
// polls that job and asks the view to reload its RefMod list when the file is there.
(function () {
  "use strict";

  var API = "/api/ng/h3";
  var CANVAS_HELP = { square: "Square 512 (people and faces, 256 rows a picture)",
    landscape: "Landscape 768x448 (scenes, 336 rows a picture)" };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function init(host, hooks) {
    if (!host || host.dataset.ready) return;
    host.dataset.ready = "1";
    var files = [];
    var details = el("details");
    details.appendChild(el("summary", "ng-gen-label", "Make a RefMod from pictures"));
    var name = el("input", "ng-gen-input");
    name.type = "text"; name.placeholder = "name, e.g. lake_v1";
    var drop = el("div", "ng-gen-label", "Drop pictures here, or click to choose (about 20 of one subject)");
    drop.style.cssText = "border: 1px dashed currentColor; padding: 12px; text-align: center; cursor: pointer; " +
      "text-transform: none; letter-spacing: 0";
    var picker = el("input"); picker.type = "file"; picker.accept = "image/*"; picker.multiple = true; picker.hidden = true;
    var canvas = el("select", "ng-gen-input");
    Object.keys(CANVAS_HELP).forEach(function (k) {
      var o = el("option", "", CANVAS_HELP[k]); o.value = k; canvas.appendChild(o);
    });
    var limit = el("input", "ng-gen-input");
    limit.type = "number"; limit.min = "1"; limit.max = "24"; limit.placeholder = "use at most N pictures, evenly spaced";
    var go = el("button", "ng-gen-btn", "Make RefMod");
    go.type = "button"; go.style.width = "auto";
    var note = el("div", "ng-gen-label", "");
    note.style.cssText = "text-transform: none; letter-spacing: 0";

    function say(text) { note.textContent = text || ""; }
    function setFiles(list) {
      files = [].filter.call(list, function (f) { return /^image\//.test(f.type) || /\.(png|jpe?g|webp)$/i.test(f.name); });
      drop.textContent = files.length ? files.length + " pictures chosen (drop or click to change)"
        : "Drop pictures here, or click to choose (about 20 of one subject)";
    }
    drop.onclick = function () { picker.click(); };
    picker.onchange = function () { setFiles(picker.files); };
    drop.ondragover = function (e) { e.preventDefault(); };
    drop.ondrop = function (e) { e.preventDefault(); setFiles(e.dataTransfer.files); };

    function poll(jobId, started) {
      var timer = setInterval(function () {
        fetch(API + "/jobs/" + jobId).then(function (r) { return r.json(); }).then(function (job) {
          if (job.status === "completed") {
            clearInterval(timer); go.disabled = false;
            say("Made " + job.meta.name + " (" + job.meta.pictures + " pictures, " + job.meta.rows.toLocaleString() + " rows). It is in the list now.");
            if (hooks && hooks.reload) hooks.reload();
          } else if (job.status === "failed" || !job.status) {
            clearInterval(timer); go.disabled = false;
            say("Failed: " + (job.error || "the job is gone (the server restarted?)"));
          } else {
            var s = Math.floor((Date.now() - started) / 1000);
            say((job.status === "queued" ? "Waiting for the GPU" : "Encoding the pictures") + " · " +
              Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2));
          }
        }).catch(function () {});
      }, 3000);
    }

    go.onclick = function () {
      if (!files.length) { say("Add some pictures first."); return; }
      var form = new FormData();
      form.append("name", name.value.trim());
      form.append("canvas", canvas.value);
      if (limit.value) form.append("max_pictures", limit.value);
      files.forEach(function (f) { form.append("files", f, f.name); });
      go.disabled = true;
      say("Uploading " + files.length + " pictures...");
      fetch(API + "/refmods/create", { method: "POST", body: form })
        .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
        .then(function (r) {
          if (!r.ok || !r.body.job_id) { go.disabled = false; say(r.body.error || "Could not start."); return; }
          say(r.body.pictures + " pictures, " + r.body.rows.toLocaleString() + " reference rows. Starting...");
          poll(r.body.job_id, Date.now());
        })
        .catch(function (e) { go.disabled = false; say(e.message); });
    };

    [name, drop, picker, canvas, limit, go, note].forEach(function (e) { details.appendChild(e); });
    host.appendChild(details);
  }

  window.H3RefmodMakeNG = { init: init };
})();
