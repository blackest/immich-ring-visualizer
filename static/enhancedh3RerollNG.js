// Enhanced H3: re-roll one window of a finished chained render and choose how each seam is joined.
// The panel sits under the video. It lists the windows (prompt, seed, a start-frame option, a
// Reroll button), a join choice for each seam, and the film versions made so far. The server side
// is routes/h3_rerollNG.py; the one-window job it starts is polled by enhancedh3NG.js like any
// other render, which calls show() again when it ends.
(function () {
  "use strict";

  var API = "/api/ng/h3/jobs/";
  var host = null;     // the panel's container, set by init()
  var hooks = null;    // {startPolling(jobId), playUrl(url), say(text)} from enhancedh3NG.js
  var current = null;  // source job id the panel is showing

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function api(path, method, body) {
    return fetch(API + path, {
      method: method || "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); });
  }

  // The seams as the person has set them now: [{seam, type, seconds}]
  function readJoins() {
    return [].map.call(host.querySelectorAll(".eh3-seam"), function (row) {
      return { seam: parseInt(row.dataset.seam, 10), type: row.querySelector("select").value,
        seconds: parseFloat(row.querySelector("input").value) || 0.5 };
    });
  }

  function seamRow(join, info) {
    var row = el("div", "ng-eh3-grid eh3-seam");
    row.dataset.seam = String(join.seam);
    row.appendChild(el("span", "ng-gen-label", "Join " + join.seam + " → " + (join.seam + 1)));
    var sel = el("select", "ng-gen-input");
    var names = { cut: "Hard cut", dissolve: "Dissolve", fade: "Fade through black", audio_lead: "Sound leads picture" };
    info.join_types.forEach(function (t) {
      var o = el("option", "", names[t] || t);
      o.value = t;
      if (t === join.type) o.selected = true;
      sel.appendChild(o);
    });
    var secs = el("input", "ng-gen-input");
    secs.type = "number"; secs.min = info.min_seconds; secs.max = info.max_seconds; secs.step = "0.25";
    secs.value = join.seconds; secs.title = "seconds";
    var sync = function () { secs.disabled = sel.value === "cut"; };
    sel.onchange = sync; sync();
    row.appendChild(sel); row.appendChild(secs);
    return row;
  }

  function windowRow(w, info) {
    var box = el("div", "ng-eh3-window");
    box.appendChild(el("div", "ng-gen-label", "Window " + w.index + (w.rerolled ? " (rerolled)" : "") +
      (w.seed != null ? " · seed " + w.seed : "")));
    var ta = el("textarea", "ng-gen-input");
    ta.rows = 4; ta.value = w.prompt || "";
    var controls = el("div", "ng-eh3-grid");
    var seed = el("input", "ng-gen-input");
    seed.type = "number"; seed.placeholder = "new random seed";
    controls.appendChild(seed);
    var from = null;
    if (w.index > 1) {
      var lab = el("label", "ng-checkbox-row");
      from = el("input"); from.type = "checkbox"; from.checked = true;
      lab.appendChild(from);
      lab.appendChild(document.createTextNode(" Start from the previous window's last frame"));
      controls.appendChild(lab);
    }
    var play = el("button", "ng-gen-btn", "Play this window");
    play.type = "button"; play.style.width = "auto";
    play.onclick = function () { hooks.playUrl(API + current + "/windows/" + w.index + "/video?v=" + Date.now()); };
    var go = el("button", "ng-gen-btn", "Reroll window " + w.index);
    go.type = "button"; go.style.width = "auto";
    go.onclick = function () {
      var body = { window: w.index, prompt: ta.value, from_previous: from ? from.checked : false, joins: readJoins() };
      if (seed.value) body.seed = parseInt(seed.value, 10);
      go.disabled = true;
      hooks.say("Starting the re-roll of window " + w.index + "...");
      api(current + "/reroll", "POST", body).then(function (r) {
        go.disabled = false;
        if (!r.ok || !r.body.job_id) { hooks.say(r.body.error || "Could not start the re-roll."); return; }
        hooks.startPolling(r.body.job_id);
      });
    };
    controls.appendChild(play); controls.appendChild(go);
    box.appendChild(ta); box.appendChild(controls);
    return box;
  }

  function draw(info) {
    host.textContent = "";
    host.appendChild(el("div", "ng-gen-label", "Reroll one window"));
    host.appendChild(el("div", "ng-gen-status",
      "A re-roll renders only that window again (about a sixth of the original time) and joins it back in. " +
      "The windows after it are kept as they are, so the join is a cut, dissolve or fade, not a continuation."));
    var windows = el("div");
    info.windows.forEach(function (w, i) {
      windows.appendChild(windowRow(w, info));
      if (i < info.joins.length) windows.appendChild(seamRow(info.joins[i], info));
    });
    host.appendChild(windows);
    var bar = el("div", "ng-gen-action-bar");
    var apply = el("button", "ng-gen-btn", "Join again with these joins");
    apply.type = "button"; apply.style.width = "auto";
    apply.onclick = function () {
      apply.disabled = true;
      hooks.say("Joining...");
      api(current + "/restitch", "POST", { joins: readJoins() }).then(function (r) {
        apply.disabled = false;
        if (!r.ok) { hooks.say(r.body.error || "Could not join."); return; }
        hooks.say("Joined: version " + r.body.version.n + ".");
        hooks.playUrl(API + current + "/video?v=" + Date.now());
        draw(r.body);
      });
    };
    bar.appendChild(apply);
    host.appendChild(bar);
    if (info.versions.length > 1) {
      var vrow = el("div", "ng-eh3-grid");
      vrow.appendChild(el("span", "ng-gen-label", "Versions"));
      info.versions.forEach(function (v) {
        var b = el("button", "ng-gen-btn", v.n === 0 ? "Original" : "v" + v.n);
        b.type = "button"; b.style.width = "auto"; b.title = v.label;
        b.onclick = function () { hooks.playUrl(API + current + "/versions/" + v.n + "/video?v=" + Date.now()); };
        vrow.appendChild(b);
      });
      host.appendChild(vrow);
    }
    host.style.display = "";
  }

  // Show the panel for a finished render; a single-window render has nothing to reroll.
  function show(jobId) {
    if (!host) return;
    current = jobId;
    api(jobId + "/windows").then(function (r) {
      if (!r.ok) { host.style.display = "none"; return; }
      draw(r.body);
    }).catch(function () { host.style.display = "none"; });
  }

  function hide() { if (host) { host.style.display = "none"; host.textContent = ""; } }

  function init(container, h) { host = container; hooks = h; }

  window.EnhancedH3RerollNG = { init: init, show: show, hide: hide };
})();
