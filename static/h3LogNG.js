/**
 * h3LogNG.js -- the H3 view's "Job Log" panel.
 *
 * A durable history of finished H3 renders (h3_job_logsNG.py), separate
 * from the ephemeral render queue in the rail -- the queue evicts (and
 * deletes) a job's files once 20 newer jobs have finished and is lost on
 * a server restart, so this is the only place a render's prompt/seed/
 * settings, its video and its full log can still be recovered afterwards.
 * Replaces #ng-h3-compose while open; "Recent" shows the current month,
 * "Archive" drills down year -> month -> a day-entries grid using the same
 * card component. Twin of musicNG.js's Job Log panel.
 *
 * Loaded after h3NG.js (uses H3NG.loadRecipe for "Use this").
 */
(function () {
  "use strict";

  var API = "/api/ng/h3";

  var els = {};
  var bound = false;
  var logMode = "recent"; // "recent" | "years" | "months" | "days"
  var logYear = null;
  var logMonth = null;
  var logNotesTimers = {};

  function refreshEls() {
    els.toggle = document.getElementById("ng-h3-log-toggle");
    els.compose = document.getElementById("ng-h3-compose");
    els.view = document.getElementById("ng-h3-log-view");
    els.back = document.getElementById("ng-h3-log-back");
    els.tabRecent = document.getElementById("ng-h3-log-tab-recent");
    els.tabArchive = document.getElementById("ng-h3-log-tab-archive");
    els.breadcrumb = document.getElementById("ng-h3-log-breadcrumb");
    els.empty = document.getElementById("ng-h3-log-empty");
    els.list = document.getElementById("ng-h3-log-list");
    els.status = document.getElementById("ng-h3-status");
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function entryUrl(entry, tail) {
    return API + "/logs/" + encodeURIComponent(entry.date) + "/" +
      encodeURIComponent(entry.job_id) + (tail || "");
  }

  function toggleView() {
    if (!els.view || !els.compose) return;
    if (els.view.style.display === "none") {
      els.compose.style.display = "none";
      els.view.style.display = "";
      setTab("recent");
    } else {
      closeView();
    }
  }

  function closeView() {
    if (!els.view || !els.compose) return;
    els.view.style.display = "none";
    els.compose.style.display = "";
  }

  function setTab(mode) {
    logMode = mode;
    logYear = null;
    logMonth = null;
    if (els.tabRecent) els.tabRecent.classList.toggle("ng-vg-log-tab-active", mode === "recent");
    if (els.tabArchive) els.tabArchive.classList.toggle("ng-vg-log-tab-active", mode !== "recent");
    if (mode === "recent") fetchRecent(); else fetchYears();
  }

  function renderBreadcrumb() {
    if (!els.breadcrumb) return;
    els.breadcrumb.innerHTML = "";
    if (logMode === "recent" || logMode === "years") {
      els.breadcrumb.style.display = "none";
      return;
    }
    els.breadcrumb.style.display = "";
    var crumbs = [{ label: "Archive", fn: fetchYears }];
    if (logYear) crumbs.push({ label: logYear, fn: function () { fetchMonths(logYear); } });
    if (logMonth) crumbs.push({ label: logYear + "-" + logMonth, fn: null });
    crumbs.forEach(function (c, i) {
      if (i > 0) els.breadcrumb.appendChild(document.createTextNode(" / "));
      var node;
      if (c.fn) {
        node = document.createElement("a");
        node.href = "#";
        node.addEventListener("click", function (e) { e.preventDefault(); c.fn(); });
      } else {
        node = document.createElement("span");
      }
      node.textContent = c.label;
      els.breadcrumb.appendChild(node);
    });
  }

  function load(url, errMsg, onPayload) {
    if (els.list) els.list.innerHTML = "Loading&hellip;";
    fetch(url)
      .then(function (res) { return res.json(); })
      .then(onPayload)
      .catch(function () { if (els.list) els.list.textContent = errMsg; });
  }

  function fetchRecent() {
    logMode = "recent";
    renderBreadcrumb();
    load(API + "/logs", "Couldn't load the job log.", function (p) {
      renderEntries((p && p.entries) || []);
    });
  }

  function fetchYears() {
    logMode = "years";
    logYear = null;
    logMonth = null;
    renderBreadcrumb();
    load(API + "/logs/archive", "Couldn't load the archive.", function (p) {
      renderButtons((p && p.years) || [], "No archived years yet.", fetchMonths);
    });
  }

  function fetchMonths(year) {
    logMode = "months";
    logYear = year;
    logMonth = null;
    renderBreadcrumb();
    load(API + "/logs/archive/" + encodeURIComponent(year), "Couldn't load that year.", function (p) {
      renderButtons((p && p.months) || [], "No archived months in " + year + ".", function (month) {
        fetchDays(year, month);
      });
    });
  }

  function fetchDays(year, month) {
    logMode = "days";
    logYear = year;
    logMonth = month;
    renderBreadcrumb();
    load(API + "/logs/archive/" + encodeURIComponent(year) + "/" + encodeURIComponent(month),
      "Couldn't load that month.", function (p) {
        renderEntries((p && p.entries) || []);
      });
  }

  function renderButtons(items, emptyMsg, onPick) {
    if (!els.list) return;
    els.list.innerHTML = "";
    if (els.empty) {
      els.empty.textContent = emptyMsg;
      els.empty.style.display = items.length ? "none" : "";
    }
    var grid = document.createElement("div");
    grid.className = "ng-vg-log-grid";
    items.forEach(function (item) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ng-gen-btn ng-gen-btn-quiet";
      btn.textContent = item;
      btn.addEventListener("click", function () { onPick(item); });
      grid.appendChild(btn);
    });
    els.list.appendChild(grid);
  }

  function renderEntries(entries) {
    if (!els.list) return;
    els.list.innerHTML = "";
    if (els.empty) {
      els.empty.textContent = "Nothing logged here yet.";
      els.empty.style.display = entries.length ? "none" : "";
    }
    var byDate = {};
    var order = [];
    entries.forEach(function (e) {
      if (!byDate[e.date]) { byDate[e.date] = []; order.push(e.date); }
      byDate[e.date].push(e);
    });
    order.forEach(function (date) {
      var heading = document.createElement("h4");
      heading.className = "ng-vg-log-day-heading";
      heading.textContent = date;
      els.list.appendChild(heading);
      var grid = document.createElement("div");
      grid.className = "ng-vg-log-grid";
      byDate[date].forEach(function (entry) { grid.appendChild(buildCard(entry)); });
      els.list.appendChild(grid);
    });
  }

  function useEntry(entry) {
    if (!window.H3NG || !window.H3NG.loadRecipe) return;
    window.H3NG.loadRecipe(entry, entry.has_ref ? entryUrl(entry, "/ref") : null);
    closeView();
  }

  function deleteEntry(entry, card, btn) {
    if (!confirm("Delete this job log entry? This deletes its video and log files permanently.")) return;
    btn.disabled = true;
    btn.textContent = "Deleting…";
    fetch(entryUrl(entry), { method: "DELETE" })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, payload: p }; }); })
      .then(function (r) {
        if (r.ok) { card.remove(); return; }
        btn.disabled = false;
        btn.textContent = "🗑 Delete";
        setStatus("Could not delete: " + ((r.payload && r.payload.error) || "unknown error"));
      })
      .catch(function (e) {
        btn.disabled = false;
        btn.textContent = "🗑 Delete";
        setStatus("Could not delete: " + e.message);
      });
  }

  function saveNotes(entry, notes) {
    fetch(entryUrl(entry, "/notes"), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: notes }),
    }).catch(function () { /* best-effort -- notes stay in the textarea either way */ });
  }

  // Toggleable full-log view under the card: fetched on first open so a
  // long month's cards don't each pull their log text up front.
  function buildLogToggle(entry) {
    var wrap = document.createElement("div");
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ng-gen-btn ng-gen-btn-quiet";
    btn.textContent = "Show log";
    var pre = document.createElement("pre");
    pre.className = "ng-vg-mainlog";
    pre.style.display = "none";
    pre.style.maxHeight = "240px";
    var loaded = false;
    btn.addEventListener("click", function () {
      var open = pre.style.display === "none";
      pre.style.display = open ? "" : "none";
      btn.textContent = open ? "Hide log" : "Show log";
      if (open && !loaded) {
        loaded = true;
        pre.textContent = "Loading…";
        fetch(entryUrl(entry, "/text"))
          .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
          .then(function (t) { pre.textContent = t || "(empty log)"; })
          .catch(function (e) { loaded = false; pre.textContent = "Couldn't load the log: " + e.message; });
      }
    });
    wrap.appendChild(btn);
    wrap.appendChild(pre);
    return wrap;
  }

  function buildCard(entry) {
    var card = document.createElement("div");
    card.className = "ng-vg-log-card";

    var status = document.createElement("span");
    status.className = "ng-vg-log-status ng-vg-log-status-" + entry.status;
    status.textContent = entry.status;
    card.appendChild(status);

    if (entry.prompt) {
      var prompt = document.createElement("div");
      prompt.className = "ng-vg-log-prompt";
      prompt.textContent = entry.prompt;
      card.appendChild(prompt);
    }

    var meta = document.createElement("div");
    meta.className = "ng-vg-log-meta";
    var when = entry.finished_at ? new Date(entry.finished_at * 1000).toLocaleTimeString() : "";
    var took = (entry.dispatched_at && entry.finished_at)
      ? Math.round((entry.finished_at - entry.dispatched_at) / 60) + " min" : "";
    meta.textContent = (entry.model || "") + (entry.turbo ? " turbo" : "") +
      (entry.width ? ", " + entry.width + "×" + entry.height : "") +
      (entry.steps != null ? ", " + entry.steps + " steps" : "") +
      (entry.duration_s != null ? ", " + entry.duration_s + "s" : "") +
      (entry.seed != null ? ", seed " + entry.seed : "") +
      (took ? ", took " + took : "") +
      (when ? ", " + when : "");
    card.appendChild(meta);

    if (entry.error) {
      var err = document.createElement("div");
      err.className = "ng-vg-log-error";
      err.textContent = entry.error;
      card.appendChild(err);
    }

    if (entry.has_video) {
      var video = document.createElement("video");
      video.src = entryUrl(entry, "/video");
      video.controls = true;
      video.preload = "metadata";
      video.style.width = "100%";
      card.appendChild(video);
    }

    var actions = document.createElement("div");
    actions.className = "ng-vg-log-actions";
    var useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    useBtn.textContent = "↺ Use this";
    useBtn.addEventListener("click", function () { useEntry(entry); });
    actions.appendChild(useBtn);
    if (entry.has_video) {
      var dl = document.createElement("a");
      dl.href = entryUrl(entry, "/video");
      dl.download = "h3-" + entry.job_id + ".mp4";
      dl.className = "ng-gen-btn ng-gen-btn-quiet";
      dl.textContent = "Download";
      actions.appendChild(dl);
    }
    var deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    deleteBtn.textContent = "🗑 Delete";
    deleteBtn.addEventListener("click", function () { deleteEntry(entry, card, deleteBtn); });
    actions.appendChild(deleteBtn);
    card.appendChild(actions);

    card.appendChild(buildLogToggle(entry));

    var notes = document.createElement("textarea");
    notes.className = "ng-vg-log-notes";
    notes.rows = 2;
    notes.placeholder = "Notes...";
    notes.value = entry.notes || "";
    notes.addEventListener("input", function () {
      var key = entry.date + "/" + entry.job_id;
      if (logNotesTimers[key]) clearTimeout(logNotesTimers[key]);
      logNotesTimers[key] = setTimeout(function () { saveNotes(entry, notes.value); }, 800);
    });
    card.appendChild(notes);

    return card;
  }

  function bind() {
    refreshEls();
    if (bound || !els.toggle) return;
    bound = true;
    els.toggle.addEventListener("click", toggleView);
    if (els.back) els.back.addEventListener("click", closeView);
    if (els.tabRecent) els.tabRecent.addEventListener("click", function () { setTab("recent"); });
    if (els.tabArchive) els.tabArchive.addEventListener("click", function () { setTab("years"); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
