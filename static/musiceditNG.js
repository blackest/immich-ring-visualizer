/**
 * musiceditNG.js -- the "Music Edit" view (bottom-bar Music Edit task).
 *
 * Splits the ABC load/edit/save/render tools that used to sit as a
 * standalone panel on the Music page (#ng-music-abc-tools) out into
 * their own workflow, per the "music page is overloaded" call: this is
 * a single-item edit-then-render surface, not a queue, so its rail pane
 * (#ng-musicedit-pane) stays a one-line hint and every control lives in
 * the main stage (#ng-musicedit-main).
 *
 * Reuses musicNG.js's toggleAbcRender/downloadAbcAsMidi/printAbcAsPdf
 * via window.MusicNG (see musicNG.js's own export) instead of
 * duplicating that render/MIDI/PDF core -- they're fully generic
 * (wrap/btn/getText callbacks), so this file only supplies its own
 * textarea instead of a fetched queue-row URL.
 *
 * The one genuinely new capability here: "Render audio from this
 * score" actually feeds the (BPM/Key-patched) ABC text to the model as
 * a real conditioning input via POST /api/ng/music/generate's `abc`
 * field -> generate_music_ng's `--abc` (see music_engineNG.py). BPM/Key
 * are parsed from the loaded score's own Q:/K: header lines on load,
 * and rewrite that line in the textarea on edit -- the textarea is
 * always the single source of truth sent to the server.
 *
 * Loaded after musicNG.js and before bootstrapWiringNG.js (which fires
 * the first ProjectManager.render(), which calls MusicEditNG.sync()).
 */
(function () {
  "use strict";

  var API = "/api/ng/music";
  var POLL_MS = 3000;

  var inited = false;
  var loadedFileName = null;
  var currentJobId = null;
  var pollTimer = null;
  var els = {};

  // ---- "Write lyrics (AI)" rail tool -- a local Ollama model (picked by
  // name, not this app's own choice) writing into the main stage's Lyrics
  // box. Same lyricist system prompt Phosphene's own Gemma write_lyrics
  // action uses (mlx_warm_helper.py); reuses Chat's existing
  // /api/ng/chat/models + /api/ng/chat/send rather than a new backend
  // route -- this is just a different system prompt over the same Ollama
  // proxy Chat already has. ----
  var aiModels = [];
  var aiModelsLoaded = false;
  var aiSystemMode = "write"; // "write" | "revise" -- which default's currently loaded
  var aiChatMessages = []; // {role: "user"|"assistant", content, pending?, error?}

  // ---- Job Log panel state -- twin of musicNG.js's own panel, same
  // shared history (music_job_logsNG.py / /api/ng/music/logs*), just a
  // different "use this" action (load into this page's own textarea
  // instead of jumping to Music). See "Job Log panel" section below. ----
  var logMode = "recent"; // "recent" | "years" | "months" | "days"
  var logYear = null;
  var logMonth = null;
  var logNotesTimers = {}; // job_id -> debounce timer for the notes PATCH

  function refreshEls() {
    els.pane = document.getElementById("ng-musicedit-pane");
    els.main = document.getElementById("ng-musicedit-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.editor = document.getElementById("ng-musicedit-editor");

    els.logToggle = document.getElementById("ng-musicedit-log-toggle");
    els.logView = document.getElementById("ng-musicedit-log-view");
    els.logBack = document.getElementById("ng-musicedit-log-back");
    els.logTabRecent = document.getElementById("ng-musicedit-log-tab-recent");
    els.logTabArchive = document.getElementById("ng-musicedit-log-tab-archive");
    els.logBreadcrumb = document.getElementById("ng-musicedit-log-breadcrumb");
    els.logEmpty = document.getElementById("ng-musicedit-log-empty");
    els.logList = document.getElementById("ng-musicedit-log-list");

    els.useInMusicBtn = document.getElementById("ng-musicedit-use-in-music-btn");

    els.fileWrap = document.getElementById("ng-musicedit-file-wrap");
    els.file = document.getElementById("ng-musicedit-file");
    els.fileBtn = document.getElementById("ng-musicedit-file-btn");
    els.fileEmpty = document.getElementById("ng-musicedit-file-empty");
    els.fileName = document.getElementById("ng-musicedit-file-name");
    els.saveBtn = document.getElementById("ng-musicedit-save-btn");
    els.text = document.getElementById("ng-musicedit-text");
    els.selectAll = document.getElementById("ng-musicedit-select-all");

    els.bpm = document.getElementById("ng-musicedit-bpm");
    els.key = document.getElementById("ng-musicedit-key");

    els.staffToggle = document.getElementById("ng-musicedit-staff-toggle");
    els.tabToggle = document.getElementById("ng-musicedit-tab-toggle");
    els.midiBtn = document.getElementById("ng-musicedit-midi-btn");
    els.pdfBtn = document.getElementById("ng-musicedit-pdf-btn");
    els.tabPdfBtn = document.getElementById("ng-musicedit-tab-pdf-btn");
    els.staffWrap = document.getElementById("ng-musicedit-staff-wrap");
    els.tabWrap = document.getElementById("ng-musicedit-tab-wrap");

    els.style = document.getElementById("ng-musicedit-style");
    els.lyrics = document.getElementById("ng-musicedit-lyrics");
    els.lyricsExpand = document.getElementById("ng-musicedit-lyrics-expand");
    els.lyricsSelectAll = document.getElementById("ng-musicedit-lyrics-select-all");
    els.duration = document.getElementById("ng-musicedit-duration");
    els.durationVal = document.getElementById("ng-musicedit-duration-val");
    els.seed = document.getElementById("ng-musicedit-seed");
    els.precision = document.getElementById("ng-musicedit-precision");
    els.temperature = document.getElementById("ng-musicedit-temperature");
    els.temperatureVal = document.getElementById("ng-musicedit-temperature-val");
    els.renderBtn = document.getElementById("ng-musicedit-render-btn");
    els.status = document.getElementById("ng-musicedit-status");
    els.audio = document.getElementById("ng-musicedit-audio");

    els.aiModel = document.getElementById("ng-musicedit-ai-model");
    els.aiConcept = document.getElementById("ng-musicedit-ai-concept");
    els.aiConceptExpand = document.getElementById("ng-musicedit-ai-concept-expand");
    els.aiSystem = document.getElementById("ng-musicedit-ai-system");
    els.aiSystemExpand = document.getElementById("ng-musicedit-ai-system-expand");
    els.aiSystemReset = document.getElementById("ng-musicedit-ai-system-reset");
    els.aiWriteBtn = document.getElementById("ng-musicedit-ai-write");
    els.aiRevise = document.getElementById("ng-musicedit-ai-revise");
    els.aiStatus = document.getElementById("ng-musicedit-ai-status");

    els.aiChatTranscript = document.getElementById("ng-musicedit-ai-chat-transcript");
    els.aiChatExpand = document.getElementById("ng-musicedit-ai-chat-expand");
    els.aiChatClear = document.getElementById("ng-musicedit-ai-chat-clear");
    els.aiChatInput = document.getElementById("ng-musicedit-ai-chat-input");
    els.aiChatSend = document.getElementById("ng-musicedit-ai-chat-send");
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function formatDuration(sec) {
    sec = Math.round(sec);
    var m = Math.floor(sec / 60);
    var s = sec % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function getText() {
    return Promise.resolve(els.text ? els.text.value : "");
  }

  function getLyrics() {
    return els.lyrics ? els.lyrics.value : "";
  }

  function filenameBase() {
    return (loadedFileName || "score").replace(/\.abc$/i, "");
  }

  function resetRenders() {
    [els.staffWrap, els.tabWrap].forEach(function (wrap) {
      if (!wrap) return;
      wrap.dataset.loaded = "";
      wrap.style.display = "none";
    });
    if (els.staffToggle) els.staffToggle.textContent = "Show sheet music";
    if (els.tabToggle) els.tabToggle.textContent = "Show guitar tab";
  }

  // ---- BPM/Key <-> the textarea's own Q:/K: header lines ----

  function parseAbcHeader(text) {
    var q = /^Q:\s*(?:\S*=)?\s*(\d+)/m.exec(text || "");
    var k = /^K:\s*(\S+)/m.exec(text || "");
    return { bpm: q ? q[1] : "", key: k ? k[1] : "" };
  }

  function applyLoadedHeader() {
    var parsed = parseAbcHeader(els.text.value);
    if (els.bpm) els.bpm.value = parsed.bpm;
    if (els.key) els.key.value = parsed.key;
  }

  function rewriteTempo(text, bpm) {
    if (!bpm) return text;
    var line = /^Q:.*$/m;
    var existingPrefix = /^Q:\s*(\S*=)/m.exec(text);
    var newLine = "Q:" + (existingPrefix ? existingPrefix[1] : "1/4=") + bpm;
    if (line.test(text)) return text.replace(line, newLine);
    var kLine = /^K:.*$/m.exec(text);
    return kLine ? text.replace(kLine[0], newLine + "\n" + kLine[0]) : newLine + "\n" + text;
  }

  function rewriteKey(text, key) {
    if (!key) return text;
    var line = /^K:.*$/m;
    var newLine = "K:" + key;
    if (line.test(text)) return text.replace(line, newLine);
    return text + (/\n$/.test(text) ? "" : "\n") + newLine + "\n";
  }

  function onBpmChange() {
    if (!els.text || !els.bpm) return;
    els.text.value = rewriteTempo(els.text.value, (els.bpm.value || "").trim());
    resetRenders();
  }

  function onKeyChange() {
    if (!els.text || !els.key) return;
    els.text.value = rewriteKey(els.text.value, (els.key.value || "").trim());
    resetRenders();
  }

  // ---- file load/save ----

  function applyLoadedText(text, name, statusMsg) {
    els.text.value = text || "";
    loadedFileName = name || null;
    if (els.fileName) {
      els.fileName.textContent = name || "";
      els.fileName.style.display = name ? "" : "none";
    }
    if (els.fileEmpty) els.fileEmpty.style.display = name ? "none" : "";
    applyLoadedHeader();
    resetRenders();
    setStatus(statusMsg || "");
  }

  // Called from musicNG.js's own "-> Edit score in Music Edit" button
  // (a finished queue row or Job Log card) -- same page, no navigation,
  // so this just hands the already-fetched score text straight to the
  // textarea instead of round-tripping through a file input.
  function loadAbcText(text, filenameBase) {
    refreshEls();
    if (!els.text) return;
    applyLoadedText(text, filenameBase ? filenameBase + ".abc" : null,
      "Loaded from " + (filenameBase || "the job log") + ".");
  }

  function onFileChosen() {
    var f = els.file.files && els.file.files[0];
    if (!f) return;
    var reader = new FileReader();
    reader.onload = function () {
      applyLoadedText(String(reader.result || ""), f.name, "Loaded " + f.name + ".");
    };
    reader.onerror = function () {
      setStatus("Could not read file: " + (reader.error && reader.error.message));
    };
    reader.readAsText(f);
  }

  // Same drag-and-drop dropzone pattern as musicNG.js's cover-file
  // picker -- iOS's file-picker sheet filters accept= unreliably, so
  // dragging a file in from Files/Photos sidesteps that step entirely.
  function wireDropzone(zone, input, onFiles) {
    if (!zone || !input) return;
    ["dragenter", "dragover"].forEach(function (evt) {
      zone.addEventListener(evt, function (e) {
        e.preventDefault();
        zone.classList.add("ng-dropzone-active");
      });
    });
    ["dragleave", "dragend", "drop"].forEach(function (evt) {
      zone.addEventListener(evt, function () {
        zone.classList.remove("ng-dropzone-active");
      });
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) {
        input.files = files;
        onFiles();
      }
    });
  }

  function onNoteClick(abcelem) {
    if (!els.text || !abcelem) return;
    var start = abcelem.startChar;
    var end = abcelem.endChar;
    if (typeof start !== "number" || typeof end !== "number" || end <= start) return;
    els.text.focus();
    els.text.setSelectionRange(start, end);
    var before = els.text.value.slice(0, start);
    var lineNum = before.split("\n").length;
    var lineHeight = parseFloat(getComputedStyle(els.text).lineHeight) || 16;
    els.text.scrollTop = Math.max(0, (lineNum - 3) * lineHeight);
  }

  function saveFile() {
    var text = els.text.value;
    if (!text.trim()) {
      setStatus("Nothing to save -- load or type ABC first.");
      return;
    }
    var blob = new Blob([text], { type: "text/plain" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filenameBase() + ".abc";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus("Saved " + a.download + ".");
  }

  // ---- render from this score ----

  function stopPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function pollRenderJob() {
    if (!currentJobId) return;
    fetch(API + "/jobs/" + currentJobId)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        if (!job || !job.status) return;
        if (job.status === "completed") {
          stopPolling();
          setStatus("Done.");
          if (els.audio && job.audio_url) {
            els.audio.src = job.audio_url + "?v=" + Date.now();
            els.audio.style.display = "";
          }
        } else if (job.status === "failed") {
          stopPolling();
          setStatus("Render failed: " + (job.error || "unknown error"));
        } else {
          setStatus("Rendering (" + job.status + ")\u2026");
        }
      })
      .catch(function () { /* keep polling -- a transient fetch error isn't fatal */ });
  }

  function renderFromScore() {
    var abc = (els.text.value || "").trim();
    if (!abc) {
      setStatus("Load or paste an ABC score first.");
      return;
    }
    var style = (els.style && els.style.value || "").trim();
    var lyrics = (els.lyrics && els.lyrics.value || "").trim();
    var durationS = parseFloat(els.duration.value) || 240;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    var precision = els.precision ? els.precision.value : "8bit";
    var temperature = els.temperature ? parseFloat(els.temperature.value) : null;

    var body = { abc: abc, style: style, lyrics: lyrics, duration_s: durationS, precision: precision };
    if (seed !== null && !isNaN(seed)) body.seed = seed;
    if (temperature !== null && !isNaN(temperature)) body.temperature = temperature;

    els.renderBtn.disabled = true;
    if (els.audio) els.audio.style.display = "none";
    setStatus("Submitting\u2026");

    fetch(API + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json().then(function (payload) { return { ok: res.ok, payload: payload }; });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          currentJobId = r.payload.job_id;
          setStatus("Queued\u2026");
          stopPolling();
          pollTimer = setInterval(pollRenderJob, POLL_MS);
          pollRenderJob();
        } else {
          setStatus("Could not start render: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        setStatus("Could not start render: " + e.message);
      })
      .then(function () {
        els.renderBtn.disabled = false;
      });
  }

  // ---- send this score back to the Music page ----

  // Reverse of musicNG.js's own sendScoreToMusicEdit -- hands the
  // (BPM/Key-patched) textarea text, plus whatever style/lyrics are
  // sitting in this page's render panel, to Music's own Compose form
  // via window.MusicNG.useAbcScore, then switches the active task so
  // the user lands there with it ready. Lets a score edited here take
  // advantage of Music's queue/cover/Job Log tooling instead of this
  // page's single-shot render.
  function useInMusic() {
    var abc = (els.text.value || "").trim();
    if (!abc) {
      setStatus("Load or paste an ABC score first.");
      return;
    }
    if (!window.MusicNG || !window.MusicNG.useAbcScore) {
      setStatus("Music page isn't available.");
      return;
    }
    var style = (els.style && els.style.value || "").trim();
    var lyrics = (els.lyrics && els.lyrics.value || "").trim();
    window.MusicNG.useAbcScore(abc, filenameBase(), style, lyrics);
    if (window.ProjectManager) window.ProjectManager.setTask("music");
  }

  // ---- Job Log panel ----
  // Durable history of finished Music renders (music_job_logsNG.py),
  // shared with the Music page's own panel -- see routes/musicNG.py's
  // /logs* routes. Twin of musicNG.js's own toggleLogView/setLogTab/
  // fetchLog*/renderLog*/buildLogCard; the only real difference is
  // useLogEntry below, which loads into this page's own fields instead
  // of jumping to another task.

  function toggleLogView() {
    if (!els.logView || !els.editor) return;
    var opening = els.logView.style.display === "none";
    if (opening) {
      els.editor.style.display = "none";
      els.logView.style.display = "";
      setLogTab("recent");
    } else {
      closeLogView();
    }
  }

  function closeLogView() {
    if (!els.logView || !els.editor) return;
    els.logView.style.display = "none";
    els.editor.style.display = "";
  }

  function setLogTab(mode) {
    logMode = mode;
    logYear = null;
    logMonth = null;
    if (els.logTabRecent) els.logTabRecent.classList.toggle("ng-vg-log-tab-active", mode === "recent");
    if (els.logTabArchive) els.logTabArchive.classList.toggle("ng-vg-log-tab-active", mode !== "recent");
    if (mode === "recent") {
      fetchLogRecent();
    } else {
      fetchLogArchiveYears();
    }
  }

  function renderLogBreadcrumb() {
    if (!els.logBreadcrumb) return;
    if (logMode === "recent" || logMode === "years") {
      els.logBreadcrumb.style.display = "none";
      els.logBreadcrumb.innerHTML = "";
      return;
    }
    els.logBreadcrumb.style.display = "";
    els.logBreadcrumb.innerHTML = "";
    var crumbs = [{ label: "Archive", fn: function () { fetchLogArchiveYears(); } }];
    if (logYear) {
      crumbs.push({ label: logYear, fn: function () { fetchLogArchiveMonths(logYear); } });
    }
    if (logMonth) {
      crumbs.push({ label: logYear + "-" + logMonth, fn: null });
    }
    crumbs.forEach(function (c, i) {
      if (i > 0) els.logBreadcrumb.appendChild(document.createTextNode(" / "));
      if (c.fn) {
        var a = document.createElement("a");
        a.href = "#";
        a.textContent = c.label;
        a.addEventListener("click", function (e) { e.preventDefault(); c.fn(); });
        els.logBreadcrumb.appendChild(a);
      } else {
        var span = document.createElement("span");
        span.textContent = c.label;
        els.logBreadcrumb.appendChild(span);
      }
    });
  }

  function fetchLogRecent() {
    logMode = "recent";
    renderLogBreadcrumb();
    if (els.logList) els.logList.innerHTML = "Loading&hellip;";
    fetch(API + "/logs")
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        renderLogEntries((payload && payload.entries) || []);
      })
      .catch(function () {
        if (els.logList) els.logList.textContent = "Couldn't load the job log.";
      });
  }

  function fetchLogArchiveYears() {
    logMode = "years";
    logYear = null;
    logMonth = null;
    renderLogBreadcrumb();
    if (els.logList) els.logList.innerHTML = "Loading&hellip;";
    fetch(API + "/logs/archive")
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        renderLogButtons((payload && payload.years) || [], "No archived years yet.", function (year) {
          fetchLogArchiveMonths(year);
        });
      })
      .catch(function () {
        if (els.logList) els.logList.textContent = "Couldn't load the archive.";
      });
  }

  function fetchLogArchiveMonths(year) {
    logMode = "months";
    logYear = year;
    logMonth = null;
    renderLogBreadcrumb();
    if (els.logList) els.logList.innerHTML = "Loading&hellip;";
    fetch(API + "/logs/archive/" + encodeURIComponent(year))
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        renderLogButtons((payload && payload.months) || [], "No archived months in " + year + ".", function (month) {
          fetchLogArchiveDays(year, month);
        });
      })
      .catch(function () {
        if (els.logList) els.logList.textContent = "Couldn't load that year.";
      });
  }

  function fetchLogArchiveDays(year, month) {
    logMode = "days";
    logYear = year;
    logMonth = month;
    renderLogBreadcrumb();
    if (els.logList) els.logList.innerHTML = "Loading&hellip;";
    fetch(API + "/logs/archive/" + encodeURIComponent(year) + "/" + encodeURIComponent(month))
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        renderLogEntries((payload && payload.entries) || []);
      })
      .catch(function () {
        if (els.logList) els.logList.textContent = "Couldn't load that month.";
      });
  }

  function renderLogButtons(items, emptyMsg, onPick) {
    if (!els.logList) return;
    els.logList.innerHTML = "";
    if (els.logEmpty) els.logEmpty.style.display = items.length ? "none" : "";
    if (els.logEmpty) els.logEmpty.textContent = emptyMsg;
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
    els.logList.appendChild(grid);
  }

  function renderLogEntries(entries) {
    if (!els.logList) return;
    els.logList.innerHTML = "";
    if (els.logEmpty) {
      els.logEmpty.textContent = "Nothing logged here yet.";
      els.logEmpty.style.display = entries.length ? "none" : "";
    }

    var byDate = {};
    var order = [];
    entries.forEach(function (e) {
      if (!byDate[e.date]) {
        byDate[e.date] = [];
        order.push(e.date);
      }
      byDate[e.date].push(e);
    });

    order.forEach(function (date) {
      var heading = document.createElement("h4");
      heading.className = "ng-vg-log-day-heading";
      heading.textContent = date;
      els.logList.appendChild(heading);

      var grid = document.createElement("div");
      grid.className = "ng-vg-log-grid";
      byDate[date].forEach(function (entry) {
        grid.appendChild(buildLogCard(entry));
      });
      els.logList.appendChild(grid);
    });
  }

  // Loads style/lyrics/seed/etc straight into this page's own render
  // panel, and -- when the entry has a score -- fetches it into the
  // main textarea via applyLoadedText, same as loading a file from
  // disk. Unlike musicNG.js's own useLogEntry, there's no task-switch:
  // we're already on Music Edit.
  function useLogEntry(entry) {
    if (els.style) els.style.value = entry.style || "";
    if (els.lyrics) els.lyrics.value = entry.lyrics || "";
    if (els.seed) els.seed.value = entry.seed != null ? entry.seed : "";
    if (els.precision && entry.precision) els.precision.value = entry.precision;
    if (els.temperature && els.temperatureVal && entry.temperature != null) {
      els.temperature.value = entry.temperature;
      els.temperatureVal.textContent = parseFloat(entry.temperature).toFixed(2);
    }
    if (typeof entry.duration_s === "number" && els.duration) {
      els.duration.value = entry.duration_s;
      if (els.durationVal) els.durationVal.textContent = formatDuration(els.duration.value);
    }
    closeLogView();

    if (!entry.has_score) {
      setStatus("Loaded style/lyrics from the job log -- this entry has no score attached.");
      return;
    }
    var scoreUrl = API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/score";
    fetch(scoreUrl)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (text) {
        applyLoadedText(text, "music-" + entry.job_id + ".abc",
          "Loaded score + style/lyrics from the job log.");
      })
      .catch(function (e) {
        setStatus("Loaded style/lyrics, but couldn't fetch the score: " + e.message);
      });
  }

  function deleteLogEntry(entry, card, btn) {
    if (!confirm("Delete this job log entry? This deletes its audio/score files permanently.")) return;
    btn.disabled = true;
    btn.textContent = "Deleting…";
    fetch(API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id), {
      method: "DELETE",
    })
      .then(function (res) { return res.json().then(function (p) { return { ok: res.ok, payload: p }; }); })
      .then(function (r) {
        if (r.ok) {
          card.remove();
        } else {
          btn.disabled = false;
          btn.textContent = "🗑 Delete";
          setStatus("Could not delete: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        btn.disabled = false;
        btn.textContent = "🗑 Delete";
        setStatus("Could not delete: " + e.message);
      });
  }

  function saveLogNotes(entry, notes) {
    fetch(API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/notes", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: notes }),
    }).catch(function () { /* best-effort -- notes stay in the textarea either way */ });
  }

  function buildLogCard(entry) {
    var card = document.createElement("div");
    card.className = "ng-vg-log-card";

    var status = document.createElement("span");
    status.className = "ng-vg-log-status ng-vg-log-status-" + entry.status;
    status.textContent = entry.status;
    card.appendChild(status);

    if (entry.style) {
      var style = document.createElement("div");
      style.className = "ng-vg-log-prompt";
      style.textContent = entry.style;
      card.appendChild(style);
    }
    if (entry.lyrics) {
      var lyrics = document.createElement("div");
      lyrics.className = "ng-vg-log-prompt";
      lyrics.style.whiteSpace = "pre-wrap";
      lyrics.textContent = entry.lyrics;
      card.appendChild(lyrics);
    }

    var meta = document.createElement("div");
    meta.className = "ng-vg-log-meta";
    var when = entry.finished_at ? new Date(entry.finished_at * 1000).toLocaleTimeString() : "";
    meta.textContent = (entry.is_cover ? "cover" + (entry.task ? " (" + entry.task + ")" : "") : (entry.mode || "full") + (entry.instrumental ? ", instrumental" : "")) +
      (entry.duration_s != null ? ", " + Math.round(entry.audio_seconds || entry.duration_s) + "s" : "") +
      (entry.resolved_seed != null ? ", seed " + entry.resolved_seed : "") +
      (entry.precision ? ", " + entry.precision : "") +
      (entry.temperature != null ? ", temp " + entry.temperature : "") +
      (when ? ", " + when : "");
    card.appendChild(meta);

    if (entry.error) {
      var err = document.createElement("div");
      err.className = "ng-vg-log-error";
      err.textContent = entry.error;
      card.appendChild(err);
    }

    if (entry.has_audio) {
      var audio = document.createElement("audio");
      audio.src = API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/audio";
      audio.controls = true;
      card.appendChild(audio);
    }

    var actions = document.createElement("div");
    actions.className = "ng-vg-log-actions";
    var useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    useBtn.textContent = "↺ Use this";
    useBtn.addEventListener("click", function () { useLogEntry(entry); });
    actions.appendChild(useBtn);
    if (entry.has_score) {
      var scoreLink = document.createElement("a");
      var scoreUrl = API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/score";
      scoreLink.href = scoreUrl;
      scoreLink.download = "music-" + entry.job_id + ".abc";
      scoreLink.className = "ng-gen-btn ng-gen-btn-quiet";
      scoreLink.textContent = "Score (.abc)";
      actions.appendChild(scoreLink);
    }
    var deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    deleteBtn.textContent = "🗑 Delete";
    deleteBtn.addEventListener("click", function () { deleteLogEntry(entry, card, deleteBtn); });
    actions.appendChild(deleteBtn);
    card.appendChild(actions);

    var notes = document.createElement("textarea");
    notes.className = "ng-vg-log-notes";
    notes.rows = 2;
    notes.placeholder = "Notes...";
    notes.value = entry.notes || "";
    notes.addEventListener("click", function (e) { e.stopPropagation(); });
    notes.addEventListener("input", function () {
      var key = entry.date + "/" + entry.job_id;
      if (logNotesTimers[key]) clearTimeout(logNotesTimers[key]);
      logNotesTimers[key] = setTimeout(function () {
        saveLogNotes(entry, notes.value);
      }, 800);
    });
    card.appendChild(notes);

    return card;
  }

  function setAiStatus(msg, isError) {
    if (!els.aiStatus) return;
    els.aiStatus.style.display = msg ? "" : "none";
    els.aiStatus.textContent = msg || "";
    els.aiStatus.style.color = isError ? "var(--ng-danger)" : "";
  }

  function loadAiModels() {
    if (aiModelsLoaded || !els.aiModel) return;
    aiModelsLoaded = true;
    fetch("/api/ng/chat/models")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || d.error || !Array.isArray(d.models)) {
          aiModelsLoaded = false; // let a later sync() retry
          els.aiModel.innerHTML = "";
          var o = document.createElement("option");
          o.value = "";
          o.textContent = "Ollama unavailable";
          els.aiModel.appendChild(o);
          els.aiModel.disabled = true;
          return;
        }
        aiModels = d.models || [];
        els.aiModel.innerHTML = "";
        if (!aiModels.length) {
          var e = document.createElement("option");
          e.value = "";
          e.textContent = "No models installed";
          els.aiModel.appendChild(e);
          els.aiModel.disabled = true;
          return;
        }
        els.aiModel.disabled = false;
        aiModels.forEach(function (m) {
          var opt = document.createElement("option");
          opt.value = m.name;
          var size = m.parameter_size ? " · " + m.parameter_size : "";
          opt.textContent = m.name + size;
          els.aiModel.appendChild(opt);
        });
      })
      .catch(function () {
        aiModelsLoaded = false;
      });
  }

  // Same craft/format lyricist prompt as Phosphene's own Gemma
  // write_lyrics action (mlx_warm_helper.py) -- full-song mode only, no
  // per-section rewrite here. {verses} mirrors that same
  // length -> verse-count table: 2 under 2min, 3 under 3:30, else 4.
  // Shared by write/revise -- the "don't sanitize" instructions John
  // supplied after seeing the write-mode default too readily reach for
  // storms/shadows/generic-hope language on anything heavy. Same root
  // problem as the whole reason this feature routes through a picked
  // Ollama model instead of a fixed Gemma call: a model's own instinct to
  // soften charged material needs to be explicitly overridden, not just
  // hoped away.
  var HONESTY_BLOCK = [
    "IMPORTANT -- TREAT THE SUBJECT HONESTLY:",
    "- Do not sanitise, soften or sentimentalise the subject.",
    "- Preserve uncomfortable details from the source material when they matter.",
    "- Prefer specific, uncomfortable reality over vague emotional language.",
    "- Do not automatically replace difficult subjects with euphemisms.",
    "- Do not turn poverty, addiction, neglect, abuse, institutional failure",
    "  or desperation into generic metaphors about darkness, storms or shadows.",
    "- People are allowed to be flawed, angry, frightened, selfish,",
    "  contradictory or unpleasant. Do not make everyone noble or inspirational.",
    "- Institutions and people can be criticised when the material supports it.",
    "  Let the criticism emerge from what happens rather than turning the",
    "  song into a political speech.",
    "- Do not manufacture hope simply because the song is sad. If hope exists,",
    "  let it be small, uncertain and earned.",
    "- Do not clean up the language merely to make the song more commercially",
    "  comfortable. A blunt line can be more powerful than a polished one.",
    "- Do not confuse \"poetic\" with \"beautiful.\" Sometimes the strongest line",
    "  is the one that simply tells the uncomfortable truth.",
    "- Do not add generic inspirational language that is absent from the",
    "  source material.",
    "- If the source material is angry, let it sound angry. If it is ugly,",
    "  let it be ugly. If it is unresolved, leave it unresolved.",
    "- The goal is not to make the subject more pleasant -- it's to make the",
    "  listener feel they're inside the situation, seeing the people, hearing",
    "  the conversations, understanding what's actually at stake.",
  ];

  function buildLyricsSystemPrompt(style, seconds) {
    var verses = seconds < 120 ? 2 : seconds < 210 ? 3 : 4;
    var lines = [
      "You are a professional lyricist writing for a song generator.",
      "Write complete, singable lyrics for the concept the user gives you.",
      "",
      "FORMAT -- follow exactly, the generator parses it:",
      "- Section tags on their own line in square brackets: [Intro], [Verse],",
      "  [Pre-Chorus], [Chorus], [Bridge], [Outro]. Use [Verse] and [Chorus]",
      "  as the tag text every time (no numbering like [Verse 2]).",
      "- Under each tag, the sung lines, one per row. Short lines: 4 to 9 words.",
      "- A tag with NO lines under it is an instrumental passage. Use that for",
      "  [Intro] and [Outro] unless the concept calls for words there.",
      "- Length: about " + verses + " verses, a chorus that returns after each,",
      "  optionally one bridge. Nothing else.",
      "",
      "STYLE:",
      "- Concrete images over abstractions. No clichés about hearts on fire,",
      "  broken wings, or the night sky unless the user asked for them.",
      "- The chorus is the hook: repeatable, memorable, the same words each",
      "  time it returns.",
      "- Rhyme when it lands naturally; never force a rhyme with a weak line.",
      style ? "- Match this musical style: " + style + "." : "- Fit a contemporary song.",
      "- Write in the language of the concept.",
      "",
    ].concat(HONESTY_BLOCK, [
      "- You are not required to rewrite every line the user's concept already",
      "  gave you. If something in it is already strong, specific and singable,",
      "  keep it. Improve only what actually needs improving.",
      "",
      "OUTPUT: only the lyrics with their tags. No title, no commentary,",
      "no quotation marks, no explanation before or after.",
    ]);
    return lines.join("\n");
  }

  // Twin of buildLyricsSystemPrompt for the Revise action -- the user
  // turn carries the current lyrics + what to change; this keeps
  // whatever the instructions don't mention instead of writing fresh.
  function buildReviseSystemPrompt(style, seconds) {
    var lines = [
      "You are a professional lyricist revising an existing song.",
      "The user's message has the current lyrics and instructions for what",
      "to change. Keep whatever the instructions don't mention -- this is a",
      "revision, not a rewrite from scratch.",
      "",
      "FORMAT -- follow exactly, the generator parses it:",
      "- Section tags on their own line in square brackets: [Intro], [Verse],",
      "  [Pre-Chorus], [Chorus], [Bridge], [Outro]. Use [Verse] and [Chorus]",
      "  as the tag text every time (no numbering like [Verse 2]).",
      "- Under each tag, the sung lines, one per row. Short lines: 4 to 9 words.",
      "- A tag with NO lines under it is an instrumental passage.",
      "- Keep the same overall structure and length as the current lyrics",
      "  unless the instructions ask for a different length.",
      "",
      "STYLE:",
      "- Concrete images over abstractions. No clichés about hearts on fire,",
      "  broken wings, or the night sky unless the instructions ask for them.",
      "- The chorus is the hook: repeatable, memorable, the same words each",
      "  time it returns.",
      "- Rhyme when it lands naturally; never force a rhyme with a weak line.",
      style ? "- Match this musical style: " + style + "." : "- Fit a contemporary song.",
      "",
    ].concat(HONESTY_BLOCK, [
      "",
      "OUTPUT: only the revised lyrics with their tags. No title, no",
      "commentary, no quotation marks, no explanation before or after.",
    ]);
    return lines.join("\n");
  }

  // Built fresh on every Discuss send -- pastes in the CURRENT style and
  // lyrics so the model is always talking about the song as it stands
  // right now, not a stale snapshot from earlier in the conversation.
  function buildChatSystemPrompt() {
    var style = els.style ? (els.style.value || "").trim() : "";
    var lyrics = els.lyrics ? els.lyrics.value.trim() : "";
    var lines = [
      "You are a collaborative songwriting partner discussing a song with",
      "the user. Talk naturally -- answer questions, brainstorm, push back",
      "on weak ideas, suggest specific lines or changes. Don't dump a full",
      "rewritten lyric sheet unless the user actually asks for one; this is",
      "a conversation, not a generation call.",
      "Treat the subject honestly -- don't sanitise, soften or sentimentalise",
      "it, and don't manufacture hope or reach for generic metaphors just to",
      "make heavy material more comfortable.",
      "",
      style ? "Current style: " + style : "No style set yet.",
      "",
      lyrics ? "Current lyrics:\n" + lyrics : "No lyrics written yet.",
    ];
    return lines.join("\n");
  }

  function looksLikeLyrics(text) {
    return /^\s*\[[A-Za-z][\w -]*\]/m.test(text || "");
  }

  // Strips a code fence and anything before the first [Tag] line -- same
  // cleanup Phosphene's own action does to whatever the model hands back.
  function cleanLyricsOutput(text) {
    var lines = (text || "").replace(/\r\n/g, "\n").split("\n")
      .map(function (l) { return l.replace(/\s+$/, ""); })
      .filter(function (l) { return l.trim().indexOf("```") !== 0; });
    while (lines.length && lines[0].trim().indexOf("[") !== 0) lines.shift();
    return lines.join("\n").trim();
  }

  // Twin of chatNG.js's own readStream. Always resolves with the full
  // accumulated reply; an optional onDelta(full) callback also gets
  // called after every token for callers that want to live-paint (the
  // discussion chat) -- the write/revise callers just omit it and use
  // the final return value once, filling a textarea in one shot.
  function readNdjsonStream(res, onDelta) {
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = "";
    var full = "";
    var streamErr = null;

    function handleLine(line) {
      line = line.trim();
      if (!line) return;
      var obj;
      try { obj = JSON.parse(line); } catch (e) { return; }
      if (obj.error) { streamErr = obj.error; return; }
      if (obj.message && typeof obj.message.content === "string") {
        full += obj.message.content;
        if (onDelta) onDelta(full);
      }
    }

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          var last = buf.trim();
          if (last) handleLine(last);
          if (streamErr) throw new Error(streamErr);
          return full;
        }
        buf += decoder.decode(chunk.value, { stream: true });
        var nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          handleLine(buf.slice(0, nl));
          buf = buf.slice(nl + 1);
        }
        return pump();
      });
    }
    return pump();
  }

  function systemPromptDefaultFor(mode) {
    var style = els.style ? (els.style.value || "").trim() : "";
    var seconds = els.duration ? Number(els.duration.value) || 240 : 240;
    return mode === "revise"
      ? buildReviseSystemPrompt(style, seconds)
      : buildLyricsSystemPrompt(style, seconds);
  }

  // "reset to default" -- always resets to whichever mode ("write" or
  // "revise") last ran or is about to.
  function resetAiSystemPrompt() {
    if (!els.aiSystem) return;
    els.aiSystem.value = systemPromptDefaultFor(aiSystemMode);
  }

  // Called right before Write or Revise runs. If the box is empty, or
  // still holds the OTHER mode's default verbatim (i.e. the user hasn't
  // customized it away from a known default), swap in this mode's
  // default -- otherwise a hand-edited prompt is respected and sent as-is
  // for whichever action just ran. Either way the box becomes this mode's
  // "reset to default" target from here on.
  function ensureAiSystemPromptFor(mode) {
    if (!els.aiSystem) return;
    var current = els.aiSystem.value.trim();
    var otherMode = mode === "revise" ? "write" : "revise";
    if (!current || current === systemPromptDefaultFor(otherMode).trim()) {
      els.aiSystem.value = systemPromptDefaultFor(mode);
    }
    aiSystemMode = mode;
  }

  function setAiWorking(working) {
    if (els.aiWriteBtn) els.aiWriteBtn.disabled = working;
    if (els.aiRevise) els.aiRevise.disabled = working;
  }

  function writeLyricsWithAi() {
    if (!els.aiModel || !els.aiWriteBtn) return;
    var model = els.aiModel.value;
    var concept = (els.aiConcept.value || "").trim();
    if (!model) { setAiStatus("No Ollama model selected.", true); return; }
    if (!concept) { setAiStatus("Say what the song is about first.", true); return; }
    if (els.lyrics && els.lyrics.value.trim() &&
        !window.confirm("Replace the current Lyrics box with AI-written lyrics?")) {
      return;
    }

    ensureAiSystemPromptFor("write");
    var system = els.aiSystem.value.trim() || systemPromptDefaultFor("write");

    setAiWorking(true);
    els.aiWriteBtn.textContent = "Writing...";
    setAiStatus("Writing lyrics with " + model + "...");

    fetch("/api/ng/chat/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: "Concept: " + concept },
        ],
      }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (d) {
            throw new Error((d && d.error) || "HTTP " + res.status);
          });
        }
        return readNdjsonStream(res);
      })
      .then(function (full) {
        var cleaned = cleanLyricsOutput(full);
        if (!cleaned) throw new Error("model returned nothing usable");
        if (els.lyrics) els.lyrics.value = cleaned;
        setAiStatus("Done.");
      })
      .catch(function (e) {
        setAiStatus("Failed: " + ((e && e.message) || e), true);
      })
      .then(function () {
        setAiWorking(false);
        els.aiWriteBtn.textContent = "Write lyrics";
      });
  }

  function reviseLyricsWithAi() {
    if (!els.aiModel || !els.aiRevise) return;
    var model = els.aiModel.value;
    var instructions = (els.aiConcept.value || "").trim();
    var current = els.lyrics ? els.lyrics.value.trim() : "";
    if (!model) { setAiStatus("No Ollama model selected.", true); return; }
    if (!current) {
      setAiStatus("Nothing to revise -- write some lyrics first, or use Write lyrics.", true);
      return;
    }

    ensureAiSystemPromptFor("revise");
    var system = els.aiSystem.value.trim() || systemPromptDefaultFor("revise");
    var user = "Instructions: " +
      (instructions || "General polish pass -- tighten weak lines, keep the meaning and structure.") +
      "\n\nCurrent lyrics:\n" + current;

    setAiWorking(true);
    els.aiRevise.textContent = "Revising...";
    setAiStatus("Revising lyrics with " + model + "...");

    fetch("/api/ng/chat/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: model,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
      }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (d) {
            throw new Error((d && d.error) || "HTTP " + res.status);
          });
        }
        return readNdjsonStream(res);
      })
      .then(function (full) {
        var cleaned = cleanLyricsOutput(full);
        if (!cleaned) throw new Error("model returned nothing usable");
        if (els.lyrics) els.lyrics.value = cleaned;
        setAiStatus("Done.");
      })
      .catch(function (e) {
        setAiStatus("Failed: " + ((e && e.message) || e), true);
      })
      .then(function () {
        setAiWorking(false);
        els.aiRevise.textContent = "Revise lyrics";
      });
  }

  // ---- Discuss the song -- inline chat, shares the model picker above.
  // Not persisted across a page reload; aiChatMessages is in-memory only
  // for this pane's lifetime. ----

  function renderAiChatTranscript() {
    if (!els.aiChatTranscript) return;
    els.aiChatTranscript.innerHTML = "";
    aiChatMessages.forEach(function (m) {
      var wrap = document.createElement("div");
      wrap.className = "ng-chat-msg " + (m.role === "user" ? "ng-chat-user" : "ng-chat-assistant");
      var who = document.createElement("div");
      who.className = "ng-chat-who";
      who.textContent = m.role === "user" ? "You" : "AI";
      var body = document.createElement("div");
      body.className = "ng-chat-body" +
        (m.error ? " ng-chat-error" : "") +
        (m.pending && !m.content ? " ng-chat-typing" : "");
      body.textContent = m.content || (m.pending ? "..." : "");
      wrap.appendChild(who);
      wrap.appendChild(body);
      if (m.role === "assistant" && m.content && !m.pending && !m.error && looksLikeLyrics(m.content)) {
        var apply = document.createElement("button");
        apply.type = "button";
        apply.className = "ng-musicedit-ai-chat-apply";
        apply.textContent = "Use as lyrics";
        apply.addEventListener("click", function () {
          var cleaned = cleanLyricsOutput(m.content);
          if (cleaned && els.lyrics) els.lyrics.value = cleaned;
        });
        wrap.appendChild(apply);
      }
      els.aiChatTranscript.appendChild(wrap);
    });
    els.aiChatTranscript.scrollTop = els.aiChatTranscript.scrollHeight;
  }

  function clearAiChat() {
    aiChatMessages = [];
    renderAiChatTranscript();
  }

  function sendAiChatMessage() {
    if (!els.aiModel || !els.aiChatInput) return;
    var model = els.aiModel.value;
    var text = (els.aiChatInput.value || "").trim();
    if (!model) { setAiStatus("No Ollama model selected.", true); return; }
    if (!text) return;

    aiChatMessages.push({ role: "user", content: text });
    els.aiChatInput.value = "";
    var assistant = { role: "assistant", content: "", pending: true };
    aiChatMessages.push(assistant);
    renderAiChatTranscript();

    var system = buildChatSystemPrompt();
    var messages = [{ role: "system", content: system }].concat(
      aiChatMessages
        .filter(function (m) { return m !== assistant; })
        .map(function (m) { return { role: m.role, content: m.content }; })
    );

    if (els.aiChatSend) els.aiChatSend.disabled = true;

    fetch("/api/ng/chat/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: model, messages: messages }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().catch(function () { return {}; }).then(function (d) {
            throw new Error((d && d.error) || "HTTP " + res.status);
          });
        }
        return readNdjsonStream(res, function (full) {
          assistant.content = full;
          renderAiChatTranscript();
        });
      })
      .then(function (full) {
        assistant.content = full;
        assistant.pending = false;
        renderAiChatTranscript();
      })
      .catch(function (e) {
        assistant.content = "Failed: " + ((e && e.message) || e);
        assistant.pending = false;
        assistant.error = true;
        renderAiChatTranscript();
      })
      .then(function () {
        if (els.aiChatSend) els.aiChatSend.disabled = false;
      });
  }

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane || !els.main) return;
    inited = true;

    if (els.fileBtn) els.fileBtn.addEventListener("click", function () { els.file.click(); });
    if (els.file) els.file.addEventListener("change", onFileChosen);
    wireDropzone(els.fileWrap, els.file, onFileChosen);
    if (els.saveBtn) els.saveBtn.addEventListener("click", saveFile);
    if (els.text) els.text.addEventListener("input", resetRenders);
    if (els.selectAll) {
      els.selectAll.addEventListener("click", function (e) {
        e.preventDefault();
        els.text.focus();
        els.text.select();
      });
    }
    if (els.bpm) els.bpm.addEventListener("change", onBpmChange);
    if (els.key) els.key.addEventListener("change", onKeyChange);

    if (els.staffToggle) {
      els.staffToggle.addEventListener("click", function () {
        if (window.MusicNG) window.MusicNG.toggleAbcRender(els.staffWrap, els.staffToggle, "staff", getText, onNoteClick);
      });
    }
    if (els.tabToggle) {
      els.tabToggle.addEventListener("click", function () {
        if (window.MusicNG) window.MusicNG.toggleAbcRender(els.tabWrap, els.tabToggle, "tab", getText, onNoteClick);
      });
    }
    if (els.midiBtn) {
      els.midiBtn.addEventListener("click", function () {
        if (window.MusicNG) window.MusicNG.downloadAbcAsMidi(els.midiBtn, filenameBase(), getText);
      });
    }
    if (els.pdfBtn) {
      els.pdfBtn.addEventListener("click", function () {
        if (window.MusicNG) window.MusicNG.printAbcAsPdf(els.pdfBtn, filenameBase(), "staff", getText, getLyrics());
      });
    }
    if (els.tabPdfBtn) {
      els.tabPdfBtn.addEventListener("click", function () {
        if (window.MusicNG) window.MusicNG.printAbcAsPdf(els.tabPdfBtn, filenameBase(), "tab", getText, getLyrics());
      });
    }
    if (els.lyricsExpand) {
      els.lyricsExpand.addEventListener("click", function (e) {
        e.preventDefault();
        var expanded = els.lyrics.classList.toggle("ng-textarea-expanded");
        els.lyricsExpand.textContent = expanded ? "collapse" : "expand";
      });
    }
    if (els.lyricsSelectAll) {
      els.lyricsSelectAll.addEventListener("click", function (e) {
        e.preventDefault();
        els.lyrics.focus();
        els.lyrics.select();
      });
    }

    if (els.duration && els.durationVal) {
      els.duration.addEventListener("input", function () {
        els.durationVal.textContent = formatDuration(els.duration.value);
      });
    }
    if (els.temperature && els.temperatureVal) {
      els.temperature.addEventListener("input", function () {
        els.temperatureVal.textContent = parseFloat(els.temperature.value).toFixed(2);
      });
    }
    if (els.renderBtn) els.renderBtn.addEventListener("click", renderFromScore);
    if (els.useInMusicBtn) els.useInMusicBtn.addEventListener("click", useInMusic);

    if (els.logToggle) els.logToggle.addEventListener("click", toggleLogView);
    if (els.logBack) els.logBack.addEventListener("click", closeLogView);
    if (els.logTabRecent) els.logTabRecent.addEventListener("click", function () { setLogTab("recent"); });
    if (els.logTabArchive) els.logTabArchive.addEventListener("click", function () { setLogTab("years"); });

    if (els.aiWriteBtn) els.aiWriteBtn.addEventListener("click", writeLyricsWithAi);
    if (els.aiRevise) els.aiRevise.addEventListener("click", reviseLyricsWithAi);
    if (els.aiSystemReset) {
      els.aiSystemReset.addEventListener("click", function (e) {
        e.preventDefault();
        resetAiSystemPrompt();
      });
    }
    // "expand" links -- same toggle-a-class-on-the-target pattern as the
    // main stage's own Lyrics box (els.lyricsExpand above), just three
    // more targets: two textareas and the chat transcript. Per John's
    // "peering through a letterbox" complaint about the rail's cramped
    // default sizing.
    [
      [els.aiConceptExpand, els.aiConcept],
      [els.aiSystemExpand, els.aiSystem],
      [els.aiChatExpand, els.aiChatTranscript],
    ].forEach(function (pair) {
      var link = pair[0], target = pair[1];
      if (!link || !target) return;
      link.addEventListener("click", function (e) {
        e.preventDefault();
        var expanded = target.classList.toggle("ng-textarea-expanded");
        link.textContent = expanded ? "collapse" : "expand";
      });
    });

    if (els.aiChatSend) els.aiChatSend.addEventListener("click", sendAiChatMessage);
    if (els.aiChatInput) {
      els.aiChatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendAiChatMessage();
        }
      });
    }
    if (els.aiChatClear) {
      els.aiChatClear.addEventListener("click", function (e) {
        e.preventDefault();
        clearAiChat();
      });
    }
  }

  // ---- called from ProjectManager.render() every tick ----
  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    init();
    var on = !!(active && active.task === "musicedit");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (on) {
      loadAiModels();
      if (els.aiSystem && !els.aiSystem.value.trim()) ensureAiSystemPromptFor("write");
    }
  }

  window.MusicEditNG = { sync: sync, loadAbcText: loadAbcText };
})();
