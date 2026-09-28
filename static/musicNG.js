/**
 * musicNG.js -- the "Music" view (bottom-bar Music task).
 *
 * YuE2 style+lyrics-to-song, wired to the NG backend at /api/ng/music/*
 * (routes/musicNG.py, music_jobsNG.py, music_engineNG.py). Twin of
 * h3NG.js's queue-based shape, but text in / audio out -- no reference
 * image, no keyframe, no paste/drop tray at all, unlike every other
 * Generate-shaped view here.
 *
 * Two submit paths share one queue: plain Compose (JSON POST to
 * /generate, style+lyrics -> song) and Cover (multipart POST to
 * /cover, an existing track + your style/lyrics -> a fresh render over
 * its transcribed melody/chords, see music_engineNG.generate_cover_ng).
 * #ng-music-cover-toggle switches the form between them; queue items
 * carry isCover/task so buildQueueRow can tell them apart.
 *
 * A finished job's ABC score (item.scoreUrl) can be rendered as an
 * actual staff inline in its queue row -- see toggleAbcRender -- via
 * vendored abcjs (static/vendor/abcjs-basic-min.js, loaded just before
 * this file, window.ABCJS). Fetched and rendered once per row, on
 * first click of its "Show sheet music" toggle, not eagerly for every
 * completed job.
 *
 * toggleAbcRender/downloadAbcAsMidi/printAbcAsPdf all take a getText()
 * callback rather than assuming a queue row, so they're exported on
 * window.MusicNG (see the bottom of this file) for the standalone
 * "Music Edit" task (static/musiceditNG.js) to reuse for its own
 * load/edit/save ABC panel -- that used to live here as
 * #ng-music-abc-tools, split out per the "music page is overloaded" call.
 *
 * Loaded after h3NG.js and before bootstrapWiringNG.js (which fires the
 * first ProjectManager.render(), which calls MusicNG.sync()).
 */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var API = "/api/ng/music";

  var inited = false;
  var statusChecked = false;
  var generationDisabled = false;
  var queue = []; // [{localId, jobId, status, style, lyrics, durationS, seed, mode, instrumental, isCover, task, error, audioUrl, scoreUrl, logTail}]
  var localSeq = 0;
  var pollTimer = null;
  var rowCache = {};
  var currentCoverBlob = null; // File picked for Cover mode's source track

  // Set by useAbcScore() (called from Music Edit's "→ Use in Music"
  // button, see window.MusicNG export at the bottom of this file) --
  // an edited ABC score waiting to ride along on the next plain
  // Compose submit as a real conditioning input (body.abc, see
  // routes/musicNG.py's /generate). Persists across submits, same as
  // the style/lyrics textareas, until cleared via the banner's "clear"
  // link or a fresh useAbcScore() call.
  var pendingAbc = null;
  var pendingAbcName = null;

  // ---- Job Log panel state (see "Job Log panel" section below) ----
  var logMode = "recent"; // "recent" | "years" | "months" | "days"
  var logYear = null;
  var logMonth = null;
  var logNotesTimers = {}; // job_id -> debounce timer for the notes PATCH

  // Length bounds -- match music_engineNG.py's MUSIC_MIN/MAX_SECONDS;
  // refreshed from /status once reachable so the two never drift apart.
  var durationBounds = { min: 8.0, max: 900.0 };

  var els = {};

  function refreshEls() {
    els.pane = document.getElementById("ng-music-pane");
    els.main = document.getElementById("ng-music-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.unavailable = document.getElementById("ng-music-unavailable");
    els.unavailableText = document.getElementById("ng-music-unavailable-text");
    els.installBtn = document.getElementById("ng-music-install-btn");
    els.installLog = document.getElementById("ng-music-install-log");

    els.queueCount = document.getElementById("ng-music-queue-count");
    els.queueEmpty = document.getElementById("ng-music-queue-empty");
    els.queueClear = document.getElementById("ng-music-queue-clear");
    els.queue = document.getElementById("ng-music-queue");

    els.instrumental = document.getElementById("ng-music-instrumental");
    els.instrumentalRow = document.getElementById("ng-music-instrumental-row");
    els.style = document.getElementById("ng-music-style");
    els.lyrics = document.getElementById("ng-music-lyrics");
    els.lyricsExpand = document.getElementById("ng-music-lyrics-expand");
    els.lyricsSelectAll = document.getElementById("ng-music-lyrics-select-all");
    els.mode = document.getElementById("ng-music-mode");
    els.modeRow = document.getElementById("ng-music-mode-row");
    els.duration = document.getElementById("ng-music-duration");
    els.durationVal = document.getElementById("ng-music-duration-val");
    els.seed = document.getElementById("ng-music-seed");
    els.precision = document.getElementById("ng-music-precision");
    els.temperature = document.getElementById("ng-music-temperature");
    els.temperatureVal = document.getElementById("ng-music-temperature-val");

    els.coverToggle = document.getElementById("ng-music-cover-toggle");
    els.coverFileWrap = document.getElementById("ng-music-cover-file-wrap");
    els.coverFileEmpty = document.getElementById("ng-music-cover-file-empty");
    els.coverFileName = document.getElementById("ng-music-cover-file-name");
    els.coverFile = document.getElementById("ng-music-cover-file");
    els.coverFileBtn = document.getElementById("ng-music-cover-file-btn");
    els.task = document.getElementById("ng-music-task");
    els.taskRow = document.getElementById("ng-music-task-row");

    els.generateBtn = document.getElementById("ng-music-generate-btn");
    els.status = document.getElementById("ng-music-status");
    els.mainLog = document.getElementById("ng-music-log");

    els.abcBanner = document.getElementById("ng-music-abc-banner");
    els.abcBannerText = document.getElementById("ng-music-abc-banner-text");
    els.abcBannerClear = document.getElementById("ng-music-abc-banner-clear");

    els.compose = document.getElementById("ng-music-compose");
    els.logToggle = document.getElementById("ng-music-log-toggle");
    els.logView = document.getElementById("ng-music-log-view");
    els.logBack = document.getElementById("ng-music-log-back");
    els.logTabRecent = document.getElementById("ng-music-log-tab-recent");
    els.logTabArchive = document.getElementById("ng-music-log-tab-archive");
    els.logBreadcrumb = document.getElementById("ng-music-log-breadcrumb");
    els.logEmpty = document.getElementById("ng-music-log-empty");
    els.logList = document.getElementById("ng-music-log-list");
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

  function onInstrumentalChange() {
    var on = !!(els.instrumental && els.instrumental.checked);
    // Lyrics are ignored server-side when instrumental (see
    // music_engineNG.generate_music_ng) -- dim the box in place rather
    // than hide it, same reasoning as h3NG.js/videogenNG.js's own
    // mode-toggle handling of their ref column.
    if (els.lyrics) els.lyrics.classList.toggle("ng-vg-ref-col-disabled", on);
    setStatus("");
  }

  function isCoverMode() {
    return !!(els.coverToggle && els.coverToggle.checked);
  }

  function onCoverToggleChange() {
    var on = isCoverMode();
    if (els.coverFileWrap) els.coverFileWrap.style.display = on ? "" : "none";
    // Score/mode applies to a plain compose; Cover task replaces it
    // (task determines the render mode too -- see
    // music_engineNG._cover_mode_for_task_ng). Instrumental is a
    // plain-compose-only convenience (a cover with empty lyrics is
    // already an instrumental re-render, no separate switch needed).
    if (els.modeRow) els.modeRow.style.display = on ? "none" : "";
    if (els.taskRow) els.taskRow.style.display = on ? "" : "none";
    if (els.instrumentalRow) els.instrumentalRow.style.display = on ? "none" : "";
    if (els.generateBtn) els.generateBtn.textContent = on ? "Cover track" : "Compose";
    setStatus("");
  }

  function onCoverFile() {
    var f = els.coverFile.files && els.coverFile.files[0];
    if (!f) return;
    currentCoverBlob = f;
    if (els.coverFileName) {
      els.coverFileName.textContent = f.name;
      els.coverFileName.style.display = "";
    }
    if (els.coverFileEmpty) els.coverFileEmpty.style.display = "none";
  }

  // iOS's file-picker sheet is unreliable at filtering by accept= (see
  // the accept= broadening on this same input) -- dragging a file in
  // from Files/Photos sidesteps that filtering step entirely. Setting
  // the hidden <input>'s .files to the dropped FileList lets the
  // existing change-driven handler do the rest, same as a real pick.
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

  function setGenerationDisabled(disabled, health) {
    generationDisabled = disabled;
    if (els.generateBtn) els.generateBtn.disabled = disabled;
    if (els.unavailable) {
      els.unavailable.style.display = disabled ? "" : "none";
      if (disabled && els.unavailableText) {
        els.unavailableText.textContent =
          "YuE2 pipeline not reachable (venv/generator/vae missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
      if (els.installBtn) {
        els.installBtn.style.display = (disabled && health && health.installable) ? "" : "none";
        if (!disabled || !(health && health.installable)) {
          els.installBtn.disabled = false;
          els.installBtn.textContent = "Install now (~11 GB)";
        }
      }
    }
  }

  // ---- self-service install (see music_installNG.py / engine_installNG.py) ----
  var installPollTimer = null;

  function pollInstallJob(jobId) {
    fetch(API + "/install/" + jobId)
      .then(function (r) { return r.json(); })
      .then(function (job) {
        if (!job || !job.status) return;
        if (els.installLog) {
          els.installLog.style.display = "";
          els.installLog.textContent = (job.log_tail || []).join("\n");
          els.installLog.scrollTop = els.installLog.scrollHeight;
        }
        if (job.status === "queued" || job.status === "installing") {
          if (els.installBtn) {
            els.installBtn.textContent = "Installing" + (job.step ? " (" + job.step + ")…" : "…");
          }
          return;
        }
        clearInterval(installPollTimer);
        installPollTimer = null;
        if (job.status === "completed") {
          if (els.installBtn) {
            els.installBtn.disabled = true;
            els.installBtn.textContent = "Installed — restart Ring Visualizer to use it";
          }
          setStatus("YuE2 installed. Restart the app to pick it up.");
        } else {
          if (els.installBtn) {
            els.installBtn.disabled = false;
            els.installBtn.textContent = "Install now (~11 GB)";
          }
          setStatus("Install failed: " + (job.error || "unknown error"));
        }
      })
      .catch(function () { /* keep polling -- a transient fetch error isn't fatal */ });
  }

  function startInstall() {
    if (!els.installBtn) return;
    els.installBtn.disabled = true;
    els.installBtn.textContent = "Starting…";
    fetch(API + "/install", { method: "POST" })
      .then(function (res) {
        return res.json().then(function (payload) { return { ok: res.ok, payload: payload }; });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          if (installPollTimer) clearInterval(installPollTimer);
          installPollTimer = setInterval(function () { pollInstallJob(r.payload.job_id); }, 2500);
          pollInstallJob(r.payload.job_id);
        } else {
          els.installBtn.disabled = false;
          els.installBtn.textContent = "Install now (~11 GB)";
          setStatus("Could not start install: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        els.installBtn.disabled = false;
        els.installBtn.textContent = "Install now (~11 GB)";
        setStatus("Could not start install: " + e.message);
      });
  }

  function ensureStatusChecked() {
    if (statusChecked) return;
    statusChecked = true;
    var precision = els.precision ? els.precision.value : "8bit";
    fetch(API + "/status?precision=" + encodeURIComponent(precision))
      .then(function (r) { return r.json(); })
      .then(function (h) {
        setGenerationDisabled(!!(h && h.reachable === false), h);
        if (h && h.min_duration_s != null) {
          durationBounds = { min: h.min_duration_s, max: h.max_duration_s };
        }
      })
      .catch(function () {
        statusChecked = false; // let a later sync retry
      });
  }

  function generateMusic() {
    if (isCoverMode()) {
      submitCover();
      return;
    }

    var style = (els.style.value || "").trim();
    var lyrics = (els.lyrics.value || "").trim();
    var instrumental = !!(els.instrumental && els.instrumental.checked);
    if (!instrumental && !style && !lyrics) {
      setStatus("Give it something to work with -- style, lyrics, or both.");
      els.style.focus();
      return;
    }

    var durationS = parseFloat(els.duration.value) || 240;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    var mode = els.mode ? els.mode.value : "full";
    var precision = els.precision ? els.precision.value : "8bit";

    var localId = "m" + ++localSeq;
    var item = {
      localId: localId,
      jobId: null,
      status: "submitting",
      style: style,
      lyrics: lyrics,
      durationS: durationS,
      seed: seed,
      mode: mode,
      instrumental: instrumental,
      isCover: false,
      task: null,
      error: null,
      audioUrl: null,
      scoreUrl: null,
      logTail: [],
    };
    queue.unshift(item);
    renderQueue();

    var body = {
      style: style, lyrics: lyrics, duration_s: durationS, mode: mode,
      instrumental: instrumental, precision: precision,
    };
    if (seed !== null && !isNaN(seed)) body.seed = seed;
    if (els.temperature) body.temperature = parseFloat(els.temperature.value);
    if (pendingAbc) body.abc = pendingAbc;

    fetch(API + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          item.jobId = r.payload.job_id;
          item.status = "queued";
          setStatus("Queued.");
          ensurePolling();
        } else {
          item.status = "failed";
          item.error = (r.payload && r.payload.error) || "could not start job";
        }
        renderQueue();
      })
      .catch(function (e) {
        item.status = "failed";
        item.error = e.message;
        renderQueue();
      });
  }

  function submitCover() {
    if (!currentCoverBlob) {
      setStatus("Pick a track to cover first.");
      return;
    }
    var style = (els.style.value || "").trim();
    var lyrics = (els.lyrics.value || "").trim();
    var durationS = parseFloat(els.duration.value) || 240;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    var task = els.task ? els.task.value : "melody-full";
    var precision = els.precision ? els.precision.value : "8bit";

    var localId = "m" + ++localSeq;
    var item = {
      localId: localId,
      jobId: null,
      status: "submitting",
      style: style,
      lyrics: lyrics,
      durationS: durationS,
      seed: seed,
      mode: null,
      instrumental: false,
      isCover: true,
      task: task,
      error: null,
      audioUrl: null,
      scoreUrl: null,
      logTail: [],
    };
    queue.unshift(item);
    renderQueue();

    var form = new FormData();
    form.append("file", currentCoverBlob, currentCoverBlob.name || "source");
    form.append("style", style);
    form.append("lyrics", lyrics);
    form.append("task", task);
    form.append("duration_s", String(durationS));
    form.append("precision", precision);
    if (seed !== null && !isNaN(seed)) form.append("seed", String(seed));
    if (els.temperature) form.append("temperature", els.temperature.value);

    fetch(API + "/cover", { method: "POST", body: form })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.job_id) {
          item.jobId = r.payload.job_id;
          item.status = "queued";
          setStatus("Queued.");
          ensurePolling();
        } else {
          item.status = "failed";
          item.error = (r.payload && r.payload.error) || "could not start job";
        }
        renderQueue();
      })
      .catch(function (e) {
        item.status = "failed";
        item.error = e.message;
        renderQueue();
      });
  }

  function ensurePolling() {
    if (pollTimer) return;
    pollTimer = setInterval(pollOnce, POLL_MS);
    pollOnce();
  }

  function pollOnce() {
    var active = queue.filter(function (q) {
      return q.jobId && (q.status === "queued" || q.status === "rendering");
    });
    if (!active.length) {
      clearInterval(pollTimer);
      pollTimer = null;
      return;
    }
    active.forEach(function (item) {
      fetch(API + "/jobs/" + item.jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (!job || !job.status) return;
          if (job.log_tail) item.logTail = job.log_tail;
          if (job.status === "completed") {
            item.status = "done";
            item.audioUrl = job.audio_url + "?v=" + Date.now();
            item.audioSeconds = job.audio_seconds;
            item.scoreUrl = job.score_url || null;
          } else if (job.status === "failed") {
            item.status = "failed";
            item.error = job.error || "generation failed";
          } else {
            item.status = job.status; // "queued" | "rendering"
          }
          renderQueue();
        })
        .catch(function () {
          /* transient -- try again next tick */
        });
    });
  }

  function removeItem(localId) {
    var item = queue.filter(function (q) { return q.localId === localId; })[0];
    if (!item) return;
    queue = queue.filter(function (q) { return q.localId !== localId; });
    renderQueue();
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  function rowSignature(item) {
    return JSON.stringify([
      item.status, item.style, item.lyrics, item.durationS, item.seed,
      item.mode, item.instrumental, item.isCover, item.task,
      item.error, item.audioUrl, item.scoreUrl, item.jobId,
    ]);
  }

  // Fetched/rendered once per row (cached on the wrap element itself,
  // not on `item` -- rowCache reuses this same DOM node across
  // renderQueue() calls as long as the row's signature is unchanged,
  // so a plain data flag on the node survives those re-renders).
  // mode "staff" (default) renders standard notation; mode "tab" adds
  // abcjs's tablature option, converting the same ABC to guitar fret
  // numbers -- ABC text is unchanged, this is purely a renderAbc option.
  // getText() returns a Promise<string> of the ABC source -- a fetch of
  // a finished job's score for queue rows, or the (possibly hand-
  // edited) ABC tools textarea for a loaded file; see fetchAbcText and
  // abcToolsText below.
  function fetchAbcText(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.text();
    });
  }

  function updateAbcBanner() {
    if (!els.abcBanner) return;
    if (!pendingAbc) {
      els.abcBanner.style.display = "none";
      return;
    }
    els.abcBanner.style.display = "";
    if (els.abcBannerText) {
      els.abcBannerText.textContent = "Using edited score “" + (pendingAbcName || "score") +
        "” as this Compose's conditioning input.";
    }
  }

  function clearAbcScore() {
    pendingAbc = null;
    pendingAbcName = null;
    updateAbcBanner();
  }

  // Called from Music Edit's "→ Use in Music" button (see
  // static/musiceditNG.js) -- the reverse of sendScoreToMusicEdit below.
  // Same page, no navigation: just stash the score for the next plain
  // Compose submit, fill in style/lyrics if Music Edit had any, and
  // switch the active task so the user lands here with it ready.
  function useAbcScore(abcText, filenameBase, style, lyrics) {
    refreshEls();
    if (!abcText || !abcText.trim()) return;
    pendingAbc = abcText;
    pendingAbcName = filenameBase || "score";
    if (style) els.style.value = style;
    if (lyrics) els.lyrics.value = lyrics;
    if (isCoverMode() && els.coverToggle) {
      els.coverToggle.checked = false;
      onCoverToggleChange();
    }
    updateAbcBanner();
    setStatus("Loaded score from Music Edit -- Compose will render over it.");
  }

  // One page, all tasks' DOM already present -- no navigation, no
  // query-param/localStorage handoff needed, just fetch the score, hand
  // it to Music Edit's own loader, then switch the active task so the
  // user lands there with it already loaded.
  function sendScoreToMusicEdit(scoreUrl, filenameBase, btn) {
    var origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Loading...";
    fetchAbcText(scoreUrl)
      .then(function (text) {
        if (!window.MusicEditNG || !window.MusicEditNG.loadAbcText) {
          throw new Error("Music Edit isn't available");
        }
        window.MusicEditNG.loadAbcText(text, filenameBase);
        if (window.ProjectManager) window.ProjectManager.setTask("musicedit");
      })
      .catch(function (e) {
        setStatus("Could not load score into Music Edit: " + e.message);
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = origText;
      });
  }

  // clickListener is optional -- only the ABC tools panel passes one
  // (see onAbcNoteClick below); queue rows have no textarea to jump to,
  // so they render read-only. Confirmed live against the vendored
  // abcjs build: clicking a rendered note DOES select its exact
  // startChar/endChar range in a paired textarea -- drag-to-retranspose
  // (the dragging/dragColor machinery also in that build) did NOT, in
  // several real attempts, so it's deliberately not wired up here.
  function toggleAbcRender(wrap, btn, mode, getText, clickListener) {
    var showLabel = mode === "tab" ? "Show guitar tab" : "Show sheet music";
    var hideLabel = mode === "tab" ? "Hide guitar tab" : "Hide sheet music";
    if (wrap.dataset.loaded === "1") {
      var showing = wrap.style.display !== "none";
      wrap.style.display = showing ? "none" : "";
      btn.textContent = showing ? showLabel : hideLabel;
      return;
    }
    if (!window.ABCJS || !window.ABCJS.renderAbc) {
      wrap.textContent = "Sheet music renderer failed to load.";
      wrap.style.display = "";
      return;
    }
    btn.disabled = true;
    btn.textContent = "Loading...";
    getText()
      .then(function (abcText) {
        var opts = { responsive: "resize" };
        if (mode === "tab") {
          opts.tablature = [{ instrument: "guitar", tuning: ["E,", "A,", "D", "G", "B", "e"] }];
        }
        if (clickListener) opts.clickListener = clickListener;
        window.ABCJS.renderAbc(wrap, abcText, opts);
        wrap.dataset.loaded = "1";
        wrap.style.display = "";
        btn.textContent = hideLabel;
      })
      .catch(function (e) {
        wrap.textContent = "Could not load sheet music: " + e.message;
        wrap.style.display = "";
        btn.textContent = showLabel;
      })
      .then(function () {
        btn.disabled = false;
      });
  }

  // ABCJS.synth.getMidiFile(abcText, {midiOutputType:"encoded"}) returns
  // an array of data-URI strings, one per tune in the ABC source (our
  // scores are always exactly one tune) -- confirmed live: a real
  // standard MIDI file (MThd/MTrk header bytes), not abcjs's own
  // playback synth. A real MIDI file in a DAW/notation editor is the
  // actual point here -- editable, re-scoreable, yours, same as the
  // ABC text download above, just in the format every music tool
  // already opens.
  // iOS Safari doesn't reliably honor the `download` attribute on a
  // `data:` URI -- when it doesn't, it hands the tap off to Files/another
  // app instead of downloading in-page. A blob: URL (same-origin, no
  // navigation) is what saveAbcFile() below already uses successfully,
  // so MIDI export decodes ABCJS's data-URI into a Blob first.
  function dataUriToBlob(dataUri) {
    var comma = dataUri.indexOf(",");
    var meta = dataUri.slice(0, comma);
    var mimeMatch = /data:([^;]+)/.exec(meta);
    var mime = mimeMatch ? mimeMatch[1] : "application/octet-stream";
    // ABCJS's "encoded" MIDI output is percent-escaped raw bytes, not
    // UTF-8 text -- decodeURIComponent validates %XX sequences as UTF-8
    // and throws "URI malformed" on the very first non-ASCII MIDI byte.
    // unescape() has no such validation: it maps each %XX straight to
    // its byte value, which is exactly what a %XX-per-byte encoding needs.
    var raw = /;base64/.test(meta) ? atob(dataUri.slice(comma + 1)) : unescape(dataUri.slice(comma + 1));
    var bytes = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }

  function downloadAbcAsMidi(btn, filenameBase, getText) {
    if (!window.ABCJS || !window.ABCJS.synth || !window.ABCJS.synth.getMidiFile) {
      setStatus("MIDI export not available (renderer failed to load).");
      return;
    }
    var origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Converting...";
    getText()
      .then(function (abcText) {
        var midiUris = window.ABCJS.synth.getMidiFile(abcText, { midiOutputType: "encoded" });
        if (!midiUris || !midiUris[0]) throw new Error("no MIDI data produced");
        var blob = dataUriToBlob(midiUris[0]);
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = filenameBase + ".mid";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      })
      .catch(function (e) {
        setStatus("Could not export MIDI: " + e.message);
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = origText;
      });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Section-level lyric placement, not syllable alignment -- neither
  // pipeline (plain generate or cover transcription) ever tracks which
  // word lands on which note, so per-syllable w: lines aren't possible
  // without a real forced-alignment step. What IS reliable: the ABC's
  // own "% sectionname" comments (yue2_run.py's own structure markers)
  // line up 1:1, in order, with the lyrics' own [SectionName] tags --
  // verified against real plain-generate job output, not assumed.
  // Tested and rejected: using rests in the Vocal voice to split into
  // per-LINE phrases -- phrase count didn't match lyric line count (the
  // model freely compresses/stretches your line breaks when it sings),
  // so that finer-grained mapping would confidently mislead a singer
  // rather than help one.
  //
  // Requires an EXACT section-count match, not just zipping to the
  // shorter list -- a plain generate's structure is driven entirely by
  // your own [Tag]s so counts always match, but a cover's structure
  // comes from the SOURCE track's transcription, which has no
  // guaranteed relationship to how many sections your own replacement
  // lyrics use. A count mismatch there means the pairing can't be
  // trusted, so it's skipped rather than guessed (same reasoning as
  // rejecting phrase-level: wrong-but-confident is worse than absent).
  function zipAbcSectionsWithLyrics(abcText, lyricsText) {
    var abcLabels = [];
    var re = /^%\s*(.+)$/gm, m;
    while ((m = re.exec(abcText))) abcLabels.push(m[1].trim());
    if (!abcLabels.length) return [];

    var parts = lyricsText.split(/\n?\[([A-Za-z][A-Za-z0-9 ]*)\]\n?/);
    var lyricBlocks = [];
    if (parts[0] && parts[0].trim()) lyricBlocks.push(parts[0].trim());
    for (var i = 1; i < parts.length; i += 2) {
      lyricBlocks.push((parts[i + 1] || "").trim());
    }
    if (lyricBlocks.length !== abcLabels.length) return [];

    var out = [];
    for (var j = 0; j < abcLabels.length; j++) {
      var label = abcLabels[j];
      out.push({ label: label.charAt(0).toUpperCase() + label.slice(1), text: lyricBlocks[j] });
    }
    return out;
  }

  // Prefixes the first note/rest line after each "% sectionname"
  // comment with a quoted free-text annotation ("^Label") -- standard
  // ABC decoration syntax, renders as a rehearsal-mark-style label
  // above the staff at that point. Skips the "V: ..." voice header
  // line(s) in between to land on actual note content.
  function injectSectionAnnotations(abcText) {
    var lines = abcText.split("\n");
    var out = [];
    var pendingLabel = null;
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      var m = /^%\s*(.+)$/.exec(line);
      if (m) {
        pendingLabel = m[1].trim();
        out.push(line);
        continue;
      }
      if (pendingLabel && line.trim() && !/^V:/.test(line.trim())) {
        var label = pendingLabel.charAt(0).toUpperCase() + pendingLabel.slice(1);
        out.push("\"^" + label.replace(/"/g, "") + "\"" + line);
        pendingLabel = null;
        continue;
      }
      out.push(line);
    }
    return out.join("\n");
  }

  // Opens a bare print-friendly popup, renders the ABC into it with the
  // same vendored abcjs, and triggers window.print() -- "Save as PDF" in
  // the browser's print dialog is the export. No server-side rendering,
  // no new vendored PDF library, matches the client-side-only pattern
  // toggleAbcRender/downloadAbcAsMidi already use. mode "tab" adds the
  // same tablature option toggleAbcRender uses for the on-page tab view.
  // lyrics is optional (queue rows have item.lyrics; the standalone ABC
  // tools panel doesn't know any lyrics, so it's omitted there) -- when
  // given, adds rehearsal-mark section labels to the notation plus a
  // matching lyrics-by-section block after it. See
  // zipAbcSectionsWithLyrics's own comment for why this is section-
  // level, not per-line/per-syllable.
  function printAbcAsPdf(btn, filenameBase, mode, getText, lyrics) {
    var origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Preparing...";
    getText()
      .then(function (abcText) {
        var renderText = abcText;
        var lyricsHtml = "";
        if (lyrics && lyrics.trim()) {
          var sections = zipAbcSectionsWithLyrics(abcText, lyrics);
          if (sections.length) {
            renderText = injectSectionAnnotations(abcText);
            lyricsHtml = "<div id=\"lyrics-block\">" + sections.map(function (s) {
              return "<h3>" + escapeHtml(s.label) + "</h3><pre>" + escapeHtml(s.text) + "</pre>";
            }).join("") + "</div>";
          }
        }
        var w = window.open("", "_blank", "width=900,height=1200");
        if (!w) throw new Error("popup blocked -- allow popups to print sheet music");
        var title = filenameBase + (mode === "tab" ? "-tab" : "");
        w.document.write(
          "<!DOCTYPE html><html><head><title>" + title + "</title>" +
          "<style>body{margin:24px;font-family:sans-serif;}" +
          "#abc-target{max-width:800px;margin:0 auto;}" +
          "#lyrics-block{max-width:800px;margin:24px auto 0;}" +
          "#lyrics-block h3{margin:16px 0 4px;font-size:1em;text-transform:uppercase;" +
          "letter-spacing:0.05em;color:#555;}" +
          "#lyrics-block h3:first-child{margin-top:0;}" +
          "#lyrics-block pre{margin:0;font-family:inherit;white-space:pre-wrap;" +
          "font-size:0.95em;line-height:1.4;}" +
          "@media print{body{margin:0;}#lyrics-block{page-break-before:auto;}}" +
          "</style></head>" +
          "<body><div id=\"abc-target\">Loading sheet music...</div>" +
          lyricsHtml +
          "<script src=\"/static/vendor/abcjs-basic-min.js\"><\/script></body></html>"
        );
        w.document.close();
        var waited = 0;
        var poll = setInterval(function () {
          waited += 50;
          if (w.closed) {
            clearInterval(poll);
            return;
          }
          if (w.ABCJS && w.ABCJS.renderAbc) {
            clearInterval(poll);
            var opts = { responsive: "resize" };
            if (mode === "tab") {
              opts.tablature = [{ instrument: "guitar", tuning: ["E,", "A,", "D", "G", "B", "e"] }];
            }
            w.ABCJS.renderAbc("abc-target", renderText, opts);
            w.onafterprint = function () { w.close(); };
            setTimeout(function () {
              w.focus();
              w.print();
            }, 150);
          } else if (waited > 5000) {
            clearInterval(poll);
            var target = w.document.getElementById("abc-target");
            if (target) target.textContent = "Sheet music renderer failed to load.";
          }
        }, 50);
      })
      .catch(function (e) {
        setStatus("Could not prepare PDF: " + e.message);
      })
      .then(function () {
        btn.disabled = false;
        btn.textContent = origText;
      });
  }

  function buildQueueRow(item) {
    var row = document.createElement("div");
    row.className = "ng-gen-queue-row st-" + item.status;

    var meta = document.createElement("div");
    meta.className = "meta";
    meta.style.gridColumn = "1 / -1";
    var title = document.createElement("div");
    title.className = "title";
    var label = item.instrumental
      ? "(instrumental) " + (item.style || "untitled")
      : (item.style || item.lyrics || "untitled").slice(0, 80);
    title.textContent = (item.isCover ? "(cover) " : "") + label;
    meta.appendChild(title);
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = "up to " + formatDuration(item.durationS) + ", " +
      (item.isCover ? "task " + item.task : item.mode) +
      (item.seed !== null ? ", seed " + item.seed : "");
    meta.appendChild(sub);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
    statusRow.style.gridColumn = "1 / -1";
    var st = document.createElement("span");
    st.className = "st";
    st.textContent =
      item.status === "done" ? "done" :
      item.status === "failed" ? "failed" :
      item.status === "rendering" ? "rendering..." :
      item.status === "submitting" ? "submitting..." : "queued";
    statusRow.appendChild(st);

    var rm = document.createElement("button");
    rm.type = "button";
    rm.className = "pose-remove";
    rm.title = "remove from queue";
    rm.textContent = "✕";
    rm.addEventListener("click", function () { removeItem(item.localId); });
    statusRow.appendChild(rm);

    row.appendChild(statusRow);

    if (item.status === "failed" && item.error) {
      var err = document.createElement("div");
      err.className = "log";
      err.style.color = "var(--ng-danger)";
      err.style.gridColumn = "1 / -1";
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
      dl.download = "music-" + item.jobId + ".flac";
      dl.textContent = "Download song";
      dl.style.gridColumn = "1 / -1";
      dl.style.marginTop = "4px";
      row.appendChild(dl);

      if (item.scoreUrl) {
        var scoreDl = document.createElement("a");
        scoreDl.className = "ng-btn";
        scoreDl.href = item.scoreUrl;
        scoreDl.download = "music-" + item.jobId + ".abc";
        scoreDl.textContent = "Download sheet music (ABC)";
        scoreDl.style.gridColumn = "1 / -1";
        scoreDl.style.marginTop = "4px";
        row.appendChild(scoreDl);

        var editDl = document.createElement("button");
        editDl.type = "button";
        editDl.className = "ng-btn";
        editDl.textContent = "→ Edit score in Music Edit";
        editDl.title = "Loads this score into Music Edit so you can adjust tempo/key and render over it.";
        editDl.style.gridColumn = "1 / -1";
        editDl.style.marginTop = "4px";
        editDl.addEventListener("click", function () {
          sendScoreToMusicEdit(item.scoreUrl, "music-" + item.jobId, editDl);
        });
        row.appendChild(editDl);

        var midiDl = document.createElement("button");
        midiDl.type = "button";
        midiDl.className = "ng-btn";
        midiDl.textContent = "Download MIDI";
        midiDl.title = "A real, editable MIDI file -- open it in any DAW or notation editor.";
        midiDl.style.gridColumn = "1 / -1";
        midiDl.style.marginTop = "4px";
        midiDl.addEventListener("click", function () {
          downloadAbcAsMidi(midiDl, "music-" + item.jobId, function () {
            return fetchAbcText(item.scoreUrl);
          });
        });
        row.appendChild(midiDl);

        var pdfDl = document.createElement("button");
        pdfDl.type = "button";
        pdfDl.className = "ng-btn";
        pdfDl.textContent = "Download sheet music (PDF)";
        pdfDl.title = "Opens a print-friendly view -- choose \"Save as PDF\" in the print dialog.";
        pdfDl.style.gridColumn = "1 / -1";
        pdfDl.style.marginTop = "4px";
        pdfDl.addEventListener("click", function () {
          printAbcAsPdf(pdfDl, "music-" + item.jobId, "staff", function () {
            return fetchAbcText(item.scoreUrl);
          }, item.lyrics);
        });
        row.appendChild(pdfDl);

        var tabPdfDl = document.createElement("button");
        tabPdfDl.type = "button";
        tabPdfDl.className = "ng-btn";
        tabPdfDl.textContent = "Download guitar tab (PDF)";
        tabPdfDl.title = "Same print-to-PDF flow, rendered as guitar fret numbers (standard tuning).";
        tabPdfDl.style.gridColumn = "1 / -1";
        tabPdfDl.style.marginTop = "4px";
        tabPdfDl.addEventListener("click", function () {
          printAbcAsPdf(tabPdfDl, "music-" + item.jobId, "tab", function () {
            return fetchAbcText(item.scoreUrl);
          }, item.lyrics);
        });
        row.appendChild(tabPdfDl);

        var staffWrap = document.createElement("div");
        staffWrap.className = "ng-music-abc-staff";
        staffWrap.style.gridColumn = "1 / -1";
        staffWrap.style.display = "none";
        row.appendChild(staffWrap);

        var staffToggle = document.createElement("button");
        staffToggle.type = "button";
        staffToggle.className = "ng-btn";
        staffToggle.textContent = "Show sheet music";
        staffToggle.style.gridColumn = "1 / -1";
        staffToggle.style.marginTop = "4px";
        staffToggle.addEventListener("click", function () {
          toggleAbcRender(staffWrap, staffToggle, "staff", function () {
            return fetchAbcText(item.scoreUrl);
          });
        });
        row.appendChild(staffToggle);

        var tabWrap = document.createElement("div");
        tabWrap.className = "ng-music-abc-staff";
        tabWrap.style.gridColumn = "1 / -1";
        tabWrap.style.display = "none";
        row.appendChild(tabWrap);

        var tabToggle = document.createElement("button");
        tabToggle.type = "button";
        tabToggle.className = "ng-btn";
        tabToggle.textContent = "Show guitar tab";
        tabToggle.title = "Same score, rendered as guitar fret numbers (standard tuning).";
        tabToggle.style.gridColumn = "1 / -1";
        tabToggle.style.marginTop = "4px";
        tabToggle.addEventListener("click", function () {
          toggleAbcRender(tabWrap, tabToggle, "tab", function () {
            return fetchAbcText(item.scoreUrl);
          });
        });
        row.appendChild(tabToggle);
      }
    }

    return row;
  }

  function renderQueue() {
    if (!els.queue) return;
    if (els.queueCount) els.queueCount.textContent = queue.length ? "(" + queue.length + ")" : "";
    if (els.queueEmpty) els.queueEmpty.style.display = queue.length ? "none" : "";

    els.queue.innerHTML = "";
    var newCache = {};
    queue.forEach(function (item) {
      var sig = rowSignature(item);
      var cached = rowCache[item.localId];
      var row = (cached && cached.sig === sig) ? cached.el : buildQueueRow(item);
      newCache[item.localId] = { sig: sig, el: row };
      els.queue.appendChild(row);
    });
    rowCache = newCache;

    updateMainLog();
  }

  function updateMainLog() {
    if (!els.mainLog) return;
    var active = queue.filter(function (q) { return q.status === "rendering"; })[0];
    if (!active || !active.logTail || !active.logTail.length) {
      els.mainLog.style.display = "none";
      els.mainLog.textContent = "";
      return;
    }
    els.mainLog.style.display = "";
    els.mainLog.textContent = active.logTail.join("\n");
    els.mainLog.scrollTop = els.mainLog.scrollHeight;
  }

  // ABC load/edit/save/render tools used to live here as a standalone
  // panel -- moved to static/musiceditNG.js (the "Music Edit" task),
  // which reuses toggleAbcRender/downloadAbcAsMidi/printAbcAsPdf above
  // via window.MusicNG, same as the queue rows below still do.

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane) return;
    inited = true;

    if (els.instrumental) els.instrumental.addEventListener("change", onInstrumentalChange);
    if (els.coverToggle) els.coverToggle.addEventListener("change", onCoverToggleChange);
    if (els.coverFileBtn) els.coverFileBtn.addEventListener("click", function () { els.coverFile.click(); });
    if (els.coverFile) els.coverFile.addEventListener("change", onCoverFile);
    wireDropzone(els.coverFileWrap, els.coverFile, onCoverFile);

    els.duration.addEventListener("input", function () {
      els.durationVal.textContent = formatDuration(els.duration.value);
    });
    if (els.temperature && els.temperatureVal) {
      els.temperature.addEventListener("input", function () {
        els.temperatureVal.textContent = parseFloat(els.temperature.value).toFixed(2);
      });
    }
    if (els.precision) {
      els.precision.addEventListener("change", function () {
        statusChecked = false; // re-check reachability for the newly picked precision
        ensureStatusChecked();
      });
    }

    if (els.logToggle) els.logToggle.addEventListener("click", toggleLogView);
    if (els.logBack) els.logBack.addEventListener("click", closeLogView);
    if (els.logTabRecent) els.logTabRecent.addEventListener("click", function () { setLogTab("recent"); });
    if (els.logTabArchive) els.logTabArchive.addEventListener("click", function () { setLogTab("years"); });

    if (els.installBtn) els.installBtn.addEventListener("click", startInstall);

    if (els.abcBannerClear) {
      els.abcBannerClear.addEventListener("click", function (e) {
        e.preventDefault();
        clearAbcScore();
      });
    }
    updateAbcBanner();

    if (els.lyricsExpand) {
      els.lyricsExpand.addEventListener("click", function (e) {
        e.preventDefault();
        var expanded = els.lyrics.classList.toggle("ng-textarea-expanded");
        els.lyricsExpand.textContent = expanded ? "collapse" : "expand";
      });
    }
    // iOS's drag-handle text selection is fiddly enough that a plain
    // "select all" beats asking the user to drag handles by touch.
    if (els.lyricsSelectAll) {
      els.lyricsSelectAll.addEventListener("click", function (e) {
        e.preventDefault();
        els.lyrics.focus();
        els.lyrics.select();
      });
    }

    els.generateBtn.addEventListener("click", generateMusic);
    els.queueClear.addEventListener("click", function () {
      queue = queue.filter(function (q) {
        return q.status !== "done" && q.status !== "failed";
      });
      renderQueue();
    });
  }

  // ---- "Job Log" panel ----
  // A durable history of finished Music renders (music_job_logsNG.py),
  // separate from the ephemeral render queue in the rail -- the queue
  // evicts (and deletes) a job's files once 20 newer jobs have finished,
  // so this is the only place a song's style/lyrics/seed ("recipe") can
  // still be recovered afterwards. Replaces #ng-music-compose while
  // open; "Recent" shows the current month, "Archive" drills down
  // year -> month -> a day-entries grid using the same card component.
  // Twin of videogenNG.js's own Job Log panel.

  function toggleLogView() {
    if (!els.logView || !els.compose) return;
    var opening = els.logView.style.display === "none";
    if (opening) {
      els.compose.style.display = "none";
      els.logView.style.display = "";
      setLogTab("recent");
    } else {
      closeLogView();
    }
  }

  function closeLogView() {
    if (!els.logView || !els.compose) return;
    els.logView.style.display = "none";
    els.compose.style.display = "";
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
      .then(function (res) { return res.json(); })
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
      .then(function (res) { return res.json(); })
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
      .then(function (res) { return res.json(); })
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
      .then(function (res) { return res.json(); })
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

  function useLogEntry(entry) {
    els.style.value = entry.style || "";
    els.lyrics.value = entry.lyrics || "";
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

    if (entry.is_cover) {
      if (els.coverToggle && !els.coverToggle.checked) {
        els.coverToggle.checked = true;
        onCoverToggleChange();
      }
      if (els.task && entry.task) els.task.value = entry.task;
      closeLogView();
      setStatus("Loaded style/lyrics/seed from the job log -- this was a cover, so pick the source track again before composing.");
      return;
    }

    if (els.coverToggle && els.coverToggle.checked) {
      els.coverToggle.checked = false;
      onCoverToggleChange();
    }
    if (els.mode && entry.mode) els.mode.value = entry.mode;
    if (els.instrumental) els.instrumental.checked = !!entry.instrumental;
    closeLogView();
    setStatus("Loaded style/lyrics/seed from the job log.");
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

      var editLink = document.createElement("button");
      editLink.type = "button";
      editLink.className = "ng-gen-btn ng-gen-btn-quiet";
      editLink.textContent = "→ Edit score";
      editLink.title = "Loads this score into Music Edit so you can adjust tempo/key and render over it.";
      editLink.addEventListener("click", function () {
        sendScoreToMusicEdit(scoreUrl, "music-" + entry.job_id, editLink);
      });
      actions.appendChild(editLink);
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

  // ---- called from ProjectManager.render() every tick ----
  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "music");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (on && els.controlsPane) els.controlsPane.style.display = "none";
    if (!on) return;

    init();
    ensureStatusChecked();
    renderQueue();
  }

  // toggleAbcRender/downloadAbcAsMidi/printAbcAsPdf are fully generic
  // (wrap/btn/getText callbacks, no hardcoded elements) -- musiceditNG.js
  // reuses them for its own textarea instead of duplicating this core.
  // useAbcScore is musiceditNG.js's "→ Use in Music" handoff, the
  // reverse of this file's own sendScoreToMusicEdit.
  window.MusicNG = {
    sync: sync,
    toggleAbcRender: toggleAbcRender,
    downloadAbcAsMidi: downloadAbcAsMidi,
    printAbcAsPdf: printAbcAsPdf,
    useAbcScore: useAbcScore,
  };
})();
