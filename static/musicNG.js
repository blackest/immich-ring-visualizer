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
 * A finished cover job's ABC score (item.scoreUrl) can be rendered as
 * an actual staff inline in its queue row -- see renderAbcStaff -- via
 * vendored abcjs (static/vendor/abcjs-basic-min.js, loaded just before
 * this file, window.ABCJS). Fetched and rendered once per row, on
 * first click of its "Show sheet music" toggle, not eagerly for every
 * completed job.
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

  // Length bounds -- match music_engineNG.py's MUSIC_MIN/MAX_SECONDS;
  // refreshed from /status once reachable so the two never drift apart.
  var durationBounds = { min: 8.0, max: 360.0 };

  var els = {};

  function refreshEls() {
    els.pane = document.getElementById("ng-music-pane");
    els.main = document.getElementById("ng-music-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.unavailable = document.getElementById("ng-music-unavailable");

    els.queueCount = document.getElementById("ng-music-queue-count");
    els.queueEmpty = document.getElementById("ng-music-queue-empty");
    els.queueClear = document.getElementById("ng-music-queue-clear");
    els.queue = document.getElementById("ng-music-queue");

    els.instrumental = document.getElementById("ng-music-instrumental");
    els.instrumentalRow = document.getElementById("ng-music-instrumental-row");
    els.style = document.getElementById("ng-music-style");
    els.lyrics = document.getElementById("ng-music-lyrics");
    els.mode = document.getElementById("ng-music-mode");
    els.modeRow = document.getElementById("ng-music-mode-row");
    els.duration = document.getElementById("ng-music-duration");
    els.durationVal = document.getElementById("ng-music-duration-val");
    els.seed = document.getElementById("ng-music-seed");
    els.precision = document.getElementById("ng-music-precision");

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

  function setGenerationDisabled(disabled, health) {
    generationDisabled = disabled;
    if (els.generateBtn) els.generateBtn.disabled = disabled;
    if (els.unavailable) {
      els.unavailable.style.display = disabled ? "" : "none";
      if (disabled) {
        els.unavailable.textContent =
          "YuE2 pipeline not reachable (venv/generator/vae missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
    }
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
  function toggleAbcStaff(item, wrap, btn) {
    if (wrap.dataset.loaded === "1") {
      var showing = wrap.style.display !== "none";
      wrap.style.display = showing ? "none" : "";
      btn.textContent = showing ? "Show sheet music" : "Hide sheet music";
      return;
    }
    if (!window.ABCJS || !window.ABCJS.renderAbc) {
      wrap.textContent = "Sheet music renderer failed to load.";
      wrap.style.display = "";
      return;
    }
    btn.disabled = true;
    btn.textContent = "Loading...";
    fetch(item.scoreUrl)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (abcText) {
        window.ABCJS.renderAbc(wrap, abcText, { responsive: "resize" });
        wrap.dataset.loaded = "1";
        wrap.style.display = "";
        btn.textContent = "Hide sheet music";
      })
      .catch(function (e) {
        wrap.textContent = "Could not load sheet music: " + e.message;
        wrap.style.display = "";
        btn.textContent = "Show sheet music";
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
  function downloadAbcAsMidi(item, btn) {
    if (!window.ABCJS || !window.ABCJS.synth || !window.ABCJS.synth.getMidiFile) {
      setStatus("MIDI export not available (renderer failed to load).");
      return;
    }
    var origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Converting...";
    fetch(item.scoreUrl)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (abcText) {
        var midiUris = window.ABCJS.synth.getMidiFile(abcText, { midiOutputType: "encoded" });
        if (!midiUris || !midiUris[0]) throw new Error("no MIDI data produced");
        var a = document.createElement("a");
        a.href = midiUris[0];
        a.download = "music-" + item.jobId + ".mid";
        document.body.appendChild(a);
        a.click();
        a.remove();
      })
      .catch(function (e) {
        setStatus("Could not export MIDI: " + e.message);
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

        var midiDl = document.createElement("button");
        midiDl.type = "button";
        midiDl.className = "ng-btn";
        midiDl.textContent = "Download MIDI";
        midiDl.title = "A real, editable MIDI file -- open it in any DAW or notation editor.";
        midiDl.style.gridColumn = "1 / -1";
        midiDl.style.marginTop = "4px";
        midiDl.addEventListener("click", function () {
          downloadAbcAsMidi(item, midiDl);
        });
        row.appendChild(midiDl);

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
          toggleAbcStaff(item, staffWrap, staffToggle);
        });
        row.appendChild(staffToggle);
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

  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane) return;
    inited = true;

    if (els.instrumental) els.instrumental.addEventListener("change", onInstrumentalChange);
    if (els.coverToggle) els.coverToggle.addEventListener("change", onCoverToggleChange);
    if (els.coverFileBtn) els.coverFileBtn.addEventListener("click", function () { els.coverFile.click(); });
    if (els.coverFile) els.coverFile.addEventListener("change", onCoverFile);
    els.duration.addEventListener("input", function () {
      els.durationVal.textContent = formatDuration(els.duration.value);
    });
    if (els.precision) {
      els.precision.addEventListener("change", function () {
        statusChecked = false; // re-check reachability for the newly picked precision
        ensureStatusChecked();
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

  window.MusicNG = { sync: sync };
})();
