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
 * Below the composer, a separate ABC tools panel (#ng-music-abc-tools)
 * lets you load any .abc file from disk, edit it in place, and save it
 * back out -- independent of the job queue. It shares its render/MIDI/
 * PDF core with the queue rows (toggleAbcRender/downloadAbcAsMidi/
 * printAbcAsPdf all take a getText() callback now, fed either from a
 * fetch of item.scoreUrl or straight from the panel's own textarea).
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
  var loadedAbcName = null; // filename of the last file loaded into the ABC tools panel

  // Length bounds -- match music_engineNG.py's MUSIC_MIN/MAX_SECONDS;
  // refreshed from /status once reachable so the two never drift apart.
  var durationBounds = { min: 8.0, max: 900.0 };

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

    els.abcFile = document.getElementById("ng-music-abc-file");
    els.abcFileBtn = document.getElementById("ng-music-abc-file-btn");
    els.abcFileEmpty = document.getElementById("ng-music-abc-file-empty");
    els.abcFileName = document.getElementById("ng-music-abc-file-name");
    els.abcSaveBtn = document.getElementById("ng-music-abc-save-btn");
    els.abcText = document.getElementById("ng-music-abc-text");
    els.abcStaffToggle = document.getElementById("ng-music-abc-staff-toggle");
    els.abcTabToggle = document.getElementById("ng-music-abc-tab-toggle");
    els.abcMidiBtn = document.getElementById("ng-music-abc-midi-btn");
    els.abcPdfBtn = document.getElementById("ng-music-abc-pdf-btn");
    els.abcTabPdfBtn = document.getElementById("ng-music-abc-tab-pdf-btn");
    els.abcStaffWrap = document.getElementById("ng-music-abc-staff-wrap");
    els.abcTabWrap = document.getElementById("ng-music-abc-tab-wrap");
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
        var a = document.createElement("a");
        a.href = midiUris[0];
        a.download = filenameBase + ".mid";
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

  // ---- ABC tools panel -- load/edit/save/render a plain ABC file,
  // independent of the AI job queue above. Shares its render/MIDI/PDF
  // core with the queue rows via toggleAbcRender/downloadAbcAsMidi/
  // printAbcAsPdf, just fed from the textarea instead of a fetched URL.
  function abcToolsText() {
    return els.abcText ? els.abcText.value : "";
  }

  function abcToolsGetText() {
    return Promise.resolve(abcToolsText());
  }

  function abcToolsFilenameBase() {
    return (loadedAbcName || "music").replace(/\.abc$/i, "");
  }

  // Clears the cached staff/tab render so the next toggle click re-
  // renders from the current textarea contents instead of reusing a
  // stale render from before a file load or an edit.
  function resetAbcToolsRenders() {
    [els.abcStaffWrap, els.abcTabWrap].forEach(function (wrap) {
      if (!wrap) return;
      wrap.dataset.loaded = "";
      wrap.style.display = "none";
    });
    if (els.abcStaffToggle) els.abcStaffToggle.textContent = "Show sheet music";
    if (els.abcTabToggle) els.abcTabToggle.textContent = "Show guitar tab";
  }

  function onAbcFileChosen() {
    var f = els.abcFile.files && els.abcFile.files[0];
    if (!f) return;
    var reader = new FileReader();
    reader.onload = function () {
      if (els.abcText) els.abcText.value = String(reader.result || "");
      loadedAbcName = f.name;
      if (els.abcFileName) {
        els.abcFileName.textContent = f.name;
        els.abcFileName.style.display = "";
      }
      if (els.abcFileEmpty) els.abcFileEmpty.style.display = "none";
      resetAbcToolsRenders();
      setStatus("Loaded " + f.name + ".");
    };
    reader.onerror = function () {
      setStatus("Could not read file: " + (reader.error && reader.error.message));
    };
    reader.readAsText(f);
  }

  // abcelem.startChar/endChar are character offsets into the ABC
  // string that was actually rendered -- confirmed live against the
  // vendored abcjs build (clicking a notehead selected the right
  // substring in a paired textarea). Selecting rather than just moving
  // the caret makes the hit visible without hunting for a blinking
  // cursor in a wall of ABC syntax.
  function onAbcNoteClick(abcelem) {
    if (!els.abcText || !abcelem) return;
    var start = abcelem.startChar;
    var end = abcelem.endChar;
    if (typeof start !== "number" || typeof end !== "number" || end <= start) return;
    els.abcText.focus();
    els.abcText.setSelectionRange(start, end);
    var before = els.abcText.value.slice(0, start);
    var lineNum = before.split("\n").length;
    var lineHeight = parseFloat(getComputedStyle(els.abcText).lineHeight) || 16;
    els.abcText.scrollTop = Math.max(0, (lineNum - 3) * lineHeight);
  }

  function saveAbcFile() {
    var text = abcToolsText();
    if (!text.trim()) {
      setStatus("Nothing to save -- load or type ABC first.");
      return;
    }
    var blob = new Blob([text], { type: "text/plain" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = abcToolsFilenameBase() + ".abc";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus("Saved " + a.download + ".");
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

    if (els.abcFileBtn) els.abcFileBtn.addEventListener("click", function () { els.abcFile.click(); });
    if (els.abcFile) els.abcFile.addEventListener("change", onAbcFileChosen);
    if (els.abcSaveBtn) els.abcSaveBtn.addEventListener("click", saveAbcFile);
    if (els.abcText) els.abcText.addEventListener("input", resetAbcToolsRenders);
    if (els.abcStaffToggle) {
      els.abcStaffToggle.addEventListener("click", function () {
        toggleAbcRender(els.abcStaffWrap, els.abcStaffToggle, "staff", abcToolsGetText, onAbcNoteClick);
      });
    }
    if (els.abcTabToggle) {
      els.abcTabToggle.addEventListener("click", function () {
        toggleAbcRender(els.abcTabWrap, els.abcTabToggle, "tab", abcToolsGetText, onAbcNoteClick);
      });
    }
    if (els.abcMidiBtn) {
      els.abcMidiBtn.addEventListener("click", function () {
        downloadAbcAsMidi(els.abcMidiBtn, abcToolsFilenameBase(), abcToolsGetText);
      });
    }
    if (els.abcPdfBtn) {
      els.abcPdfBtn.addEventListener("click", function () {
        printAbcAsPdf(els.abcPdfBtn, abcToolsFilenameBase(), "staff", abcToolsGetText);
      });
    }
    if (els.abcTabPdfBtn) {
      els.abcTabPdfBtn.addEventListener("click", function () {
        printAbcAsPdf(els.abcTabPdfBtn, abcToolsFilenameBase(), "tab", abcToolsGetText);
      });
    }

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
