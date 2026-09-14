/**
 * videogenNG.js -- the "Animate" view (bottom-bar Animate task).
 *
 * Self-contained image-to-video generation, wired to the NG backend at
 * /api/ng/videogen/* (routes/videogenNG.py, video_jobsNG.py, LTX-2.5 via
 * ltx_engineNG.py). One reference image + a motion prompt + a duration in,
 * one mp4 out -- deliberately no pose grid, no character registration,
 * no draft-character bookkeeping like Generate has; a video render isn't
 * part of a character's sheet.
 *
 * The ONLY external input read from the rest of the app is a convenience
 * button that pulls whatever image is currently the active reference in
 * GenerateNG's ref tray (window.GenerateNG.getActiveReference()) -- handy
 * since opening a character from the picker grid already lands its
 * avatar there (see characterPickerNG.js). Otherwise this view is fully
 * self-contained: pick a photo from disk, type a prompt, go.
 *
 * Model: a flat FIFO render queue, same shape as Generate's -- each entry
 * is one (reference, prompt, duration) job. The backend's single LTX
 * worker (video_jobsNG.py) renders one at a time; queue rows here just
 * reflect that. In-memory only, lost on refresh, matching the backend's
 * own job store.
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after generateNG.js and before bootstrapWiringNG.js.
 */
(function () {
  "use strict";

  var POLL_MS = 3000;
  var API = "/api/ng/videogen";
  var MAX_IMAGE_DIM = 1568; // downscale above this so payloads stay sane

  // ---- module state ----
  var inited = false;
  var statusChecked = false;
  var generationDisabled = false;
  var currentRefBlob = null; // File/Blob picked from disk or pulled from Generate
  var currentRefPreviewUrl = null; // object URL for the <img> preview
  // Set only when currentRefBlob came from a live curation-session frame
  // (pulled via useGenerateReference(), which fetched it from
  // /api/ng/framefile/<id> -- an in-memory cache, see video_analysisNG.
  // find_cache_frame_ng). When set, submit ref_frame_id instead of
  // re-uploading currentRefBlob: the backend already has these bytes in
  // the same process, no need to round-trip them over HTTP first. A
  // local disk pick (onDiskFile) has no server-side frame, so this stays
  // null for that case -- the blob is genuinely the only copy.
  var currentRefFrameId = null;
  var queue = []; // [{localId, jobId, status, refPreviewUrl, prompt, durationS, seed, error, videoUrl, logTail}]
  var localSeq = 0;
  // "Discuss next scene" panel: [{role, content, discussion?, options?,
  // error?}, ...] -- role/content are what get resent to /discuss as the
  // conversation history; discussion/options/error are render-only.
  var discussTurns = [];
  var discussBusy = false;
  // "Chat with Gemma" panel: same shape as discussTurns, resent to /chat
  // instead of /discuss -- plain conversation, no options/discussion
  // fields since Gemma isn't steered toward any fixed reply shape here.
  var chatTurns = [];
  var chatBusy = false;
  // Images attached to the NEXT chat message -- [{dataUrl, base64}], same
  // shape as chatNG.js/rachelNG.js's pendingImages. Only the current
  // (about-to-be-sent) turn's images are ever sent -- see
  // ltx_engineNG.chat_with_gemma_ng's docstring for why an image
  // attached to an earlier turn couldn't be seen by Gemma again anyway.
  var pendingChatImages = [];
  var pollTimer = null;

  // "Job Log" panel: durable history from job_logsNG.py, fetched fresh
  // on every open (recent = current month) or archive drill-down step.
  // Unlike discussTurns/chatTurns/queue above, this is NOT in-memory
  // state the page owns -- it's a read-through view of the backend's
  // permanent record; logMode/logYear/logMonth just track where the
  // archive breadcrumb currently is.
  var logMode = "recent"; // "recent" | "years" | "months" | "days"
  var logYear = null;
  var logMonth = null;
  var logNotesTimers = {}; // job_id -> debounce timer for the notes PATCH

  // Render-size/fps bounds -- match the backend defaults in
  // ltx_engineNG.py; refreshed from /status once reachable so the two
  // never silently drift apart.
  var dimBounds = { step: 64, min: 256, max: 1536, minFps: 8, maxFps: 30 };

  var els = {};

  function refreshEls() {
    els.pane = document.getElementById("ng-videogen-pane");
    els.main = document.getElementById("ng-videogen-main");
    els.controlsPane = document.getElementById("ng-controls-pane");
    els.unavailable = document.getElementById("ng-vg-unavailable");

    els.queueHead = document.getElementById("ng-vg-queue-head");
    els.queueCount = document.getElementById("ng-vg-queue-count");
    els.queueEmpty = document.getElementById("ng-vg-queue-empty");
    els.queueClear = document.getElementById("ng-vg-queue-clear");
    els.queue = document.getElementById("ng-vg-queue");

    els.modeT2v = document.getElementById("ng-vg-mode-t2v");
    els.refCol = document.getElementById("ng-vg-ref-col");
    els.refPreview = document.getElementById("ng-vg-ref-preview");
    els.refEmpty = document.getElementById("ng-vg-ref-empty");
    els.refFile = document.getElementById("ng-vg-ref-file");
    els.refFileBtn = document.getElementById("ng-vg-ref-file-btn");
    els.refUseGen = document.getElementById("ng-vg-ref-use-gen");

    els.prompt = document.getElementById("ng-vg-prompt");
    els.enhanceBtn = document.getElementById("ng-vg-enhance-btn");
    els.discussToggle = document.getElementById("ng-vg-discuss-toggle");
    els.discussPanel = document.getElementById("ng-vg-discuss-panel");
    els.discussTranscript = document.getElementById("ng-vg-discuss-transcript");
    els.discussEmpty = document.getElementById("ng-vg-discuss-empty");
    els.discussInput = document.getElementById("ng-vg-discuss-input");
    els.discussSend = document.getElementById("ng-vg-discuss-send");
    els.chatToggle = document.getElementById("ng-vg-chat-toggle");
    els.chatPanel = document.getElementById("ng-vg-chat-panel");
    els.chatTranscript = document.getElementById("ng-vg-chat-transcript");
    els.chatEmpty = document.getElementById("ng-vg-chat-empty");
    els.chatInput = document.getElementById("ng-vg-chat-input");
    els.chatSend = document.getElementById("ng-vg-chat-send");
    els.chatAttach = document.getElementById("ng-vg-chat-attach");
    els.chatFile = document.getElementById("ng-vg-chat-file");
    els.chatAttachments = document.getElementById("ng-vg-chat-attachments");
    els.duration = document.getElementById("ng-vg-duration");
    els.durationVal = document.getElementById("ng-vg-duration-val");
    els.durationAuto = document.getElementById("ng-vg-duration-auto");
    els.seed = document.getElementById("ng-vg-seed");
    els.width = document.getElementById("ng-vg-width");
    els.height = document.getElementById("ng-vg-height");
    els.fps = document.getElementById("ng-vg-fps");

    els.generateBtn = document.getElementById("ng-vg-generate-btn");
    els.status = document.getElementById("ng-vg-status");
    els.mainLog = document.getElementById("ng-vg-log");

    els.logToggle = document.getElementById("ng-vg-log-toggle");
    els.compose = document.getElementById("ng-vg-compose");
    els.logView = document.getElementById("ng-vg-log-view");
    els.logBack = document.getElementById("ng-vg-log-back");
    els.logTabRecent = document.getElementById("ng-vg-log-tab-recent");
    els.logTabArchive = document.getElementById("ng-vg-log-tab-archive");
    els.logBreadcrumb = document.getElementById("ng-vg-log-breadcrumb");
    els.logEmpty = document.getElementById("ng-vg-log-empty");
    els.logList = document.getElementById("ng-vg-log-list");
  }

  function setStatus(msg) {
    if (els.status) els.status.textContent = msg || "";
  }

  function setReference(blob, frameId) {
    if (currentRefPreviewUrl) URL.revokeObjectURL(currentRefPreviewUrl);
    currentRefBlob = blob;
    currentRefFrameId = frameId || null;
    currentRefPreviewUrl = URL.createObjectURL(blob);
    if (els.refPreview) {
      els.refPreview.src = currentRefPreviewUrl;
      els.refPreview.style.display = "";
    }
    if (els.refEmpty) els.refEmpty.style.display = "none";
  }

  // /api/ng/framefile/<job_id>_anchor and /api/ng/framefile/<job_id>_<frame>
  // are the only URLs Generate's ref tray ever hands back for a "proj"-kind
  // ref (anchor or a selected frame) -- both resolve, server-side, to
  // video_analysisNG.find_cache_frame_ng's in-memory cache. A "disk"-kind
  // ref (a local file the user picked) never matches this.
  function frameIdFromUrl(url) {
    var m = /\/api\/ng\/framefile\/([^/?#]+)/.exec(url || "");
    return m ? m[1] : null;
  }

  function isT2vMode() {
    return !!(els.modeT2v && els.modeT2v.checked);
  }

  function onModeChange() {
    var t2v = isT2vMode();
    if (els.refCol) els.refCol.style.display = t2v ? "none" : "";
    setStatus("");
  }

  function onDiskFile() {
    var f = els.refFile.files && els.refFile.files[0];
    if (f) setReference(f);
  }

  function useGenerateReference() {
    var gen = window.GenerateNG && window.GenerateNG.getActiveReference
      ? window.GenerateNG.getActiveReference()
      : null;
    if (!gen || !gen.url) {
      setStatus("No reference set in Generate -- open a character or add a photo there first.");
      return;
    }
    setStatus("Pulling reference from Generate...");
    fetch(gen.url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.blob();
      })
      .then(function (blob) {
        // Still fetched for the preview thumbnail either way, but when
        // this resolves to a frame id we submit that instead of
        // re-uploading the blob -- see setReference's comment.
        setReference(blob, frameIdFromUrl(gen.url));
        setStatus("");
      })
      .catch(function (e) {
        setStatus("Could not pull Generate's reference: " + e.message);
      });
  }

  function setGenerationDisabled(disabled, health) {
    generationDisabled = disabled;
    if (els.generateBtn) els.generateBtn.disabled = disabled;
    if (els.unavailable) {
      els.unavailable.style.display = disabled ? "" : "none";
      if (disabled) {
        els.unavailable.textContent =
          "LTX pipeline not reachable (binary/model/gemma missing at " +
          ((health && health.repo_dir) || "?") +
          "). Queuing is disabled until it's fixed.";
      }
    }
  }

  function ensureStatusChecked() {
    if (statusChecked) return;
    statusChecked = true;
    fetch(API + "/status")
      .then(function (r) { return r.json(); })
      .then(function (h) {
        setGenerationDisabled(!!(h && h.reachable === false), h);
        if (els.enhanceBtn) {
          var enhanceOk = !!(h && h.enhance_gemma_ok);
          els.enhanceBtn.disabled = !enhanceOk;
          els.enhanceBtn.title = enhanceOk
            ? ""
            : "Chat Gemma-3 checkpoint not found at " + ((h && h.enhance_gemma_path) || "?");
        }
        if (h && h.dim_step) {
          dimBounds = {
            step: h.dim_step, min: h.min_dim, max: h.max_dim,
            minFps: h.min_fps, maxFps: h.max_fps,
          };
          [els.width, els.height].forEach(function (el) {
            if (!el) return;
            el.step = dimBounds.step;
            el.min = dimBounds.min;
            el.max = dimBounds.max;
          });
          if (els.fps) {
            els.fps.min = dimBounds.minFps;
            els.fps.max = dimBounds.maxFps;
          }
        }
      })
      .catch(function () {
        statusChecked = false; // let a later sync retry
      });
  }

  function enhancePrompt() {
    var prompt = (els.prompt.value || "").trim();
    if (!prompt) {
      setStatus("Describe the motion first.");
      els.prompt.focus();
      return;
    }
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);
    var t2v = isT2vMode();
    var hasRef = !t2v && !!currentRefBlob;

    els.enhanceBtn.disabled = true;
    setStatus(hasRef
      ? "Enhancing prompt with Gemma (looking at the reference image)..."
      : "Enhancing prompt with Gemma...");

    var form = new FormData();
    form.append("prompt", prompt);
    if (seed !== null && !isNaN(seed)) form.append("seed", String(seed));
    if (hasRef) {
      if (currentRefFrameId) {
        // Already cached server-side (a curation-session frame) -- send
        // the reference, not the bytes we just downloaded.
        form.append("ref_frame_id", currentRefFrameId);
      } else {
        var ext = currentRefBlob.type === "image/png" ? ".png" : ".jpg";
        form.append("file", currentRefBlob, "reference" + ext);
      }
    }

    fetch(API + "/enhance", { method: "POST", body: form })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.prompt) {
          els.prompt.value = r.payload.prompt; // replaces the given prompt
          setStatus("Prompt enhanced.");
        } else {
          setStatus("Enhance failed: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function (e) {
        setStatus("Enhance failed: " + e.message);
      })
      .then(function () {
        els.enhanceBtn.disabled = false;
      });
  }

  // Rough "how long does this beat need" estimate from the prompt text
  // alone -- LTX itself can't self-time a render (it denoises a fixed
  // frame count, it doesn't stop when a line "feels done"), so this is
  // a stand-in: quoted dialogue is timed at a words-per-minute pace
  // (slower for drawl/deliberate cues, faster for rapid/blurted ones),
  // and any leftover action text (lead-in before the line, a cut after
  // it) is split into short clauses and given a flat hold each.
  var VG_PACE_WPM = { slow: 90, normal: 130, fast: 180 };
  var VG_SLOW_RE = /\b(slow(ly)?|drawl(s|ing)?|deliberate(ly)?|measured|hesitant(ly)?|halting(ly)?|languid(ly)?)\b/i;
  var VG_FAST_RE = /\b(fast|rapid(ly)?|quick(ly)?|hurried(ly)?|frantic(ally)?|blurts?|snaps?|clipped)\b/i;
  var VG_QUOTE_RE = /["“]([^"”]+)["”]|'([^']+)'/g;
  var VG_BEAT_SECONDS = 0.8; // rough hold for a short non-dialogue action clause

  function estimateDurationSeconds(text) {
    text = (text || "").trim();
    if (!text) return null;

    var dialogueWords = 0;
    var rest = text;
    var m;
    VG_QUOTE_RE.lastIndex = 0;
    while ((m = VG_QUOTE_RE.exec(text)) !== null) {
      var line = m[1] || m[2] || "";
      dialogueWords += (line.match(/\S+/g) || []).length;
      rest = rest.replace(m[0], " ");
    }

    var wpm = VG_SLOW_RE.test(text) ? VG_PACE_WPM.slow
      : VG_FAST_RE.test(text) ? VG_PACE_WPM.fast
      : VG_PACE_WPM.normal;
    var dialogueSeconds = dialogueWords > 0
      ? Math.max(0.6, dialogueWords / (wpm / 60))
      : 0;

    var beatText = rest.replace(/\s+/g, " ").trim();
    var clauses = beatText
      ? beatText.split(/,|\bthen\b|\band then\b/i).map(function (s) { return s.trim(); }).filter(Boolean)
      : [];
    var beatSeconds = clauses.length * VG_BEAT_SECONDS;

    var total = dialogueSeconds + beatSeconds;
    if (total <= 0) return null;

    var min = parseFloat(els.duration.min) || 0.5;
    var max = parseFloat(els.duration.max) || 30;
    var step = parseFloat(els.duration.step) || 0.5;
    total = Math.round(total / step) * step;
    return Math.min(max, Math.max(min, total));
  }

  function applyEstimatedDuration(seconds, source) {
    var min = parseFloat(els.duration.min) || 0.5;
    var max = parseFloat(els.duration.max) || 30;
    var step = parseFloat(els.duration.step) || 0.5;
    seconds = Math.round(seconds / step) * step;
    seconds = Math.min(max, Math.max(min, seconds));
    els.duration.value = String(seconds);
    els.durationVal.textContent = seconds.toFixed(1) + "s";
    setStatus("Estimated " + seconds.toFixed(1) + "s (" + source + ") -- drag the slider to override.");
  }

  // Primary path: ask the enhance Gemma checkpoint to read the shot
  // description (dialogue + delivery note + surrounding beats) and time
  // it directly -- it can judge pacing far better than a fixed WPM
  // table. Falls back to the local estimateDurationSeconds() heuristic
  // above when Gemma isn't reachable (checkpoint missing, LTX lab down,
  // request failed), so "Time it" still does something useful offline.
  function estimateDuration() {
    var promptText = (els.prompt.value || "").trim();
    var localEst = estimateDurationSeconds(promptText);
    if (!promptText) {
      setStatus("Nothing to time -- add a motion prompt or a quoted dialogue line first.");
      return;
    }

    if (els.durationAuto) els.durationAuto.disabled = true;
    setStatus("Asking Gemma to time the scene...");

    fetch(API + "/estimate-duration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText }),
    })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && typeof r.payload.seconds === "number") {
          applyEstimatedDuration(r.payload.seconds, "Gemma");
          return;
        }
        if (localEst != null) {
          applyEstimatedDuration(localEst, "local heuristic -- Gemma unavailable");
        } else {
          setStatus("Gemma couldn't time that: " + ((r.payload && r.payload.error) || "unknown error"));
        }
      })
      .catch(function () {
        if (localEst != null) {
          applyEstimatedDuration(localEst, "local heuristic -- Gemma unreachable");
        } else {
          setStatus("Could not estimate a duration for that prompt.");
        }
      })
      .then(function () {
        if (els.durationAuto) els.durationAuto.disabled = false;
      });
  }

  // ---- "Discuss next scene" panel ----
  // A small chat, scoped to this view, for storyboarding the next shot
  // with Gemma before committing to a prompt/duration: describe the
  // last shot + a rough idea, get back 3 labeled options, either apply
  // one straight to the form or keep refining across more turns. State
  // is in-memory only (discussTurns above), cleared on reload -- same as
  // the render queue.

  function toggleDiscussPanel() {
    if (!els.discussPanel) return;
    var opening = els.discussPanel.style.display === "none";
    if (opening && els.chatPanel) els.chatPanel.style.display = "none";
    els.discussPanel.style.display = opening ? "" : "none";
    if (opening && els.discussInput) els.discussInput.focus();
  }

  function applyDiscussOption(opt) {
    els.prompt.value = opt.prompt;
    applyEstimatedDuration(opt.seconds, "Gemma option " + opt.label);
  }

  function buildDiscussTurnEl(turn) {
    var msg = document.createElement("div");
    msg.className = "ng-chat-msg " + (turn.role === "user" ? "ng-chat-user" : "ng-chat-assistant");

    var who = document.createElement("div");
    who.className = "ng-chat-who";
    who.textContent = turn.role === "user" ? "You" : "Gemma";
    msg.appendChild(who);

    var body = document.createElement("div");
    body.className = "ng-chat-body" + (turn.error ? " ng-chat-error" : "");
    body.textContent = turn.role === "user" ? turn.content : (turn.discussion || turn.content || "");
    msg.appendChild(body);

    if (turn.options && turn.options.length) {
      var opts = document.createElement("div");
      opts.className = "ng-vg-discuss-options";
      turn.options.forEach(function (o) {
        var card = document.createElement("div");
        card.className = "ng-vg-discuss-option";

        var head = document.createElement("div");
        head.className = "ng-vg-discuss-option-head";
        head.textContent = o.label + " · " + o.seconds.toFixed(1) + "s";
        card.appendChild(head);

        var cardBody = document.createElement("div");
        cardBody.className = "ng-vg-discuss-option-body";
        cardBody.textContent = o.prompt;
        card.appendChild(cardBody);

        var useBtn = document.createElement("button");
        useBtn.type = "button";
        useBtn.className = "ng-gen-btn ng-gen-btn-quiet";
        useBtn.textContent = "Use this";
        useBtn.addEventListener("click", function () { applyDiscussOption(o); });
        card.appendChild(useBtn);

        opts.appendChild(card);
      });
      msg.appendChild(opts);
    }
    return msg;
  }

  function renderDiscussTranscript() {
    if (!els.discussTranscript) return;
    els.discussTranscript.innerHTML = "";
    discussTurns.forEach(function (t) {
      els.discussTranscript.appendChild(buildDiscussTurnEl(t));
    });
    if (els.discussEmpty) els.discussEmpty.style.display = discussTurns.length ? "none" : "";
    els.discussTranscript.scrollTop = els.discussTranscript.scrollHeight;
  }

  function setDiscussBusy(busy) {
    discussBusy = busy;
    if (els.discussSend) els.discussSend.disabled = busy;
    if (els.discussInput) els.discussInput.disabled = busy;
    if (!els.discussTranscript) return;
    var existing = document.getElementById("ng-vg-discuss-typing");
    if (busy && !existing) {
      var typing = document.createElement("div");
      typing.className = "ng-chat-msg ng-chat-assistant";
      typing.id = "ng-vg-discuss-typing";
      var who = document.createElement("div");
      who.className = "ng-chat-who";
      who.textContent = "Gemma";
      var body = document.createElement("div");
      body.className = "ng-chat-body ng-chat-typing";
      body.textContent = "thinking…";
      typing.appendChild(who);
      typing.appendChild(body);
      els.discussTranscript.appendChild(typing);
      els.discussTranscript.scrollTop = els.discussTranscript.scrollHeight;
    } else if (!busy && existing) {
      existing.remove();
    }
  }

  function sendDiscussMessage() {
    if (discussBusy) return;
    var raw = (els.discussInput.value || "").trim();
    if (!raw) return;

    // First turn only: fold in the current prompt box as "the last
    // scene" -- matches how you'd naturally describe it ("last scene
    // was X, now we need Y") without a separate field to fill in.
    var content = raw;
    if (discussTurns.length === 0) {
      var lastScene = (els.prompt.value || "").trim();
      content = lastScene
        ? "Previous shot: \"" + lastScene + "\"\n\nNow I need the next shot. My rough idea: " + raw
        : "This is the first shot. My rough idea: " + raw;
    }

    discussTurns.push({ role: "user", content: content });
    els.discussInput.value = "";
    renderDiscussTranscript();
    setDiscussBusy(true);

    var wireTurns = discussTurns
      .filter(function (t) { return !t.error; })
      .map(function (t) { return { role: t.role, content: t.content }; });

    fetch(API + "/discuss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: wireTurns }),
    })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.raw) {
          discussTurns.push({
            role: "assistant",
            content: r.payload.raw,
            discussion: r.payload.discussion,
            options: r.payload.options,
          });
        } else {
          discussTurns.push({
            role: "assistant",
            content: "",
            discussion: "Couldn't get suggestions: " + ((r.payload && r.payload.error) || "unknown error"),
            error: true,
          });
        }
        renderDiscussTranscript();
      })
      .catch(function (e) {
        discussTurns.push({
          role: "assistant",
          content: "",
          discussion: "Request failed: " + e.message,
          error: true,
        });
        renderDiscussTranscript();
      })
      .then(function () {
        setDiscussBusy(false);
      });
  }

  // ---- "Chat with Gemma" panel ----
  // A free-form chat, scoped to this view, against the same Gemma
  // checkpoint as Discuss -- but no forced 3-option JSON shape. State is
  // in-memory only (chatTurns above), cleared on reload -- same as the
  // render queue and discussTurns.

  function toggleChatPanel() {
    if (!els.chatPanel) return;
    var opening = els.chatPanel.style.display === "none";
    if (opening && els.discussPanel) els.discussPanel.style.display = "none";
    els.chatPanel.style.display = opening ? "" : "none";
    if (opening && els.chatInput) els.chatInput.focus();
  }

  // ---- "Chat with Gemma" image attachments ----
  // Same shape/behavior as chatNG.js's/rachelNG.js's own pendingImages --
  // duplicated rather than shared since each *NG.js file is self-contained.
  function addChatImageFiles(fileList) {
    Array.prototype.forEach.call(fileList || [], function (f) {
      if (!/^image\//.test(f.type)) return;
      var reader = new FileReader();
      reader.onload = function () {
        downscaleChatImage(reader.result, function (dataUrl) {
          pendingChatImages.push({ dataUrl: dataUrl, base64: dataUrl.split(",")[1] });
          renderChatAttachments();
        });
      };
      reader.readAsDataURL(f);
    });
  }

  function downscaleChatImage(dataUrl, cb) {
    var img = new Image();
    img.onload = function () {
      var w = img.naturalWidth, h = img.naturalHeight;
      if (w <= MAX_IMAGE_DIM && h <= MAX_IMAGE_DIM) {
        cb(dataUrl);
        return;
      }
      var scale = MAX_IMAGE_DIM / Math.max(w, h);
      var canvas = document.createElement("canvas");
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      cb(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = function () {
      cb(dataUrl); // fall back to the original rather than dropping it
    };
    img.src = dataUrl;
  }

  function renderChatAttachments() {
    if (!els.chatAttachments) return;
    els.chatAttachments.innerHTML = "";
    els.chatAttachments.hidden = !pendingChatImages.length;
    pendingChatImages.forEach(function (im, idx) {
      var thumb = document.createElement("div");
      thumb.className = "ng-chat-attach-thumb";
      var img = document.createElement("img");
      img.src = im.dataUrl;
      var rm = document.createElement("button");
      rm.type = "button";
      rm.textContent = "×";
      rm.title = "Remove";
      rm.addEventListener("click", function () {
        pendingChatImages.splice(idx, 1);
        renderChatAttachments();
      });
      thumb.appendChild(img);
      thumb.appendChild(rm);
      els.chatAttachments.appendChild(thumb);
    });
  }

  function buildChatTurnEl(turn) {
    var msg = document.createElement("div");
    msg.className = "ng-chat-msg " + (turn.role === "user" ? "ng-chat-user" : "ng-chat-assistant");

    var who = document.createElement("div");
    who.className = "ng-chat-who";
    who.textContent = turn.role === "user" ? "You" : "Gemma";
    msg.appendChild(who);

    if (turn.images && turn.images.length) {
      var thumbs = document.createElement("div");
      thumbs.className = "ng-chat-msg-thumbs";
      turn.images.forEach(function (im) {
        var img = document.createElement("img");
        img.src = im.dataUrl;
        thumbs.appendChild(img);
      });
      msg.appendChild(thumbs);
    }

    var body = document.createElement("div");
    body.className = "ng-chat-body" + (turn.error ? " ng-chat-error" : "");
    body.textContent = turn.content || "";
    msg.appendChild(body);

    return msg;
  }

  function renderChatTranscript() {
    if (!els.chatTranscript) return;
    els.chatTranscript.innerHTML = "";
    chatTurns.forEach(function (t) {
      els.chatTranscript.appendChild(buildChatTurnEl(t));
    });
    if (els.chatEmpty) els.chatEmpty.style.display = chatTurns.length ? "none" : "";
    els.chatTranscript.scrollTop = els.chatTranscript.scrollHeight;
  }

  function setChatBusy(busy) {
    chatBusy = busy;
    if (els.chatSend) els.chatSend.disabled = busy;
    if (els.chatInput) els.chatInput.disabled = busy;
    if (!els.chatTranscript) return;
    var existing = document.getElementById("ng-vg-chat-typing");
    if (busy && !existing) {
      var typing = document.createElement("div");
      typing.className = "ng-chat-msg ng-chat-assistant";
      typing.id = "ng-vg-chat-typing";
      var who = document.createElement("div");
      who.className = "ng-chat-who";
      who.textContent = "Gemma";
      var body = document.createElement("div");
      body.className = "ng-chat-body ng-chat-typing";
      body.textContent = "thinking…";
      typing.appendChild(who);
      typing.appendChild(body);
      els.chatTranscript.appendChild(typing);
      els.chatTranscript.scrollTop = els.chatTranscript.scrollHeight;
    } else if (!busy && existing) {
      existing.remove();
    }
  }

  function sendChatMessage() {
    if (chatBusy) return;
    var raw = (els.chatInput.value || "").trim();
    if (!raw && !pendingChatImages.length) return;

    var userTurn = { role: "user", content: raw };
    if (pendingChatImages.length) userTurn.images = pendingChatImages.slice();
    chatTurns.push(userTurn);
    els.chatInput.value = "";
    pendingChatImages = [];
    renderChatAttachments();
    renderChatTranscript();
    setChatBusy(true);

    var wireTurns = chatTurns
      .filter(function (t) { return !t.error; })
      .map(function (t) { return { role: t.role, content: t.content }; });
    if (userTurn.images && userTurn.images.length) {
      // Only the LAST wire turn's images can ever matter -- mlx_vlm's
      // chat template only places image tokens on the current turn (see
      // ltx_engineNG.chat_with_gemma_ng's docstring) -- so this is
      // always the one we just pushed.
      wireTurns[wireTurns.length - 1].images = userTurn.images.map(function (im) {
        return im.base64;
      });
    }

    fetch(API + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: wireTurns }),
    })
      .then(function (res) {
        return res.json().then(function (payload) {
          return { ok: res.ok, payload: payload };
        });
      })
      .then(function (r) {
        if (r.ok && r.payload && r.payload.reply) {
          chatTurns.push({ role: "assistant", content: r.payload.reply });
        } else {
          chatTurns.push({
            role: "assistant",
            content: "Couldn't reach Gemma: " + ((r.payload && r.payload.error) || "unknown error"),
            error: true,
          });
        }
        renderChatTranscript();
      })
      .catch(function (e) {
        chatTurns.push({ role: "assistant", content: "Request failed: " + e.message, error: true });
        renderChatTranscript();
      })
      .then(function () {
        setChatBusy(false);
      });
  }

  // ---- "Job Log" panel ----
  // A durable history of finished Animate runs (job_logsNG.py), separate
  // from the ephemeral render queue in the rail. Replaces #ng-vg-compose
  // while open; "Recent" shows the current month, "Archive" drills down
  // year -> month -> a day-entries grid using the same card component.

  function toggleLogView() {
    if (!els.logView || !els.compose) return;
    var opening = els.logView.style.display === "none";
    if (opening) {
      if (els.discussPanel) els.discussPanel.style.display = "none";
      if (els.chatPanel) els.chatPanel.style.display = "none";
      els.compose.style.display = "none";
      els.logView.style.display = "";
      setLogTab("recent");
    } else {
      els.logView.style.display = "none";
      els.compose.style.display = "";
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
    els.prompt.value = entry.prompt || "";
    if (typeof entry.duration_s === "number") {
      applyEstimatedDuration(entry.duration_s, "Job Log");
    }
    if (entry.width) els.width.value = entry.width;
    if (entry.height) els.height.value = entry.height;
    if (entry.frame_rate) els.fps.value = entry.frame_rate;
    closeLogView();
    setStatus("Loaded prompt/duration from the job log -- pick a reference image before generating.");
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

    if (entry.has_ref) {
      var img = document.createElement("img");
      img.className = "ng-vg-log-thumb";
      img.src = API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/ref";
      card.appendChild(img);
    }

    var status = document.createElement("span");
    status.className = "ng-vg-log-status ng-vg-log-status-" + entry.status;
    status.textContent = entry.status;
    card.appendChild(status);

    var prompt = document.createElement("div");
    prompt.className = "ng-vg-log-prompt";
    prompt.textContent = entry.prompt || "";
    card.appendChild(prompt);

    var meta = document.createElement("div");
    meta.className = "ng-vg-log-meta";
    var when = entry.finished_at ? new Date(entry.finished_at * 1000).toLocaleTimeString() : "";
    meta.textContent = (entry.duration_s != null ? entry.duration_s + "s" : "") +
      (entry.width && entry.height ? ", " + entry.width + "×" + entry.height : "") +
      (entry.frame_rate ? ", " + entry.frame_rate + "fps" : "") +
      (entry.resolved_seed != null ? ", seed " + entry.resolved_seed : "") +
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
      video.src = API + "/logs/" + encodeURIComponent(entry.date) + "/" + encodeURIComponent(entry.job_id) + "/video";
      video.controls = true;
      card.appendChild(video);
    }

    var actions = document.createElement("div");
    actions.className = "ng-vg-log-actions";
    var useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.className = "ng-gen-btn ng-gen-btn-quiet";
    useBtn.textContent = "↺ Use this";
    useBtn.addEventListener("click", function () { useLogEntry(entry); });
    actions.appendChild(useBtn);
    card.appendChild(actions);

    var notes = document.createElement("textarea");
    notes.className = "ng-vg-log-notes";
    notes.rows = 2;
    notes.placeholder = "Storyboarding notes...";
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

  // Multiple of dimBounds.step, within [dimBounds.min, dimBounds.max].
  // Mirrors ltx_engineNG._validate_ltx_dims_ng -- checked client-side too
  // so a bad value fails fast instead of round-tripping to the server.
  function validDim(v) {
    return v % dimBounds.step === 0 && v >= dimBounds.min && v <= dimBounds.max;
  }

  function generateVideo() {
    var prompt = (els.prompt.value || "").trim();
    var t2v = isT2vMode();
    if (!t2v && !currentRefBlob) {
      setStatus("Pick a reference image first.");
      return;
    }
    if (!prompt) {
      setStatus("Describe the motion first.");
      els.prompt.focus();
      return;
    }
    var durationS = parseFloat(els.duration.value) || 3.0;
    var seedRaw = (els.seed.value || "").trim();
    var seed = seedRaw === "" ? null : parseInt(seedRaw, 10);

    var width = parseInt(els.width.value, 10) || dimBounds.min;
    var height = parseInt(els.height.value, 10) || dimBounds.min;
    var fps = parseFloat(els.fps.value) || dimBounds.minFps;
    if (!validDim(width) || !validDim(height)) {
      setStatus(
        "Width and height must be multiples of " + dimBounds.step +
        ", between " + dimBounds.min + " and " + dimBounds.max + "."
      );
      return;
    }
    if (fps < dimBounds.minFps || fps > dimBounds.maxFps) {
      setStatus("FPS must be between " + dimBounds.minFps + " and " + dimBounds.maxFps + ".");
      return;
    }

    var localId = "v" + ++localSeq;
    var item = {
      localId: localId,
      jobId: null,
      status: "submitting",
      refPreviewUrl: t2v ? null : currentRefPreviewUrl,
      prompt: prompt,
      durationS: durationS,
      seed: seed,
      width: width,
      height: height,
      fps: fps,
      error: null,
      videoUrl: null,
      logTail: [],
    };
    queue.unshift(item);
    renderQueue();

    var form = new FormData();
    if (!t2v) {
      if (currentRefFrameId) {
        form.append("ref_frame_id", currentRefFrameId);
      } else {
        var ext = currentRefBlob.type === "image/png" ? ".png" : ".jpg";
        form.append("file", currentRefBlob, "reference" + ext);
      }
    }
    form.append("prompt", prompt);
    form.append("duration_s", String(durationS));
    if (seed !== null && !isNaN(seed)) form.append("seed", String(seed));
    form.append("width", String(width));
    form.append("height", String(height));
    form.append("frame_rate", String(fps));

    fetch(API + "/generate", { method: "POST", body: form })
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
            item.videoUrl = job.video_url + "?v=" + Date.now();
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
    // Best-effort: tell the backend to drop/kill the job too. Fire and
    // forget -- the row is already gone from view either way, and a
    // job that's already done/failed has nothing left to cancel.
    if (item.jobId && (item.status === "queued" || item.status === "rendering")) {
      fetch(API + "/jobs/" + item.jobId, { method: "DELETE" }).catch(function () {});
    }
  }

  // Keyed by localId -> {sig, el}. Polling re-renders the WHOLE queue every
  // few seconds while a job is in flight, but most items in it haven't
  // changed -- rebuilding their DOM from scratch would recreate their
  // <video>/<img> nodes every tick, which forces the browser to reload/
  // restart them (a finished video visibly flashes/restarts while the
  // next job renders, same class of bug this app has hit before with
  // queue thumbnails). Reuse the existing node when an item's rendered
  // state hasn't changed since the last render instead of rebuilding it.
  var rowCache = {};

  function rowSignature(item) {
    return JSON.stringify([
      item.status, item.prompt, item.durationS, item.seed,
      item.width, item.height, item.fps,
      item.error, item.videoUrl, item.jobId, item.refPreviewUrl,
    ]);
  }

  function buildQueueRow(item) {
    var row = document.createElement("div");
    row.className = "ng-gen-queue-row st-" + item.status;

    var thumbs = document.createElement("div");
    thumbs.className = "thumbs";
    if (item.refPreviewUrl) {
      var img = document.createElement("img");
      img.src = item.refPreviewUrl;
      thumbs.appendChild(img);
    }
    row.appendChild(thumbs);

    var meta = document.createElement("div");
    meta.className = "meta";
    var title = document.createElement("div");
    title.className = "title";
    title.textContent = item.prompt;
    meta.appendChild(title);
    var sub = document.createElement("div");
    sub.className = "sub";
    sub.textContent = item.durationS + "s, " + item.width + "×" + item.height +
      ", " + item.fps + "fps" + (item.seed !== null ? ", seed " + item.seed : "");
    meta.appendChild(sub);
    row.appendChild(meta);

    var statusRow = document.createElement("div");
    statusRow.className = "status-row";
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
      err.textContent = item.error;
      row.appendChild(err);
    } else if (item.status === "done" && item.videoUrl) {
      var video = document.createElement("video");
      video.src = item.videoUrl;
      video.controls = true;
      video.style.gridColumn = "1 / -1";
      video.style.width = "100%";
      video.style.borderRadius = "4px";
      video.style.marginTop = "4px";
      row.appendChild(video);

      // Explicit download link, forced as an attachment server-side --
      // more reliable than relying on the <video> controls' own save
      // option (which iPad Safari in particular often lacks for
      // inline playback).
      var dl = document.createElement("a");
      dl.className = "ng-btn ng-vg-download";
      dl.href = API + "/jobs/" + item.jobId + "/download";
      dl.download = "animate-" + item.jobId + ".mp4";
      dl.textContent = "Download video";
      dl.style.gridColumn = "1 / -1";
      dl.style.marginTop = "4px";
      row.appendChild(dl);
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

  // Only one job ever renders at a time (single backend worker), so the
  // main-stage log just tails whichever queue item is currently
  // "rendering" -- plenty of room there to actually read it, unlike the
  // narrow rail queue row it used to live in.
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

    els.refFileBtn.addEventListener("click", function () { els.refFile.click(); });
    els.refFile.addEventListener("change", onDiskFile);
    els.refUseGen.addEventListener("click", useGenerateReference);
    if (els.modeT2v) els.modeT2v.addEventListener("change", onModeChange);

    els.duration.addEventListener("input", function () {
      els.durationVal.textContent = parseFloat(els.duration.value).toFixed(1) + "s";
    });

    els.enhanceBtn.addEventListener("click", enhancePrompt);
    if (els.durationAuto) els.durationAuto.addEventListener("click", estimateDuration);
    if (els.discussToggle) els.discussToggle.addEventListener("click", toggleDiscussPanel);
    if (els.discussSend) els.discussSend.addEventListener("click", sendDiscussMessage);
    if (els.discussInput) {
      els.discussInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendDiscussMessage();
        }
      });
    }
    if (els.chatToggle) els.chatToggle.addEventListener("click", toggleChatPanel);
    if (els.chatSend) els.chatSend.addEventListener("click", sendChatMessage);
    if (els.chatInput) {
      els.chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendChatMessage();
        }
      });
      // Paste an image straight from the clipboard; let plain text paste
      // through untouched.
      els.chatInput.addEventListener("paste", function (e) {
        var items = (e.clipboardData && e.clipboardData.items) || [];
        var files = [];
        for (var i = 0; i < items.length; i++) {
          if (items[i].kind === "file" && /^image\//.test(items[i].type)) {
            files.push(items[i].getAsFile());
          }
        }
        if (files.length) {
          e.preventDefault();
          addChatImageFiles(files);
        }
      });
    }
    if (els.chatAttach && els.chatFile) {
      els.chatAttach.addEventListener("click", function () {
        els.chatFile.click();
      });
      els.chatFile.addEventListener("change", function () {
        addChatImageFiles(els.chatFile.files);
        els.chatFile.value = "";
      });
    }
    if (els.logToggle) els.logToggle.addEventListener("click", toggleLogView);
    if (els.logBack) els.logBack.addEventListener("click", closeLogView);
    if (els.logTabRecent) els.logTabRecent.addEventListener("click", function () { setLogTab("recent"); });
    if (els.logTabArchive) els.logTabArchive.addEventListener("click", function () { setLogTab("years"); });
    els.generateBtn.addEventListener("click", generateVideo);
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
    var on = !!(active && active.task === "videogen");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (els.controlsPane && (!active || (active.task !== "generate" && active.task !== "chat" && active.task !== "rachel"))) {
      els.controlsPane.style.display = on ? "none" : "";
    }
    if (!on) return;

    init();
    ensureStatusChecked();
    renderQueue();
  }

  window.VideoGenNG = { sync: sync };
})();
