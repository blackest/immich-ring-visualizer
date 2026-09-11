/**
 * rachelNG.js -- the "Rachel" view (bottom-bar Rachel task).
 *
 * A plain chat interface in front of a Hermes agent gateway (NousResearch
 * hermes-agent), wired to the NG backend proxy at /api/ng/rachel/*
 * (routes/rachelNG.py). Structurally a trimmed copy of chatNG.js:
 *
 *   * ONE fixed agent -- no model picker. The rail pane is just the
 *     status banner, an optional system prompt, a temperature slider and
 *     "New chat".
 *   * The backend translates the gateway's SSE stream into the same
 *     NDJSON line shape chatNG.js uses, so readStream() here is identical.
 *   * Replies can take minutes to start (long agent loop + big context on
 *     a slow local backend); the Stop button aborts via AbortController.
 *
 * ONE global conversation, module state, not per-project -- switch tabs
 * or tasks and come back to the same thread. Lost on refresh (the agent
 * keeps its own server-side memory regardless, via the session headers
 * the backend sends).
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after chatNG.js and before bootstrapWiringNG.js (whose first
 * ProjectManager.render() calls RachelNG.sync()).
 */
(function () {
  "use strict";

  var API = "/api/ng/rachel";

  // ---- module state (the one conversation) ----
  var inited = false;
  var healthChecked = false;
  var conversation = []; // [{role: "user"|"assistant", content, error?}]
  var streaming = false;
  var abortCtl = null;
  var unavailable = false;

  // ---- DOM (owned here; queried lazily so load order can't bite) ----
  var els = {};
  function $(id) {
    return document.getElementById(id);
  }
  function refreshEls() {
    els.pane = $("ng-rachel-pane");
    els.main = $("ng-rachel-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-rachel-unavailable");
    els.system = $("ng-rachel-system");
    els.temp = $("ng-rachel-temp");
    els.tempVal = $("ng-rachel-temp-val");
    els.newBtn = $("ng-rachel-new");
    els.transcript = $("ng-rachel-transcript");
    els.empty = $("ng-rachel-empty");
    els.composer = $("ng-rachel-composer");
    els.input = $("ng-rachel-input");
    els.send = $("ng-rachel-send");
    els.stop = $("ng-rachel-stop");
  }

  // ---- one-time wiring ----
  function init() {
    if (inited) return;
    refreshEls();
    if (!els.pane) return;
    inited = true;

    els.composer.addEventListener("submit", function (e) {
      e.preventDefault();
      submit();
    });

    // Enter sends, Shift+Enter is a newline.
    els.input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });
    els.input.addEventListener("input", autoGrow);

    els.temp.addEventListener("input", function () {
      els.tempVal.textContent = Number(els.temp.value).toFixed(1);
    });

    els.stop.addEventListener("click", function () {
      if (abortCtl) abortCtl.abort();
    });

    els.newBtn.addEventListener("click", function () {
      if (streaming && abortCtl) abortCtl.abort();
      conversation = [];
      renderTranscript();
      els.input.focus();
    });
  }

  function autoGrow() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
  }

  // ---- called from ProjectManager.render() every tick ----
  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "rachel");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "flex" : "none";
    // #ng-controls-pane is shared with the Generate and Chat views. Only
    // hide it when Rachel is on; when Rachel is off, don't un-hide it out
    // from under whichever of those owns the active task. RachelNG.sync
    // runs last in render(), so forcing "none" here wins for "rachel".
    if (els.controlsPane) {
      if (on) els.controlsPane.style.display = "none";
      else if (!active || (active.task !== "generate" && active.task !== "chat"))
        els.controlsPane.style.display = "";
    }
    if (!on) return;

    init();
    ensureHealth();
    renderTranscript();
  }

  // ---- gateway health, checked once ----
  function ensureHealth() {
    if (healthChecked) return;
    healthChecked = true;
    fetch(API + "/status")
      .then(function (r) {
        return r.json();
      })
      .then(function (h) {
        setUnavailable(!(h && h.reachable), h);
      })
      .catch(function () {
        healthChecked = false; // let a later sync retry
      });
  }

  function setUnavailable(off, health) {
    unavailable = !!off;
    if (els.send) els.send.disabled = unavailable || streaming;
    if (els.input) els.input.disabled = unavailable;
    if (!els.unavailable) return;
    if (unavailable) {
      var base = (health && health.base_url) || "the Hermes gateway";
      els.unavailable.textContent =
        "Can't reach Rachel at " +
        base +
        (health && health.error ? " (" + health.error + ")" : "") +
        ". Check the gateway is up and reopen this view.";
      els.unavailable.style.display = "block";
    } else {
      els.unavailable.style.display = "none";
    }
  }

  // ---- send a turn ----
  function submit() {
    if (streaming || unavailable) return;
    var text = (els.input.value || "").trim();
    if (!text) return;

    conversation.push({ role: "user", content: text });
    els.input.value = "";
    autoGrow();

    var assistant = { role: "assistant", content: "" };
    conversation.push(assistant);
    renderTranscript();

    var messages = [];
    var sys = (els.system.value || "").trim();
    if (sys) messages.push({ role: "system", content: sys });
    conversation.forEach(function (m) {
      if (m === assistant) return; // don't send the empty placeholder
      messages.push({ role: m.role, content: m.content });
    });

    var body = {
      messages: messages,
      options: { temperature: Number(els.temp.value) },
    };

    setStreaming(true);
    abortCtl = new AbortController();

    fetch(API + "/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: abortCtl.signal,
    })
      .then(function (res) {
        if (!res.ok) {
          return res
            .json()
            .catch(function () {
              return { error: "HTTP " + res.status };
            })
            .then(function (d) {
              throw new Error((d && d.error) || "HTTP " + res.status);
            });
        }
        return readStream(res, assistant);
      })
      .catch(function (e) {
        if (e && e.name === "AbortError") {
          assistant.content += assistant.content ? "\n\n[stopped]" : "[stopped]";
        } else {
          assistant.error = String((e && e.message) || e);
        }
      })
      .then(function () {
        setStreaming(false);
        abortCtl = null;
        renderTranscript();
        els.input.focus();
      });
  }

  // Read the NDJSON stream from /send, appending message deltas onto
  // `assistant` and repainting just its bubble as they arrive.
  function readStream(res, assistant) {
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = "";

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          flush(buf, assistant);
          return;
        }
        buf += decoder.decode(chunk.value, { stream: true });
        var nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          var line = buf.slice(0, nl);
          buf = buf.slice(nl + 1);
          handleLine(line, assistant);
        }
        return pump();
      });
    }
    return pump();
  }

  function flush(line, assistant) {
    if (line && line.trim()) handleLine(line, assistant);
  }

  function handleLine(line, assistant) {
    line = line.trim();
    if (!line) return;
    var obj;
    try {
      obj = JSON.parse(line);
    } catch (e) {
      return; // partial / non-JSON -- ignore
    }
    if (obj.error) {
      assistant.error = obj.error;
      return;
    }
    if (obj.message && typeof obj.message.content === "string") {
      assistant.content += obj.message.content;
      paintLastBubble(assistant);
    }
  }

  function setStreaming(on) {
    streaming = on;
    if (els.send) els.send.disabled = on || unavailable;
    if (els.stop) els.stop.style.display = on ? "" : "none";
  }

  // ---- transcript view (reuses the .ng-chat-* bubble styles) ----
  function renderTranscript() {
    if (!els.transcript) return;
    var atBottom = nearBottom();
    els.transcript.innerHTML = "";
    if (!conversation.length) {
      if (els.empty) {
        els.transcript.appendChild(els.empty);
        els.empty.style.display = "";
      }
      return;
    }
    conversation.forEach(function (m) {
      els.transcript.appendChild(bubbleFor(m));
    });
    if (atBottom) scrollToBottom();
  }

  function bubbleFor(m) {
    var row = document.createElement("div");
    row.className = "ng-chat-msg ng-chat-" + m.role;
    var who = document.createElement("div");
    who.className = "ng-chat-who";
    who.textContent = m.role === "user" ? "You" : "Rachel";
    var body = document.createElement("div");
    body.className = "ng-chat-body";
    if (m.error) {
      body.classList.add("ng-chat-error");
      body.textContent = (m.content ? m.content + "\n\n" : "") + "⚠ " + m.error;
    } else if (!m.content && m.role === "assistant") {
      body.classList.add("ng-chat-typing");
      body.textContent = "…";
    } else {
      body.textContent = m.content;
    }
    row.appendChild(who);
    row.appendChild(body);
    return row;
  }

  // Repaint only the last bubble's body during streaming (cheap, no full
  // re-render on every token).
  function paintLastBubble(assistant) {
    var rows = els.transcript.querySelectorAll(".ng-chat-msg");
    var last = rows[rows.length - 1];
    if (!last) return;
    var body = last.querySelector(".ng-chat-body");
    if (!body) return;
    body.classList.remove("ng-chat-typing");
    body.textContent = assistant.content;
    if (nearBottom()) scrollToBottom();
  }

  function nearBottom() {
    var t = els.transcript;
    if (!t) return true;
    return t.scrollHeight - t.scrollTop - t.clientHeight < 80;
  }
  function scrollToBottom() {
    if (els.transcript) els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  window.RachelNG = { sync: sync };
})();
