/**
 * chatNG.js -- the "Chat" view (bottom-bar Chat task).
 *
 * A plain chat interface in front of the local Ollama daemon, wired to
 * the NG backend proxy at /api/ng/chat/* (routes/chatNG.py). Self-
 * contained: it reads nothing from the rest of the app -- no project, no
 * ring, no reference image. The bottom-bar button just flips the active
 * project's task to "chat"; this view then owns the whole left rail
 * (#ng-chat-pane) and main stage (#ng-chat-main).
 *
 * ONE global conversation. It is module state, not per-project -- switch
 * tabs, switch tasks, come back, it's the same thread. This is
 * deliberate: a single Ollama model is loaded at a time, and juggling
 * several independent histories against it just invites contention and a
 * confused model. Lost on refresh (same as generateNG.js's queue).
 *
 * Streaming: POST the full message list to /api/ng/chat/send, read the
 * response body as a stream, parse the NDJSON Ollama emits line by line,
 * and append each message-content delta into the live assistant bubble.
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after projectManagerNG.js and before bootstrapWiringNG.js (whose first
 * ProjectManager.render() calls ChatNG.sync()).
 */
(function () {
  "use strict";

  var API = "/api/ng/chat";

  // ---- module state (the one conversation) ----
  var inited = false;
  var bootstrapped = false; // models + health fetched once
  var models = []; // [{name, parameter_size, ...}]
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
    els.pane = $("ng-chat-pane");
    els.main = $("ng-chat-main");
    els.controlsPane = $("ng-controls-pane");
    els.unavailable = $("ng-chat-unavailable");
    els.model = $("ng-chat-model");
    els.modelHint = $("ng-chat-model-hint");
    els.system = $("ng-chat-system");
    els.temp = $("ng-chat-temp");
    els.tempVal = $("ng-chat-temp-val");
    els.newBtn = $("ng-chat-new");
    els.transcript = $("ng-chat-transcript");
    els.empty = $("ng-chat-empty");
    els.composer = $("ng-chat-composer");
    els.input = $("ng-chat-input");
    els.send = $("ng-chat-send");
    els.stop = $("ng-chat-stop");
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
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
    var on = !!(active && active.task === "chat");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "flex" : "none";
    // #ng-controls-pane is shared: the Generate view hides it too. Only
    // hide it when Chat is on; when Chat is off, let whoever owns the
    // active task decide (don't un-hide it out from under Generate).
    if (els.controlsPane) {
      if (on) els.controlsPane.style.display = "none";
      else if (!active || active.task !== "generate")
        els.controlsPane.style.display = "";
    }
    if (!on) return;

    init();
    ensureModelsAndHealth();
    renderTranscript();
  }

  // ---- models + daemon health, fetched once ----
  function ensureModelsAndHealth() {
    if (bootstrapped) return;
    bootstrapped = true;

    fetch(API + "/status")
      .then(function (r) {
        return r.json();
      })
      .then(function (h) {
        setUnavailable(!(h && h.reachable), h);
      })
      .catch(function () {
        /* fail open -- the models fetch below will show the real problem */
      });

    fetch(API + "/models")
      .then(function (r) {
        return r.json().then(function (d) {
          return { ok: r.ok, data: d };
        });
      })
      .then(function (r) {
        if (!r.ok || !r.data || r.data.error) {
          bootstrapped = false; // let a later sync retry
          setUnavailable(true, { error: r.data && r.data.error });
          return;
        }
        models = r.data.models || [];
        renderModelOptions(r.data.loaded || []);
      })
      .catch(function () {
        bootstrapped = false;
      });
  }

  function renderModelOptions(loaded) {
    if (!els.model) return;
    var cur = els.model.value;
    els.model.innerHTML = "";
    if (!models.length) {
      var o = document.createElement("option");
      o.value = "";
      o.textContent = "No models installed";
      els.model.appendChild(o);
      els.model.disabled = true;
      return;
    }
    els.model.disabled = false;
    models.forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m.name;
      var size = m.parameter_size ? " · " + m.parameter_size : "";
      var warm = loaded.indexOf(m.name) >= 0 ? " · loaded" : "";
      opt.textContent = m.name + size + warm;
      els.model.appendChild(opt);
    });

    // Preselect: keep the user's pick if still valid, else the model the
    // running agent already has warm in memory, else the first one.
    var pick =
      (cur && modelExists(cur) && cur) ||
      (loaded && loaded.filter(modelExists)[0]) ||
      models[0].name;
    els.model.value = pick;
    updateModelHint(loaded);
    els.model.addEventListener("change", function () {
      updateModelHint(loaded);
    });
  }

  function modelExists(name) {
    return models.some(function (m) {
      return m.name === name;
    });
  }

  function updateModelHint(loaded) {
    if (!els.modelHint) return;
    var name = els.model.value;
    if (loaded && loaded.indexOf(name) >= 0) {
      els.modelHint.textContent = "Already loaded in memory.";
    } else {
      els.modelHint.textContent =
        "Not loaded yet — the first reply will wait on the model loading.";
    }
  }

  function setUnavailable(off, health) {
    unavailable = !!off;
    if (els.send) els.send.disabled = unavailable || streaming;
    if (els.input) els.input.disabled = unavailable;
    if (!els.unavailable) return;
    if (unavailable) {
      var base = (health && health.base_url) || "http://localhost:11434";
      els.unavailable.textContent =
        "Ollama isn't reachable at " +
        base +
        (health && health.error ? " (" + health.error + ")" : "") +
        ". Start it with `ollama serve` (or the Ollama app) and reopen this view.";
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
    var model = els.model && els.model.value;
    if (!model) {
      setUnavailable(true, { error: "no model selected" });
      return;
    }

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
      model: model,
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
          flush(buf, assistant, true);
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
      return; // partial / non-JSON keepalive -- ignore
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
    if (els.model) els.model.disabled = on || !models.length;
  }

  // ---- transcript view ----
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
    who.textContent = m.role === "user" ? "You" : "Assistant";
    var body = document.createElement("div");
    body.className = "ng-chat-body";
    if (m.error) {
      body.classList.add("ng-chat-error");
      body.textContent =
        (m.content ? m.content + "\n\n" : "") + "⚠ " + m.error;
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

  window.ChatNG = { sync: sync };
})();
