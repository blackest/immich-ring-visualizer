// H3 view: the LoRA rows. The library is whatever sits in the H3 loras folder (GET /api/ng/h3/loras);
// the rows are built here so their count can follow what the installed runner can stack, a LoRA the
// runner cannot read yet but could after converting gets a "Make usable" button, and selected() is
// what the render form sends. Made once by h3NG.js with its shared element map:
//   var loraUi = H3LorasNG({ els: els, api: "/api/ng/h3" });
(function () {
  "use strict";

  window.H3LorasNG = function (ctx) {
    var els = ctx.els, API = ctx.api;

    // ---- User style/character LoRAs (GET /api/ng/h3/loras) ----
    // The library is whatever sits in the H3 loras folder; rows are built
    // here rather than in the HTML so the count can follow what the installed
    // runner can stack (max_stack, or max_stack_with_speed while turbo / v1.3
    // is ticked).
    var loraLib = { loras: [], max: 0, maxSpeed: 0, loaded: false };

    function loraLimit() {
      var speed = !!((els.turbo && els.turbo.checked) || (els.v13 && els.v13.checked)
        || (els.tristep && els.tristep.checked) || (els.flex && els.flex.checked));
      return speed ? loraLib.maxSpeed : loraLib.max;
    }

    function loraEntry(name) {
      for (var i = 0; i < loraLib.loras.length; i++) {
        if (loraLib.loras[i].name === name) return loraLib.loras[i];
      }
      return null;
    }

    function updateLoraHint(row) {
      var hint = row.querySelector(".ng-h3-lora-hint");
      var entry = loraEntry(row.querySelector("select").value);
      var words = entry && entry.trigger_words && entry.trigger_words.length
        ? "Trigger: " + entry.trigger_words.join(" / ") : "";
      hint.textContent = words;
      hint.style.display = words ? "" : "none";
    }

    function syncLoraControls() {
      if (!els.lorasBlock) return;
      var limit = loraLimit();
      els.lorasBlock.hidden = !(loraLib.loras.length && (limit > 0 || els.lorasRows.children.length));
      var rows = els.lorasRows.children;
      // Over the limit (e.g. turbo got ticked on a single-slot runner): drop
      // the surplus rows rather than let a doomed request be sent.
      while (rows.length > limit) els.lorasRows.removeChild(rows[rows.length - 1]);
      if (els.loraAdd) els.loraAdd.disabled = rows.length >= limit;
    }

    function addLoraRow(name, strength) {
      if (!els.lorasRows) return;
      var row = document.createElement("div");
      row.className = "ng-h3-lora-row";
      row.style.cssText = "margin-bottom: 6px";
      var line = document.createElement("div");
      line.style.cssText = "display: flex; gap: 6px; align-items: center";
      var select = document.createElement("select");
      select.className = "ng-gen-input";
      select.style.cssText = "flex: 1; margin-bottom: 0";
      loraLib.loras.forEach(function (entry) {
        var opt = document.createElement("option");
        opt.value = entry.name;
        opt.textContent = (entry.ok ? "" : "(unusable) ") + (entry.title || entry.name);
        opt.disabled = !entry.ok;
        opt.title = entry.reason || "";
        select.appendChild(opt);
      });
      var strengthInput = document.createElement("input");
      strengthInput.type = "number";
      strengthInput.className = "ng-gen-input";
      strengthInput.step = "0.05";
      strengthInput.min = "0";
      strengthInput.max = "2";
      strengthInput.style.cssText = "width: 76px; margin-bottom: 0";
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ng-gen-btn";
      remove.style.cssText = "width: auto";
      remove.textContent = "\u00d7";
      remove.title = "Remove this LoRA";
      var hint = document.createElement("div");
      hint.className = "ng-gen-label ng-h3-lora-hint";
      hint.style.cssText = "margin-top: 2px; text-transform: none; letter-spacing: 0; display: none";
      line.appendChild(select);
      line.appendChild(strengthInput);
      line.appendChild(remove);
      row.appendChild(line);
      row.appendChild(hint);
      els.lorasRows.appendChild(row);

      var first = loraLib.loras.filter(function (e) { return e.ok; })[0];
      if (name && loraEntry(name) && loraEntry(name).ok) select.value = name;
      else if (first) select.value = first.name;
      var chosen = loraEntry(select.value);
      strengthInput.value = strength != null ? strength
        : (chosen && chosen.recommended_strength) || 1;
      select.addEventListener("change", function () {
        var e = loraEntry(select.value);
        if (e && e.recommended_strength) strengthInput.value = e.recommended_strength;
        updateLoraHint(row);
      });
      remove.addEventListener("click", function () {
        els.lorasRows.removeChild(row);
        syncLoraControls();
      });
      updateLoraHint(row);
      syncLoraControls();
    }

    // [{name, strength}] for the rows currently on screen.
    function selectedLoras() {
      if (!els.lorasRows) return [];
      var out = [];
      Array.prototype.forEach.call(els.lorasRows.children, function (row) {
        var name = row.querySelector("select").value;
        var strength = parseFloat(row.querySelector("input").value);
        if (name) out.push({ name: name, strength: isNaN(strength) ? 1.0 : strength });
      });
      return out;
    }

    // LoRAs the runner cannot read as they are but that can be converted (kohya-format): one row
    // each with a button. The original is kept in the loras folder's _originals.
    function drawLoraFixes() {
      if (!els.lorasFix) return;
      var fixable = loraLib.loras.filter(function (e) { return !e.ok && e.convertible; });
      els.lorasFix.hidden = !fixable.length;
      els.lorasFix.textContent = "";
      if (!fixable.length) return;
      var head = document.createElement("div");
      head.className = "ng-gen-label";
      head.textContent = "LoRAs that need converting";
      els.lorasFix.appendChild(head);
      fixable.forEach(function (entry) {
        var line = document.createElement("div");
        line.style.cssText = "display: flex; gap: 6px; align-items: center; margin-bottom: 4px";
        var name = document.createElement("span");
        name.style.cssText = "flex: 1; overflow: hidden; text-overflow: ellipsis";
        name.textContent = (entry.title || entry.name) + " (kohya, " + entry.size_mb + " MB)";
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ng-gen-btn";
        btn.style.cssText = "width: auto";
        btn.textContent = "Make usable";
        btn.title = "Rewrites it for the H3 runner. The original is kept in _originals.";
        var note = document.createElement("div");
        note.className = "ng-gen-label";
        note.style.cssText = "text-transform: none; letter-spacing: 0";
        btn.addEventListener("click", function () {
          btn.disabled = true;
          note.textContent = "Converting...";
          fetch(API + "/loras/" + encodeURIComponent(entry.name) + "/convert", { method: "POST" })
            .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
            .then(function (r) {
              if (!r.ok) { note.textContent = r.body.error || "Could not convert it."; btn.disabled = false; return; }
              loraLib.loaded = false;
              ensureLorasLoaded();
            })
            .catch(function (e) { note.textContent = e.message; btn.disabled = false; });
        });
        line.appendChild(name);
        line.appendChild(btn);
        els.lorasFix.appendChild(line);
        els.lorasFix.appendChild(note);
      });
    }

    function ensureLorasLoaded() {
      if (loraLib.loaded) return;
      loraLib.loaded = true;
      fetch(API + "/loras")
        .then(function (r) { return r.json(); })
        .then(function (payload) {
          loraLib.loras = (payload && payload.loras) || [];
          loraLib.max = (payload && payload.max_stack) || 0;
          loraLib.maxSpeed = (payload && payload.max_stack_with_speed) || 0;
          // Rebuild the rows so a just-converted LoRA appears in their dropdowns.
          var kept = selectedLoras();
          if (els.lorasRows) els.lorasRows.textContent = "";
          kept.forEach(function (k) { addLoraRow(k.name, k.strength); });
          syncLoraControls();
          drawLoraFixes();
        })
        .catch(function () {
          loraLib.loaded = false; // let a later sync retry
        });
    }

    return { limit: loraLimit, entry: loraEntry, sync: syncLoraControls, addRow: addLoraRow,
      selected: selectedLoras, ensureLoaded: ensureLorasLoaded };
  };
})();
