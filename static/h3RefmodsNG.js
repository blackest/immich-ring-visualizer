// H3 view: the RefMod rows (GET /api/ng/h3/refmods), shown only on the "H3 Ref2VA Q8" model. A RefMod is a
// saved reference latent the Ref2VA model reads as extra rows; each row costs denoise time in
// proportion to its rows. Also locks off the speed adapters that were not made for those weights.
// Made once by h3NG.js with its shared element map:
//   var refmodUi = H3RefmodsNG({ els: els, api: "/api/ng/h3" });
(function () {
  "use strict";

  window.H3RefmodsNG = function (ctx) {
    var els = ctx.els, API = ctx.api;

    // ---- RefMod references (GET /api/ng/h3/refmods) ----
    // Saved reference latents that the Ref2VA model reads as extra rows. Shown only
    // on the "H3 Ref2VA Q8" model; each row costs denoise time in proportion to its
    // `rows`, and a render is one 5 s window with no keyframe or speed adapter, so
    // the speed checkboxes are locked off while this model is picked.
    var refLib = { refmods: [], ready: false, ref2vaOk: false, supported: false,
                   maxVideos: 3, loaded: false };

    function refmodEntry(name) {
      for (var i = 0; i < refLib.refmods.length; i++) {
        if (refLib.refmods[i].name === name) return refLib.refmods[i];
      }
      return null;
    }

    function refmodsOn() { return !!(els.model && els.model.value === "h3ref"); }

    function refmodsNoteText() {
      if (refLib.ready) {
        return refLib.refmods.length
          ? "One 5 s window. Each reference adds denoise time (about 3x for a 2,800-row file)."
          : "No RefMod files yet -- put .safetensors RefMods in the H3 refmods folder.";
      }
      var missing = [];
      if (!refLib.ref2vaOk) missing.push("the Ref2VA Q8 weights");
      if (!refLib.supported) missing.push("a runner build that takes --refmod");
      return "Not available yet: this install is missing " + missing.join(" and ") + ".";
    }

    function updateRefmodHint(row) {
      var hint = row.querySelector(".ng-h3-refmod-hint");
      var entry = refmodEntry(row.querySelector("select").value);
      var bits = [];
      if (entry && entry.rows) bits.push(entry.rows.toLocaleString() + " reference rows");
      if (entry && entry.has_lora) bits.push("also carries a LoRA (not applied)");
      hint.textContent = bits.join(" \u00b7 ");
      hint.style.display = bits.length ? "" : "none";
    }

    function syncRefmodControls() {
      if (!els.refmodsBlock) return;
      var on = refmodsOn();
      els.refmodsBlock.hidden = !on;
      if (els.refmodsNote) els.refmodsNote.textContent = on ? refmodsNoteText() : "";
      var rows = els.refmodsRows.children;
      if (!on || !refLib.ready) {
        while (rows.length) els.refmodsRows.removeChild(rows[0]);
      }
      if (els.refmodAdd) {
        els.refmodAdd.disabled = !refLib.ready || !refLib.refmods.some(function (e) { return e.ok; })
          || rows.length >= refLib.maxVideos;
      }
      // Turbo / v1.3 were not made for the Ref2VA weights: lock them off here.
      // Fast (the TaoMate 3-step adapter) is the exception -- tested with a RefMod
      // on the Ref2VA Q8 -- so it is left alone.
      [els.turbo, els.v13].forEach(function (box) {
        if (!box) return;
        if (on) { box.checked = false; box.disabled = true; }
      });
    }

    // Strength a new RefMod row starts at, and what an unreadable box falls back to. 1.0 copied the
    // reference too hard (the clip opened on the reference photo, its wall colour bled into the
    // scene); 0.7 is the usual recommendation.
    var REFMOD_DEFAULT_STRENGTH = 0.7;

    function addRefmodRow(name, strength) {
      if (!els.refmodsRows) return;
      var row = document.createElement("div");
      row.className = "ng-h3-refmod-row";
      row.style.cssText = "margin-bottom: 6px";
      var line = document.createElement("div");
      line.style.cssText = "display: flex; gap: 6px; align-items: center";
      var select = document.createElement("select");
      select.className = "ng-gen-input";
      select.style.cssText = "flex: 1; margin-bottom: 0";
      // The shared library can hold ~1,500 files: a filter box narrows the list
      // (name or description), keeping the current pick if it still matches.
      var filter = document.createElement("input");
      filter.type = "search";
      filter.className = "ng-gen-input";
      filter.placeholder = "filter " + refLib.refmods.length + " RefMods…";
      filter.style.cssText = "width: 11em; margin-bottom: 0";
      function populate() {
        var keep = select.value;
        var q = filter.value.trim().toLowerCase();
        select.textContent = "";
        refLib.refmods.forEach(function (entry) {
          if (q && (entry.name + " " + (entry.description || "")).toLowerCase().indexOf(q) < 0) return;
          var opt = document.createElement("option");
          opt.value = entry.name;
          opt.textContent = (entry.ok ? "" : "(unusable) ") + entry.name;
          opt.disabled = !entry.ok;
          opt.title = entry.reason || "";
          select.appendChild(opt);
        });
        var opts = Array.prototype.slice.call(select.options);
        var pick = opts.filter(function (o) { return o.value === keep; })[0]
          || opts.filter(function (o) { return !o.disabled; })[0];
        if (pick) select.value = pick.value;
        if (row.parentNode) updateRefmodHint(row);
      }
      populate();
      var strengthInput = document.createElement("input");
      strengthInput.type = "number";
      strengthInput.className = "ng-gen-input";
      strengthInput.step = "0.05";
      strengthInput.min = "0";
      strengthInput.max = "1";
      strengthInput.style.cssText = "width: 76px; margin-bottom: 0";
      var remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ng-gen-btn";
      remove.style.cssText = "width: auto";
      remove.textContent = "\u00d7";
      remove.title = "Remove this RefMod";
      var hint = document.createElement("div");
      hint.className = "ng-gen-label ng-h3-refmod-hint";
      hint.style.cssText = "margin-top: 2px; text-transform: none; letter-spacing: 0; display: none";
      line.appendChild(filter);
      line.appendChild(select);
      line.appendChild(strengthInput);
      line.appendChild(remove);
      row.appendChild(line);
      row.appendChild(hint);
      els.refmodsRows.appendChild(row);

      var first = refLib.refmods.filter(function (e) { return e.ok; })[0];
      if (name && refmodEntry(name) && refmodEntry(name).ok) select.value = name;
      else if (first) select.value = first.name;
      strengthInput.value = strength != null ? strength : REFMOD_DEFAULT_STRENGTH;
      select.addEventListener("change", function () { updateRefmodHint(row); });
      filter.addEventListener("input", populate);
      remove.addEventListener("click", function () {
        els.refmodsRows.removeChild(row);
        syncRefmodControls();
      });
      updateRefmodHint(row);
      syncRefmodControls();
    }

    // [{name, strength}] for the rows on screen -- only meaningful on h3ref.
    function selectedRefmods() {
      if (!els.refmodsRows || !refmodsOn()) return [];
      var out = [];
      Array.prototype.forEach.call(els.refmodsRows.children, function (row) {
        var name = row.querySelector("select").value;
        // The row's FIRST <input> is the filter box, so ask for the number box by type: reading
        // the first one gave NaN every time and sent strength 1.0 whatever was typed.
        var strength = parseFloat(row.querySelector('input[type="number"]').value);
        if (name) out.push({ name: name, strength: isNaN(strength) ? REFMOD_DEFAULT_STRENGTH : strength });
      });
      return out;
    }

    function ensureRefmodsLoaded() {
      if (refLib.loaded) return;
      refLib.loaded = true;
      fetch(API + "/refmods")
        .then(function (r) { return r.json(); })
        .then(function (payload) {
          refLib.refmods = (payload && payload.refmods) || [];
          refLib.ready = !!(payload && payload.ready);
          refLib.ref2vaOk = !!(payload && payload.ref2va_ok);
          refLib.supported = !!(payload && payload.refmod_supported);
          refLib.maxVideos = (payload && payload.max_videos) || 3;
          syncRefmodControls();
        })
        .catch(function () {
          refLib.loaded = false; // let a later sync retry
        });
    }

    return {
      on: refmodsOn, sync: syncRefmodControls, addRow: addRefmodRow, selected: selectedRefmods,
      ensureLoaded: ensureRefmodsLoaded,
      reload: function () { refLib.loaded = false; ensureRefmodsLoaded(); },
      hasRoom: function () { return els.refmodsRows.children.length < refLib.maxVideos; }
    };
  };
})();
