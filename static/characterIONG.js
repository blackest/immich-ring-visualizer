/**
 * characterIONG.js -- Save / Load a character as a single .json file.
 *
 * "Save" serializes the active ProjectManager tab to <name>.character.json
 * and downloads it; "Load" reads such a file back into a fresh tab.
 *
 * Design (see the discussion that led here): the file stores POINTERS, not
 * pixels, for everything heavy --
 *   - source: which video (filename + byte size) / folder / Immich, plus
 *     the anchor and the selected/rejected frame list. On Load the user
 *     re-supplies the video/folder; the selection is re-derivable from it.
 *   - generation: just the character id -- the rendered PNGs stay on disk
 *     under exportsNG/<id>/character/, rediscovered via sheet-meta.
 * The ONE image it embeds is a small THUMBNAIL of the current reference
 * (<=256px, JPEG) -- enough to recognise who the character is at a glance
 * (so you don't open Mary's file while hunting for Jayne), not a
 * generation-quality reference. Re-supplying the source restores the real
 * reference frame.
 *
 * Everything past identity.name is best-effort: a field that's absent
 * (older file, nothing analyzed yet) is simply skipped on Load.
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after generateNG.js, before bootstrapWiringNG.js.
 */
(function () {
  "use strict";

  var SCHEMA = "ringviz/character@1";

  // Keys copied verbatim between project <-> doc.settings (all plain
  // numbers/enums/bools already on CharacterProject).
  var SETTING_KEYS = [
    "simThreshold", "blurThreshold", "cacheFormatPng", "ringScale",
    "squeezeMinPct", "squeezeUserOverridden", "ringSortMetric",
    "sharpCutoffEnabled", "sharpMinVal", "folderRefIndex",
    "immichAnalyzeRefIndex", "rankedSortMetric", "immichRankedSortMetric",
  ];

  function $(id) {
    return document.getElementById(id);
  }

  function slugName(s) {
    var v = String(s || "")
      .replace(/[^A-Za-z0-9 _-]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    return v || "character";
  }

  function fileSafe(s) {
    return slugName(s).replace(/\s+/g, "_") || "character";
  }

  function num(v, dflt) {
    var n = parseInt(v, 10);
    return isNaN(n) ? dflt : n;
  }

  // ---- render-settings panel (#ng-gen-*) read/apply -------------------
  function readRenderInputs() {
    var il = $("ng-gen-identity-lock");
    return {
      width: num($("ng-gen-width") && $("ng-gen-width").value, 2048),
      height: num($("ng-gen-height") && $("ng-gen-height").value, 2048),
      steps: num($("ng-gen-steps") && $("ng-gen-steps").value, 28),
      style: ($("ng-gen-style") && $("ng-gen-style").value) || "none",
      wardrobe: ($("ng-gen-wardrobe") && $("ng-gen-wardrobe").value) || "",
      hair_color: ($("ng-gen-hair") && $("ng-gen-hair").value) || "",
      identity_lock: il ? !!il.checked : true,
    };
  }

  function applyRenderInputs(r) {
    if (!r) return;
    var set = function (id, v) {
      var e = $(id);
      if (e && v !== undefined && v !== null && v !== "") e.value = v;
    };
    set("ng-gen-width", r.width);
    set("ng-gen-height", r.height);
    set("ng-gen-steps", r.steps);
    set("ng-gen-wardrobe", r.wardrobe);
    set("ng-gen-hair", r.hair_color);
    // The style <select> is populated async by GenerateNG's /presets
    // fetch -- add the saved value as an option now so it sticks;
    // renderStyleOptions() preserves the current value when it rebuilds.
    var st = $("ng-gen-style");
    if (st && r.style) {
      var has = Array.prototype.some.call(st.options, function (o) {
        return o.value === r.style;
      });
      if (!has) {
        var opt = document.createElement("option");
        opt.value = r.style;
        opt.textContent = r.style.replace(/_/g, " ");
        st.appendChild(opt);
      }
      st.value = r.style;
    }
    var il = $("ng-gen-identity-lock");
    if (il && r.identity_lock !== undefined) il.checked = !!r.identity_lock;
    var sp = $("ng-gen-size-preset");
    if (sp) sp.value = "custom";
  }

  // ---- source pointer ----------------------------------------------------
  function framesToNames(project, frameSet) {
    var base = (project.ring && project.ring.baseResults) || [];
    return Array.from(frameSet || []).map(function (f) {
      var hit = base.find(function (x) {
        return x.frame === f;
      });
      return hit ? hit.filename : f;
    });
  }

  function buildSource(p) {
    if (p.immichRing || (p.ring && p.ring.sourceType === "immich")) {
      return {
        kind: "immich",
        anchorAssetId: p.immichRing ? p.immichRing.centerAssetId : null,
        anchorFilename: p.immichRing ? p.immichRing.centerFilename : null,
        selectedAssetIds: Array.from(p.selectedAssetIds || []),
        selectedFrames: Array.from(p.selectedFrames || []),
      };
    }
    if (p.ring && p.ring.sourceType === "folder") {
      return {
        kind: "folder",
        name: (p.job && p.job.sourceName) || null,
        anchorIndex: p.ring.refFrameIdx,
        selected: framesToNames(p, p.selectedFrames),
        rejected: framesToNames(p, p.excludedFrames),
      };
    }
    if (
      p.video ||
      (p.ring && p.ring.sourceType === "video") ||
      (p.job && p.job.sourceType === "video")
    ) {
      return {
        kind: "video",
        name:
          (p.job && p.job.sourceName) ||
          (p.videoFile && p.videoFile.name) ||
          null,
        bytes: p.videoFile ? p.videoFile.size : null,
        anchor: p.ring
          ? p.ring.refFrameIdx
          : p.job
          ? p.job.refFrame
          : null,
        selected: Array.from(p.selectedFrames || []),
        rejected: Array.from(p.excludedFrames || []),
      };
    }
    return null;
  }

  function sourceHint(src) {
    if (!src) return "";
    if (src.kind === "video") {
      return (
        "Re-load " +
        (src.name || "the source video") +
        " to rebuild the frame selection (" +
        (src.selected || []).length +
        " frames)."
      );
    }
    if (src.kind === "folder") {
      return (
        "Re-load the folder / zip “" +
        (src.name || "?") +
        "” to rebuild the image selection."
      );
    }
    if (src.kind === "immich") {
      return "Re-run the Immich search to rebuild the selection.";
    }
    return "";
  }

  // ---- reference thumbnail embed --------------------------------------
  var THUMB_MAX_PX = 256;

  // blob -> small JPEG data URI (longest edge <= THUMB_MAX_PX). Falls
  // back to null on any decode/encode failure -- the embed is optional.
  function blobToThumbDataUri(blob) {
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(blob);
      var img = new Image();
      img.onload = function () {
        try {
          var scale = Math.min(
            1,
            THUMB_MAX_PX / Math.max(img.naturalWidth, img.naturalHeight)
          );
          var w = Math.max(1, Math.round(img.naturalWidth * scale));
          var h = Math.max(1, Math.round(img.naturalHeight * scale));
          var cv = document.createElement("canvas");
          cv.width = w;
          cv.height = h;
          cv.getContext("2d").drawImage(img, 0, 0, w, h);
          resolve(cv.toDataURL("image/jpeg", 0.8));
        } catch (e) {
          resolve(null);
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      img.onerror = function () {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    });
  }

  function resolveReferenceThumb(p) {
    var url = null;
    var origin = null;
    var gref =
      window.GenerateNG &&
      window.GenerateNG.getActiveReference &&
      window.GenerateNG.getActiveReference();
    if (gref && gref.url) {
      url = gref.url;
      origin = gref.origin || "generate";
    } else if (p.ring && p.ring.anchorUrl) {
      url = p.ring.anchorUrl;
      origin = "anchor";
    } else if (p.immichRing && p.immichRing.centerAssetId) {
      url = "/api/ng/thumb/" + p.immichRing.centerAssetId;
      origin = "immich:" + p.immichRing.centerAssetId;
    }
    if (!url) return Promise.resolve(null);
    return fetch(url)
      .then(function (r) {
        return r.ok ? r.blob() : null;
      })
      .then(function (b) {
        return b ? blobToThumbDataUri(b) : null;
      })
      .then(function (d) {
        return d ? { thumb: d, origin: origin } : null;
      })
      .catch(function () {
        return null;
      });
  }

  // ---- save ------------------------------------------------------------
  function buildDoc(project) {
    var settings = {};
    SETTING_KEYS.forEach(function (k) {
      if (project[k] !== undefined) settings[k] = project[k];
    });
    var doc = {
      schema: SCHEMA,
      savedAt: new Date().toISOString(),
      identity: { name: project.name },
      task: project.task || null,
      settings: settings,
      render: readRenderInputs(),
      generation: { characterId: slugName(project.name) },
    };
    var src = buildSource(project);
    if (src) doc.source = src;
    return doc;
  }

  function downloadJson(filename, obj) {
    var blob = new Blob([JSON.stringify(obj, null, 2)], {
      type: "application/json",
    });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 0);
  }

  function save(project) {
    if (!project) {
      alert("No character open to save.");
      return;
    }
    var doc = buildDoc(project);
    resolveReferenceThumb(project).then(function (ref) {
      if (ref) doc.reference = ref; // { thumb: <data URI>, origin }
      downloadJson(fileSafe(project.name) + ".character.json", doc);
    });
  }

  // ---- load -----------------------------------------------------------
  function applyDoc(doc) {
    var pm = window.ProjectManager;
    if (!pm) {
      alert("Project manager not ready.");
      return;
    }
    pm.createProject(doc.identity.name); // creates + setActive + render
    var p = pm.getActive();
    if (!p) return;

    if (doc.task) p.task = doc.task;
    var s = doc.settings || {};
    SETTING_KEYS.forEach(function (k) {
      if (s[k] !== undefined) p[k] = s[k];
    });
    applyRenderInputs(doc.render);

    // Kept for a later pass / manual rehydrate -- Load doesn't re-run
    // analysis or restore a generation-quality reference (the embed is a
    // recognition thumbnail only); the user re-supplies the source.
    p._savedSource = doc.source || null;
    p._savedThumb = (doc.reference && doc.reference.thumb) || null;

    pm.saveState();
    pm.render();

    var hint = sourceHint(doc.source);
    alert("Loaded “" + doc.identity.name + "”." + (hint ? "\n\n" + hint : ""));
  }

  function load(file) {
    var fr = new FileReader();
    fr.onload = function () {
      var doc;
      try {
        doc = JSON.parse(fr.result);
      } catch (e) {
        alert("Not a valid character file: " + e);
        return;
      }
      if (
        !doc ||
        typeof doc !== "object" ||
        !doc.identity ||
        !doc.identity.name
      ) {
        alert("Not a character file (missing identity.name).");
        return;
      }
      applyDoc(doc);
    };
    fr.onerror = function () {
      alert("Could not read that file.");
    };
    fr.readAsText(file);
  }

  // ---- wiring -------------------------------------------------------------
  function wire() {
    var saveBtn = $("ng-save-project");
    var loadBtn = $("ng-load-project");
    var fileInput = $("ng-load-project-file");

    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.title = "Load a character .json file";
      loadBtn.addEventListener("click", function () {
        if (fileInput) fileInput.click();
      });
    }
    if (fileInput) {
      fileInput.addEventListener("change", function () {
        var f = fileInput.files && fileInput.files[0];
        if (f) load(f);
        fileInput.value = "";
      });
    }
    if (saveBtn) {
      saveBtn.addEventListener("click", function () {
        save(window.ProjectManager && window.ProjectManager.getActive());
      });
    }
  }

  window.CharacterIO = { save: save, load: load };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
