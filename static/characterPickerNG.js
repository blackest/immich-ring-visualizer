/**
 * characterPickerNG.js -- the Generate view's landing-state picker grid.
 *
 * Shown by projectManagerNG.js's renderMain() in place of the plain "No
 * project open" placeholder whenever no project is active. Lists every
 * known character (GET /api/ng/characters -- see routes/charactersNG.py /
 * character_sheetNG.list_characters_ng()) as a clickable tile; clicking
 * one either loads its Saved project doc (GET /api/ng/characters/<id>/doc,
 * fed to window.CharacterIO.loadDoc) or, for a generation-only character
 * with no Saved doc yet, opens a fresh project under that name on the
 * Generate task so its existing sheet/shots are reachable immediately.
 *
 * Classic script sharing page scope with the other *NG.js files. Loaded
 * after characterIONG.js (needs window.CharacterIO.loadDoc), before
 * bootstrapWiringNG.js.
 */
(function () {
  "use strict";

  var els = {};
  var characters = [];

  function $(id) {
    return document.getElementById(id);
  }

  function refreshEls() {
    els.wrap = $("ng-character-picker");
    els.empty = $("ng-char-picker-empty");
    els.grid = $("ng-char-picker-grid");
  }

  function render() {
    if (!els.grid) return;
    els.grid.innerHTML = "";
    if (els.empty) els.empty.style.display = characters.length ? "none" : "";
    characters.forEach(function (c) {
      var tile = document.createElement("div");
      tile.className = "ng-char-tile";
      tile.tabIndex = 0;

      var thumb = document.createElement("div");
      thumb.className = "ng-char-tile-thumb";
      if (c.hasAvatar) {
        var img = document.createElement("img");
        img.src = "/api/ng/generate/characters/" + encodeURIComponent(c.id) + "/avatar";
        img.alt = "";
        thumb.appendChild(img);
      } else {
        thumb.textContent = (c.name || "?").slice(0, 1).toUpperCase();
      }
      tile.appendChild(thumb);

      var name = document.createElement("div");
      name.className = "ng-char-tile-name";
      name.textContent = c.name;
      tile.appendChild(name);

      var sub = document.createElement("div");
      sub.className = "ng-char-tile-sub";
      var bits = [];
      if (c.shotCount) bits.push(c.shotCount + " shot" + (c.shotCount === 1 ? "" : "s"));
      bits.push(c.hasProjectDoc ? "saved project" : "generated only");
      sub.textContent = bits.join(" · ");
      tile.appendChild(sub);

      var open = function () {
        openCharacter(c);
      };
      tile.addEventListener("click", open);
      tile.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      });

      els.grid.appendChild(tile);
    });
  }

  function openCharacter(c) {
    var pm = window.ProjectManager;
    if (!pm) return;
    if (!c.hasProjectDoc) {
      // Nothing was ever explicitly Saved for this id -- just open a
      // fresh project under its name, on the Generate task, so its
      // existing sheet/shots are reachable via the ref tray + queue.
      pm.createProject(c.name);
      var p = pm.getActive();
      if (p) {
        p.task = "generate";
        pm.saveState();
        pm.render();
      }
      return;
    }
    fetch("/api/ng/characters/" + encodeURIComponent(c.id) + "/doc")
      .then(function (r) {
        return r.json();
      })
      .then(function (doc) {
        if (doc && doc.identity && doc.identity.name && window.CharacterIO) {
          window.CharacterIO.loadDoc(doc);
        }
      })
      .catch(function (e) {
        console.warn("[characterPicker] failed to load doc:", e);
      });
  }

  function show() {
    refreshEls();
    if (!els.wrap) return;
    els.wrap.style.display = "";
    render(); // whatever we already have, so repeat opens don't flash empty
    fetch("/api/ng/characters")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        characters = (data && data.characters) || [];
        render();
      })
      .catch(function (e) {
        console.warn("[characterPicker] failed to load characters:", e);
      });
  }

  function hide() {
    refreshEls();
    if (!els.wrap) return;
    els.wrap.style.display = "none";
  }

  window.CharacterPickerNG = { show: show, hide: hide };
})();
