/**
 * kleinLayoutNG.js -- box layout for the Klein pane.
 *
 * FLUX 3 Klein reads placement from the prompt: a JSON list appended to the caption, each element
 * { id, bbox: [top, left, bottom, right] on a 0..1000 grid, desc }, with the caption tagging the
 * element as <id>. This file turns boxes drawn on a canvas into that text. No backend involved.
 *
 *   KleinLayoutNG.compose(caption, boxes)  -> prompt text (pure; also used by tests)
 *   KleinLayoutNG.build(host, getSize)     -> { enabled(), apply(caption), refresh() }
 */
(function (root) {
  "use strict";

  var GRID = 1000;
  var MIN_BOX = 20; // grid units; smaller drags are treated as clicks, not boxes

  function clamp(v) { return Math.max(0, Math.min(GRID, Math.round(v))); }

  // Two corner points as fractions (0..1) of the canvas -> [top, left, bottom, right] on the grid.
  function toBbox(x0, y0, x1, y1) {
    return [clamp(Math.min(y0, y1) * GRID), clamp(Math.min(x0, x1) * GRID),
            clamp(Math.max(y0, y1) * GRID), clamp(Math.max(x0, x1) * GRID)];
  }

  function cleanId(s) { return String(s || "").trim().toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, ""); }

  function compose(caption, boxes) {
    var list = (boxes || []).filter(function (b) { return b && b.id && b.bbox; });
    var text = String(caption || "").trim();
    if (!list.length) return text;
    // Every element must be tagged in the caption; add any tag the user did not place themselves.
    var missing = list.filter(function (b) { return text.indexOf("<" + b.id + ">") < 0; });
    if (missing.length) {
      text += (text && !/[.!?]$/.test(text) ? "." : "") + " " +
        missing.map(function (b) { return (b.desc ? b.desc.split(/[.,]/)[0].trim() + " " : "") + "<" + b.id + ">"; }).join(", ") + ".";
    }
    var json = list.map(function (b) { return { id: b.id, bbox: b.bbox, desc: b.desc || "" }; });
    return text.trim() + " " + JSON.stringify(json);
  }

  function nextId(boxes) {
    var n = boxes.length + 1;
    while (boxes.some(function (b) { return b.id === "element_" + n; })) n++;
    return "element_" + n;
  }

  function build(host, getSize) {
    var boxes = [];
    var drag = null;
    var enabledBox = document.createElement("input"); enabledBox.type = "checkbox";
    var wrap = document.createElement("label"); wrap.className = "ng-klein-lora";
    wrap.appendChild(enabledBox);
    wrap.appendChild(document.createTextNode(" Place things with boxes (layout)"));
    host.appendChild(wrap);

    var panel = document.createElement("div"); panel.style.display = "none";
    var help = document.createElement("div"); help.className = "ng-gen-status";
    help.textContent = "Drag on the canvas to draw a box, then say what goes in it. Mention <name> in your prompt where it belongs, or it is added for you.";
    panel.appendChild(help);
    var canvas = document.createElement("div"); canvas.className = "ng-klein-layout-canvas";
    panel.appendChild(canvas);
    var list = document.createElement("div"); list.className = "ng-klein-layout-list";
    panel.appendChild(list);
    host.appendChild(panel);

    function frac(ev) {
      var r = canvas.getBoundingClientRect();
      return [Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (ev.clientY - r.top) / r.height))];
    }

    function rect(b, cls) {
      var d = document.createElement("div"); d.className = cls;
      d.style.top = b.bbox[0] / 10 + "%"; d.style.left = b.bbox[1] / 10 + "%";
      d.style.height = (b.bbox[2] - b.bbox[0]) / 10 + "%"; d.style.width = (b.bbox[3] - b.bbox[1]) / 10 + "%";
      return d;
    }

    function draw() {
      var s = getSize(); // [w, h]
      canvas.style.aspectRatio = s[0] + " / " + s[1];
      canvas.textContent = "";
      boxes.forEach(function (b) {
        var d = rect(b, "ng-klein-layout-box"); d.textContent = b.id; canvas.appendChild(d);
      });
      list.textContent = "";
      boxes.forEach(function (b, i) {
        var row = document.createElement("div"); row.className = "ng-klein-lora";
        var id = document.createElement("input"); id.className = "ng-gen-input"; id.value = b.id; id.style.width = "9em";
        id.addEventListener("change", function () {
          var v = cleanId(id.value);
          if (v && !boxes.some(function (o) { return o !== b && o.id === v; })) b.id = v;
          draw();
        });
        var desc = document.createElement("input"); desc.className = "ng-gen-input"; desc.style.flex = "1";
        desc.placeholder = "what is in this box"; desc.value = b.desc;
        desc.addEventListener("input", function () { b.desc = desc.value; });
        var del = document.createElement("button"); del.className = "ng-gen-btn"; del.textContent = "Remove";
        del.addEventListener("click", function () { boxes.splice(i, 1); draw(); });
        row.appendChild(id); row.appendChild(desc); row.appendChild(del); list.appendChild(row);
      });
    }

    canvas.addEventListener("pointerdown", function (ev) {
      var p = frac(ev); drag = { p0: p, el: null };
      canvas.setPointerCapture(ev.pointerId);
    });
    canvas.addEventListener("pointermove", function (ev) {
      if (!drag) return;
      var p = frac(ev);
      if (drag.el) drag.el.remove();
      drag.el = rect({ bbox: toBbox(drag.p0[0], drag.p0[1], p[0], p[1]) }, "ng-klein-layout-box ng-klein-layout-draft");
      canvas.appendChild(drag.el);
    });
    canvas.addEventListener("pointerup", function (ev) {
      if (!drag) return;
      var p = frac(ev), bb = toBbox(drag.p0[0], drag.p0[1], p[0], p[1]);
      drag = null;
      if (bb[2] - bb[0] >= MIN_BOX && bb[3] - bb[1] >= MIN_BOX) boxes.push({ id: nextId(boxes), bbox: bb, desc: "" });
      draw();
    });

    enabledBox.addEventListener("change", function () { panel.style.display = enabledBox.checked ? "" : "none"; draw(); });

    return {
      enabled: function () { return enabledBox.checked && boxes.length > 0; },
      apply: function (caption) { return enabledBox.checked ? compose(caption, boxes) : caption; },
      refresh: draw,
      boxes: function () { return boxes.slice(); },
    };
  }

  var api = { compose: compose, toBbox: toBbox, cleanId: cleanId, build: build };
  root.KleinLayoutNG = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
