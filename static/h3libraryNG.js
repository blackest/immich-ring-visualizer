/**
 * h3libraryNG.js -- "Prompt Library" bottom-bar view: browse the 1,000 MiniMax-H3 prompt/clip pairs and reuse one.
 *
 * Backend: routes/h3libraryNG.py (list / clip / video). Filters live in the rail pane, a grid of looping
 * previews and a detail card in the main stage.
 *
 *   Use in H3            -> the full three-field text into the H3 prompt box (#ng-h3-prompt), then opens H3
 *   Use in Enhanced H3   -> the visual description into "What happens" (#ng-enhancedh3-intent), then opens
 *                           Enhanced H3, whose compiler rewrites it (soundscape and music are written fresh)
 *
 * window.H3LibraryNG = { sync } -- sync(active) is called on every ProjectManager.render().
 */
(function () {
  "use strict";

  var API = "/api/ng/h3-library";
  var PAGE = 60;
  var els = null, inited = false;
  var offset = 0, total = 0, loadedOnce = false, reqSeq = 0, debounce = null;

  function $(id) { return document.getElementById(id); }
  function node(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function refreshEls() {
    els = {
      pane: $("ng-h3library-pane"), main: $("ng-h3library-main"),
      q: $("ng-h3lib-q"), look: $("ng-h3lib-look"), shots: $("ng-h3lib-shots"),
      dialogue: $("ng-h3lib-dialogue"), music: $("ng-h3lib-music"), status: $("ng-h3lib-status"),
      grid: $("ng-h3lib-grid"), count: $("ng-h3lib-count"), more: $("ng-h3lib-more"), detail: $("ng-h3lib-detail")
    };
  }

  function query(off) {
    var p = new URLSearchParams();
    if (els.q.value.trim()) p.set("q", els.q.value.trim());
    if (els.look.value) p.set("look", els.look.value);
    if (els.shots.value !== "0") p.set("shots", els.shots.value);
    if (els.dialogue.value) p.set("dialogue", els.dialogue.value);
    if (els.music.value) p.set("music", els.music.value);
    p.set("offset", off);
    p.set("limit", PAGE);
    return p.toString();
  }

  function fillLooks(looks) {
    var keep = els.look.value;
    els.look.innerHTML = "";
    var any = node("option", null, "any"); any.value = ""; els.look.appendChild(any);
    Object.keys(looks).sort().forEach(function (k) {
      var o = node("option", null, k + " (" + looks[k] + ")"); o.value = k; els.look.appendChild(o);
    });
    els.look.value = keep;
  }

  // Previews are ~5 MB each, so only a few first-frames load at a time and only for cards that are on
  // (or near) the screen. Slow links (the iPad over a Tailscale relay) then fill in gradually.
  var MAX_LOADING = 4, pending = [], loading = 0, observer = null;

  function pump() {
    while (loading < MAX_LOADING && pending.length) {
      var v = pending.shift();
      if (!v.isConnected || v.src) continue;
      loading++;
      var done = function () {
        v.removeEventListener("loadeddata", done);
        v.removeEventListener("error", done);
        loading--;
        pump();
      };
      v.addEventListener("loadeddata", done);
      v.addEventListener("error", done);
      v.src = v.dataset.src;
    }
  }

  function watch(v) {
    if (!("IntersectionObserver" in window)) { pending.push(v); pump(); return; }
    if (!observer) {
      observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          observer.unobserve(e.target);
          pending.push(e.target);
        });
        pump();
      }, { root: els && els.main, rootMargin: "300px" });
    }
    observer.observe(v);
  }

  function card(item) {
    var c = node("div", "ng-h3lib-card");
    c.tabIndex = 0;
    var v = node("video", "ng-h3lib-thumb");
    v.muted = true; v.loop = true; v.playsInline = true; v.preload = "metadata";
    v.dataset.src = API + "/video/" + item.id + ".mp4#t=0.1";
    watch(v);
    c.addEventListener("mouseenter", function () {
      if (!v.src) { v.src = v.dataset.src; }  // hovered before its turn in the queue: load it now
      v.play().catch(function () {});
    });
    c.addEventListener("mouseleave", function () { v.pause(); });
    c.appendChild(v);
    var tags = node("div", "ng-h3lib-tags");
    [item.look, item.shots + (item.shots === 1 ? " shot" : " shots")]
      .concat(item.dialogue ? ["dialogue"] : [], item.music ? ["score"] : [])
      .forEach(function (t) { tags.appendChild(node("span", "ng-h3lib-tag", t)); });
    c.appendChild(tags);
    c.appendChild(node("div", "ng-h3lib-sum", item.summary));
    c.addEventListener("click", function () { openDetail(item.id); });
    c.addEventListener("keydown", function (e) { if (e.key === "Enter") openDetail(item.id); });
    return c;
  }

  function load(reset) {
    var seq = ++reqSeq;
    if (reset) { offset = 0; els.grid.innerHTML = ""; pending = []; }
    els.status.textContent = "Loading…";
    fetch(API + "/list?" + query(offset)).then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, j: j }; });
    }).then(function (res) {
      if (seq !== reqSeq) return;
      if (!res.ok) { els.status.textContent = res.j.error || "Could not load the library."; return; }
      loadedOnce = true;
      total = res.j.total;
      if (reset) fillLooks(res.j.looks);
      res.j.items.forEach(function (it) { els.grid.appendChild(card(it)); });
      offset += res.j.items.length;
      els.count.textContent = total + (total === 1 ? " prompt" : " prompts");
      els.more.hidden = offset >= total;
      els.status.textContent = "";
    }).catch(function () {
      if (seq === reqSeq) els.status.textContent = "Could not reach the server.";
    });
  }

  function field(label, text) {
    var d = node("div", "ng-h3lib-field");
    d.appendChild(node("div", "ng-gen-label", label));
    d.appendChild(node("div", "ng-h3lib-text", text));
    return d;
  }

  function useIn(task, targetId, text) {
    var t = $(targetId);
    if (!t) return;
    t.value = text;
    t.dispatchEvent(new Event("input", { bubbles: true }));
    if (window.ProjectManager) window.ProjectManager.setTask(task);
    var back = $(targetId);
    if (back) { back.value = text; back.focus(); }
  }

  function openDetail(id) {
    fetch(API + "/clip/" + id).then(function (r) { return r.json(); }).then(function (c) {
      if (c.error) { els.status.textContent = c.error; return; }
      var d = els.detail;
      d.innerHTML = "";
      d.hidden = false;
      var head = node("div", "ng-h3lib-dhead");
      head.appendChild(node("strong", null, "#" + c.id + " · " + c.look));
      var close = node("button", "ng-gen-btn", "Close"); close.style.width = "auto";
      close.addEventListener("click", function () { d.hidden = true; d.innerHTML = ""; });
      head.appendChild(close);
      d.appendChild(head);
      var body = node("div", "ng-h3lib-dbody");
      var v = node("video", "ng-h3lib-dvideo");
      v.controls = true; v.loop = true; v.autoplay = true; v.muted = true; v.playsInline = true;
      v.src = API + "/video/" + c.id + ".mp4";
      body.appendChild(v);
      var text = node("div", "ng-h3lib-dtext");
      text.appendChild(field("Description", c.integrated_multimodal_description));
      text.appendChild(field("Soundscape", c.overall_soundscape));
      text.appendChild(field("Music", c.non_diegetic_music));
      body.appendChild(text);
      d.appendChild(body);
      var row = node("div", "ng-h3lib-actions");
      var b1 = node("button", "ng-gen-btn", "Use in H3");
      b1.title = "Put the whole prompt (description, soundscape, music) in the H3 prompt box";
      b1.addEventListener("click", function () { useIn("h3", "ng-h3-prompt", c.full); });
      var b2 = node("button", "ng-gen-btn", "Use in Enhanced H3");
      b2.title = "Put the description in “What happens” so the compiler can rewrite it";
      b2.addEventListener("click", function () {
        useIn("enhancedh3", "ng-enhancedh3-intent", c.integrated_multimodal_description);
      });
      var b3 = node("button", "ng-gen-btn", "Copy");
      b3.addEventListener("click", function () {
        var done = function () { b3.textContent = "Copied"; setTimeout(function () { b3.textContent = "Copy"; }, 1200); };
        if (navigator.clipboard) navigator.clipboard.writeText(c.full).then(done, function () {});
      });
      [b1, b2, b3].forEach(function (b) { b.style.width = "auto"; row.appendChild(b); });
      d.appendChild(row);
      d.scrollIntoView({ block: "start", behavior: "smooth" });
    });
  }

  function init() {
    if (inited || !els.q) return;
    inited = true;
    var again = function () { clearTimeout(debounce); debounce = setTimeout(function () { load(true); }, 250); };
    els.q.addEventListener("input", again);
    [els.look, els.shots, els.dialogue, els.music].forEach(function (s) { s.addEventListener("change", again); });
    els.more.addEventListener("click", function () { load(false); });
  }

  function sync(active) {
    refreshEls();
    if (!els.pane || !els.main) return;
    var on = !!(active && active.task === "h3library");
    els.pane.style.display = on ? "" : "none";
    els.main.style.display = on ? "" : "none";
    if (!on) return;
    var controls = $("ng-controls-pane");
    if (controls) controls.style.display = "none";
    init();
    if (!loadedOnce) load(true);
  }

  window.H3LibraryNG = { sync: sync };
})();
