/**
 * h3EnhancedPopoutNG.js -- the Enhanced prompt writer as a pop-out window over the H3 view.
 *
 * The writer is the Enhanced H3 workspace (#ng-enhancedh3-main, static/enhancedh3NG.js). Opening the
 * pop-out lifts that workspace into a window; "Send to H3" copies its prompts back: one window's text
 * goes into the H3 prompt box, several go into a list of prompt windows (one per 5 s) that the H3
 * render sends as `chain_prompts`. The clip length follows the brief. Nothing is rendered here.
 *
 *   H3EnhancedPopoutNG.windows()  -> the prompt windows to send (an array longer than 1), else []
 */
(function () {
  "use strict";

  var chain = [];
  var overlay = null, body = null, status = null;
  var home = null; // where the workspace lives when it is not popped out

  function $(id) { return document.getElementById(id); }

  function node(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  function build() {
    overlay = node("div", "ng-eh3-pop");
    overlay.hidden = true;
    var box = node("div", "ng-eh3-pop-box");
    var head = node("div", "ng-eh3-pop-head");
    head.appendChild(node("strong", "", "Enhanced prompt"));
    status = node("span", "ng-gen-status");
    head.appendChild(status);
    var send = node("button", "ng-gen-btn", "Send to H3");
    send.type = "button"; send.style.width = "auto";
    send.title = "Put these prompts into the H3 view";
    send.addEventListener("click", sendBack);
    var close = node("button", "ng-gen-btn", "Close");
    close.type = "button"; close.style.width = "auto";
    close.addEventListener("click", closePop);
    head.appendChild(send); head.appendChild(close);
    body = node("div", "ng-eh3-pop-body");
    box.appendChild(head); box.appendChild(body);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !overlay.hidden) closePop(); });
  }

  function openPop() {
    var E = window.EnhancedH3NG, main = $("ng-enhancedh3-main");
    if (!E || !main) return;
    if (!overlay) build();
    E.ensureInit();
    home = { parent: main.parentNode, next: main.nextSibling };
    E.setPopped(true);
    body.appendChild(main);
    main.style.display = "";
    document.body.classList.add("ng-eh3-popped");
    overlay.hidden = false;
    status.textContent = "";
  }

  function closePop() {
    var E = window.EnhancedH3NG, main = $("ng-enhancedh3-main");
    if (!overlay || overlay.hidden) return;
    if (home && main) home.parent.insertBefore(main, home.next);
    if (main) main.style.display = "none"; // the H3 view is the active task, not this one
    if (E) E.setPopped(false);
    document.body.classList.remove("ng-eh3-popped");
    overlay.hidden = true;
  }

  // ---- the prompt windows shown in the H3 view ------------------------------------------------
  function drawWindows() {
    var box = $("ng-h3-windows"), prompt = $("ng-h3-prompt");
    box.textContent = "";
    box.hidden = chain.length < 2;
    prompt.disabled = chain.length > 1;
    if (chain.length < 2) return;
    box.appendChild(node("div", "ng-gen-label", "Prompt windows (" + chain.length + " x 5 s), from the Enhanced prompt"));
    chain.forEach(function (text, i) {
      var ta = node("textarea", "ng-gen-input");
      ta.rows = 4; ta.value = text;
      ta.title = "Window " + (i + 1);
      ta.addEventListener("input", function () { chain[i] = ta.value; });
      box.appendChild(node("div", "ng-gen-status", "Window " + (i + 1)));
      box.appendChild(ta);
    });
    var clear = node("button", "ng-gen-btn", "Clear windows");
    clear.type = "button"; clear.style.width = "auto";
    clear.addEventListener("click", function () { chain = []; drawWindows(); });
    box.appendChild(clear);
    if (window.H3BatchNG) window.H3BatchNG.mount(box, chain.length);
  }

  function setDuration(seconds) {
    var d = $("ng-h3-duration");
    if (!d || !seconds) return;
    d.value = String(Math.min(parseFloat(d.max) || 15, Math.max(parseFloat(d.min) || 3, seconds)));
    d.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function sendBack() {
    var got = window.EnhancedH3NG && window.EnhancedH3NG.windows();
    if (!got) { status.textContent = "Write a brief first."; return; }
    var prompt = $("ng-h3-prompt");
    if (got.prompts.length === 1) {
      chain = [];
      prompt.value = got.prompts[0];
    } else {
      chain = got.prompts.slice();
      prompt.value = chain[0];
    }
    setDuration(got.seconds);
    drawWindows();
    closePop();
    prompt.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function init() {
    var btn = $("ng-h3-enhance-open");
    if (btn) btn.addEventListener("click", openPop);
  }

  window.H3EnhancedPopoutNG = { windows: function () { return chain.length > 1 ? chain.slice() : []; } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
