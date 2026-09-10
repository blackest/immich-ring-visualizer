/**
 * settingsNG.js -- settings-cog modal, top-right of the top bar.
 *
 * First slice of the "settings cog" plan: a single curated venv-
 * maintenance action (update yt-dlp), proving the pattern -- modal +
 * backend action + visible success/failure output -- before adding
 * more actions later. Deliberately not an open-ended "run any command"
 * panel; each action gets its own reviewed backend endpoint
 * (routes/settingsNG.py).
 *
 * Self-contained: only touches its own DOM (#ng-settings-cog,
 * #ng-settings-modal and children), so it doesn't need to sit in the
 * WIRING_ORDER chain -- loaded after bootstrapWiringNG.js in
 * indexNG.html, same as any other standalone feature file.
 */

(function () {
  const cogBtn = document.getElementById("ng-settings-cog");
  const overlay = document.getElementById("ng-settings-modal");
  const closeBtn = document.getElementById("ng-settings-modal-close");
  const updateBtn = document.getElementById("ng-settings-update-ytdlp");
  const outputEl = document.getElementById("ng-settings-ytdlp-output");

  function openModal() {
    overlay.style.display = "flex";
  }

  function closeModal() {
    overlay.style.display = "none";
  }

  cogBtn.addEventListener("click", openModal);
  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });

  updateBtn.addEventListener("click", async () => {
    updateBtn.disabled = true;
    updateBtn.textContent = "Updating...";
    outputEl.style.display = "block";
    outputEl.textContent = "Running pip install --upgrade yt-dlp ...";

    try {
      const res = await fetch("/api/ng/settings/update-ytdlp", { method: "POST" });
      const data = await res.json();

      if (data.ok) {
        outputEl.textContent = "Success.\n\n" + (data.output || "");
      } else {
        outputEl.textContent = "Failed: " + (data.error || "unknown error") +
          (data.output ? "\n\n" + data.output : "");
      }
    } catch (e) {
      outputEl.textContent = "Failed: " + e.message;
    } finally {
      updateBtn.disabled = false;
      updateBtn.textContent = "Update";
    }
  });
})();
