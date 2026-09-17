/**
 * settingsNG.js -- settings-cog modal, top-right of the top bar.
 *
 * Curated actions (update yt-dlp, launch Suno Vault) plus an "Addresses"
 * section for the machine-specific endpoints in configNG.NG_ADDRESS_SETTINGS
 * (Ollama URL, Hermes gateway URL/key/model/session, Tailscale hostname).
 * Addresses are fetched/saved via routes/settingsNG.py, which persists
 * them to a gitignored JSON file (configNG.NG_SETTINGS_FILE) -- edits
 * take effect immediately, no server restart. Deliberately not an
 * open-ended "run any command" panel; each action gets its own reviewed
 * backend endpoint.
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
  const launchSunoBtn = document.getElementById("ng-settings-launch-suno");
  const sunoOutputEl = document.getElementById("ng-settings-suno-output");
  const launchComfyBtn = document.getElementById("ng-settings-launch-comfyui");
  const comfyOutputEl = document.getElementById("ng-settings-comfyui-output");
  const addressesEl = document.getElementById("ng-settings-addresses");
  const saveAddressesBtn = document.getElementById("ng-settings-save-addresses");
  const addressesOutputEl = document.getElementById("ng-settings-addresses-output");

  const SOURCE_LABEL = {
    env: "env var",
    saved: "saved",
    default: "default",
  };

  function renderAddressField(setting) {
    const wrap = document.createElement("div");
    wrap.className = "ng-settings-field";

    const head = document.createElement("div");
    head.className = "ng-settings-field-head";
    const label = document.createElement("strong");
    label.textContent = setting.label;
    const source = document.createElement("span");
    source.className = "ng-settings-field-source";
    source.textContent = SOURCE_LABEL[setting.source] || setting.source;
    head.appendChild(label);
    head.appendChild(source);

    const desc = document.createElement("span");
    desc.className = "ng-settings-row-desc";
    desc.textContent = setting.description || "";

    const input = document.createElement("input");
    input.type = setting.secret ? "password" : "text";
    input.dataset.settingKey = setting.key;
    input.value = setting.value || "";
    input.placeholder = setting.default || "";
    if (setting.source === "env") {
      // An env var always wins over a saved override -- editing here
      // would silently do nothing, so disable it and say why.
      input.disabled = true;
      source.textContent += " (takes precedence, edit not possible here)";
    }

    wrap.appendChild(head);
    wrap.appendChild(desc);
    wrap.appendChild(input);
    return wrap;
  }

  async function loadAddresses() {
    addressesEl.textContent = "Loading...";
    try {
      const res = await fetch("/api/ng/settings/addresses");
      const data = await res.json();
      addressesEl.textContent = "";
      (data.settings || []).forEach((setting) => {
        addressesEl.appendChild(renderAddressField(setting));
      });
    } catch (e) {
      addressesEl.textContent = "Failed to load: " + e.message;
    }
  }

  function openModal() {
    overlay.style.display = "flex";
    loadAddresses();
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

  launchSunoBtn.addEventListener("click", async () => {
    launchSunoBtn.disabled = true;
    launchSunoBtn.textContent = "Launching...";
    sunoOutputEl.style.display = "block";
    sunoOutputEl.textContent = "Starting Suno Vault server...";

    try {
      const res = await fetch("/api/ng/settings/launch-suno", { method: "POST" });
      const data = await res.json();

      if (data.ok) {
        sunoOutputEl.textContent =
          (data.already_running ? "Already running at " : "Started at ") +
          data.url;
        window.open(data.url, "_blank");
      } else {
        sunoOutputEl.textContent = "Failed: " + (data.error || "unknown error");
      }
    } catch (e) {
      sunoOutputEl.textContent = "Failed: " + e.message;
    } finally {
      launchSunoBtn.disabled = false;
      launchSunoBtn.textContent = "Launch";
    }
  });

  launchComfyBtn.addEventListener("click", async () => {
    launchComfyBtn.disabled = true;
    launchComfyBtn.textContent = "Launching...";
    comfyOutputEl.style.display = "block";
    comfyOutputEl.textContent = "Starting ComfyUI server (can take a while -- it's loading torch)...";

    try {
      const res = await fetch("/api/ng/settings/launch-comfyui", { method: "POST" });
      const data = await res.json();

      if (data.ok) {
        // No window.open here (unlike Suno above) -- the ComfyUI tab talks
        // to this server over its own API, nothing needs to navigate
        // anywhere, and popping an external origin is exactly what forced
        // a kill-and-restart of the installed iPad app after Suno's launch.
        comfyOutputEl.textContent =
          (data.already_running ? "Already running at " : "Started at ") +
          data.url +
          (data.logPath ? "\nLog: tail -f " + data.logPath : "");
      } else {
        comfyOutputEl.textContent = "Failed: " + (data.error || "unknown error");
      }
    } catch (e) {
      comfyOutputEl.textContent = "Failed: " + e.message;
    } finally {
      launchComfyBtn.disabled = false;
      launchComfyBtn.textContent = "Launch";
    }
  });

  saveAddressesBtn.addEventListener("click", async () => {
    const values = {};
    addressesEl.querySelectorAll("input[data-setting-key]").forEach((input) => {
      if (!input.disabled) values[input.dataset.settingKey] = input.value;
    });

    saveAddressesBtn.disabled = true;
    saveAddressesBtn.textContent = "Saving...";
    addressesOutputEl.style.display = "block";
    addressesOutputEl.textContent = "Saving...";

    try {
      const res = await fetch("/api/ng/settings/addresses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const data = await res.json();

      if (data.ok) {
        addressesOutputEl.textContent = "Saved. Takes effect immediately.";
        addressesEl.textContent = "";
        (data.settings || []).forEach((setting) => {
          addressesEl.appendChild(renderAddressField(setting));
        });
      } else {
        addressesOutputEl.textContent = "Failed: " + (data.error || "unknown error");
      }
    } catch (e) {
      addressesOutputEl.textContent = "Failed: " + e.message;
    } finally {
      saveAddressesBtn.disabled = false;
      saveAddressesBtn.textContent = "Save";
    }
  });
})();
