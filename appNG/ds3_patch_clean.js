(function () {
  if (window.__ds3_patch_loaded) {
    console.log("DS3 patch already running.");
    return;
  }
  window.__ds3_patch_loaded = true;

  let lastAudio = null;
  let audioPlayPatched = false;
  let currentMuteState = true;
  let nudgeGiven = false;
  let alarmIntervalStarted = false;
  const playingAudios = [];
  const originalPlay = Audio.prototype.play;
  let alarmStartTime = null;
  let commsAlarmCount = 0;
  let isAnyAlarmActive = false;
  let lastIsAnyAlarmActive = false;
  let lastAlarmCount = 0;
  let lastNudgeTime = Date.now();

  const sarcasticMessages = [
    "Is this going to fix itself?",
    "Oh look, another blinking light. Must be nothing.",
    "Hey, it's not like it's your job or anything.",
    "This one's been crying out for 12 minutes. Shall we keep ignoring it?",
    "Imagine this was a real alarm. Just imagine.",
    "Don't worry, maybe it'll fix itself.",
    "That red button's not going to click itself.",
    "Mrs. Doyle is now drinking straight from the bottle",
  ];

  const style = document.createElement("style");
  style.textContent = `
    .comms-alert-count {
      position: absolute;
      top: -6px;
      right: -6px;
      background: red;
      color: white;
      border-radius: 10px;
      padding: 0 6px;
      font-size: 10px;
      font-weight: bold;
      line-height: 16px;
      height: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
  `;
  document.head.appendChild(style);

  function updateIsAnyAlarmActiveFromButtons() {
    const siteLabels = [
      "EE1", "VS2", "EE3", "EE4", "EE5", "EE6", "EE7", "EE8", "EE9", "VN1",
    ];
    const btnBar = document.getElementById("joj");
    if (!btnBar) {
      console.warn("Button bar (id='joj') not found.");
      return;
    }
    isAnyAlarmActive = siteLabels.some((label) => {
      const btn = Array.from(btnBar.querySelectorAll("button")).find(
        (b) => b.textContent.trim() === label
      );
      if (btn) {
        const bg = getComputedStyle(btn).backgroundColor;
        return bg !== "rgb(0, 0, 0)";
      }
      return false;
    });
  }

  function muteDangerRowsInTable(table) {
    if (!table) return;
    const rows = table.querySelectorAll("tbody > tr");
    rows.forEach((row, index) => {
      if (index < 7) {
        const isDanger =
          row.classList.contains("table-danger") ||
          getComputedStyle(row).backgroundColor === "rgb(248, 215, 218)";
        const isMuted = row.classList.contains("table-warning");
        if (isDanger && !isMuted) {
          const muteCell = row.querySelector("td[aria-colindex='7']");
          if (muteCell) {
            const muteBtn = muteCell.querySelector("button");
            if (muteBtn) muteBtn.click();
          }
        }
      }
    });
  }

  function speakMessage(message) {
    if (!window.speechSynthesis) {
      console.warn("Speech Synthesis not supported");
      return;
    }
    const utterance = new SpeechSynthesisUtterance(message);
    speechSynthesis.speak(utterance);
  }

  function utilityPatchAudioPlay(mute = true) {
    currentMuteState = mute;
    if (audioPlayPatched) {
      console.log(`Audio.play already patched. Switching mute to ${mute}`);
      if (lastAudio) {
        lastAudio.muted = mute;
      }
      return;
    }
    audioPlayPatched = true;
    Audio.prototype.play = function () {
      this.muted = currentMuteState;
      lastAudio = this;
      playingAudios.push(this);
      commsAlarmCount++;
      updateCommsAlert(true, commsAlarmCount);
      console.log("Comms alarm count:", commsAlarmCount);
      const now = new Date().toLocaleTimeString();
      console.log(
        `%c${now} — Audio started (${currentMuteState ? "muted" : "audible"}): ${this.src}`,
        currentMuteState ? "color: red;" : "color: blue;"
      );
      return originalPlay.call(this);
    };
  }

  function patchAudioPlay(mute = true) {
    utilityPatchAudioPlay(mute);
    const motivationBtn = document.getElementById("motivation");
    if (motivationBtn) {
      motivationBtn.textContent = "Mrs. Doyle is screaming!";
      motivationBtn.style.backgroundColor = "#dc3545";
    }
    if (window.location.href.includes("ds3Controllers")) {
      const allBlack = [...document.querySelectorAll("button")]
        .filter((b) => b.textContent && b.textContent.match(/^EE\d|VN\d/))
        .every((btn) => btn.style.backgroundColor === "rgb(0, 0, 0)");
      if (allBlack) {
      }
    }
    commsAlarmCount++;
    updateCommsAlert(true, commsAlarmCount);
    console.log("Comms alarm count:", commsAlarmCount);
    return Promise.resolve();
  }

  function utilityPatchAudioPause() {
    const originalPause = Audio.prototype.pause;
    Audio.prototype.pause = function () {
      const now = new Date().toLocaleTimeString();
      console.log(
        `%c[${now}] [Audio.pause()] Alarm stopped: %s`,
        "color: green;",
        this.src || this.currentSrc
      );
      const motivationBtn = document.getElementById("motivation");
      if (motivationBtn) {
        motivationBtn.textContent = "DS3 System Stable";
        motivationBtn.style.backgroundColor = "#28a745";
      }
      return originalPause.apply(this, arguments);
    };
  }

  function patchAudioPause() {
    utilityPatchAudioPause();
  }

  function addADelay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function playSiren() {
    const siren = new Audio(
      "https://ds3.viotas.com/static/media/siren.00751e6.mp3"
    );
    siren.play().catch((err) => {
      console.warn("Failed to play siren:", err);
    });
  }

  function setMuteState(mute) {
    currentMuteState = mute;
    if (lastAudio) {
      lastAudio.muted = mute;
    }
    console.log(`Audio mute set to ${mute}`);
  }

  async function testSirenSequence(durationMs = 2000, mute = true, message = null) {
    utilityPatchAudioPause();
    if (!audioPlayPatched) {
      utilityPatchAudioPlay(mute);
    } else {
      setMuteState(mute);
    }
    if (message) {
      if (window.speechSynthesis) {
        setMuteState(false);
        await new Promise((resolve) => {
          const utterance = new SpeechSynthesisUtterance(message);
          utterance.onend = resolve;
          speechSynthesis.speak(utterance);
        });
      } else {
        console.warn("Speech synthesis not supported — skipping audio.");
      }
    } else {
      if (durationMs > 0) {
        playSiren();
        await addADelay(durationMs);
        if (lastAudio) lastAudio.pause();
      } else {
        console.log("Siren duration 0, skipping siren playback.");
      }
    }
    setMuteState(true);
    console.log(
      `🔕 Siren sequence completed (duration ${durationMs / 1000}s, mute: ${mute}, message: ${message ? "yes" : "no"})`
    );
  }

  function createButtonBar() {
    function getFullyVisibleTable() {
      const tables = document.querySelectorAll("table");
      let bestTable = null;
      let bestTop = Infinity;
      tables.forEach((table) => {
        const rect = table.getBoundingClientRect();
        if (
          rect.top >= 0 &&
          rect.bottom <= window.innerHeight &&
          rect.top < bestTop
        ) {
          bestTop = rect.top;
          bestTable = table;
        }
      });
      return bestTable;
    }

    function triggerSubmit(table) {
      if (!table) return;
      const rows = table.querySelectorAll("tbody > tr");
      rows.forEach((row, index) => {
        if (index < 7) {
          const bg = getComputedStyle(row).backgroundColor;
          const isDanger =
            bg === "rgb(248, 215, 218)" ||
            row.classList.contains("table-danger") ||
            row.classList.contains("table-warning");
          if (isDanger) {
            const resetCell = row.cells[7];
            if (resetCell) {
              const resetButton = resetCell.querySelector("button");
              if (resetButton) {
                resetButton.click();
              }
            }
          }
        }
      });
    }

    function getFilteredValues(table, columnIndex) {
      const values = [];
      const rows = table.querySelectorAll("tbody > tr");
      rows.forEach((row, index) => {
        if (index < 7) {
          const cells = row.querySelectorAll("td");
          if (cells.length > columnIndex) {
            const value = cells[columnIndex].textContent.trim();
            const match = value.match(/^\d+(\.\d+)?/);
            if (match) values.push(match[0]);
          }
        }
      });
      return values;
    }

    function copyToClipboard(values) {
      const clipboardText = values.join("\n");
      const textarea = document.createElement("textarea");
      textarea.value = clipboardText;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      alert("Values copied to clipboard!");
    }

    const buttons = [
      { id: "declaration-table-EE1", label: "EE1", key: "1" },
      { id: "declaration-table-VS2", label: "VS2", key: "2" },
      { id: "declaration-table-VN1", label: "VN1", key: "-" },
      { id: "declaration-table-EE3", label: "EE3", key: "3" },
      { id: "declaration-table-EE4", label: "EE4", key: "4" },
      { id: "declaration-table-EE5", label: "EE5", key: "5" },
      { id: "declaration-table-EE6", label: "EE6", key: "6" },
      { id: "declaration-table-EE7", label: "EE7", key: "7" },
      { id: "declaration-table-EE8", label: "EE8", key: "8" },
      { id: "declaration-table-EE9", label: "EE9", key: "9" },
      { id: "submit", label: "Submit", key: "0" },
      { id: "copy", label: "<u>C</u>opy", key: "c" },
      { id: "comms", label: "Comms" },
      { id: "motivation", label: "DS3 System Stable" },
    ];

    const nav = document.createElement("div");
    nav.id = "joj";
    nav.style.position = "fixed";
    nav.style.top = "4px";
    nav.style.left = "260px";
    nav.style.zIndex = "9999";
    nav.style.background = "#f8f9fa";
    nav.style.padding = "10px";
    nav.style.display = "flex";
    nav.style.gap = "10px";
    nav.style.fontFamily = "sans-serif";
    document.body.appendChild(nav);

    const refs = {};

    buttons.forEach(({ id, label, key }) => {
      const btn = document.createElement("button");
      btn.innerHTML = label;
      btn.style.backgroundColor = id === "copy" ? "#28a745" : "#000";
      btn.style.color = "#fff";
      btn.style.border = "none";
      btn.style.padding = "6px 10px";
      btn.style.borderRadius = "4px";
      btn.style.cursor = "pointer";
      btn.style.outline = "none";

      if (id === "copy") {
        btn.onclick = () => {
          const table = getFullyVisibleTable();
          if (table) {
            const values = getFilteredValues(table, 4);
            if (values.length > 0) {
              copyToClipboard(values);
            } else {
              alert("No values found.");
            }
          }
        };
      } else if (id === "submit") {
        btn.onclick = () => {
          const table = getFullyVisibleTable();
          if (table) triggerSubmit(table);
        };
      } else if (id === "comms") {
        btn.id = "comms";
        btn.innerHTML = `<i class="fa fa-volume-down"></i> Comms`;
        btn.style.position = "relative";
        const countBadge = document.createElement("span");
        countBadge.id = "comms-alert-count";
        countBadge.className = "comms-alert-count";
        countBadge.textContent = "0";
        countBadge.style.display = "none";
        btn.appendChild(countBadge);
        btn.onclick = () => {
          btn.innerHTML = `<i class="fa fa-volume-down"></i> Comms`;
          initialChecks();
        };
      } else if (id === "motivation") {
        btn.id = "motivation";
        btn.onclick = () => {
          btn.textContent = "DS3 System Stable";
          lastNudgeTime = Date.now();
        };
      } else {
        btn.onclick = () => {
          const table = document.getElementById(id);
          if (table) {
            const rect = table.getBoundingClientRect();
            const scrollToY = window.scrollY + rect.top - 100;
            scrollTo({ top: scrollToY, behavior: "auto" });
          }
        };
      }

      refs[id] = btn;
      nav.appendChild(btn);
    });

    document.addEventListener("keydown", (e) => {
      const key = e.key.toLowerCase();
      if (key === "c") {
        const table = getFullyVisibleTable();
        if (table) {
          const values = getFilteredValues(table, 4);
          if (values.length > 0) {
            copyToClipboard(values);
          } else {
            alert("No values found.");
          }
        }
        return;
      }
      const button = buttons.find((b) => b.key === key);
      if (!button) return;
      const table = getFullyVisibleTable();
      switch (button.id) {
        case "submit":
          if (table) triggerSubmit(table);
          break;
        case "copy":
          if (table) {
            const values = getFilteredValues(table, 4);
            if (values.length > 0) {
              copyToClipboard(values);
            } else {
              alert("No values found.");
            }
          }
          break;
        default: {
          const targetTable = document.getElementById(button.id);
          if (targetTable) {
            const rect = targetTable.getBoundingClientRect();
            const scrollToY = window.scrollY + rect.top - 100;
            scrollTo({ top: scrollToY, behavior: "auto" });
          }
          break;
        }
      }
    });

    setInterval(() => {
      buttons.forEach(({ id }) => {
        const table = document.getElementById(id);
        if (!table) return;

        muteDangerRowsInTable(table);

        let redCount = 0;
        const rows = table.querySelectorAll("tr");
        rows.forEach((row) => {
          const bg = getComputedStyle(row).backgroundColor;
          if (
            bg === "rgb(248, 215, 218)" ||
            row.classList.contains("table-danger") ||
            row.classList.contains("table-warning")
          ) {
            redCount++;
          }
        });

        const btn = refs[id];
        if (!btn) return;

        if (redCount >= 4) {
          btn.style.backgroundColor = "#dc3545";
        } else if (redCount > 0) {
          btn.style.backgroundColor = "#ffc107";
        } else {
          btn.style.backgroundColor = id === "copy" ? "green" : "#000";
        }
      });

      updateIsAnyAlarmActiveFromButtons();
      if (isAnyAlarmActive !== lastIsAnyAlarmActive) {
        if (isAnyAlarmActive) {
          alarmStartTime = Date.now();
        } else {
          alarmStartTime = null;
        }
        lastIsAnyAlarmActive = isAnyAlarmActive;
      }
    }, 2000);
  }

  function lazyNudge() {
    return sarcasticMessages[Math.floor(Math.random() * sarcasticMessages.length)];
  }

  utilityPatchAudioPlay(true);
  patchAudioPause();

  function checkSiteName() {
    const isDeclarationPage = window.location.href.includes("ds3Controllers");
    if (!isDeclarationPage || document.getElementById("declaration-toolbar"))
      return;
    createButtonBar();
  }
  setTimeout(checkSiteName, 0);

  function updateCommsAlert(active, count = 0, attempt = 0) {
    const btn = document.getElementById("comms");
    const icon = btn && btn.querySelector("i");
    let badge = document.getElementById("comms-alert-count");

    if (btn && !badge) {
      badge = document.createElement("span");
      badge.id = "comms-alert-count";
      badge.className = "comms-alert-count";
      badge.textContent = "0";
      badge.style.display = "none";
      btn.appendChild(badge);
      console.log("🧷 Reattached comms badge to button.");
    }

    if (!btn || !icon || !badge) {
      if (attempt < 5) {
        setTimeout(() => updateCommsAlert(active, count, attempt + 1), 200);
      } else {
        console.warn("🔕 Comms UI not ready after multiple attempts.");
      }
      return;
    }

    if (active) {
      icon.className = "fa fa-volume-up";
      badge.style.display = "block";
      badge.textContent = count;
    } else {
      icon.className = "fa fa-volume-down";
      badge.style.display = "none";
      badge.textContent = "0";
    }
  }

  function showModal(message) {
    const overlay = document.createElement("div");
    overlay.style = `
      position: fixed; top: 0; left: 0; width: 125vw; height: 100vh;
      background: rgba(0,0,0,0.7); display: flex; justify-content: center; align-items: center;
      z-index: 10000;
    `;
    const modal = document.createElement("div");
    modal.style = `
      background: #fff; color: #000; padding: 20px; max-width: 800px; max-height: 80vh;
      overflow-y: auto; border-radius: 8px; font-family: monospace; white-space: pre-wrap;
    `;
    const messageNode = document.createElement("div");
    messageNode.textContent = message;
    modal.appendChild(messageNode);
    const hr = document.createElement("hr");
    hr.style.marginTop = "20px";
    modal.appendChild(hr);
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.style = `
      margin-top: 15px; padding: 6px 12px; font-size: 14px; cursor: pointer;
    `;
    closeBtn.onclick = () => document.body.removeChild(overlay);
    modal.appendChild(closeBtn);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
  }

  function checkAlarmDuration() {
    if (!alarmStartTime) {
      console.log("No alarm start time set yet.");
      nudgeGiven = false;
      return;
    }
    const now = Date.now();
    const elapsedSeconds = (now - alarmStartTime) / 1000;
    console.log(elapsedSeconds, "seconds");
    const btn = document.getElementById("motivation");
    if (!btn) {
      console.warn("⚠️ Motivation button not found.");
      return;
    }
    btn.style.backgroundColor = "red";
    let message = "";
    if (elapsedSeconds > 600) {
      testSirenSequence(20000, false);
    } else if (elapsedSeconds > 120) {
      testSirenSequence(5000, false);
    } else if (elapsedSeconds >= 90) {
      if (elapsedSeconds >= 90 && !nudgeGiven) {
        message = lazyNudge();
        testSirenSequence(2000, false, message);
        const btn = document.getElementById("motivation");
        if (btn) btn.textContent = message;
        nudgeGiven = true;
      }
    } else if (elapsedSeconds >= 60) {
      message = "📉 Possible data loss. Check site comms.";
    } else if (elapsedSeconds >= 30) {
      message = "🟠 Declarations likely needed.";
    } else {
      message = "Mrs. Doyle sobs. Why, Father, why?";
    }
    btn.textContent = message;
  }

  function initialChecks() {
    const btn = document.getElementById("motivation");
    if (btn) {
      btn.textContent = "DS3 System Stable";
      btn.style.backgroundColor = "#28a745";
    }

    if (window.location.href.includes("ds3Controllers")) {
      const dangerClass = "table-danger";
      const mutedClass = "table-warning";
      const rows = document.querySelectorAll("tbody > tr");
      let mutedCount = 0;
      rows.forEach((row) => {
        const isDanger = row.classList.contains(dangerClass);
        const isMuted = row.classList.contains(mutedClass);
        if (isDanger && !isMuted) {
          const btn = row.querySelector("td[aria-colindex='9'] button");
          if (btn) {
            const iconSpan = btn.querySelector("span.fa");
            const isCurrentlyMuted = iconSpan.classList.contains("fa-volume-off");
            if (!isCurrentlyMuted) {
              btn.click();
              mutedCount++;
            }
          }
        }
      });
      console.log(`Muted ${mutedCount} alarming rows.`);
    }

    setTimeout(() => {
      const allRows = Array.from(document.querySelectorAll("table tr"));
      let totalMuted = 0;
      let totalActive = 0;
      const recentMutedSites = [];

      allRows.forEach((row) => {
        const cells = row.querySelectorAll("td");
        if (cells.length < 6) return;
        const muteIcon = row.querySelector("span.fa-volume-off");
        const isMuted = !!muteIcon;
        const dataAgeText = cells[5].textContent.trim();
        const isRecent = /^\d{1,2}:\d{2}:\d{2}$/.test(dataAgeText);
        if (isMuted) {
          totalMuted++;
          if (isRecent) {
            const siteName = cells[0].textContent.trim();
            const fullId = row.closest("table")?.id || "Unknown table";
            const table = fullId.replace(/^powerData-table-/, "");
            recentMutedSites.push({ site: siteName, table, dataAge: dataAgeText });
          }
        } else {
          totalActive++;
        }
      });

      const totalSites = totalMuted + totalActive;
      const timestamp = new Date().toLocaleTimeString();

      let summary = `🛰️ Comms check @ ${timestamp}\n\n`;
      summary += `Total sites: ${totalSites}\nActive: ${totalActive}\nMuted: ${totalMuted}\n`;

      if (recentMutedSites.length > 0) {
        summary += `Recent comms issues (<24h): ${recentMutedSites.length}\n\n`;
        summary += `🔻 Recent Issues:\n`;
        recentMutedSites.forEach(({ site, table, dataAge }) => {
          summary += `• ${table}, ${site}, ${dataAge}\n`;
        });
      } else {
        summary += `✅ No recent comms loss in the last 24 hours detected.\n`;
      }

      const scheduledSites = [
        "Kerry Food Carrigaline Caterpillar Generator",
        "Kerry Food Carrigaline FGWilson Generator",
        "Newmarket Co-Op Cork Combined Generators",
      ];

      const now = new Date();
      const hour = now.getHours();
      const withinWindow = hour >= 8 && hour < 20;

      const rows = Array.from(document.querySelectorAll("table tr"));
      const scheduleReport = [];

      rows.forEach((row) => {
        const cells = row.querySelectorAll("td");
        if (cells.length < 8) return;
        const siteName = cells[0].textContent.trim();
        if (!scheduledSites.includes(siteName)) return;
        const icon = cells[7].querySelector("span.fa-check.text-success");
        const isActive = !!icon;
        scheduleReport.push(
          `${siteName}: ${isActive ? "ACTIVE ✅" : "INACTIVE ⛔️"} (expected: ${withinWindow ? "ACTIVE" : "INACTIVE"})`
        );
      });

      if (scheduleReport.length > 0) {
        summary += `\n🕗 Scheduled site status:\n`;
        summary += scheduleReport.join("\n");
      }

      showModal(summary);
    }, 1500);

    setTimeout(() => {
      const motivationBtn = document.getElementById("motivation");
      if (motivationBtn && !alarmIntervalStarted) {
        setInterval(checkAlarmDuration, 1000);
        alarmIntervalStarted = true;
      } else {
        console.warn("⚠️ Motivation button not yet available. Retrying...");
        const retryInterval = setInterval(() => {
          const btn = document.getElementById("motivation");
          if (btn && !alarmIntervalStarted) {
            btn.textContent = "DS3 System Stable";
            btn.style.backgroundColor = "#28a745";
            clearInterval(retryInterval);
            setInterval(checkAlarmDuration, 1000);
            alarmIntervalStarted = true;
            console.log("✅ Motivation button found. Alarm duration check started.");
          }
        }, 1000);
      }
    }, 1000);
  }

  initialChecks();
})();
