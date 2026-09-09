/**
 * projectManagerNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 1234-2544 (as of this
 * extraction) -- the `const ProjectManager = {...}` object in full.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 * A faithful copy-out for study, not a working module yet -- appNG.js is
 * untouched, still has its own copy of this object.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * The tab/instance manager: owns the list of CharacterProject instances
 * (Tom, Mary, default1, default2...), which one is active, persistence
 * (localStorage save/load), and EVERY render function in the app -- tabs,
 * bottom bar, main stage, ranked lists, left rail, the selection modal,
 * frame preview. This is genuinely the single largest and most tangled
 * piece in appNG.js -- 1,310 lines, more than CharacterProjectNG.js's 755.
 *
 * ============================================================
 * *** THIS IS THE "MONOLITH-INSIDE-THE-MONOLITH" appNG-module-contracts.md
 * *** ALREADY IDENTIFIED. That doc's findings apply directly here and are
 * *** NOT re-derived in this pass -- see "KNOWN INTERNAL CUT LINES" below
 * *** for exactly where and why a future split should happen.
 * ============================================================
 *
 * Method inventory (all are properties of the ProjectManager object
 * literal, called as ProjectManager.xxx()), with their approximate
 * position in this file:
 *
 *   getActive()              -- line 7    -- returns the active project
 *   isNameTaken()             -- line 11
 *   uniquePlaceholderName()   -- line 16
 *   createProject()           -- line 26   -- "+" button handoff target
 *                                              (see initchar.js's monkey-
 *                                              patch of this method)
 *   closeProject()            -- line 33
 *   setActive()                -- line 49
 *   reorderProject()           -- line 65
 *   renameProject()            -- line 75   -- called from initchar.js's
 *                                              wireNamingFlow blur handler
 *   setTask()                  -- line 87
 *   saveState()                -- line 97   -- localStorage persistence
 *   loadState()                -- line 112
 *   render()                   -- line 128  -- top-level render, calls the
 *                                              others below in sequence
 *   renderTabs()                -- line 138
 *   renderBottomBar()           -- line 204  -- task button disabled state
 *                                              (referenced in initguing.js)
 *   renderMain()                 -- line 212  -- ***SEE CUT LINE 1 BELOW***
 *   renderStage()                 -- line 273  -- clean dispatcher, barely
 *                                                needs touching per the
 *                                                contracts doc
 *   renderVideoStage()             -- line 292  -- ***SEE CUT LINE 2 BELOW,
 *                                                THE BIG ONE***
 *   renderRankedList()              -- line 404
 *   renderImmichStage()              -- line 451  -- not yet fully read/
 *                                                verified per the
 *                                                contracts doc
 *   renderRankedListImmich()          -- line 566
 *   renderLeftRail()                   -- line 612
 *   renderImmichSearch()                -- line 711
 *   renderVideoAnalysisBody()            -- line 732
 *   renderAnalysisStatus()                -- line 764
 *   renderImmichAnalysisStatus()           -- line 864
 *   openSelectedModal()                     -- line 938
 *   findNeutralPose()                        -- line 1196
 *   renderFramePreview()                      -- line 1265 -- see
 *                                                FrameScrubber in
 *                                                appNG-module-contracts.md,
 *                                                not yet extracted; calls
 *                                                out to drawFrame()
 *
 * ============================================================
 * KNOWN INTERNAL CUT LINES (from appNG-module-contracts.md, restated here
 * against this file's own line numbers so it's self-contained)
 * ============================================================
 *
 * CUT LINE 1 -- renderMain() (this file's line ~212):
 *   Functionally already a correct state-decision tree:
 *     no project open -> placeholder ("press + to start one")
 *     project exists, no task picked -> placeholder, task buttons disabled
 *     task picked, no analysis yet -> placeholder prompting to run analysis
 *     analysis done -> hands off to renderStage()
 *   FLAW: fused to specific DOM elements (mainPlaceholderEl, stageWrapEl,
 *   sidebarEl -- all module-level globals from appNG.js, NOT redefined in
 *   this extraction) instead of returning a state name and letting a
 *   separate renderer do the showing/hiding. Future cut: separate the
 *   state decision (pure function) from the DOM toggle (thin renderer).
 *
 * CUT LINE 2 -- renderVideoStage() (this file's line ~292) -- THE actual
 *   monolith-inside-the-monolith. One function doing at least four
 *   unrelated jobs:
 *     1. HUD text derivation (mode label, anchor label, match %)
 *     2. Sidebar "currently selected" panel update
 *     3. Mode branch: pose-list view vs. ring view (two different
 *        visualizations sharing one function)
 *     4. Ring view's DOM construction -- band circles, center node,
 *        per-result node placement, hover-preview wiring, all built
 *        imperatively inline
 *   Future cut lines: HUD updater, sidebar panel updater, PoseListView
 *   (semi-separate already via renderPoseListNG elsewhere in appNG.js --
 *   not part of this extraction, just needs the branch removed from here),
 *   and a proper RingVisualizer component (anchor + banded/scored results
 *   + ringScale in, rendered nodes + hover/select events out -- toolkit-
 *   tier: "place items by score, closest = center" isn't face-specific).
 *
 * Not yet verified against source (per the contracts doc, still open):
 *   renderImmichStage() and whatever comes after it in the original file
 *   that wasn't read this session.
 *
 * ============================================================
 * COUPLING POINTS (everything this object reaches OUTSIDE itself)
 * ============================================================
 *   - CharacterProject      -- constructs instances of it (createProject),
 *                               references it as a type/class (referenced
 *                               3x in this slice) -- see characterProjectNG.js
 *   - PlaybackModal          -- opens/closes the video popout (5 refs) --
 *                               see appNG-module-contracts.md's VideoPopout
 *                               entry, not yet extracted
 *   - drawFrame(project, frameNo) -- called from renderFramePreview() --
 *                               see appNG-module-contracts.md's
 *                               FrameScrubber entry, not yet extracted
 *   - applyResolutionSummaryNG() -- called during render
 *   - gatherExportParamsNG()      -- called during render
 *   - setupPosePickerNG() / setupScalePickerNG() / renderPosePickerGridNG()
 *     / renderScalePickerGridNG() -- called from render() -- see
 *     pickersNG.js, already extracted. THIS FILE IS THE CALLER documented
 *     in pickersNG.js's "CALL-SITE WIRING" section -- if/when both files
 *     are actually wired in, this confirms that documentation was accurate.
 *   - A large number of module-level DOM element globals defined
 *     elsewhere in appNG.js (mainPlaceholderEl, stageWrapEl, sidebarEl,
 *     and many more inside the render* methods) -- NOT enumerated
 *     exhaustively here; they are the same pattern as pickersNG.js's DOM
 *     globals, just far more numerous given this file's size. A full
 *     enumeration would need a dedicated pass (grep for
 *     document.getElementById across this file) before this could
 *     actually run standalone.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Enumerate and either import or re-declare every DOM-element global
 *      this file's render* methods depend on (large task, not attempted
 *      here -- this extraction is for reading/study, not deployment).
 *   2. Load after CharacterProjectNG.js, PlaybackModal, drawFrame,
 *      applyResolutionSummaryNG, gatherExportParamsNG, and pickersNG.js.
 *   3. Remove the corresponding object literal from appNG.js.
 *   4. Apply Cut Line 1 and Cut Line 2 above as their own follow-up
 *      extraction passes -- do NOT attempt both the lift-out AND the
 *      internal split in one step.
 */

  const ProjectManager = {
    projects: [],
    activeId: null,
    nextId: 1,
    dragSourceId: null,

    getActive() {
      return this.projects.find((p) => p.id === this.activeId) || null;
    },

    isNameTaken(name, excludeId) {
      const lower = name.trim().toLowerCase();
      return this.projects.some((p) => p.id !== excludeId && p.name.toLowerCase() === lower);
    },

    uniquePlaceholderName(base) {
      let candidate = base;
      let n = 1;
      while (this.isNameTaken(candidate, null)) {
        candidate = base + n;
        n++;
      }
      return candidate;
    },

    createProject(name) {
      const project = new CharacterProject(name);
      this.projects.push(project);
      this.activeId = project.id;
      this.render();
    },

    closeProject(id) {
      // Placeholder for "closing saves state" -- no disk-backed persistence
      // layer yet (see APP_ARCHITECTURE_NOTES.md's shopping-list/SQLite
      // plan), so this just removes it from the in-memory tab list for now.
      const closing = this.projects.find((p) => p.id === id);
      if (closing) closing.destroy();
      if (typeof PlaybackModal !== "undefined" && PlaybackModal.projectId === id) PlaybackModal.close();
      const idx = this.projects.findIndex((p) => p.id === id);
      if (idx === -1) return;
      this.projects.splice(idx, 1);
      if (this.activeId === id) {
        this.activeId = this.projects.length ? this.projects[this.projects.length - 1].id : null;
      }
      this.render();
    },

    setActive(id) {
      if (this.activeId === id) return;
      const outgoing = this.getActive();
      if (outgoing) outgoing.stopPlayIfRunning();
      // the playback modal is a pop-out for one specific project's build --
      // don't leave it open showing the outgoing project's clip once a
      // different tab becomes active.
      if (typeof PlaybackModal !== "undefined" && PlaybackModal.projectId && PlaybackModal.projectId !== id) {
        PlaybackModal.close();
      }
      neutralPoseReadoutEl.style.display = "none";
      neutralPoseReadoutEl.innerHTML = "";
      this.activeId = id;
      this.render();
    },

    reorderProject(sourceId, targetId) {
      if (sourceId === targetId) return;
      const fromIdx = this.projects.findIndex((p) => p.id === sourceId);
      const toIdx = this.projects.findIndex((p) => p.id === targetId);
      if (fromIdx === -1 || toIdx === -1) return;
      const [moved] = this.projects.splice(fromIdx, 1);
      this.projects.splice(toIdx, 0, moved);
      this.render();
    },

    renameProject(id, name) {
      const project = this.projects.find((p) => p.id === id);
      if (!project) return;
      const trimmed = name.trim();
      if (!trimmed || trimmed === project.name) return;
      if (this.isNameTaken(trimmed, id)) {
        alert(trimmed + " -- there can only be one.");
        return;
      }
      project.name = trimmed;
    },

    setTask(task) {
      const project = this.getActive();
      if (!project) return;
      if (project.task === "video" && task !== "video") project.stopPlayIfRunning();
      project.task = task;
      this.render();
    },

    // ---- persistence (browser-local convenience: survives a refresh, not
    // a real save/load layer -- see APP_ARCHITECTURE_NOTES.md) ----
    saveState() {
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            projects: this.projects.map((p) => p.toPlain()),
            activeId: this.activeId,
            nextId: this.nextId,
          })
        );
      } catch (e) {
        // localStorage can throw (private mode, quota, disabled) -- non-fatal
      }
    },

    loadState() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const data = JSON.parse(raw);
        if (Array.isArray(data.projects)) {
          data.projects.forEach((p) => this.projects.push(CharacterProject.fromPlain(p)));
          this.activeId = data.activeId || null;
          this.nextId = data.nextId || this.projects.length + 1;
        }
      } catch (e) {
        // corrupt/missing state -- just start empty
      }
    },

    // ---- rendering ----
    render() {
      this.renderTabs();
      this.renderBottomBar();
      this.renderLeftRail();
      this.renderMain();
      const active = this.getActive();
      applyResolutionSummaryNG(active && active.job ? active.job.resolutionSummary : null);
      this.saveState();
    },

    renderTabs() {
      tabsEl.innerHTML = "";
      this.projects.forEach((project) => {
        const tab = document.createElement("div");
        tab.className = "ng-tab" + (project.id === this.activeId ? " ng-tab-active" : "");
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", project.id === this.activeId ? "true" : "false");

        tab.draggable = true;
        tab.addEventListener("dragstart", (e) => {
          this.dragSourceId = project.id;
          tab.classList.add("ng-tab-dragging");
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", project.id);
        });
        tab.addEventListener("dragend", () => {
          this.dragSourceId = null;
          tab.classList.remove("ng-tab-dragging");
        });
        tab.addEventListener("dragover", (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
        });
        tab.addEventListener("drop", (e) => {
          e.preventDefault();
          if (this.dragSourceId) this.reorderProject(this.dragSourceId, project.id);
        });

        const label = document.createElement("span");
        label.className = "ng-tab-label";
        label.textContent = project.name;
        label.title = "Double-click to rename";
        label.addEventListener("dblclick", (e) => {
          e.stopPropagation();
          label.contentEditable = "true";
          label.focus();
          document.execCommand("selectAll", false, null);
        });
        label.addEventListener("blur", () => {
          label.contentEditable = "false";
          this.renameProject(project.id, label.textContent);
          this.render();
        });
        label.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            label.blur();
          }
        });

        const closeBtn = document.createElement("button");
        closeBtn.className = "ng-tab-close";
        closeBtn.textContent = "×";
        closeBtn.title = "Close (saves state)";
        closeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.closeProject(project.id);
        });

        tab.addEventListener("click", () => this.setActive(project.id));
        tab.appendChild(label);
        tab.appendChild(closeBtn);
        tabsEl.appendChild(tab);
      });
    },

    renderBottomBar() {
      const active = this.getActive();
      taskButtons.forEach((btn) => {
        btn.disabled = !active;
        btn.classList.toggle("ng-task-active", !!active && active.task === btn.dataset.task);
      });
    },

    renderMain() {
      const active = this.getActive();

      const showPlaceholder = (text) => {
        mainPlaceholderEl.style.display = "";
        mainPlaceholderEl.textContent = text;
        stageWrapEl.style.display = "none";
        sidebarEl.style.display = "none";
      };

      if (!active) {
        showPlaceholder("No project open. Press + to start one.");
        return;
      }
      if (!active.task) {
        showPlaceholder("Pick Video, Immich, or Folder / Zip below to get started with “" + active.name + "”.");
        return;
      }
      if (active.task === "video") {
        if (!active.ring) {
          showPlaceholder("No analysis yet for “" + active.name + "” — load a video, pick a frame, and press Run Analysis.");
          return;
        }
      } else if (active.task === "folderzip") {
        if (!active.ring) {
          showPlaceholder("No analysis yet for “" + active.name + "” — load a folder or .zip in the left rail to begin.");
          return;
        }
      } else if (active.task === "immich") {
        // A completed/running batch analyze-immich job takes priority --
        // it shares project.ring/job with Video and Folder/Zip (see the
        // comment on CharacterProject's ring/job fields), and running it
        // is a deliberate action producing a richer, pose-annotated pool
        // (feeds Pose Picker / Shot Scale Picker) that supersedes the
        // plain neighbor-browsing view below.
        const hasBatchRing = active.ring && active.ring.sourceType === "immich";
        if (!hasBatchRing) {
          if (active.immichLoading && !active.immichRing) {
            showPlaceholder("Loading Immich neighbors…");
            return;
          }
          if (active.immichError && !active.immichRing) {
            showPlaceholder("Immich error: " + active.immichError);
            return;
          }
          if (!active.immichRing) {
            showPlaceholder("Search for a face by filename in the left rail’s Search section to begin, or tick assets and press Analyze.");
            return;
          }
        }
      } else {
        showPlaceholder("Pick Video, Immich, or Folder / Zip below to get started with “" + active.name + "”.");
        return;
      }

      mainPlaceholderEl.style.display = "none";
      stageWrapEl.style.display = "";
      sidebarEl.style.display = "";
      this.renderStage(active);
    },

    renderStage(project) {
      // renderLeftRail() covers everything selection-count-dependent in
      // the left rail (the Search panel's Analyze button text/disabled
      // state, the sticky Immich selection bar) -- every call site below
      // fires after a selection change (checkbox toggle, node dblclick,
      // Select All/Deselect All, Clear), so keeping it in lockstep here
      // means callers don't each need to remember to call both.
      this.renderLeftRail();
      // "immich" task shows the batch-analysis ring (pose-annotated,
      // shared with Video/Folder/Zip) when one exists, falling back to
      // the lighter neighbor-browsing immichRing view otherwise -- see
      // the matching comment in renderMain().
      if (project.task === "immich" && !(project.ring && project.ring.sourceType === "immich")) {
        this.renderImmichStage(project);
      } else {
        this.renderVideoStage(project);
      }
    },

    renderVideoStage(project) {
      immichSectionEl.style.display = "none";
      const ring = project.ring;
      const anchorLabel = `Frame ${ring.refFrameIdx} (Anchor)`;
      const metricLabel = { sim: "Similarity", yaw: "Yaw", pitch: "Pitch", roll: "Roll", blur: "Sharpness" }[project.ringSortMetric];
      const modeLabel = ring.sourceType === "folder"
        ? "FOLDER / IMAGE-SET ANALYSIS (local, not in Immich)"
        : ring.sourceType === "immich"
        ? "IMMICH BATCH ANALYSIS (pose data for selected Immich assets)"
        : "VIDEO FRAME ANALYSIS (local, not in Immich)";
      if (framesSectionTitleEl) {
        framesSectionTitleEl.textContent = ring.sourceType === "immich" ? "Immich batch results" : "Local frames";
      }
      hudModeEl.textContent = modeLabel;
      hudFilenameEl.textContent = `${anchorLabel} · sorted by ${metricLabel}`;

      // currently-selected panel always shows the anchor for this slice --
      // there's no recenter target since video-frame results carry no
      // assetId (matches the original's behavior for local-frame nodes).
      sidebarCurrentImgEl.src = ring.anchorUrl;
      sidebarCurrentFnameEl.textContent = anchorLabel;
      sidebarCurrentModeEl.textContent = modeLabel;
      sidebarCurrentDetailEl.textContent = "match: 100.0%";

      const sorted = project.sortedRanked();
      const { kept, total } = project.squeezeFiltered(sorted);
      squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        stageEl.style.display = "none";
        poseListViewEl.style.display = "flex";
        poseListScrubberEl.style.display = "flex";
        renderPoseListNG(project, ring.anchorUrl, anchorLabel, kept, (r) => project.toggleFrameSelection(r.frame), (r) => project.selectedFrames.has(r.frame));
        this.renderRankedList(project, kept);
        return;
      }
      stageEl.style.display = "";
      poseListViewEl.style.display = "none";
      poseListScrubberEl.style.display = "none";

      stageEl.innerHTML = "";
      const ringScale = project.ringScale / 100;

      [0.9, 0.7, 0.5, 0.35].forEach((band) => {
        const r = radiusForSim(band, ringScale);
        const ringEl = document.createElement("div");
        ringEl.className = "ng-ring";
        ringEl.style.width = r * 2 + "px";
        ringEl.style.height = r * 2 + "px";
        stageEl.appendChild(ringEl);
      });

      const center = document.createElement("div");
      center.className = "ng-node ng-node-center";
      center.style.width = CENTER_SIZE + "px";
      center.style.height = CENTER_SIZE + "px";
      center.style.transform = "translate(-50%, -50%)";
      center.dataset.baseX = 0;
      center.dataset.baseY = 0;
      center.innerHTML = `<img src="${ring.anchorUrl}">`;
      center.addEventListener("mouseenter", () => showHoverPreview({ filename: "Reference (anchor)", thumbUrl: ring.anchorUrl, similarity: 1 }));
      center.addEventListener("mouseleave", hideHoverPreview);
      stageEl.appendChild(center);

      const bands = Array.from({ length: BAND_COUNT }, () => []);
      kept.forEach((r) => {
        const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
        const bandIdx = Math.min(BAND_COUNT - 1, Math.floor((1 - t) * BAND_COUNT));
        bands[bandIdx].push(r);
      });

      bands.forEach((bandResults, bandIdx) => {
        if (!bandResults.length) return;
        const t = 1 - bandIdx / (BAND_COUNT - 1);
        const avgSim = 0.25 + t * 0.75;
        const radius = radiusForSim(avgSim, ringScale);
        const size = sizeForSim(avgSim);
        const angleOffset = bandIdx * 0.6;

        bandResults.forEach((r, i) => {
          const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          const node = document.createElement("div");
          node.className = "ng-node" + (project.selectedFrames.has(r.frame) ? " ng-node-selected" : "");
          node.style.width = size + "px";
          node.style.height = size + "px";
          node.style.left = `calc(50% + ${x}px)`;
          node.style.top = `calc(50% + ${y}px)`;
          node.style.transform = "translate(-50%, -50%)";
          node.dataset.baseX = x;
          node.dataset.baseY = y;
          node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}%`;
          node.innerHTML = `<img src="${thumbUrlFor(r)}">`;

          node.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.toggleFrameSelection(r.frame);
            this.renderStage(project);
          };
          node.addEventListener("mouseenter", () => showHoverPreview(r));
          node.addEventListener("mouseleave", hideHoverPreview);
          stageEl.appendChild(node);
        });
      });

      this.renderRankedList(project, kept);
    },

    renderRankedList(project, kept) {
      listBodyFramesEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.frame = r.frame;
        const pct = (r.similarity * 100).toFixed(1);

        let poseHtml = "";
        if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
          let poseLine = `pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
          if (typeof r.blur === "number") poseLine += ` · sharp: ${r.blur.toFixed(0)}`;
          if (typeof r.vertFillPct === "number") poseLine += ` · ${(r.vertFillPct * 100).toFixed(0)}% frame ht`;
          poseHtml = `<div style="font-size:10px;color:var(--ng-text-dim);margin-top:2px;">${poseLine}</div>`;
        }

        row.innerHTML = `
          <input type="checkbox" class="ng-frame-select-cb" data-frame="${r.frame}" ${project.selectedFrames.has(r.frame) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${thumbUrlFor(r)}" loading="lazy">
          <div class="info">
            <div class="fname">${r.filename}</div>
            <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
            ${poseHtml}
          </div>
          <div class="simpct">${pct}%</div>
        `;
        const cb = row.querySelector(".ng-frame-select-cb");
        cb.addEventListener("change", () => {
          project.toggleFrameSelection(r.frame);
          this.renderStage(project);
        });
        row.addEventListener("mouseenter", () => showHoverPreview(r));
        row.addEventListener("mouseleave", hideHoverPreview);
        listBodyFramesEl.appendChild(row);
      });

      framesSectionEl.style.display = kept.length ? "" : "none";
      framesSectionCountEl.textContent = kept.length ? `(${kept.length})` : "";
    },

    // ---- Immich stage: same ring-layout approach as the video stage, but
    // centered on an Immich asset instead of a video frame, with a
    // click-to-recenter interaction the video ring has no equivalent for
    // (video frames aren't independently-addressable "assets"). Kept as
    // its own method (some duplication vs renderVideoStage) rather than a
    // shared abstraction -- matches this codebase's existing NG-duplicate
    // philosophy and is safer than reshaping the working video path. ----
    renderImmichStage(project) {
      framesSectionEl.style.display = "none";
      const ring = project.immichRing;
      const anchorUrl = `/api/ng/preview/${ring.centerAssetId}`;
      const modeLabel = ring.mode === "clip" ? "CLIP image embedding (no face match found)" : "face embedding";
      const metricLabel = { sim: "Similarity", yaw: "Yaw", pitch: "Pitch", roll: "Roll", blur: "Sharpness" }[project.ringSortMetric];

      hudModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      hudFilenameEl.textContent = `${ring.centerFilename} · sorted by ${metricLabel}`;

      sidebarCurrentImgEl.src = anchorUrl;
      sidebarCurrentFnameEl.textContent = ring.centerFilename;
      sidebarCurrentModeEl.textContent = `IMMICH NEIGHBORS (${modeLabel})`;
      if (ring.centerPose) {
        const p = ring.centerPose;
        sidebarCurrentDetailEl.textContent =
          `pitch: ${p.pitch.toFixed(1)} yaw: ${p.yaw.toFixed(1)} roll: ${p.roll.toFixed(1)}` +
          (typeof p.blur === "number" ? ` · sharpness: ${p.blur.toFixed(0)}` : "");
      } else {
        sidebarCurrentDetailEl.textContent = "match: 100.0%";
      }

      const sorted = project.sortedRankedImmich();
      const { kept, total } = project.squeezeFiltered(sorted);
      squeezeVal.textContent = `${project.squeezeMinPct}% (${kept.length}/${total})`;
      sharpVal.textContent = `${project.sharpMinVal} (${kept.length}/${total})`;

      if (project.ringSortMetric !== "sim") {
        stageEl.style.display = "none";
        poseListViewEl.style.display = "flex";
        poseListScrubberEl.style.display = "flex";
        renderPoseListNG(project, anchorUrl, ring.centerFilename, kept, (r) => project.toggleAssetSelection(r.assetId), (r) => project.selectedAssetIds.has(r.assetId), (r) => project.recenterImmich(r.assetId, r.filename));
        this.renderRankedListImmich(project, kept);
        return;
      }
      stageEl.style.display = "";
      poseListViewEl.style.display = "none";
      poseListScrubberEl.style.display = "none";

      stageEl.innerHTML = "";
      const ringScale = project.ringScale / 100;

      [0.9, 0.7, 0.5, 0.35].forEach((band) => {
        const r = radiusForSim(band, ringScale);
        const ringEl = document.createElement("div");
        ringEl.className = "ng-ring";
        ringEl.style.width = r * 2 + "px";
        ringEl.style.height = r * 2 + "px";
        stageEl.appendChild(ringEl);
      });

      const center = document.createElement("div");
      center.className = "ng-node ng-node-center";
      center.style.width = CENTER_SIZE + "px";
      center.style.height = CENTER_SIZE + "px";
      center.style.transform = "translate(-50%, -50%)";
      center.dataset.baseX = 0;
      center.dataset.baseY = 0;
      center.innerHTML = `<img src="${anchorUrl}">`;
      center.addEventListener("mouseenter", () => showHoverPreview({ filename: ring.centerFilename + " (centered)", thumbUrl: anchorUrl, similarity: 1 }));
      center.addEventListener("mouseleave", hideHoverPreview);
      stageEl.appendChild(center);

      const bands = Array.from({ length: BAND_COUNT }, () => []);
      kept.forEach((r) => {
        const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
        const bandIdx = Math.min(BAND_COUNT - 1, Math.floor((1 - t) * BAND_COUNT));
        bands[bandIdx].push(r);
      });

      bands.forEach((bandResults, bandIdx) => {
        if (!bandResults.length) return;
        const t = 1 - bandIdx / (BAND_COUNT - 1);
        const avgSim = 0.25 + t * 0.75;
        const radius = radiusForSim(avgSim, ringScale);
        const size = sizeForSim(avgSim);
        const angleOffset = bandIdx * 0.6;

        bandResults.forEach((r, i) => {
          const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          const node = document.createElement("div");
          node.className = "ng-node" + (project.selectedAssetIds.has(r.assetId) ? " ng-node-selected" : "");
          node.style.width = size + "px";
          node.style.height = size + "px";
          node.style.left = `calc(50% + ${x}px)`;
          node.style.top = `calc(50% + ${y}px)`;
          node.style.transform = "translate(-50%, -50%)";
          node.dataset.baseX = x;
          node.dataset.baseY = y;
          node.title = `${r.filename} — ${(r.similarity * 100).toFixed(1)}% (click to recenter, dblclick to select)`;
          node.innerHTML = `<img src="${thumbUrlFor(r)}">`;

          node.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.recenterImmich(r.assetId, r.filename);
          });
          node.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            project.toggleAssetSelection(r.assetId);
            this.renderStage(project);
          };
          node.addEventListener("mouseenter", () => showHoverPreview(r));
          node.addEventListener("mouseleave", hideHoverPreview);
          stageEl.appendChild(node);
        });
      });

      this.renderRankedListImmich(project, kept);
    },

    renderRankedListImmich(project, kept) {
      listBodyImmichEl.innerHTML = "";
      kept.forEach((r) => {
        const row = document.createElement("div");
        row.className = "ng-list-row";
        row.dataset.assetId = r.assetId;
        const pct = (r.similarity * 100).toFixed(1);

        row.innerHTML = `
          <input type="checkbox" class="ng-asset-select-cb" data-asset="${r.assetId}" ${project.selectedAssetIds.has(r.assetId) ? "checked" : ""} style="margin-right:6px;flex-shrink:0;">
          <img src="${thumbUrlFor(r)}" loading="lazy">
          <div class="info">
            <div class="fname">${r.filename}</div>
            <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="simpct">${pct}%</div>
        `;
        const cb = row.querySelector(".ng-asset-select-cb");
        cb.addEventListener("change", (e) => {
          e.stopPropagation();
          project.toggleAssetSelection(r.assetId);
          this.renderStage(project);
        });
        row.querySelector("img").addEventListener("click", () => project.recenterImmich(r.assetId, r.filename));
        row.addEventListener("mouseenter", () => showHoverPreview(r));
        row.addEventListener("mouseleave", hideHoverPreview);
        listBodyImmichEl.appendChild(row);
      });

      immichSectionEl.style.display = kept.length ? "" : "none";
      immichSectionCountEl.textContent = kept.length ? `(${kept.length})` : "";

      // Refresh re-pulls fresh neighbor results from Immich for the
      // current center asset -- the NG shell deliberately never
      // auto-fetches on its own (see APP_ARCHITECTURE_NOTES.md), so
      // unlike the original app (where a plain page reload re-ran the
      // query), getting an up-to-date pull here needs an explicit action.
      immichRefreshBtn.disabled = project.immichLoading || !project.immichRing;
      immichRefreshBtn.textContent = project.immichLoading ? "Refreshing…" : "Refresh";

      const n = project.selectedAssetIds.size;
      immichSaveSelectedBtn.textContent = `Save ${n} selected to disk`;
      immichSaveSelectedBtn.disabled = n === 0;
    },

    // ---- Left rail: ported sidebar, shown whenever a project is active ----
    renderLeftRail() {
      const active = this.getActive();

      if (!active) {
        leftRailEmptyEl.style.display = "";
        leftRailBodyEl.style.display = "none";
        return;
      }
      leftRailEmptyEl.style.display = "none";
      leftRailBodyEl.style.display = "flex";

      // The three ingest sources are separate switchable "pages" within
      // the tab -- only the section(s) for the active task are shown.
      // Video and Folder/Zip share one "…Analysis" section (same shared
      // job/thresholds/status underneath, see this.job on CharacterProject)
      // but swap which entry-point controls are visible so folder/zip work
      // doesn't show video-only chrome (video picker, "Analyze window (sec)").
      videoAnalysisSectionEl.style.display = (active.task === "video" || active.task === "folderzip") ? "" : "none";
      searchSectionEl.style.display = active.task === "immich" ? "" : "none";
      if (analysisSectionTitleEl) {
        analysisSectionTitleEl.textContent = active.task === "folderzip" ? "Folder / Zip Analysis" : "Video Analysis";
      }
      videoAnalysisBodyEl.style.display = active.task === "video" ? "" : "none";
      if (videoRangeRowEl) videoRangeRowEl.style.display = active.task === "video" ? "" : "none";
      if (folderRowEl) folderRowEl.style.display = active.task === "folderzip" ? "" : "none";

      // Pose Picker / Shot Scale Picker: re-run setup (reset sliders to the
      // pool's centroid, reset sticky grid slots) only when this project's
      // ring identity actually changed since we last looked -- not on
      // every render tick (polling, unrelated toggles, etc.).
      if (active.ring !== active._posePickerRingRef) {
        active._posePickerRingRef = active.ring;
        setupPosePickerNG(active);
      }
      if (active.ring !== active._scalePickerRingRef) {
        active._scalePickerRingRef = active.ring;
        setupScalePickerNG(active);
      }
      renderPosePickerGridNG(active);
      renderScalePickerGridNG(active);

      ringScaleInput.value = active.ringScale;
      ringScaleVal.textContent = active.ringScale + "%";
      squeezeSlider.value = active.squeezeMinPct;
      ringSortCbs.forEach((cb) => { cb.checked = cb.dataset.metric === active.ringSortMetric; });
      sharpEnableCb.checked = active.sharpCutoffEnabled;
      sharpControlsEl.style.display = active.sharpCutoffEnabled ? "flex" : "none";
      sharpSlider.value = active.sharpMinVal;
      folderRefIndexInput.value = active.folderRefIndex;
      if (active.task === "immich" && active.immichRing && !(active.ring && active.ring.sourceType === "immich")) {
        const { kept, total } = active.squeezeFiltered(active.sortedRankedImmich());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
        sharpVal.textContent = `${active.sharpMinVal} (${kept.length}/${total})`;
      } else if (active.ring) {
        const { kept, total } = active.squeezeFiltered(active.sortedRanked());
        squeezeVal.textContent = `${active.squeezeMinPct}% (${kept.length}/${total})`;
        sharpVal.textContent = `${active.sharpMinVal} (${kept.length}/${total})`;
      } else {
        squeezeVal.textContent = `${active.squeezeMinPct}% (0/0)`;
        sharpVal.textContent = `${active.sharpMinVal} (0/0)`;
      }

      simThresholdInput.value = active.simThreshold;
      blurThresholdInput.value = active.blurThreshold;
      cacheFormatPngCb.checked = active.cacheFormatPng;
      if (immichSimThresholdInput) immichSimThresholdInput.value = active.simThreshold;
      if (immichBlurThresholdInput) immichBlurThresholdInput.value = active.blurThreshold;
      if (immichAnalyzeRefIndexInput) immichAnalyzeRefIndexInput.value = active.immichAnalyzeRefIndex;
      rankedSortRadios.forEach((cb) => {
        const currentMetric = active.task === "immich" ? active.immichRankedSortMetric : active.rankedSortMetric;
        cb.checked = cb.value === currentMetric;
      });

      if (immichAnalyzeBtn) {
        const n = active.selectedAssetIds.size;
        const jobRunning = active.job && active.job.status === "running";
        immichAnalyzeBtn.textContent = `Analyze ${n} selected asset${n === 1 ? "" : "s"}`;
        immichAnalyzeBtn.disabled = n === 0 || jobRunning;
        immichAnalyzeBtn.title = n === 0
          ? "Tick assets in the Immich matches list or a person-cluster grid first"
          : (jobRunning ? "Analysis already running for this tab" : "");
      }

      if (immichSelectionBarEl) {
        const n = active.selectedAssetIds.size;
        immichSelectionBarEl.style.display = n > 0 ? "block" : "none";
        if (immichSelectionCountEl) immichSelectionCountEl.textContent = n;
        const jobRunning = active.job && active.job.status === "running";
        if (immichAnalyzeSelectedBtn) immichAnalyzeSelectedBtn.disabled = n === 0 || jobRunning;
        if (immichExportSelectedBtn) immichExportSelectedBtn.disabled = n === 0;
      }

      this.renderVideoAnalysisBody(active);
      this.renderFramePreview(active);
      this.renderAnalysisStatus(active);
      this.renderImmichAnalysisStatus(active);
      this.renderImmichSearch(active);
    },

    renderImmichSearch(project) {
      if (document.activeElement !== immichSearchInput) {
        immichSearchInput.value = project.immichSearchQuery;
      }
      if (immichRandomFaceBtn) immichRandomFaceBtn.disabled = project.immichSearching;
      immichSearchStatusEl.textContent = project.immichSearching ? "Searching…" : "";
      immichSearchResultsEl.innerHTML = "";
      if (!project.immichSearchResults.length) {
        immichSearchResultsEl.style.display = "none";
        return;
      }
      immichSearchResultsEl.style.display = "";
      project.immichSearchResults.forEach((m) => {
        const row = document.createElement("div");
        row.className = "ng-search-result-row";
        row.innerHTML = `<img src="/api/ng/thumb/${m.assetId}" loading="lazy"><span class="ng-search-result-fname">${m.filename}</span>`;
        row.addEventListener("click", () => project.loadImmichNeighbors(m.assetId, m.filename));
        immichSearchResultsEl.appendChild(row);
      });
    },

    renderVideoAnalysisBody(project) {
      if (project.task !== "video") {
        videoAnalysisBodyEl.innerHTML = "";
        return;
      }
      videoAnalysisBodyEl.innerHTML = "";

      if (project.videoLoading) {
        videoAnalysisBodyEl.appendChild(placeholder("Loading video..."));
      } else if (project.video) {
        const info = document.createElement("p");
        info.className = "ng-video-hint";
        info.textContent =
          project.video.fps.toFixed(2) + " fps, " +
          project.video.totalFrames + " frames, " +
          project.video.duration.toFixed(1) + "s";
        videoAnalysisBodyEl.appendChild(info);
        videoAnalysisBodyEl.appendChild(videoPickerButton(project, "Replace Video"));
      } else {
        videoAnalysisBodyEl.appendChild(placeholder("No video loaded for this project yet."));
        videoAnalysisBodyEl.appendChild(videoPickerButton(project, "Choose Video..."));
      }

      analysisStartInput.value = project.video && project.video.rangeStartSec != null ? project.video.rangeStartSec : "";
      analysisEndInput.value = project.video && project.video.rangeEndSec != null ? project.video.rangeEndSec : "";

      startAnalysisBtn.disabled = !project.video || project.videoLoading || (project.job && project.job.status === "running");
      startAnalysisBtn.title = !project.video
        ? "Load a video and pick a frame first"
        : (project.job && project.job.status === "running" ? "Analysis already running for this tab" : "");
    },

    renderAnalysisStatus(project) {
      if (!project.job) {
        analysisStatusEl.innerHTML = "";
        return;
      }
      const j = project.job;
      if (j.status === "error") {
        analysisStatusEl.innerHTML = `Error: ${j.error || "unknown error"}`;
        return;
      }
      if (j.status === "running") {
        analysisStatusEl.innerHTML = `
          Processing… ${j.frameCount || 0} images seen<br>
          ${j.passed || 0} kept, ${j.failedSim || 0} low sim, ${j.failedBlur || 0} blurry
          <div class="ng-progress-track"><div class="ng-progress-fill" style="width:${Math.min(100, ((j.frameCount || 0) / 200) * 100)}%"></div></div>
        `;
        return;
      }

      const selCount = project.selectedFrames.size;
      const isVideo = j.sourceType === "video";

      analysisStatusEl.innerHTML = `
        Done — ${j.passed}/${j.frameCount} kept.
        <div id="ng-sim-sparkline-wrap" style="margin-top:8px;"></div>
        <button id="ng-btn-save-kept" class="ng-btn ng-btn-full ng-btn-accent">Save kept frames to disk</button>
        <button id="ng-btn-save-selected" class="ng-btn ng-btn-full ng-btn-accent" ${selCount ? "" : "disabled"}>Save ${selCount} selected frame${selCount === 1 ? "" : "s"} to disk</button>
        <button id="ng-btn-view-selected" class="ng-btn ng-btn-full">View selected</button>
        <div id="ng-export-result" class="ng-stub-note"></div>
      `;

      renderNGChart(project, analysisStatusEl.querySelector("#ng-sim-sparkline-wrap"));

      const exportResultEl = analysisStatusEl.querySelector("#ng-export-result");
      const saveKeptBtn = analysisStatusEl.querySelector("#ng-btn-save-kept");
      const saveSelectedBtn = analysisStatusEl.querySelector("#ng-btn-save-selected");
      const viewSelectedBtn = analysisStatusEl.querySelector("#ng-btn-view-selected");

      saveKeptBtn.onclick = async () => {
        saveKeptBtn.textContent = "Saving…";
        saveKeptBtn.disabled = true;
        try {
          const result = await project.exportFrames(false);
          exportResultEl.textContent = result.error
            ? `Error: ${result.error}`
            : `Saved ${result.exported} frames → ${result.path}`;
        } catch (e) {
          exportResultEl.textContent = `Error: ${e.message}`;
        }
        saveKeptBtn.textContent = "Save kept frames to disk";
        saveKeptBtn.disabled = false;
      };

      if (saveSelectedBtn) {
        saveSelectedBtn.onclick = async () => {
          saveSelectedBtn.textContent = "Saving…";
          saveSelectedBtn.disabled = true;
          try {
            const result = await project.exportFrames(true);
            exportResultEl.textContent = result.error
              ? `Error: ${result.error}`
              : `Saved ${result.exported} frames → ${result.path}`;
          } catch (e) {
            exportResultEl.textContent = `Error: ${e.message}`;
          }
          saveSelectedBtn.textContent = `Save ${project.selectedFrames.size} selected frame${project.selectedFrames.size === 1 ? "" : "s"} to disk`;
          saveSelectedBtn.disabled = project.selectedFrames.size === 0;
        };
      }

      viewSelectedBtn.onclick = () => ProjectManager.openSelectedModal(project);

      // "Pop out playback" only makes sense for a real video source --
      // folder/image-set jobs have no clip for build-playback to
      // reassemble (the backend requires job.videoBytes).
      if (isVideo) {
        const btn = document.createElement("button");
        btn.className = "ng-btn ng-btn-full ng-btn-accent";
        btn.style.marginTop = "6px";
        btn.disabled = project.playbackBuilding;
        btn.textContent = project.playbackBuilding
          ? "Building playback…"
          : project.playback
            ? "Reopen playback"
            : "Pop out playback (rejected frames blanked)";
        btn.addEventListener("click", () => {
          if (project.playback) PlaybackModal.openBuild(project);
          else project.buildPlayback();
        });
        analysisStatusEl.appendChild(btn);
      }
    },

    // ---- Immich batch-analysis status: parallel to renderAnalysisStatus,
    // targeting its own status div (ng-immich-analysis-status, inside the
    // Search panel section) rather than #ng-analysis-status, since that
    // div's container (the Video/Image-Set Analysis panel section) is
    // hidden entirely while the Immich task is active. Some duplication
    // vs renderAnalysisStatus, matching this codebase's existing
    // NG-duplicate philosophy (see the comment on renderImmichStage). ----
    renderImmichAnalysisStatus(project) {
      if (!immichAnalysisStatusEl) return;
      const j = project.job;
      if (!j || j.sourceType !== "immich") {
        immichAnalysisStatusEl.innerHTML = "";
        return;
      }
      if (j.status === "error") {
        immichAnalysisStatusEl.innerHTML = `Error: ${j.error || "unknown error"}`;
        return;
      }
      if (j.status === "running") {
        immichAnalysisStatusEl.innerHTML = `
          Fetching &amp; analyzing… ${j.frameCount || 0} assets seen<br>
          ${j.passed || 0} kept, ${j.failedSim || 0} low sim, ${j.failedBlur || 0} blurry
          <div class="ng-progress-track"><div class="ng-progress-fill" style="width:${Math.min(100, ((j.frameCount || 0) / 200) * 100)}%"></div></div>
        `;
        return;
      }

      const selCount = project.selectedFrames.size;
      const fetchErrCount = (j.fetchErrors || []).length;

      immichAnalysisStatusEl.innerHTML = `
        Done — ${j.passed}/${j.frameCount} kept${fetchErrCount ? ` (${fetchErrCount} asset${fetchErrCount === 1 ? "" : "s"} failed to fetch)` : ""}.
        <div id="ng-immich-sim-sparkline-wrap" style="margin-top:8px;"></div>
        <button id="ng-immich-btn-save-kept" class="ng-btn ng-btn-full ng-btn-accent">Save kept frames to disk</button>
        <button id="ng-immich-btn-save-selected" class="ng-btn ng-btn-full ng-btn-accent" ${selCount ? "" : "disabled"}>Save ${selCount} selected frame${selCount === 1 ? "" : "s"} to disk</button>
        <button id="ng-immich-btn-view-selected" class="ng-btn ng-btn-full">View selected</button>
        <div id="ng-immich-export-result" class="ng-stub-note"></div>
      `;

      renderNGChart(project, immichAnalysisStatusEl.querySelector("#ng-immich-sim-sparkline-wrap"));

      const exportResultEl = immichAnalysisStatusEl.querySelector("#ng-immich-export-result");
      const saveKeptBtn = immichAnalysisStatusEl.querySelector("#ng-immich-btn-save-kept");
      const saveSelectedBtn = immichAnalysisStatusEl.querySelector("#ng-immich-btn-save-selected");
      const viewSelectedBtn = immichAnalysisStatusEl.querySelector("#ng-immich-btn-view-selected");

      saveKeptBtn.onclick = async () => {
        saveKeptBtn.textContent = "Saving…";
        saveKeptBtn.disabled = true;
        try {
          const result = await project.exportFrames(false);
          exportResultEl.textContent = result.error
            ? `Error: ${result.error}`
            : `Saved ${result.exported} frames → ${result.path}`;
        } catch (e) {
          exportResultEl.textContent = `Error: ${e.message}`;
        }
        saveKeptBtn.textContent = "Save kept frames to disk";
        saveKeptBtn.disabled = false;
      };

      if (saveSelectedBtn) {
        saveSelectedBtn.onclick = async () => {
          saveSelectedBtn.textContent = "Saving…";
          saveSelectedBtn.disabled = true;
          try {
            const result = await project.exportFrames(true);
            exportResultEl.textContent = result.error
              ? `Error: ${result.error}`
              : `Saved ${result.exported} frames → ${result.path}`;
          } catch (e) {
            exportResultEl.textContent = `Error: ${e.message}`;
          }
          saveSelectedBtn.textContent = `Save ${project.selectedFrames.size} selected frame${project.selectedFrames.size === 1 ? "" : "s"} to disk`;
          saveSelectedBtn.disabled = project.selectedFrames.size === 0;
        };
      }

      viewSelectedBtn.onclick = () => ProjectManager.openSelectedModal(project);
    },

    openSelectedModal(project) {
      const overlay = document.getElementById("ng-selected-modal");
      const grid = document.getElementById("ng-selected-modal-grid");
      const title = document.getElementById("ng-selected-modal-title");
      const poseLayoutCb = document.getElementById("ng-selected-modal-pose-layout");
      const spreadWrapEl = document.getElementById("ng-selected-modal-spread-wrap");
      const spreadSliderEl = document.getElementById("ng-selected-modal-spread");
      const realPreviewCb = document.getElementById("ng-selected-modal-real-preview");
      const deselectAllBtn = document.getElementById("ng-selected-modal-deselect-all");
      if (!overlay || !grid) return;

      const POSE_SCATTER_SPREAD_BASE = 42;
      const jobId = project.job ? project.job.jobId : null;

      // Real crop preview reuses the exact same crop_resize_export call the
      // actual export does (routes/exportNG.py's export-preview[-immich]) --
      // what you see here is genuinely what gets written, not a generic
      // square thumbnail unrelated to crop mode/margin/native.
      function srcFor(it) {
        if (!realPreviewCb.checked) return it.thumb;
        const params = new URLSearchParams(gatherExportParamsNG());
        if (it.kind === "frame" && jobId) return `/api/ng/export-preview/${jobId}/${it.frame}?${params.toString()}`;
        if (it.kind === "asset") return `/api/ng/export-preview-immich/${it.assetId}?${params.toString()}`;
        return it.thumb;
      }

      // Immich assets ticked while browsing neighbors (project.selectedAssetIds)
      // are a separate pool from job-cached frames -- combined here to match
      // the original app's renderSelectionModal(), which folds both
      // selectedAssetIds and selectedFrames into one grid. assetPoseCache
      // covers assets whose pose was fetched on-demand via "Detect pose"
      // below rather than coming pre-attached from a batch analysis job.
      function buildItems() {
        const frameItems = (project.ring ? project.ring.baseResults : [])
          .filter((r) => project.selectedFrames.has(r.frame))
          .map((r) => ({
            kind: "frame", frame: r.frame, filename: r.filename, thumb: thumbUrlFor(r),
            similarity: r.similarity,
            pitch: typeof r.pitch === "number" ? r.pitch : null,
            yaw: typeof r.yaw === "number" ? r.yaw : null,
          }));
        const assetItems = Array.from(project.selectedAssetIds).map((assetId) => {
          const known = project.immichRing ? project.immichRing.baseResults.find((r) => r.assetId === assetId) : null;
          const cached = project.assetPoseCache[assetId];
          return {
            kind: "asset", assetId, filename: known ? known.filename : assetId,
            thumb: known ? thumbUrlFor(known) : `/api/ng/thumb/${assetId}`,
            similarity: known ? known.similarity : undefined,
            pitch: cached ? cached.pitch : (known && typeof known.pitch === "number" ? known.pitch : null),
            yaw: cached ? cached.yaw : (known && typeof known.yaw === "number" ? known.yaw : null),
          };
        });
        return [...assetItems, ...frameItems];
      }

      function removeItem(it) {
        if (it.kind === "asset") project.toggleAssetSelection(it.assetId);
        else project.toggleFrameSelection(it.frame);
        if (project.isActive) ProjectManager.render();
        renderModal();
      }

      function renderGridMode(items) {
        grid.classList.remove("ng-pose-scatter-mode");
        grid.innerHTML = "";
        if (!items.length) {
          grid.appendChild(placeholder("Nothing selected yet — dblclick a ring node or its checkbox to select."));
          return;
        }
        items.forEach((r) => {
          const cell = document.createElement("div");
          cell.className = "ng-selected-modal-cell";
          cell.innerHTML = `
            <img src="${srcFor(r)}" loading="lazy">
            <div class="ng-selected-modal-cell-info">${r.filename}${typeof r.similarity === "number" ? ` — ${(r.similarity * 100).toFixed(1)}%` : ""}</div>
            <button class="ng-selected-modal-remove" title="Remove from selection">&times;</button>
          `;
          cell.querySelector("img").addEventListener("click", () => {
            if (r.kind === "asset") {
              project.recenterImmich(r.assetId, r.filename);
              overlay.style.display = "none";
            } else if (project.job && project.job.sourceType === "video" && project.video) {
              project.stepAndSyncAudio(r.frame);
            } else {
              showStaticFramePreviewNG(project, r);
            }
          });
          cell.querySelector(".ng-selected-modal-remove").addEventListener("click", () => removeItem(r));
          grid.appendChild(cell);
        });
      }

      async function detectPoseForItems(items) {
        const btn = document.getElementById("ng-detect-pose-btn");
        if (btn) { btn.textContent = `Detecting 0/${items.length}\u2026`; btn.disabled = true; }
        let done = 0;
        await Promise.all(items.map(async (it) => {
          try {
            const res = await fetch(`/api/ng/asset-face-pose/${it.assetId}`);
            const data = await res.json();
            if (!data.error) {
              // full metric set (pose + blur + vertFillPct), same as a
              // video/folder analysis frame would carry -- cached so
              // sharpness sort/filtering works on on-demand-detected
              // Immich items too, matching the original's assetPoseCache.
              project.assetPoseCache[it.assetId] = { pitch: data.pitch, yaw: data.yaw, blur: data.blur, vertFillPct: data.vertFillPct };
            }
          } catch (e) {
            console.warn("Pose detection failed for", it.assetId, e);
          } finally {
            done++;
            if (btn) btn.textContent = `Detecting ${done}/${items.length}\u2026`;
          }
        }));
        renderModal();
      }

      // Pitch/yaw scatter, anchored on this selection's own pose centroid
      // (not absolute zero) -- if the whole set is consistently
      // turned/tilted, anchoring on absolute zero would pick an
      // unrepresentative outlier and huddle everything else in one corner.
      function renderPoseScatterMode(items) {
        grid.classList.add("ng-pose-scatter-mode");
        grid.innerHTML = "";
        const posed = items.filter((it) => it.pitch !== null && it.yaw !== null);
        const unposed = items.filter((it) => it.pitch === null || it.yaw === null);

        if (!posed.length) {
          const detectable = unposed.filter((it) => it.kind === "asset");
          const wrap = document.createElement("div");
          wrap.style.cssText = "color:var(--ng-text-dim);font-size:11px;text-align:center;padding:20px;";
          wrap.innerHTML = "No pose data on the current selection — pitch/yaw only comes from the pose-analysis pipeline (analyze video/folder/Immich selection).<br><br>" +
            (detectable.length ? `<button type="button" id="ng-detect-pose-btn">Detect pose for ${detectable.length} Immich item(s)</button>` : "");
          grid.appendChild(wrap);
          const detectBtn = document.getElementById("ng-detect-pose-btn");
          if (detectBtn) detectBtn.addEventListener("click", () => detectPoseForItems(detectable));
          return;
        }

        const meanPitch = posed.reduce((s, it) => s + it.pitch, 0) / posed.length;
        const meanYaw = posed.reduce((s, it) => s + it.yaw, 0) / posed.length;
        let anchor = posed[0];
        let bestScore = Infinity;
        posed.forEach((it) => {
          const score = Math.abs(it.pitch - meanPitch) + Math.abs(it.yaw - meanYaw);
          if (score < bestScore) { bestScore = score; anchor = it; }
        });

        const deltaYaw = (it) => it.yaw - anchor.yaw;
        const deltaPitch = (it) => it.pitch - anchor.pitch;
        const yawMax = Math.max(15, ...posed.map((it) => Math.abs(deltaYaw(it))));
        const pitchMax = Math.max(15, ...posed.map((it) => Math.abs(deltaPitch(it))));

        const stage = document.createElement("div");
        stage.className = "ng-pose-scatter-stage";
        const spread = parseFloat(spreadSliderEl.value) || 1;

        posed.forEach((it) => {
          const isAnchor = it === anchor;
          // pitch inverted to match the pose-list strip's convention:
          // positive pitch (nose up) moves toward the top of the stage.
          const yawRatio = isAnchor ? 0 : deltaYaw(it) / yawMax;
          const pitchRatio = isAnchor ? 0 : -deltaPitch(it) / pitchMax;
          const leftPct = 50 + yawRatio * POSE_SCATTER_SPREAD_BASE * spread;
          const topPct = 50 + pitchRatio * POSE_SCATTER_SPREAD_BASE * spread;
          const size = isAnchor ? 108 : 76;

          const cell = document.createElement("div");
          cell.className = "ng-pose-scatter-item" + (isAnchor ? " ng-pose-scatter-anchor" : "");
          // yawRatio/pitchRatio kept on the cell so the Spread slider can
          // just recompute left/top directly -- no rebuild, no image
          // reload/refetch, which matters with real-crop-preview on since
          // that hits the network per Immich asset.
          cell.dataset.yawRatio = yawRatio;
          cell.dataset.pitchRatio = pitchRatio;
          cell.dataset.baseTransform = "translate(-50%,-50%)";
          cell.dataset.baseZ = isAnchor ? "5" : "2";
          cell.style.left = `${Math.max(4, Math.min(96, leftPct))}%`;
          cell.style.top = `${Math.max(4, Math.min(96, topPct))}%`;
          cell.style.transform = cell.dataset.baseTransform;
          cell.style.zIndex = cell.dataset.baseZ;
          cell.style.width = `${size}px`;
          const titleAttr = isAnchor
            ? `pitch ${it.pitch.toFixed(1)}, yaw ${it.yaw.toFixed(1)} (closest to this selection's pose centroid, not necessarily true zero)`
            : `pitch ${it.pitch.toFixed(1)}, yaw ${it.yaw.toFixed(1)} — ${deltaPitch(it) >= 0 ? "+" : ""}${deltaPitch(it).toFixed(1)}p / ${deltaYaw(it) >= 0 ? "+" : ""}${deltaYaw(it).toFixed(1)}y from center`;
          cell.innerHTML = `
            <img src="${srcFor(it)}" loading="lazy" title="${titleAttr} — double-click to remove" style="width:${size}px;height:${size}px;">
            <div class="ng-pose-scatter-label">${isAnchor ? `center (p${it.pitch.toFixed(0)} y${it.yaw.toFixed(0)})` : `p${it.pitch.toFixed(0)} y${it.yaw.toFixed(0)}`}</div>
          `;
          cell.ondblclick = () => removeItem(it);
          stage.appendChild(cell);
        });

        grid.appendChild(stage);
        // reuse the same dock-style magnify used on the pose-list strip --
        // genuinely overlapping thumbnails at similar pitch/yaw are
        // otherwise impossible to pick apart.
        attachNgLensEffect(stage, ".ng-pose-scatter-item", { radius: 90, maxScale: 1.8 });

        if (unposed.length) {
          const detectable = unposed.filter((it) => it.kind === "asset");
          const strip = document.createElement("div");
          strip.className = "ng-pose-scatter-unposed-strip";
          strip.innerHTML = `
            <div class="ng-pose-scatter-unposed-header">
              <span>No pose data (${unposed.length}):</span>
              ${detectable.length ? `<button type="button" id="ng-detect-pose-btn">Detect pose (${detectable.length})</button>` : ""}
            </div>
          `;
          const row = document.createElement("div");
          row.className = "ng-pose-scatter-unposed-row";
          unposed.forEach((it) => {
            const cell = document.createElement("div");
            cell.className = "ng-pose-scatter-unposed-cell";
            cell.innerHTML = `<img src="${srcFor(it)}" loading="lazy" title="Double-click to remove">`;
            cell.ondblclick = () => removeItem(it);
            row.appendChild(cell);
          });
          strip.appendChild(row);
          grid.appendChild(strip);
          const detectBtn = document.getElementById("ng-detect-pose-btn");
          if (detectBtn) detectBtn.addEventListener("click", () => detectPoseForItems(detectable));
        }
      }

      function renderModal() {
        const items = buildItems();
        title.textContent = `Selected (${items.length})`;
        spreadWrapEl.style.display = poseLayoutCb.checked ? "flex" : "none";
        if (poseLayoutCb.checked) renderPoseScatterMode(items);
        else renderGridMode(items);
      }

      poseLayoutCb.onchange = renderModal;
      realPreviewCb.onchange = renderModal;
      spreadSliderEl.oninput = () => {
        // recompute positions only -- no rebuild/refetch (see note above)
        const spread = parseFloat(spreadSliderEl.value) || 1;
        grid.querySelectorAll(".ng-pose-scatter-item").forEach((cell) => {
          const yawRatio = parseFloat(cell.dataset.yawRatio) || 0;
          const pitchRatio = parseFloat(cell.dataset.pitchRatio) || 0;
          const leftPct = 50 + yawRatio * POSE_SCATTER_SPREAD_BASE * spread;
          const topPct = 50 + pitchRatio * POSE_SCATTER_SPREAD_BASE * spread;
          cell.style.left = `${Math.max(4, Math.min(96, leftPct))}%`;
          cell.style.top = `${Math.max(4, Math.min(96, topPct))}%`;
        });
      };
      deselectAllBtn.onclick = () => {
        project.selectedFrames.clear();
        project.selectedAssetIds.clear();
        if (project.isActive) ProjectManager.render();
        renderModal();
      };

      renderModal();
      overlay.style.display = "flex";
    },

    findNeutralPose(project) {
      // A completed batch analyze-immich job (project.ring.sourceType ===
      // "immich") takes the same priority here as it does in
      // renderMain()/renderStage() -- only fall back to the lighter
      // neighbor-browsing immichRing pool when no batch ring exists.
      const hasBatchRing = project.ring && project.ring.sourceType === "immich";
      const baseResults = project.task === "immich" && !hasBatchRing
        ? (project.immichRing ? project.immichRing.baseResults : null)
        : (project.ring ? project.ring.baseResults : null);
      if (!baseResults) return;

      const { kept } = project.squeezeFiltered(baseResults);
      const pool = kept.filter((r) => typeof r.yaw === "number" && typeof r.pitch === "number" && typeof r.roll === "number");

      if (!pool.length) {
        neutralPoseReadoutEl.style.display = "block";
        neutralPoseReadoutEl.textContent = "No frames with pose data in current working set.";
        return;
      }

      let best = pool[0];
      let bestScore = Math.abs(best.yaw) + Math.abs(best.pitch) + Math.abs(best.roll);
      pool.forEach((r) => {
        const score = Math.abs(r.yaw) + Math.abs(r.pitch) + Math.abs(r.roll);
        if (score < bestScore) { best = r; bestScore = score; }
      });

      neutralPoseReadoutEl.style.display = "block";
      neutralPoseReadoutEl.innerHTML = `
        Most neutral: <b>${best.filename}</b> — yaw ${best.yaw.toFixed(1)}° pitch ${best.pitch.toFixed(1)}° roll ${best.roll.toFixed(1)}° (sim ${(best.similarity * 100).toFixed(1)}%)
        <button type="button" id="ng-use-as-reference-btn" class="ng-btn ng-btn-full ng-btn-accent" style="margin-top:6px;">Use as reference &amp; re-analyze</button>
      `;
      flashHighlightNG(best.frame, best.assetId);

      const useBtn = document.getElementById("ng-use-as-reference-btn");
      useBtn.onclick = () => {
        // this readout describes the *old* anchor's neutral-pose stats,
        // invalidated by the re-analysis it's about to trigger.
        const hideReadout = () => { neutralPoseReadoutEl.style.display = "none"; neutralPoseReadoutEl.innerHTML = ""; };
        const sourceType = project.job ? project.job.sourceType : null;

        if (sourceType === "immich") {
          if (!project._lastImmichAssetIds || !project._lastImmichAssetIds.length) {
            useBtn.textContent = "Original Immich selection no longer available — re-tick assets and re-analyze";
            return;
          }
          hideReadout();
          project.startImmichAnalysis(project._lastImmichAssetIds, best.frame);
        } else if (project.task === "immich") {
          hideReadout();
          project.recenterImmich(best.assetId, best.filename);
        } else if (sourceType === "folder") {
          if (!project._lastFolderSource) {
            useBtn.textContent = "Original folder/zip no longer available — reload it first";
            return;
          }
          hideReadout();
          project.startFolderAnalysis(project._lastFolderSource, best.frame);
        } else {
          if (!project.videoFile) {
            useBtn.textContent = "Original video no longer available — reload it first";
            return;
          }
          hideReadout();
          project.startAnalysis(best.frame);
        }
      };
    },

    renderFramePreview(project) {
      if (!project.video) {
        if (project.staticPreviewFrame) {
          // folder/image-set job with a chart-click preview already shown --
          // redraw it rather than resetting to the "no video" placeholder,
          // since a full render() (selection toggles, tab switches, etc.)
          // must not silently wipe out what the person just clicked to view.
          showStaticFramePreviewNG(project, project.staticPreviewFrame);
          return;
        }
        previewHintEl.textContent = project.task === "folderzip"
          ? "No folder/zip analyzed yet for this project."
          : "No video loaded for this project yet.";
        previewCanvasEl.style.display = "none";
        previewControlsScrollEl.style.display = "none";
        return;
      }

      // Play button is a single shared element -- always resync its visual
      // state to whichever project is actually being rendered, rather than
      // trusting whatever the last togglePlay() call left it as.
      setPlayingVisual(!!project._playTimer);

      previewHintEl.textContent = "Reference frame — use ← / → to step one actual video frame";
      previewCanvasEl.style.display = "";
      previewControlsScrollEl.style.display = "";

      if (project.video.objectUrl && videoAudioEl.dataset.objectUrl !== project.video.objectUrl) {
        videoAudioEl.src = project.video.objectUrl;
        videoAudioEl.dataset.objectUrl = project.video.objectUrl;
        videoAudioEl.dataset.projectId = project.id;
        videoAudioEl.load();
      }

      // objectUrl doesn't survive a page reload (blob URLs die with the
      // page), so after a reload there's nothing left to pop out until
      // the video is re-picked -- same constraint the audio-synced
      // scrubber above already lives with.
      popoutVideoBtn.disabled = !project.video.objectUrl;
      popoutVideoBtn.title = project.video.objectUrl
        ? "Pop out the source video, with audio, before any analysis is run"
        : "Video needs to be re-loaded after a page reload before it can be popped out";

      frameCounterEl.textContent = "Frame: " + project.video.currentFrame + " / " + project.video.totalFrames;
      drawFrame(project, project.video.currentFrame);
    },
  };

// ---- exposed for appNG.js and other modules to call ----
window.ProjectManagerNG = ProjectManager;

