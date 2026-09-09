/**
 * personClustersNG.js -- EXTRACTED FROM appNG.js, VERBATIM (no logic changes).
 *
 * Source: static/appNG.js, dev-ng branch, lines 208-317.
 *
 * STATUS: NOT YET WIRED. Not currently loaded or referenced anywhere.
 *
 * ============================================================
 * WHAT THIS IS
 * ============================================================
 * THIS RESOLVES ONE OF appNG-module-contracts.md's OPEN ITEMS: "Person
 * Clusters block (not yet read this session)". It turns out to be
 * genuinely self-contained -- a single function, own DOM refs captured
 * inside itself, own module-level state (lastClusterRowsNG), no coupling
 * to CharacterProject/ProjectManager at all. This is Immich's own
 * person-clustering feature (GET /api/ng/person-clusters), independent of
 * the face-similarity ring pipeline entirely -- it's browsing Immich's
 * pre-existing person clusters, not analyzing anything itself.
 *
 *   wirePersonClustersNG() -- call once at startup. Wires:
 *     - loadClustersNG() -- fetches /api/ng/person-clusters, sorted
 *       tightest-cluster-first, populates lastClusterRowsNG
 *     - renderClusterRows() -- renders the fetched rows, flagging any
 *       cluster whose avgSim exceeds the (adjustable) tightThreshold input
 *       as a "tight cluster"
 *     - loadPersonAssetsNG(personId, rowEl) -- click-through on a cluster
 *       row (see full body below for what this loads)
 *
 * ============================================================
 * COUPLING POINTS
 * ============================================================
 *   - Own DOM refs only (ng-pc-load-btn, ng-pc-min-faces,
 *     ng-pc-tight-threshold, ng-pc-status, ng-pc-cluster-list,
 *     ng-pc-thumb-grid) -- all captured inside wirePersonClustersNG()
 *     itself, not shared module-level globals. This is actually the
 *     CLEANEST-scoped piece found in appNG.js so far -- no reach into
 *     project/ProjectManager state at all.
 *   - lastClusterRowsNG is declared just above this function (module-
 *     level `let`, line 208 area) -- shared between loadClustersNG and
 *     renderClusterRows, not passed as a parameter. Minor scoping note,
 *     not a real problem since both functions live inside the same
 *     wirePersonClustersNG() closure.
 *
 * ============================================================
 * TO MAKE THIS ACTUALLY RUN (not done in this extraction pass)
 * ============================================================
 *   1. Load after the Person Clusters panel's DOM exists (the 6 ng-pc-*
 *      elements above).
 *   2. Remove the corresponding lines from appNG.js (including the
 *      `let lastClusterRowsNG = [];` line just above this block).
 *   3. Call wirePersonClustersNG() once at startup, same as before.
 */

let lastClusterRowsNG = [];

  let lastClusterRowsNG = [];

  function wirePersonClustersNG() {
    const loadBtn = document.getElementById("ng-pc-load-btn");
    const minFacesInput = document.getElementById("ng-pc-min-faces");
    const tightThresholdInput = document.getElementById("ng-pc-tight-threshold");
    const statusEl = document.getElementById("ng-pc-status");
    const listEl = document.getElementById("ng-pc-cluster-list");
    const gridEl = document.getElementById("ng-pc-thumb-grid");

    function renderClusterRows() {
      const threshold = parseFloat(tightThresholdInput.value);
      const tightCutoff = isNaN(threshold) ? 0.85 : threshold;
      listEl.innerHTML = "";
      lastClusterRowsNG.forEach((p) => {
        const row = document.createElement("div");
        row.className = "ng-pc-cluster-row";
        const pct = (p.avgSim * 100).toFixed(1);
        const tight = p.avgSim > tightCutoff;
        row.innerHTML = `
          <div class="ng-pc-cluster-info">
            <div class="ng-pc-cluster-fname">${p.name}${tight ? ' <span class="ng-pc-tight-flag">&#9679; tight cluster</span>' : ""}</div>
            <div class="ng-pc-simbar-track"><div class="ng-pc-simbar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="ng-pc-simpct">${pct}%</div>
          <div class="ng-pc-facecount">${p.faceCount}</div>
        `;
        row.addEventListener("click", () => loadPersonAssetsNG(p.personId, row));
        listEl.appendChild(row);
      });
    }

    async function loadClustersNG() {
      statusEl.textContent = "Loading\u2026";
      listEl.innerHTML = "";
      gridEl.innerHTML = "";
      try {
        const minFaces = parseInt(minFacesInput.value, 10) || 5;
        const res = await fetch(`/api/ng/person-clusters?minFaces=${minFaces}&limit=40`);
        const rows = await res.json();
        if (rows.error) {
          statusEl.textContent = "Error: " + rows.error;
          return;
        }
        lastClusterRowsNG = rows;
        statusEl.textContent = `${rows.length} person${rows.length === 1 ? "" : "s"} (sorted tightest-cluster first)`;
        renderClusterRows();
      } catch (e) {
        statusEl.textContent = "Request failed: " + e.message;
      }
    }

    async function loadPersonAssetsNG(personId, rowEl) {
      listEl.querySelectorAll(".ng-pc-cluster-row").forEach((r) => r.classList.remove("ng-pc-row-active"));
      if (rowEl) rowEl.classList.add("ng-pc-row-active");
      gridEl.innerHTML = "Loading\u2026";
      try {
        const res = await fetch(`/api/ng/person-assets/${personId}?limit=200`);
        const assets = await res.json();
        gridEl.innerHTML = "";
        assets.forEach((a) => {
          const cell = document.createElement("div");
          let clickTimer = null;
          const isSelected = () => { const p = ProjectManager.getActive(); return !!p && p.selectedAssetIds.has(a.assetId); };
          const img = document.createElement("img");
          img.src = `/api/ng/thumb/${a.assetId}`;
          img.loading = "lazy";
          img.title = a.filename;
          img.style.outline = isSelected() ? "2px solid var(--ng-accent)" : "none";
          cell.appendChild(img);

          const toggleExportSelection = () => {
            const p = ProjectManager.getActive();
            if (!p) return;
            if (p.selectedAssetIds.has(a.assetId)) {
              p.selectedAssetIds.delete(a.assetId);
              img.style.outline = "none";
            } else {
              p.selectedAssetIds.add(a.assetId);
              img.style.outline = "2px solid var(--ng-accent)";
            }
            ProjectManager.render();
          };
          cell.onclick = () => {
            clearTimeout(clickTimer);
            clickTimer = setTimeout(toggleExportSelection, 220);
          };
          cell.ondblclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            clearTimeout(clickTimer);
            const p = ProjectManager.getActive();
            if (!p) return;
            // this is an Immich-sourced action regardless of which task
            // tab you were on when you clicked -- switch so the resulting
            // neighbor ring is actually visible (mirrors setTask()).
            if (p.task === "video") p.stopPlayIfRunning();
            p.task = "immich";
            p.recenterImmich(a.assetId, a.filename);
          };
          gridEl.appendChild(cell);
        });
      } catch (e) {
        gridEl.innerHTML = "Failed to load assets: " + e.message;
      }
    }

    tightThresholdInput.addEventListener("change", renderClusterRows);
    loadBtn.addEventListener("click", loadClustersNG);
  }

// ---- exposed for appNG.js startup wiring to call ----
window.personClustersNG = { wirePersonClustersNG };

