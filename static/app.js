const stage = document.getElementById('stage');
const CENTER_SIZE = 120;
const MIN_SIZE = 34;
const MAX_RADIUS_VW = 42;

function sizeForSim(sim) {
  const t = Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
  return MIN_SIZE + t * (CENTER_SIZE - MIN_SIZE);
}

let ringScale = 1.0;

function radiusForSim(sim) {
  const minDim = Math.min(window.innerWidth, window.innerHeight);
  const maxR = minDim * (MAX_RADIUS_VW / 100) * ringScale;
  const t = 1 - Math.max(0, Math.min(1, (sim - 0.25) / 0.75));
  return (90 * ringScale) + t * (maxR - (90 * ringScale));
}

let lastRenderArgs = null;

const originalRender = render;
render = function(centerId, data, centerThumbUrl) {
  lastRenderArgs = { centerId, data, centerThumbUrl };
  originalRender(centerId, data, centerThumbUrl);
};

document.getElementById('ring-scale-input').addEventListener('input', (e) => {
  const val = e.target.value;
  ringScale = val / 100;
  document.getElementById('ring-scale-val').textContent = `${val}%`;
  if (lastRenderArgs) {
    render(lastRenderArgs.centerId, lastRenderArgs.data, lastRenderArgs.centerThumbUrl);
  }
});

async function loadNeighbors(assetId) {
  stage.innerHTML = '<div id="loading">loading neighbors…</div>';
  const res = await fetch(`/api/neighbors?assetId=${assetId}&limit=36`);
  const data = await res.json();
  render(assetId, data);
}

function thumbUrlFor(r) {
  return r.thumbUrl || `/api/thumb/${r.assetId}`;
}

function previewUrlFor(r) {
  return r.thumbUrl || `/api/preview/${r.assetId}`;
}

let currentPoseRequestToken = 0;

function setSidebarCurrentDetail(centerId, centerResult) {
  const detailEl = document.getElementById('sidebar-current-detail');
  if (!detailEl) return;
  const requestToken = ++currentPoseRequestToken;

  const sim = centerResult && typeof centerResult.similarity === 'number'
    ? `match: ${(centerResult.similarity * 100).toFixed(1)}%`
    : '';
  detailEl.textContent = sim;

  const canShowInlinePose = centerResult &&
    centerResult.pitch !== undefined &&
    centerResult.yaw !== undefined &&
    centerResult.roll !== undefined &&
    centerResult.pitch !== null;
  if (canShowInlinePose) {
    detailEl.textContent = `${sim}${sim ? ' · ' : ''}pitch: ${centerResult.pitch.toFixed(1)} yaw: ${centerResult.yaw.toFixed(1)} roll: ${centerResult.roll.toFixed(1)}`;
    return;
  }

  if (!centerId || centerId === '__anchor__') return;

  fetch(`/api/asset-face-pose/${centerId}`)
    .then(res => res.json())
    .then(pose => {
      if (requestToken !== currentPoseRequestToken || pose.error) return;
      detailEl.textContent = `${sim}${sim ? ' · ' : ''}pitch: ${pose.pitch.toFixed(1)} yaw: ${pose.yaw.toFixed(1)} roll: ${pose.roll.toFixed(1)}`;
    })
    .catch(() => {});
}

const hoverPanel = document.getElementById('preview-hover-panel');
const hoverImg = document.getElementById('preview-hover-img');
const hoverCaption = document.getElementById('preview-hover-caption');
let hoverTimer = null;

function showHoverPreview(r) {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    hoverImg.src = previewUrlFor(r);
    const pct = (r.similarity * 100).toFixed(1);
    
    let poseText = '';
    if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
      poseText = `<br>pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
    }
    
    hoverCaption.innerHTML = `${r.filename} — ${pct}%${poseText}`;
    hoverPanel.classList.add('active');
  }, 80);
}

function hideHoverPreview() {
  clearTimeout(hoverTimer);
  hoverPanel.classList.remove('active');
}

function render(centerId, data, centerThumbUrl) {
  stage.innerHTML = '';
  const modeLabel = data.mode === 'face' ? 'FACE SIMILARITY' : 'CLIP (WHOLE-IMAGE) SIMILARITY';
  document.getElementById('hud-mode').textContent = modeLabel;

  const results = data.results.filter(r => r.assetId !== centerId);
  const centerResult = data.results.find(r => r.assetId === centerId) || data.results[0];
  document.getElementById('hud-filename').textContent = centerResult ? centerResult.filename : '';

  const curImg = document.getElementById('sidebar-current-img');
  const curFname = document.getElementById('sidebar-current-fname');
  const curMode = document.getElementById('sidebar-current-mode');
  if (curImg) curImg.src = centerThumbUrl || ('/api/thumb/' + centerId);
  if (curFname) curFname.textContent = centerResult ? centerResult.filename : '—';
  if (curMode) curMode.textContent = modeLabel;
  setSidebarCurrentDetail(centerId, centerResult);

  [0.9, 0.7, 0.5, 0.35].forEach(band => {
    const r = radiusForSim(band);
    const ring = document.createElement('div');
    ring.className = 'ring';
    ring.style.width = (r * 2) + 'px';
    ring.style.height = (r * 2) + 'px';
    stage.appendChild(ring);
  });

  const center = document.createElement('div');
  center.className = 'node center';
  center.style.width = CENTER_SIZE + 'px';
  center.style.height = CENTER_SIZE + 'px';
  center.dataset.baseX = 0;
  center.dataset.baseY = 0;
  center.style.transform = 'translate(-50%, -50%)';
  center.innerHTML = `<img src="${centerThumbUrl || ('/api/thumb/' + centerId)}" loading="lazy">`;
  
  center.addEventListener('mouseenter', () => showHoverPreview({
    assetId: centerId,
    filename: 'Reference',
    similarity: 1,
    fromImmich: false
  }));
  center.addEventListener('mouseleave', hideHoverPreview);

  stage.appendChild(center);

  const bandCount = 8;
  const bands = Array.from({length: bandCount}, () => []);
  results.forEach(r => {
    const t = Math.max(0, Math.min(1, (r.similarity - 0.25) / 0.75));
    const bandIdx = Math.min(bandCount - 1, Math.floor((1 - t) * bandCount));
    bands[bandIdx].push(r);
  });

  bands.forEach((bandResults, bandIdx) => {
    if (bandResults.length === 0) return;
    const t = 1 - (bandIdx / (bandCount - 1));
    const avgSim = 0.25 + t * 0.75;
    const radius = radiusForSim(avgSim);
    const size = sizeForSim(avgSim);

    const angleOffset = bandIdx * 0.6;
    bandResults.forEach((r, i) => {
      const angle = angleOffset + (i / bandResults.length) * 2 * Math.PI;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;

      const node = document.createElement('div');
      node.className = 'node';
      node.style.width = size + 'px';
      node.style.height = size + 'px';
      node.style.left = `calc(50% + ${x}px)`;
      node.style.top = `calc(50% + ${y}px)`;
      node.dataset.baseX = x;
      node.dataset.baseY = y;
      node.style.transform = 'translate(-50%, -50%)';
      if (r.fromImmich) {
        node.style.border = '2px solid #d4a544';
        node.style.boxShadow = '0 0 8px rgba(212,165,68,0.5)';
      }
      node.title = `${r.filename} — ${(r.similarity*100).toFixed(1)}%${r.fromImmich ? ' (from Immich library)' : ''}`;
      node.dataset.filename = r.filename;
      if (r.assetId) node.dataset.assetId = r.assetId;
      if (r.frame !== undefined) node.dataset.frame = r.frame;
      node.innerHTML = `<img src="${thumbUrlFor(r)}" loading="lazy">`;
      const isSelected = r.assetId ? selectedAssetIds.has(r.assetId) : (r.frame !== undefined && selectedFrames.has(r.frame));
      if (isSelected) node.classList.add('export-selected');
      // single click (debounced) = recenter the ring on this node; double
      // click = add/remove from the export queue. Same debounce pattern as
      // the person-clusters grid, so a dblclick doesn't also fire recenter.
      let nodeClickTimer = null;
      const toggleNodeExportSelection = () => {
        if (r.assetId) {
          if (selectedAssetIds.has(r.assetId)) { selectedAssetIds.delete(r.assetId); }
          else { selectedAssetIds.add(r.assetId); }
          updateImmichSelectionBar();
        } else if (r.frame !== undefined) {
          if (selectedFrames.has(r.frame)) { selectedFrames.delete(r.frame); }
          else { selectedFrames.add(r.frame); }
          updateSaveSelectedButton();
        }
        node.classList.toggle('export-selected');
      };
      node.onclick = () => {
        if (!r.assetId || r.fromImmich) return;
        clearTimeout(nodeClickTimer);
        nodeClickTimer = setTimeout(() => loadNeighbors(r.assetId), 220);
      };
      node.ondblclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearTimeout(nodeClickTimer);
        toggleNodeExportSelection();
      };
      node.addEventListener('mouseenter', () => showHoverPreview(r));
      node.addEventListener('mouseleave', hideHoverPreview);
      stage.appendChild(node);
    });
  });

  const framesSection = document.getElementById('frames-section');
  const immichSection = document.getElementById('immich-section');
  const listBodyFrames = document.getElementById('list-body-frames');
  const listBodyImmich = document.getElementById('list-body-immich');
  listBodyFrames.innerHTML = '';
  listBodyImmich.innerHTML = '';

  // map sourceFrame -> assetId (and back) for cross-checked pairs, so each
  // side of a linked pair can show a badge pointing at the other - visible
  // instead of silently reconciled at export time
  const frameToAsset = new Map();
  const assetToFrame = new Map();
  extraImmichNodes.forEach(n => {
    if (n.sourceFrame !== undefined) {
      frameToAsset.set(n.sourceFrame, n.assetId);
      assetToFrame.set(n.assetId, n.sourceFrame);
    }
  });

  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'list-row';
    row.dataset.assetId = r.assetId;
    const pct = (r.similarity * 100).toFixed(1);
    const isFrameRow = r.frame !== undefined;

    let checkboxHtml = '';
    if (isFrameRow) {
      checkboxHtml = `<input type="checkbox" class="frame-select-cb" data-frame="${r.frame}" ${selectedFrames.has(r.frame) ? 'checked' : ''} style="margin-right:6px;flex-shrink:0;">`;
    } else if (r.assetId) {
      checkboxHtml = `<input type="checkbox" class="asset-select-cb" data-asset-id="${r.assetId}" ${selectedAssetIds.has(r.assetId) ? 'checked' : ''} style="margin-right:6px;flex-shrink:0;">`;
    }

    let poseHtml = '';
    if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
      poseHtml = `<div style="font-size:10px;color:var(--dim);margin-top:2px;">pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}</div>`;
    }

    let linkedBadgeHtml = '';
    if (isFrameRow && frameToAsset.has(r.frame)) {
      linkedBadgeHtml = `<span class="linked-badge" data-jump-asset="${frameToAsset.get(r.frame)}" title="Same photo also matched in Immich - selecting both will export a duplicate">linked → Immich</span>`;
    } else if (!isFrameRow && r.assetId && assetToFrame.has(r.assetId)) {
      linkedBadgeHtml = `<span class="linked-badge" data-jump-frame="${assetToFrame.get(r.assetId)}" title="Same photo also present as a local frame - selecting both will export a duplicate">linked → frame ${assetToFrame.get(r.assetId)}</span>`;
    }

    row.innerHTML = `
      ${checkboxHtml}
      <img src="${thumbUrlFor(r)}" loading="lazy" style="${r.fromImmich ? 'border:1.5px solid #d4a544;' : ''}">
      <div class="info">
        <div class="fname">${r.filename}${r.fromImmich ? ' <span style="color:#d4a544;font-size:9px;">● Immich</span>' : ''}${linkedBadgeHtml}</div>
        <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
        ${poseHtml}
      </div>
      <div class="simpct">${pct}%</div>
    `;
    row.onclick = (e) => {
      if (e.target.classList.contains('frame-select-cb') || e.target.classList.contains('asset-select-cb')) return;
      if (e.target.classList.contains('linked-badge')) {
        e.stopPropagation();
        const jumpAsset = e.target.dataset.jumpAsset;
        const jumpFrame = e.target.dataset.jumpFrame;
        const targetRow = jumpAsset
          ? listBodyImmich.querySelector(`.list-row[data-asset-id="${jumpAsset}"]`)
          : listBodyFrames.querySelector(`.frame-select-cb[data-frame="${jumpFrame}"]`)?.closest('.list-row');
        if (targetRow) {
          targetRow.scrollIntoView({ block: 'center', behavior: 'smooth' });
          targetRow.classList.add('flash-highlight');
          setTimeout(() => targetRow.classList.remove('flash-highlight'), 1000);
        }
        return;
      }
      if (r.assetId && !r.fromImmich) loadNeighbors(r.assetId);
    };
    const cb = row.querySelector('.frame-select-cb');
    if (cb) {
      cb.addEventListener('change', () => {
        const frame = parseInt(cb.dataset.frame, 10);
        if (cb.checked) selectedFrames.add(frame);
        else selectedFrames.delete(frame);
        syncSelectionVisuals();
        updateSaveSelectedButton();
      });
    }
    const acb = row.querySelector('.asset-select-cb');
    if (acb) {
      acb.addEventListener('change', () => {
        if (acb.checked) selectedAssetIds.add(acb.dataset.assetId);
        else selectedAssetIds.delete(acb.dataset.assetId);
        syncSelectionVisuals();
        updateImmichSelectionBar();
      });
    }
    row.addEventListener('mouseenter', () => showHoverPreview(r));
    row.addEventListener('mouseleave', hideHoverPreview);
    (isFrameRow ? listBodyFrames : listBodyImmich).appendChild(row);
  });

  const frameRowCount = listBodyFrames.children.length;
  const immichRowCount = listBodyImmich.children.length;
  framesSection.style.display = frameRowCount ? '' : 'none';
  immichSection.style.display = immichRowCount ? '' : 'none';
  document.getElementById('frames-section-count').textContent = frameRowCount ? `(${frameRowCount})` : '';
  document.getElementById('immich-section-count').textContent = immichRowCount ? `(${immichRowCount})` : '';
}

const selectedFrames = new Set();

// ---- selection persistence: pure in-memory state gets wiped by an
// accidental refresh, which is a real cost once you're deliberately
// hand-picking a curated set rather than just poking at the tool. Mirror
// both selection Sets to localStorage on every change and restore on load.
const SELECTION_STORAGE_KEY = 'immichring_selection_v1';

function saveSelectionToStorage() {
  try {
    localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify({
      frames: Array.from(selectedFrames),
      assetIds: Array.from(selectedAssetIds),
      savedAt: Date.now(),
    }));
  } catch (e) {
    console.warn('Could not persist selection:', e);
  }
}

function restoreSelectionFromStorage() {
  try {
    const raw = localStorage.getItem(SELECTION_STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    (data.frames || []).forEach(f => selectedFrames.add(f));
    (data.assetIds || []).forEach(id => selectedAssetIds.add(id));
    if (selectedFrames.size || selectedAssetIds.size) {
      updateSaveSelectedButton();
      updateImmichSelectionBar();
    }
  } catch (e) {
    console.warn('Could not restore selection:', e);
  }
}

// ---- shared export settings (used by all three export-to-disk buttons) ----
function getExportParams() {
  const upscaleOn = document.getElementById('export-upscale').checked;
  return {
    width: parseInt(document.getElementById('export-width').value, 10) || 512,
    height: parseInt(document.getElementById('export-height').value, 10) || 512,
    cropMode: document.getElementById('export-crop-mode').value,
    minFacePx: parseFloat(document.getElementById('export-min-face').value) || 0,
    margin: parseFloat(document.getElementById('export-margin').value) || 2.2,
    interp: document.getElementById('export-interp').value,
    upscale: upscaleOn,
    maxUpscale: upscaleOn ? (parseFloat(document.getElementById('export-max-upscale').value) || null) : null,
    padMode: upscaleOn ? document.getElementById('export-pad-mode').value : 'none',
  };
}

(function () {
  const cropModeSel = document.getElementById('export-crop-mode');
  const marginRow = document.getElementById('export-margin-row');
  const maxUpscaleRow = document.getElementById('export-max-upscale-row');
  const padRow = document.getElementById('export-pad-row');
  const upscaleCb = document.getElementById('export-upscale');
  function syncRows() {
    const isFace = cropModeSel.value === 'face';
    const showCap = isFace && upscaleCb.checked;
    marginRow.style.display = isFace ? 'flex' : 'none';
    maxUpscaleRow.style.display = showCap ? 'flex' : 'none';
    padRow.style.display = showCap ? 'flex' : 'none';
  }
  cropModeSel.addEventListener('change', syncRows);
  upscaleCb.addEventListener('change', syncRows);
  syncRows();

  document.querySelectorAll('.export-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('export-width').value = btn.dataset.w;
      document.getElementById('export-height').value = btn.dataset.h;
    });
  });
})();

function updateSaveSelectedButton() {
  const btn = document.getElementById('save-selected-btn');
  if (!btn) return;
  const n = selectedFrames.size;
  const a = selectedAssetIds.size;
  const frameText = `${n} selected frame${n === 1 ? '' : 's'}`;
  const assetText = a ? ` + ${a} Immich image${a === 1 ? '' : 's'}` : '';
  btn.textContent = `Save ${frameText}${assetText} to disk`;
  btn.disabled = n === 0 && a === 0;
  btn.style.opacity = (n === 0 && a === 0) ? '0.5' : '1';
  refreshSelectionModalIfOpen();
  saveSelectionToStorage();
}

const selectedAssetIds = new Set();

function updateImmichSelectionBar() {
  const bar = document.getElementById('immich-selection-bar');
  const countEl = document.getElementById('immich-selection-count');
  if (!bar || !countEl) return;
  const n = selectedAssetIds.size;
  countEl.textContent = n;
  bar.style.display = n > 0 ? 'block' : 'none';
  updateSaveSelectedButton();
  refreshSelectionModalIfOpen();
  saveSelectionToStorage();
}

// ---- select all / deselect all: scoped per-section (local frames vs
// Immich matches) so the two categories can't get tangled together the
// way they could when they shared one flat list. Still respects whatever
// squeeze/similarity filter is currently narrowing the visible set. ----
function syncSelectionVisuals() {
  document.querySelectorAll('.node[data-asset-id], .node[data-frame]').forEach(node => {
    const isSelected = node.dataset.assetId
      ? selectedAssetIds.has(node.dataset.assetId)
      : (node.dataset.frame !== undefined && selectedFrames.has(parseInt(node.dataset.frame, 10)));
    node.classList.toggle('export-selected', !!isSelected);
  });
  document.querySelectorAll('.pose-list-item[data-asset-id], .pose-list-item[data-frame]').forEach(item => {
    const isSelected = item.dataset.assetId
      ? selectedAssetIds.has(item.dataset.assetId)
      : (item.dataset.frame !== undefined && selectedFrames.has(parseInt(item.dataset.frame, 10)));
    item.classList.toggle('selected', !!isSelected);
  });
}

function selectAllInSection(target) {
  if (target === 'frames') {
    document.querySelectorAll('#list-body-frames .frame-select-cb').forEach(cb => {
      selectedFrames.add(parseInt(cb.dataset.frame, 10));
      cb.checked = true;
    });
    updateSaveSelectedButton();
  } else {
    document.querySelectorAll('#list-body-immich .asset-select-cb').forEach(cb => {
      selectedAssetIds.add(cb.dataset.assetId);
      cb.checked = true;
    });
    updateImmichSelectionBar();
  }
  syncSelectionVisuals();
}

function deselectAllInSection(target) {
  if (target === 'frames') {
    document.querySelectorAll('#list-body-frames .frame-select-cb').forEach(cb => {
      selectedFrames.delete(parseInt(cb.dataset.frame, 10));
      cb.checked = false;
    });
    updateSaveSelectedButton();
  } else {
    document.querySelectorAll('#list-body-immich .asset-select-cb').forEach(cb => {
      selectedAssetIds.delete(cb.dataset.assetId);
      cb.checked = false;
    });
    updateImmichSelectionBar();
  }
  syncSelectionVisuals();
}

document.querySelectorAll('.section-select-all-btn').forEach(btn => {
  btn.addEventListener('click', () => selectAllInSection(btn.dataset.target));
});
document.querySelectorAll('.section-deselect-all-btn').forEach(btn => {
  btn.addEventListener('click', () => deselectAllInSection(btn.dataset.target));
});

// ---- selection review modal: lets you see everything queued for export
// across both Immich assets and local video frames in one place, and pull
// items back out by double-clicking them (same debounce-free dblclick
// convention used elsewhere for a deliberate "remove" action) ----
function findFrameResult(frame) {
  if (!lastVideoRingState) return null;
  return lastVideoRingState.baseResults.find(r => r.frame === frame)
      || extraImmichNodes.find(r => r.frame === frame)
      || null;
}

function openSelectionModal() {
  document.getElementById('selection-modal-overlay').style.display = 'flex';
  renderSelectionModal();
}

function closeSelectionModal() {
  document.getElementById('selection-modal-overlay').style.display = 'none';
}

function refreshSelectionModalIfOpen() {
  const overlay = document.getElementById('selection-modal-overlay');
  if (overlay && overlay.style.display === 'flex') renderSelectionModal();
}

function renderSelectionModal() {
  const grid = document.getElementById('selection-modal-grid');
  const countEl = document.getElementById('selection-modal-count');
  grid.innerHTML = '';

  const assetItems = Array.from(selectedAssetIds).map(id => {
    const known = [...(lastVideoRingState ? lastVideoRingState.baseResults : []), ...extraImmichNodes]
      .find(r => r.assetId === id);
    return { kind: 'asset', assetId: id, filename: known ? known.filename : id, thumb: `/api/thumb/${id}` };
  });
  const frameItems = Array.from(selectedFrames).map(frame => {
    const r = findFrameResult(frame);
    return { kind: 'frame', frame, filename: r ? r.filename : `frame ${frame}`, thumb: r ? thumbUrlFor(r) : '' };
  });
  const items = [...assetItems, ...frameItems];
  countEl.textContent = items.length;

  if (!items.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;color:var(--dim);font-size:11px;text-align:center;padding:20px;">Nothing selected yet.</div>';
    return;
  }

  items.forEach(it => {
    const cell = document.createElement('div');
    cell.style.textAlign = 'center';
    cell.style.cursor = 'pointer';
    cell.innerHTML = `
      <img src="${it.thumb}" loading="lazy" title="Double-click to remove"
           style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;border:2px solid var(--accent);display:block;">
      <div style="font-size:9px;color:var(--dim);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${it.filename}</div>
    `;
    cell.ondblclick = () => {
      if (it.kind === 'asset') {
        selectedAssetIds.delete(it.assetId);
        updateImmichSelectionBar();
      } else {
        selectedFrames.delete(it.frame);
        updateSaveSelectedButton();
      }
      // reflect removal in whichever grid/ring/strip is currently rendered
      document.querySelectorAll('.pose-list-item.selected, .node.export-selected').forEach(el => {
        if (el.dataset.filename === it.filename) el.classList.remove('selected', 'export-selected');
      });
      renderSelectionModal();
    };
    grid.appendChild(cell);
  });
}

document.getElementById('selection-modal-close').addEventListener('click', closeSelectionModal);
document.getElementById('selection-modal-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'selection-modal-overlay') closeSelectionModal();
});
document.getElementById('immich-view-selected-btn').addEventListener('click', openSelectionModal);

document.getElementById('immich-export-selected-btn').addEventListener('click', async () => {
  const btn = document.getElementById('immich-export-selected-btn');
  const assetIds = Array.from(selectedAssetIds);
  if (!assetIds.length) return;
  const prevText = btn.textContent;
  btn.textContent = 'Exporting…';
  btn.disabled = true;
  try {
    const res = await fetch('/api/export-immich-assets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assetIds, ...getExportParams() }),
    });
    const result = await res.json();
    if (result.error) {
      btn.textContent = 'Error: ' + result.error;
    } else {
      const skipped = result.errors ? result.errors.filter(e => e.includes(': skipped (')).length : 0;
      btn.textContent = `Saved ${result.exported} of ${assetIds.length} → ${result.path}` + (skipped ? ` (${skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded to avoid upscaling)` : '');
      if (result.errors && result.errors.length) {
        console.warn('Some exports failed:', result.errors);
      }
    }
  } catch (e) {
    btn.textContent = 'Request failed';
  }
  setTimeout(() => { btn.textContent = prevText; btn.disabled = false; }, 4000);
});

// ---- person clusters panel: standalone Immich browse, no video job needed ----
(function () {
  const loadBtn = document.getElementById('pc-load-btn');
  const minFacesInput = document.getElementById('pc-min-faces');
  const tightThresholdInput = document.getElementById('pc-tight-threshold');
  const statusEl = document.getElementById('pc-status');
  const listEl = document.getElementById('pc-cluster-list');
  const gridEl = document.getElementById('pc-thumb-grid');

  let lastRows = [];

  function renderRows() {
    const threshold = parseFloat(tightThresholdInput.value);
    const tightCutoff = isNaN(threshold) ? 0.85 : threshold;
    listEl.innerHTML = '';
    lastRows.forEach(p => {
      const row = document.createElement('div');
      row.className = 'list-row';
      row.style.cursor = 'pointer';
      const pct = (p.avgSim * 100).toFixed(1);
      const tight = p.avgSim > tightCutoff;
      row.innerHTML = `
        <div class="info" style="flex:1;">
          <div class="fname">${p.name}${tight ? ' <span style="color:#d4a544;font-size:9px;">● tight cluster</span>' : ''}</div>
          <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
        </div>
        <div class="simpct">${pct}%</div>
        <div style="font-size:10px;color:var(--dim);margin-left:6px;">${p.faceCount}</div>
      `;
      row.onclick = () => loadPersonAssets(p.personId, row);
      listEl.appendChild(row);
    });
  }

  async function loadClusters() {
    statusEl.textContent = 'Loading…';
    listEl.innerHTML = '';
    gridEl.innerHTML = '';
    try {
      const minFaces = parseInt(minFacesInput.value, 10) || 5;
      const res = await fetch(`/api/person-clusters?minFaces=${minFaces}&limit=40`);
      const rows = await res.json();
      if (rows.error) {
        statusEl.textContent = 'Error: ' + rows.error;
        return;
      }
      lastRows = rows;
      statusEl.textContent = `${rows.length} person${rows.length === 1 ? '' : 's'} (sorted tightest-cluster first)`;
      renderRows();
    } catch (e) {
      statusEl.textContent = 'Request failed: ' + e;
    }
  }

  tightThresholdInput.addEventListener('change', renderRows);

  async function loadPersonAssets(personId, rowEl) {
    document.querySelectorAll('#pc-cluster-list .list-row').forEach(r => r.style.background = '');
    if (rowEl) rowEl.style.background = 'rgba(212,165,68,0.12)';
    gridEl.innerHTML = 'Loading…';
    try {
      const res = await fetch(`/api/person-assets/${personId}?limit=200`);
      const assets = await res.json();
      gridEl.innerHTML = '';
      assets.forEach(a => {
        const cell = document.createElement('div');
        cell.style.position = 'relative';
        cell.style.cursor = 'pointer';
        let clickTimer = null;
        const checked = selectedAssetIds.has(a.assetId);
        cell.innerHTML = `
          <img src="/api/thumb/${a.assetId}" loading="lazy" title="${a.filename}"
               style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px;${checked ? 'outline:2px solid #d4a544;' : ''}">
        `;
        const toggleExportSelection = () => {
          if (selectedAssetIds.has(a.assetId)) {
            selectedAssetIds.delete(a.assetId);
            cell.querySelector('img').style.outline = 'none';
          } else {
            selectedAssetIds.add(a.assetId);
            cell.querySelector('img').style.outline = '2px solid #d4a544';
          }
          updateImmichSelectionBar();
        };
        cell.onclick = () => {
          clearTimeout(clickTimer);
          clickTimer = setTimeout(toggleExportSelection, 220);
        };
        cell.ondblclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          clearTimeout(clickTimer);
          const input = document.getElementById('search-input');
          const results = document.getElementById('search-results');
          if (input) input.value = a.filename;
          if (results) results.classList.remove('open');
          loadNeighbors(a.assetId);
        };
        gridEl.appendChild(cell);
      });
    } catch (e) {
      gridEl.innerHTML = 'Failed to load assets: ' + e;
    }
  }

  loadBtn.addEventListener('click', loadClusters);
})();

document.getElementById('immich-clear-selected-btn').addEventListener('click', () => {
  selectedAssetIds.clear();
  updateImmichSelectionBar();
  if (lastVideoRingState) renderVideoRing();
  else {
    document.querySelectorAll('.asset-select-cb').forEach(cb => { cb.checked = false; });
  }
});

async function runImmichAnalysis(assetIds, refIndex, statusEl) {
  if (!assetIds.length) return;
  try {
    const res = await fetch('/api/analyze-immich', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        assetIds,
        simThreshold: document.getElementById('sim-threshold').value,
        blurThreshold: document.getElementById('blur-threshold').value,
        refIndex,
      }),
    });
    const data = await res.json();
    if (data.error) {
      alert('Analyze failed: ' + data.error);
      return;
    }
    if (data.fetchErrors && data.fetchErrors.length) {
      console.warn('Some Immich assets failed to fetch for analysis:', data.fetchErrors);
    }
    selectedFrames.clear();
    lastImmichAnalysisAssetIds = assetIds;
    pollAnalysis(data.jobId, `${data.imageCount} Immich images`, refIndex, statusEl || folderStatus || videoStatus, 'immich');
  } catch (e) {
    alert('Analyze failed: ' + e);
  }
}

function reanalyzeImmichSelection(assetIds, refIndex) {
  runImmichAnalysis(assetIds, refIndex);
}

document.getElementById('immich-analyze-selected-btn').addEventListener('click', async () => {
  const btn = document.getElementById('immich-analyze-selected-btn');
  const assetIds = Array.from(selectedAssetIds);
  if (!assetIds.length) return;
  const prevText = btn.textContent;
  btn.textContent = `Fetching ${assetIds.length} from Immich…`;
  btn.disabled = true;
  const refIndex = parseInt(document.getElementById('folder-ref-index')?.value, 10) || 1;
  await runImmichAnalysis(assetIds, refIndex);
  btn.textContent = prevText;
  btn.disabled = false;
});

async function init() {
  restoreSelectionFromStorage();
  const params = new URLSearchParams(window.location.search);
  let assetId = params.get('assetId');
  if (!assetId) {
    const res = await fetch('/api/random-face');
    const data = await res.json();
    assetId = data.assetId;
  }
  loadNeighbors(assetId);
}

const FISHEYE_RADIUS = 160;
const MAX_SCALE = 2.0;
const MAX_PUSH = 46;

let rafPending = false;
let lastMouse = null;

function applyFisheye(mx, my) {
  const rect = stage.getBoundingClientRect();
  const localX = mx - rect.left;
  const localY = my - rect.top;
  const centerX = rect.width / 2;
  const centerY = rect.height / 2;

  document.querySelectorAll('.node').forEach(node => {
    const baseX = parseFloat(node.dataset.baseX || 0);
    const baseY = parseFloat(node.dataset.baseY || 0);
    const nodeScreenX = centerX + baseX;
    const nodeScreenY = centerY + baseY;

    const dx = nodeScreenX - localX;
    const dy = nodeScreenY - localY;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < FISHEYE_RADIUS) {
      const t = 1 - (dist / FISHEYE_RADIUS);
      const eased = t * t * (3 - 2 * t);
      const scale = 1 + eased * (MAX_SCALE - 1);
      const push = eased * MAX_PUSH;

      const angle = Math.atan2(baseY, baseX);
      const pushX = (baseX === 0 && baseY === 0) ? 0 : Math.cos(angle) * push;
      const pushY = (baseX === 0 && baseY === 0) ? 0 : Math.sin(angle) * push;
      node.style.transform = `translate(-50%, -50%) translate(${pushX}px, ${pushY}px) scale(${scale})`;
      node.style.zIndex = Math.round(10 + eased * 50);
    } else {
      node.style.transform = 'translate(-50%, -50%)';
      node.style.zIndex = 1;
    }
  });
  rafPending = false;
}

stage.addEventListener('mousemove', (e) => {
  lastMouse = [e.clientX, e.clientY];
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(() => applyFisheye(...lastMouse));
  }
});

stage.addEventListener('mouseleave', () => {
  document.querySelectorAll('.node').forEach(node => {
    node.style.transform = 'translate(-50%, -50%)';
    node.style.zIndex = 1;
  });
});

// ---- pose list lens effect: dock-style magnify along the horizontal strip,
// composes with each item's pitch-based translateY so the wave and the
// hover-zoom don't fight each other ----
const POSE_LENS_RADIUS = 140;
const POSE_LENS_MAX_SCALE = 1.6;
let poseLensRafPending = false;
let poseLensLastMouse = null;

function applyPoseLens(mx, my) {
  const listEl = document.getElementById('pose-list-view');
  const items = listEl.querySelectorAll('.pose-list-item');
  items.forEach(item => {
    const rect = item.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dist = Math.hypot(mx - cx, my - cy);
    const baseY = parseFloat(item.dataset.pitchY || 0);
    if (dist < POSE_LENS_RADIUS) {
      const t = 1 - (dist / POSE_LENS_RADIUS);
      const eased = t * t * (3 - 2 * t);
      const scale = 1 + eased * (POSE_LENS_MAX_SCALE - 1);
      item.style.transform = `translateY(${baseY}px) scale(${scale})`;
      item.style.zIndex = Math.round(10 + eased * 50);
    } else {
      item.style.transform = `translateY(${baseY}px)`;
      item.style.zIndex = 1;
    }
  });
  poseLensRafPending = false;
}

const poseListView = document.getElementById('pose-list-view');
poseListView.addEventListener('mousemove', (e) => {
  poseLensLastMouse = [e.clientX, e.clientY];
  if (!poseLensRafPending) {
    poseLensRafPending = true;
    requestAnimationFrame(() => applyPoseLens(...poseLensLastMouse));
  }
});
poseListView.addEventListener('mouseleave', () => {
  poseListView.querySelectorAll('.pose-list-item').forEach(item => {
    const baseY = parseFloat(item.dataset.pitchY || 0);
    item.style.transform = `translateY(${baseY}px)`;
    item.style.zIndex = 1;
  });
});

init();

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimer = null;

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (q.length < 2) {
    searchResults.classList.remove('open');
    return;
  }
  searchTimer = setTimeout(async () => {
    const res = await fetch(`/api/find-by-filename?name=${encodeURIComponent(q)}`);
    const matches = await res.json();
    searchResults.innerHTML = '';
    if (matches.length === 0) {
      searchResults.innerHTML = '<div class="search-result-row" style="color:var(--dim)">no matches</div>';
    } else {
      matches.forEach(m => {
        const row = document.createElement('div');
        row.className = 'search-result-row';
        row.innerHTML = `<img src="/api/thumb/${m.assetId}" loading="lazy"><span>${m.filename}</span>`;
        row.onclick = () => {
          searchResults.classList.remove('open');
          searchInput.value = m.filename;
          loadNeighbors(m.assetId);
        };
        searchResults.appendChild(row);
      });
    }
    searchResults.classList.add('open');
  }, 250);
});

document.addEventListener('click', (e) => {
  if (!document.getElementById('search-box').contains(e.target)) {
    searchResults.classList.remove('open');
  }
});

// ---- left panel: accordion sections, drag-resize, collapse-all ----
(function () {
  const panel = document.getElementById('left-panel');
  const handle = document.getElementById('left-resize-handle');
  const splitter = document.getElementById('left-splitter');
  const controlsPane = document.getElementById('left-controls-pane');
  const collapseAllBtn = document.getElementById('left-panel-collapse-all');

  // restore persisted width / collapsed states
  const savedWidth = localStorage.getItem('ringviz.leftPanelWidth');
  if (savedWidth) panel.style.width = savedWidth + 'px';
  const savedCollapsedAll = localStorage.getItem('ringviz.leftPanelCollapsedAll') === '1';
  if (savedCollapsedAll) {
    panel.classList.add('collapsed-all');
    collapseAllBtn.textContent = '▸';
  }
  const savedControlsPct = parseFloat(localStorage.getItem('ringviz.leftControlsPct'));
  if (!Number.isNaN(savedControlsPct)) {
    controlsPane.style.flexBasis = Math.max(24, Math.min(76, savedControlsPct)) + '%';
  }
  document.querySelectorAll('.panel-section').forEach(sec => {
    const key = 'ringviz.section.' + sec.dataset.section;
    const saved = localStorage.getItem(key);
    if (saved === '1') sec.classList.add('expanded');
    if (saved === '0') sec.classList.remove('expanded');
    const header = sec.querySelector('.panel-section-header');
    header.addEventListener('click', () => {
      sec.classList.toggle('expanded');
      localStorage.setItem(key, sec.classList.contains('expanded') ? '1' : '0');
    });
  });

  collapseAllBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const collapsed = panel.classList.toggle('collapsed-all');
    collapseAllBtn.textContent = collapsed ? '▸' : '◂';
    localStorage.setItem('ringviz.leftPanelCollapsedAll', collapsed ? '1' : '0');
  });

  let splitDragging = false;
  splitter.addEventListener('mousedown', (e) => {
    if (panel.classList.contains('collapsed-all')) return;
    splitDragging = true;
    splitter.classList.add('dragging');
    document.body.style.cursor = 'ns-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!splitDragging) return;
    const rect = document.getElementById('left-panel-content').getBoundingClientRect();
    const pct = ((e.clientY - rect.top) / rect.height) * 100;
    controlsPane.style.flexBasis = Math.max(24, Math.min(76, pct)) + '%';
  });
  document.addEventListener('mouseup', () => {
    if (!splitDragging) return;
    splitDragging = false;
    splitter.classList.remove('dragging');
    document.body.style.cursor = '';
    const rect = document.getElementById('left-panel-content').getBoundingClientRect();
    const controlsRect = controlsPane.getBoundingClientRect();
    localStorage.setItem('ringviz.leftControlsPct', Math.round((controlsRect.height / rect.height) * 100));
  });

  let dragging = false;
  let startX = 0;
  let startWidth = 0;
  handle.addEventListener('mousedown', (e) => {
    if (panel.classList.contains('collapsed-all')) return;
    dragging = true;
    handle.classList.add('dragging');
    startX = e.clientX;
    startWidth = panel.getBoundingClientRect().width;
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const newWidth = Math.max(240, Math.min(720, startWidth + (e.clientX - startX)));
    panel.style.width = newWidth + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    localStorage.setItem('ringviz.leftPanelWidth', Math.round(panel.getBoundingClientRect().width));
  });
})();

// ---- right sidebar: optionally hide ranked/Immich match list ----
(function () {
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('toggle-list-btn');
  if (!sidebar || !toggle) return;

  function applyHidden(hidden) {
    sidebar.classList.toggle('list-hidden', hidden);
    toggle.textContent = hidden ? 'Show' : 'Hide';
    toggle.title = hidden ? 'Show ranked match list' : 'Hide ranked match list';
  }

  applyHidden(localStorage.getItem('ringviz.listHidden') === '1');
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    const hidden = !sidebar.classList.contains('list-hidden');
    applyHidden(hidden);
    localStorage.setItem('ringviz.listHidden', hidden ? '1' : '0');
  });
})();

// ---- right sidebar: drag-resize ----
(function () {
  const sidebar = document.getElementById('sidebar');
  const stage = document.getElementById('stage');
  if (!sidebar || !stage) return;

  const handle = document.createElement('div');
  handle.id = 'right-resize-handle';
  handle.style.cssText = 'position:fixed;top:0;bottom:0;width:8px;cursor:ew-resize;z-index:11;';
  document.body.appendChild(handle);

  function applyWidth(w) {
    sidebar.style.width = w + 'px';
    stage.style.right = w + 'px';
    handle.style.right = (w - 4) + 'px';
  }

  const savedSidebarWidth = parseInt(localStorage.getItem('ringviz.sidebarWidth'), 10);
  applyWidth(savedSidebarWidth && !isNaN(savedSidebarWidth) ? savedSidebarWidth : 340);

  handle.addEventListener('mouseenter', () => handle.style.background = 'rgba(124,196,255,0.25)');
  handle.addEventListener('mouseleave', () => { if (!dragging) handle.style.background = ''; });

  let dragging = false;
  let startX = 0;
  let startWidth = 0;
  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    startX = e.clientX;
    startWidth = sidebar.getBoundingClientRect().width;
    document.body.style.cursor = 'ew-resize';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const newWidth = Math.max(260, Math.min(720, startWidth - (e.clientX - startX)));
    applyWidth(newWidth);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    handle.style.background = '';
    localStorage.setItem('ringviz.sidebarWidth', Math.round(sidebar.getBoundingClientRect().width));
  });
})();

const dropZone = document.getElementById('video-drop-zone');
const fileInput = document.getElementById('video-file-input');
const previewContainer = document.getElementById('preview-container');
const previewCanvas = document.getElementById('preview-canvas');
const hiddenVideo = document.getElementById('hidden-video');
const frameCounter = document.getElementById('frame-counter');
const videoStatus = document.getElementById('video-status');

let currentVideoFile = null;
let currentPreviewId = null;
let currentFrameIdx = 1;
let totalFrames = 1;
let fps = 0;

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFileSelect(e.target.files[0]);
});

window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    handleFileSelect(file);
  }
});

async function handleFileSelect(file) {
  currentVideoFile = file;
  if (referenceAudio.src) URL.revokeObjectURL(referenceAudio.src);
  referenceAudio.src = URL.createObjectURL(file);
  referenceAudio.load();
  currentPreviewId = null;
  currentFrameIdx = 1;
  totalFrames = 1;
  fps = 0;

  videoStatus.textContent = "Uploading video for exact frame preview…";

  const form = new FormData();
  form.append('video', file);

  try {
    const res = await fetch('/api/preview-video', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok || data.error) {
      videoStatus.textContent = 'Error: ' + (data.error || 'Could not prepare video');
      return;
    }

    currentPreviewId = data.previewId;
    fps = data.fps;
    totalFrames = data.totalFrames;

    const previewSection = document.querySelector('.panel-section[data-section="frame-preview"]');
    if (previewSection) previewSection.classList.add('expanded');
    videoStatus.innerHTML =
      `${data.totalFrames.toLocaleString()} frames · ${fps.toFixed(3)} fps · ${data.duration.toFixed(2)} sec`;

    await seekToFrame(1);
  } catch (err) {
    videoStatus.textContent = 'Error: ' + err;
  }
}

async function seekToFrame(idx) {
  if (!currentPreviewId) return;

  currentFrameIdx = Math.max(1, Math.min(totalFrames, idx));
  frameCounter.textContent =
    `Frame: ${currentFrameIdx.toLocaleString()} / ${totalFrames.toLocaleString()}`;

  const img = new Image();

  img.onload = () => {
    previewCanvas.width = img.naturalWidth;
    previewCanvas.height = img.naturalHeight;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
  };

  img.onerror = () => {
    videoStatus.textContent = `Could not decode frame ${currentFrameIdx}`;
  };

  img.src =
    `/api/preview-frame/${currentPreviewId}/${currentFrameIdx}?t=${Date.now()}`;
}

function renderSimSparkline(results, threshold) {
  const wrap = document.getElementById('sim-sparkline-wrap');
  if (!wrap || !results || !results.length) return;

  const sorted = [...results].sort((a, b) => a.frame - b.frame);

  const W = wrap.clientWidth || 320;
  const H = 90;
  const padL = 4, padR = 4, padT = 8, padB = 4;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const minFrame = sorted[0].frame;
  const maxFrame = sorted[sorted.length - 1].frame;
  const frameSpan = Math.max(1, maxFrame - minFrame);

  const xFor = (frame) => padL + ((frame - minFrame) / frameSpan) * plotW;
  const yFor = (sim) => padT + (1 - Math.max(0, Math.min(1, sim))) * plotH;

  const linePoints = sorted.map(r => `${xFor(r.frame).toFixed(1)},${yFor(r.sim).toFixed(1)}`).join(' ');
  const thresholdY = yFor(threshold).toFixed(1);

  const dots = sorted.map(r => {
    const cx = xFor(r.frame).toFixed(1);
    const cy = yFor(r.sim).toFixed(1);
    const color = r.passed ? '#7cc4ff' : '#d9534f';
    return `<circle cx="${cx}" cy="${cy}" r="7" fill="transparent" data-frame="${r.frame}" class="spark-hit" style="cursor:pointer;"></circle>` +
           `<circle cx="${cx}" cy="${cy}" r="2" fill="${color}" style="pointer-events:none;"></circle>`;
  }).join('');

  wrap.innerHTML = `
    <div style="font-size:10px;color:var(--dim);margin-bottom:3px;">
      Match confidence by frame — click a point to jump the preview
    </div>
    <svg width="${W}" height="${H}" style="background:#0c0c10;border:1px solid #22222a;border-radius:6px;display:block;">
      <line x1="${padL}" y1="${thresholdY}" x2="${W - padR}" y2="${thresholdY}"
            stroke="#4a4a55" stroke-width="1" stroke-dasharray="3,3"></line>
      <polyline points="${linePoints}" fill="none" stroke="#5a8fc4" stroke-width="1.5"></polyline>
      ${dots}
    </svg>
  `;

  wrap.querySelectorAll('.spark-hit').forEach(el => {
    el.addEventListener('click', () => {
      const frame = parseInt(el.dataset.frame, 10);
      stopFramePlayback();
      seekToFrame(frame);
      syncAudioToFrame(frame);
      previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  });
}

let playTimer = null;
const referenceAudio = document.getElementById('reference-audio');

function frameTime(frameIdx) {
  return Math.max(0, (frameIdx - 1) / (fps > 0 ? fps : 24));
}

function frameFromTime(timeSeconds) {
  const currentFps = fps > 0 ? fps : 24;
  return Math.min(totalFrames, Math.max(1, Math.floor(timeSeconds * currentFps) + 1));
}

function syncAudioToFrame(frameIdx) {
  if (!referenceAudio.src) return;
  const t = frameTime(frameIdx);
  try {
    referenceAudio.currentTime = t;
  } catch (_) {}
}

function stopFramePlayback() {
  if (playTimer) { 
    clearInterval(playTimer); 
    playTimer = null; 
  }
  referenceAudio.pause();
  document.getElementById('btn-play-frames').textContent = '▶ Play';
}

function startFramePlayback() {
  if (!currentPreviewId || playTimer) return;
  if (currentFrameIdx >= totalFrames) {
    currentFrameIdx = 1;
    syncAudioToFrame(1);
  } else {
    syncAudioToFrame(currentFrameIdx);
  }

  document.getElementById('btn-play-frames').textContent = '⏸ Playing…';

  const playPromise = referenceAudio.play();
  if (playPromise && playPromise.catch) playPromise.catch(() => {});

  const intervalMs = fps > 0 ? (1000 / fps) : (1000 / 24);
  playTimer = setInterval(() => {
    if (referenceAudio.paused || referenceAudio.ended) {
      stopFramePlayback();
      return;
    }

    const targetFrame = frameFromTime(referenceAudio.currentTime);
    if (targetFrame !== currentFrameIdx) {
      seekToFrame(targetFrame);
    }

    if (targetFrame >= totalFrames) {
      stopFramePlayback();
    }
  }, intervalMs / 2);
}

document.getElementById('btn-play-frames').addEventListener('click', () => {
  if (playTimer) stopFramePlayback(); else startFramePlayback();
});

document.getElementById('btn-stop-frames').addEventListener('click', stopFramePlayback);

document.getElementById('btn-prev-frame').addEventListener('click', () => { 
  stopFramePlayback(); 
  const nextIdx = Math.max(1, currentFrameIdx - 1);
  seekToFrame(nextIdx); 
  syncAudioToFrame(nextIdx); 
});

document.getElementById('btn-next-frame').addEventListener('click', () => { 
  stopFramePlayback(); 
  const nextIdx = Math.min(totalFrames, currentFrameIdx + 1);
  seekToFrame(nextIdx); 
  syncAudioToFrame(nextIdx); 
});

document.addEventListener('keydown', (e) => {
  if (!currentPreviewId) return;
  if (document.activeElement && ['INPUT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement.tagName)) return;
  
  if (e.key === 'ArrowLeft') { 
    e.preventDefault(); 
    stopFramePlayback(); 
    const nextIdx = Math.max(1, currentFrameIdx - 1);
    seekToFrame(nextIdx); 
    syncAudioToFrame(nextIdx); 
  } else if (e.key === 'ArrowRight') { 
    e.preventDefault(); 
    stopFramePlayback(); 
    const nextIdx = Math.min(totalFrames, currentFrameIdx + 1);
    seekToFrame(nextIdx); 
    syncAudioToFrame(nextIdx); 
  }
});

document.getElementById('btn-start-analysis').addEventListener('click', () => {
  if (currentVideoFile) {
    startVideoAnalysis(currentVideoFile, currentFrameIdx);
  }
});

async function startVideoAnalysis(videoFile, refFrameIdx) {
  selectedFrames.clear();
  videoStatus.textContent = 'Uploading for analysis…';
  const form = new FormData();
  form.append('video', videoFile);
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refFrame', refFrameIdx);

  const res = await fetch('/api/analyze-video', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) {
    videoStatus.textContent = 'Error: ' + data.error;
    return;
  }
  pollAnalysis(data.jobId, videoFile.name, refFrameIdx, videoStatus, 'video');
}

async function pollAnalysis(jobId, sourceLabel, refFrameIdx, statusEl, sourceType) {
  statusEl = statusEl || videoStatus;
  sourceType = sourceType || 'video';
  const res = await fetch(`/api/analysis-status/${jobId}`);
  const data = await res.json();

  if (data.status === 'error') {
    statusEl.textContent = 'Error: ' + data.error;
    return;
  }

  const passed = data.results.filter(r => r.passed).length;
  const failedSim = data.results.filter(r => !r.passed && r.failReason === 'sim').length;
  const failedBlur = data.results.filter(r => !r.passed && r.failReason === 'blur').length;

  statusEl.innerHTML = `
    Processing… ${data.frameCount} images seen<br>
    ${passed} kept, ${failedSim} low sim, ${failedBlur} blurry
    <div class="progress-track"><div class="progress-fill" style="width:${Math.min(100, (data.frameCount/200)*100)}%"></div></div>
  `;

  if (data.status === 'running') {
    setTimeout(() => pollAnalysis(jobId, sourceLabel, refFrameIdx, statusEl, sourceType), 800);
    return;
  }

  statusEl.innerHTML = `
    Done — ${passed}/${data.frameCount} kept.
    <div id="sim-sparkline-wrap" style="margin-top:8px;"></div>
    <button id="export-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Save kept frames to disk</button>
    <button id="save-selected-btn" disabled style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;opacity:0.5;">Save 0 selected frames to disk</button>
    <button id="view-selected-btn" style="margin-top:6px;width:100%;padding:6px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:6px;cursor:pointer;font-size:11px;">View selected</button>
    ${sourceType === 'folder' || sourceType === 'immich' ? '' : `
    <button id="playback-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Build playback (rejected frames blanked)</button>
    <video id="playback-video" controls style="width:100%;margin-top:8px;display:none;border-radius:6px;"></video>
    <div id="playback-frame-controls" style="display:none;margin-top:6px;gap:6px;">
      <button id="playback-prev-frame" class="btn-seek" style="flex:1;">◀ -1 frame</button>
      <button id="playback-next-frame" class="btn-seek" style="flex:1;">+1 frame ▶</button>
    </div>
    `}
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #22222a;">
      <div id="crosscheck-header" style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:11px;color:var(--dim);margin-bottom:8px;user-select:none;">
        <span class="chev" id="crosscheck-chev" style="display:inline-block;transition:transform .15s ease;font-size:10px;width:10px;">▾</span>
        <span style="flex:1;">Cross-check vs Immich library</span>
      </div>
      <div style="display:flex;gap:6px;">
        <input id="crosscheck-frame-input" type="number" min="1" style="width:70px;background:#0c0c10;border:1px solid #2a2a32;color:var(--text);border-radius:6px;padding:4px 6px;font-size:11px;" placeholder="frame #">
        <button id="crosscheck-btn" style="flex:1;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:11px;">Check vs Immich library</button>
      </div>
      <div id="crosscheck-results" style="margin-top:8px;"></div>
    </div>
  `;
  renderSimSparkline(data.results, data.simThreshold);
  const ccInput = document.getElementById('crosscheck-frame-input');
  if (ccInput) ccInput.value = currentFrameIdx;
  const ccHeader = document.getElementById('crosscheck-header');
  const ccChev = document.getElementById('crosscheck-chev');
  const ccResultsEl = document.getElementById('crosscheck-results');
  let ccCollapsed = localStorage.getItem('ringviz.crosscheckCollapsed') === '1';
  function applyCcCollapsed() {
    ccResultsEl.style.display = ccCollapsed ? 'none' : '';
    ccChev.style.transform = ccCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
  }
  applyCcCollapsed();
  ccHeader.addEventListener('click', () => {
    ccCollapsed = !ccCollapsed;
    localStorage.setItem('ringviz.crosscheckCollapsed', ccCollapsed ? '1' : '0');
    applyCcCollapsed();
  });
  document.getElementById('crosscheck-btn').onclick = async () => {
    const input = document.getElementById('crosscheck-frame-input');
    const frameNo = parseInt(input.value, 10) || currentFrameIdx;
    const btn = document.getElementById('crosscheck-btn');
    const out = document.getElementById('crosscheck-results');
    if (ccCollapsed) { ccCollapsed = false; localStorage.setItem('ringviz.crosscheckCollapsed', '0'); applyCcCollapsed(); }
    btn.textContent = 'Checking…';
    btn.disabled = true;
    out.innerHTML = '';
    try {
      const res = await fetch(`/api/immich-cross-check/${jobId}/${frameNo}`);
      const result = await res.json();
      if (result.error) {
        out.innerHTML = `<div style="font-size:11px;color:#d9534f;">${result.error}</div>`;
      } else if (!result.results.length) {
        out.innerHTML = `<div style="font-size:11px;color:var(--dim);">No faces in Immich library yet to compare against.</div>`;
      } else {
        out.innerHTML = `<div style="font-size:10px;color:var(--dim);margin-bottom:4px;">Closest matches already in Immich:</div>` +
          result.results.slice(0, 8).map((r, i) => `
            <div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #1a1a20;">
              <img src="/api/thumb/${r.assetId}" style="width:36px;height:36px;object-fit:cover;border-radius:4px;">
              <div style="flex:1;min-width:0;font-size:10px;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.filename}</div>
              <div style="font-size:10px;color:var(--accent);">${(r.similarity * 100).toFixed(1)}%</div>
              <button class="cc-add-btn" data-idx="${i}" data-asset-id="${r.assetId}" style="font-size:9px;padding:3px 6px;background:#2a1a3a;border:1px solid #4a2a6a;color:#d4a5ff;border-radius:4px;cursor:pointer;flex-shrink:0;">${extraImmichNodes.some(n => n.assetId === r.assetId) ? 'Remove ✓' : '+ Ring'}</button>
            </div>
          `).join('');
        out.querySelectorAll('.cc-add-btn').forEach(btn => {
          btn.addEventListener('click', () => {
            const match = result.results[parseInt(btn.dataset.idx, 10)];
            const isInRing = extraImmichNodes.some(n => n.assetId === match.assetId);
            if (isInRing) {
              removeImmichNodeFromRing(match.assetId);
              btn.textContent = '+ Ring';
            } else {
              addImmichNodeToRing(match, jobId, frameNo);
              btn.textContent = 'Remove ✓';
            }
          });
        });
      }
    } catch (e) {
      out.innerHTML = `<div style="font-size:11px;color:#d9534f;">Request failed: ${e}</div>`;
    }
    btn.textContent = 'Check vs Immich library';
    btn.disabled = false;
  };
  document.getElementById('export-btn').onclick = async () => {
    const btn = document.getElementById('export-btn');
    const assetIds = Array.from(selectedAssetIds);
    btn.textContent = 'Saving…';
    btn.disabled = true;
    const res = await fetch(`/api/export-job/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assetIds, ...getExportParams() }),
    });
    const result = await res.json();
    btn.textContent = `Saved ${result.frameExported ?? result.exported} frames + ${result.immichExported || 0} Immich → ${result.path}` + (result.skipped ? ` (${result.skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded to avoid upscaling)` : '');
    if (result.errors && result.errors.length) console.warn('Some Immich exports failed:', result.errors);
  };
  const viewSelectedBtn = document.getElementById('view-selected-btn');
  if (viewSelectedBtn) viewSelectedBtn.onclick = openSelectionModal;
// ---- dedupe frame/Immich pairs before export: a frame pulled into the ring
// via "+ Ring" cross-check is the *same underlying photo* as its matched
// Immich asset, just reached by two different paths. If both ends up
// selected (e.g. via Select All grabbing everything currently listed),
// exporting both writes the same image twice. Keep the Immich copy - it's
// the canonical library asset - and drop the frame side of any such pair.
function dedupeFrameImmichPairs(frames, assetIds) {
  const assetIdSet = new Set(assetIds);
  const framesToSourceAsset = new Map();
  extraImmichNodes.forEach(n => {
    if (n.sourceFrame !== undefined && assetIdSet.has(n.assetId)) {
      framesToSourceAsset.set(n.sourceFrame, n.assetId);
    }
  });
  const dedupedFrames = frames.filter(f => !framesToSourceAsset.has(f));
  const droppedCount = frames.length - dedupedFrames.length;
  return { frames: dedupedFrames, assetIds, droppedCount };
}

document.getElementById('save-selected-btn').onclick = async () => {
  const btn = document.getElementById('save-selected-btn');
  const rawFrames = Array.from(selectedFrames);
  const assetIds = Array.from(selectedAssetIds);
  const { frames, droppedCount } = dedupeFrameImmichPairs(rawFrames, assetIds);
  if (!frames.length && !assetIds.length) return;
  const prevText = btn.textContent;
  btn.textContent = 'Saving…';
  btn.disabled = true;
  const res = await fetch(`/api/export-job/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frames, assetIds, ...getExportParams() }),
  });
  const result = await res.json();
  btn.textContent = `Saved ${result.frameExported ?? 0}/${frames.length} frames + ${result.immichExported || 0}/${assetIds.length} Immich → ${result.path}` + (droppedCount ? ` (${droppedCount} duplicate frame(s) already covered by an Immich match, skipped)` : '') + (result.skipped ? ` (${result.skipped} skipped: face too small)` : '') + (result.widened ? ` (${result.widened} zoomed out, ${result.padded || 0} padded)` : '');
  if (result.errors && result.errors.length) console.warn('Some Immich exports failed:', result.errors);
  setTimeout(updateSaveSelectedButton, 3000);
};
  document.getElementById('playback-btn')?.addEventListener('click', async () => {
    const btn = document.getElementById('playback-btn');
    btn.textContent = 'Building…';
    btn.disabled = true;
    const res = await fetch(`/api/build-playback/${jobId}`, { method: 'POST' });
    const result = await res.json();
    if (result.error) {
      btn.textContent = 'Error: ' + result.error;
      return;
    }
    btn.textContent = 'Playback ready';
    const vid = document.getElementById('playback-video');
    vid.src = result.url;
    vid.style.display = 'block';

    const playbackFps = result.fps || 24.0;
    const frameStep = 1 / playbackFps;
    const stepControls = document.getElementById('playback-frame-controls');
    stepControls.style.display = 'flex';

    document.getElementById('playback-prev-frame').onclick = () => {
      vid.pause();
      vid.currentTime = Math.max(0, vid.currentTime - frameStep);
    };
    document.getElementById('playback-next-frame').onclick = () => {
      vid.pause();
      vid.currentTime = Math.min(vid.duration || Infinity, vid.currentTime + frameStep);
    };
  });

  const results = data.results
    .filter(r => r.passed)
    .map(r => ({
      filename: r.origName || `frame_${r.frame}`,
      frame: r.frame,
      similarity: r.sim,
      thumbUrl: `/api/framefile/${r.frameId}`,
      pitch: r.pitch,
      yaw: r.yaw,
      roll: r.roll,
    }))
    .sort((a, b) => b.similarity - a.similarity);

  const anchorUrl = `/api/framefile/${jobId}_anchor`;
  const hudLabel = sourceType === 'immich'
    ? 'IMMICH SELECTION ANALYSIS'
    : (sourceType === 'folder' ? 'IMAGE SET ANALYSIS (local, not in Immich)' : 'VIDEO FRAME ANALYSIS (local, not in Immich)');
  document.getElementById('hud-mode').textContent = hudLabel;
  document.getElementById('hud-filename').textContent = sourceLabel;
  const curModeEl = document.getElementById('sidebar-current-mode');
  if (curModeEl) curModeEl.textContent = hudLabel;

  lastVideoRingState = {
    anchorUrl,
    refFrameIdx,
    baseResults: results,
    sourceType,
  };
  renderVideoRing();
  updateSaveSelectedButton();
}

// ---- image folder / zip loader: runs the exact same analysis pipeline as
// video, just over a variable-count set of still images (e.g. 32 curated
// LoRA reference frames) instead of decoded video frames ----
const folderImagesInput = document.getElementById('folder-images-input');
const folderZipInput = document.getElementById('folder-zip-input');
const folderStatus = document.getElementById('folder-status');
let lastFolderSource = null;

document.getElementById('folder-images-btn').addEventListener('click', () => folderImagesInput.click());
document.getElementById('folder-zip-btn').addEventListener('click', () => folderZipInput.click());

folderImagesInput.addEventListener('change', () => {
  if (folderImagesInput.files.length) startFolderAnalysis({ images: folderImagesInput.files });
});
folderZipInput.addEventListener('change', () => {
  if (folderZipInput.files.length) startFolderAnalysis({ zip: folderZipInput.files[0] });
});

async function startFolderAnalysis({ images, zip }, refIndexOverride) {
  selectedFrames.clear();
  const refIndex = refIndexOverride || (parseInt(document.getElementById('folder-ref-index').value, 10) || 1);
  document.getElementById('folder-ref-index').value = refIndex;
  lastFolderSource = { images, zip };
  const form = new FormData();
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refIndex', refIndex);

  let sourceLabel;
  if (zip) {
    form.append('zip', zip);
    sourceLabel = zip.name;
    folderStatus.textContent = `Uploading ${zip.name}…`;
  } else {
    Array.from(images).forEach(f => form.append('images', f));
    sourceLabel = `${images.length} images`;
    form.append('sourceName', `imgset_${images.length}`);
    folderStatus.textContent = `Uploading ${images.length} images…`;
  }

  const res = await fetch('/api/analyze-folder', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) {
    folderStatus.textContent = 'Error: ' + data.error;
    return;
  }
  folderStatus.textContent = `${data.imageCount} images accepted, analyzing…`;
  pollAnalysis(data.jobId, sourceLabel, refIndex, folderStatus, 'folder');
}

let lastVideoRingState = null;
let lastImmichAnalysisAssetIds = null;
const extraImmichNodes = [];
let ringSortMetric = 'sim';

// squeeze filter: a live post-hoc min-similarity cutoff applied on top of
// whatever the original video-analysis job already loaded, so you can
// tighten (or loosen) the working set without re-running the whole job.
// Straight-on similarity browsing wants a high bar; pose-diversity browsing
// (yaw/pitch/roll) wants a low bar so marginal-confidence frames stay
// available - defaults switch automatically per metric, but the slider
// always stays user-overridable.
const SQUEEZE_DEFAULTS = { sim: 65, yaw: 20, pitch: 20, roll: 20 };
let squeezeMinPct = SQUEEZE_DEFAULTS.sim;
let squeezeUserOverridden = false;

const squeezeSlider = document.getElementById('ring-squeeze-slider');
const squeezeVal = document.getElementById('ring-squeeze-val');
squeezeSlider.value = squeezeMinPct;

squeezeSlider.addEventListener('input', () => {
  squeezeMinPct = parseFloat(squeezeSlider.value);
  squeezeUserOverridden = true;
  if (lastVideoRingState) renderVideoRing();
});

document.querySelectorAll('.ring-sort-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    if (cb.checked) {
      document.querySelectorAll('.ring-sort-cb').forEach(other => {
        if (other !== cb) other.checked = false;
      });
      ringSortMetric = cb.dataset.metric;
      if (!squeezeUserOverridden) {
        squeezeMinPct = SQUEEZE_DEFAULTS[ringSortMetric];
        squeezeSlider.value = squeezeMinPct;
      }
    } else {
      // don't allow zero selection - fall back to similarity
      cb.checked = true;
      return;
    }
    if (lastVideoRingState) renderVideoRing();
  });
});

function metricValueForRing(r) {
  const raw = r.similarity;
  return typeof raw === 'number' ? raw : 0;
}

function applySqueeze(combined) {
  const cutoff = squeezeMinPct / 100;
  const kept = combined.filter(r => {
    const sim = typeof r.similarity === 'number' ? r.similarity : (typeof r.sim === 'number' ? r.sim : 1);
    return sim >= cutoff;
  });
  squeezeVal.textContent = `${squeezeMinPct}% (${kept.length}/${combined.length})`;
  return kept;
}

document.getElementById('find-neutral-btn').addEventListener('click', () => {
  if (!lastVideoRingState) return;
  const { baseResults } = lastVideoRingState;
  const pool = applySqueeze([...baseResults, ...extraImmichNodes])
    .filter(r => typeof r.yaw === 'number' && typeof r.pitch === 'number' && typeof r.roll === 'number');
  const readout = document.getElementById('neutral-pose-readout');
  if (!pool.length) {
    readout.style.display = 'block';
    readout.textContent = 'No frames with pose data in current working set.';
    return;
  }
  let best = pool[0];
  let bestScore = Math.abs(best.yaw) + Math.abs(best.pitch) + Math.abs(best.roll);
  pool.forEach(r => {
    const score = Math.abs(r.yaw) + Math.abs(r.pitch) + Math.abs(r.roll);
    if (score < bestScore) { best = r; bestScore = score; }
  });
  readout.style.display = 'block';
  readout.innerHTML = `Most neutral: <b>${best.filename}</b> — yaw ${best.yaw.toFixed(1)}° pitch ${best.pitch.toFixed(1)}° roll ${best.roll.toFixed(1)}° (sim ${(best.similarity*100).toFixed(1)}%)
    <button type="button" id="use-as-reference-btn" style="display:block;width:100%;margin-top:6px;font-size:10px;padding:5px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:4px;cursor:pointer;">Use as reference &amp; re-analyze</button>`;
  flashHighlightFrame(best);

  document.getElementById('use-as-reference-btn').onclick = () => {
    const btn = document.getElementById('use-as-reference-btn');
    const sourceType = lastVideoRingState.sourceType;
    if (sourceType === 'folder') {
      if (!lastFolderSource) {
        btn.textContent = 'Original folder/zip no longer available — reload it first';
        return;
      }
      btn.textContent = 'Re-analyzing…';
      startFolderAnalysis(lastFolderSource, best.frame);
    } else if (sourceType === 'immich') {
      if (!lastImmichAnalysisAssetIds) {
        btn.textContent = 'Original Immich selection no longer available — re-select and analyze again';
        return;
      }
      btn.textContent = 'Re-analyzing…';
      reanalyzeImmichSelection(lastImmichAnalysisAssetIds, best.frame);
    } else {
      if (!currentVideoFile) {
        btn.textContent = 'Original video no longer available — reload it first';
        return;
      }
      btn.textContent = 'Re-analyzing…';
      startVideoAnalysis(currentVideoFile, best.frame);
    }
  };
});

// flashes/scrolls to whichever node or pose-list item represents this frame,
// in whichever view is currently active
function flashHighlightFrame(r) {
  const el = document.querySelector(`.pose-list-item[data-filename="${CSS.escape(r.filename)}"], .node[data-filename="${CSS.escape(r.filename)}"]`);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    el.style.transition = 'box-shadow 0.15s ease';
    const prevShadow = el.style.boxShadow;
    let flashes = 0;
    const flashInterval = setInterval(() => {
      el.style.boxShadow = flashes % 2 === 0 ? '0 0 0 4px #7cc4ff' : prevShadow;
      flashes++;
      if (flashes > 5) { clearInterval(flashInterval); el.style.boxShadow = prevShadow; }
    }, 200);
  }
}

function renderVideoRing() {
  if (!lastVideoRingState) return;
  const { anchorUrl, refFrameIdx, baseResults, sourceType } = lastVideoRingState;
  const combined = applySqueeze([...baseResults, ...extraImmichNodes]);
  const metricLabel = { sim: 'Similarity', yaw: 'Yaw', pitch: 'Pitch', roll: 'Roll' }[ringSortMetric];
  const baseLabel = sourceType === 'immich'
    ? 'IMMICH SELECTION ANALYSIS'
    : (sourceType === 'folder' ? 'IMAGE SET ANALYSIS (local, not in Immich)' : 'VIDEO FRAME ANALYSIS (local, not in Immich)');
  const anchorLabel = (sourceType === 'folder' || sourceType === 'immich') ? 'Reference (Anchor)' : `Frame ${refFrameIdx} (Anchor)`;

  const stageEl = document.getElementById('stage');
  const listEl = document.getElementById('pose-list-view');
  const scrubberEl = document.getElementById('pose-list-scrubber');

  if (ringSortMetric === 'sim') {
    listEl.style.display = 'none';
    scrubberEl.style.display = 'none';
    stageEl.style.display = '';
    render('__anchor__', {
      mode: 'face',
      results: [{ assetId: '__anchor__', filename: anchorLabel, similarity: 1.0, thumbUrl: anchorUrl }, ...combined],
    }, anchorUrl);
  } else {
    stageEl.style.display = 'none';
    listEl.style.display = 'flex';
    scrubberEl.style.display = 'flex';
    renderPoseList(ringSortMetric, anchorUrl, anchorLabel, combined);
  }

  const hudMode = document.getElementById('hud-mode');
  if (hudMode) hudMode.textContent = `${baseLabel} · sorted by ${metricLabel}`;
}

function renderPoseList(metric, anchorUrl, anchorLabel, combined) {
  const listEl = document.getElementById('pose-list-view');
  listEl.innerHTML = '';

  const anchorWrap = document.createElement('div');
  anchorWrap.className = 'pose-list-anchor';
  anchorWrap.innerHTML = `<img src="${anchorUrl}" loading="lazy"><div class="plabel">${anchorLabel}</div>`;
  listEl.appendChild(anchorWrap);

  // sort by raw signed value, not abs() - so one end is the most-negative
  // extreme (e.g. head turned hard left) and the other end is the
  // most-positive extreme (turned hard right), with near-neutral poses
  // sitting in the middle. Items missing this metric sort to the middle too.
  const withMetric = combined.filter(r => typeof r[metric] === 'number');
  const withoutMetric = combined.filter(r => typeof r[metric] !== 'number');
  withMetric.sort((a, b) => a[metric] - b[metric]);

  const PITCH_PX_PER_DEG = 1.6;
  const PITCH_CLAMP_DEG = 40;

  withMetric.forEach(r => {
    const item = document.createElement('div');
    const isSelected = r.assetId ? selectedAssetIds.has(r.assetId) : (r.frame !== undefined && selectedFrames.has(r.frame));
    item.className = 'pose-list-item' + (isSelected ? ' selected' : '');
    item.dataset.filename = r.filename;
    if (r.assetId) item.dataset.assetId = r.assetId;
    if (r.frame !== undefined) item.dataset.frame = r.frame;
    const thumb = r.thumbUrl || (r.fromImmich ? `/api/thumb/${r.assetId}` : `/api/thumb/${r.assetId}`);
    item.innerHTML = `<img src="${thumb}" loading="lazy"><div class="plabel">${metric}: ${r[metric].toFixed(1)}°</div>`;
    if (typeof r.pitch === 'number') {
      // pitch up (nose up) raises the thumbnail, pitch down lowers it -
      // gives the strip a wavy "head bob" feel that mirrors the pose itself.
      const clamped = Math.max(-PITCH_CLAMP_DEG, Math.min(PITCH_CLAMP_DEG, r.pitch));
      const offsetPx = -clamped * PITCH_PX_PER_DEG;
      item.dataset.pitchY = offsetPx;
      item.style.transform = `translateY(${offsetPx}px)`;
    } else {
      item.dataset.pitchY = 0;
    }

    // single click (debounced) = browse/recenter; double click = add to
    // export queue. Debounce mirrors the person-clusters grid pattern so a
    // dblclick doesn't also fire the single-click action first.
    let clickTimer = null;
    const toggleExportSelection = () => {
      if (r.assetId) {
        if (selectedAssetIds.has(r.assetId)) { selectedAssetIds.delete(r.assetId); }
        else { selectedAssetIds.add(r.assetId); }
        updateImmichSelectionBar();
      } else if (r.frame !== undefined) {
        if (selectedFrames.has(r.frame)) { selectedFrames.delete(r.frame); }
        else { selectedFrames.add(r.frame); }
        updateSaveSelectedButton();
      }
      item.classList.toggle('selected');
    };
    item.onclick = () => {
      if (!r.assetId) return; // no recenter target for local video frames
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => {
        const input = document.getElementById('search-input');
        if (input) input.value = r.filename || '';
        loadNeighbors(r.assetId);
      }, 220);
    };
    item.ondblclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      clearTimeout(clickTimer);
      toggleExportSelection();
    };
    item.addEventListener('mouseenter', () => showHoverPreview(r));
    item.addEventListener('mouseleave', hideHoverPreview);
    listEl.appendChild(item);
  });

  if (withoutMetric.length) {
    const note = document.createElement('div');
    note.className = 'pose-list-item';
    note.style.opacity = '0.4';
    note.innerHTML = `<div class="plabel">+${withoutMetric.length} no ${metric} data</div>`;
    listEl.appendChild(note);
  }

  setupPoseListScrubber(listEl, metric, withMetric);
}

// ---- horizontal scrubber: drag-anywhere navigation for the pose strip.
// Essential once you're dealing with hundreds of frames - side-scrolling
// (even with a scroll wheel that supports it) is far too slow to browse
// a 700-frame set. Two-way synced with the strip's actual scroll position,
// and a vertical-wheel fallback for anyone without horizontal scroll input.
function setupPoseListScrubber(listEl, metric, withMetric) {
  const slider = document.getElementById('pose-scrub-slider');
  const leftLabel = document.getElementById('pose-scrub-left');
  const rightLabel = document.getElementById('pose-scrub-right');

  if (withMetric.length) {
    leftLabel.textContent = `${metric}: ${withMetric[0][metric].toFixed(1)}°`;
    rightLabel.textContent = `${metric}: ${withMetric[withMetric.length - 1][metric].toFixed(1)}°`;
  } else {
    leftLabel.textContent = '';
    rightLabel.textContent = '';
  }

  const maxScroll = () => Math.max(1, listEl.scrollWidth - listEl.clientWidth);

  let syncingFromScroll = false;
  slider.value = 0;
  slider.oninput = () => {
    syncingFromScroll = true;
    listEl.scrollLeft = (parseFloat(slider.value) / 1000) * maxScroll();
    syncingFromScroll = false;
  };

  listEl.onscroll = () => {
    if (syncingFromScroll) return;
    slider.value = Math.round((listEl.scrollLeft / maxScroll()) * 1000);
  };

  // vertical wheel -> horizontal scroll, so anyone without a side-scroll
  // wheel/trackpad gesture can still move through the strip with a normal mouse
  listEl.onwheel = (e) => {
    if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      e.preventDefault();
      listEl.scrollLeft += e.deltaY;
    }
  };
}

function addImmichNodeToRing(match, jobId, frameNo) {
  if (extraImmichNodes.some(n => n.assetId === match.assetId)) return;
  const node = {
    assetId: match.assetId,
    filename: `${match.filename} (Immich)`,
    similarity: match.similarity,
    thumbUrl: `/api/thumb/${match.assetId}`,
    fromImmich: true,
    sourceFrame: frameNo,
  };
  extraImmichNodes.push(node);
  selectedAssetIds.add(match.assetId);
  updateImmichSelectionBar();
  renderVideoRing();

  if (jobId !== undefined && frameNo !== undefined) {
    fetch(`/api/immich-face-pose/${jobId}/${frameNo}/${match.assetId}`)
      .then(res => res.json())
      .then(pose => {
        if (pose.error) return;
        node.yaw = pose.yaw;
        node.pitch = pose.pitch;
        node.roll = pose.roll;
        renderVideoRing();
      })
      .catch(() => {});
  }
}

function removeImmichNodeFromRing(assetId) {
  const idx = extraImmichNodes.findIndex(n => n.assetId === assetId);
  if (idx === -1) return;
  extraImmichNodes.splice(idx, 1);
  selectedAssetIds.delete(assetId);
  updateImmichSelectionBar();
  renderVideoRing();
}
