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
function wireRenderPatch() {
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
}

let rankedSortMetric = 'sim';
function sortRankedResults(results) {
  // sort descending for sim and vertFillPct (bigger = better match / bigger
  // face), descending for blur too (sharper = better) - all three want
  // "best first" at the top of the list. Rows missing the chosen metric
  // (e.g. blur/vertFillPct only exist once a frame has gone through pose
  // analysis or on-demand detection) sort to the bottom rather than being
  // dropped, so the list stays complete either way.
  const key = rankedSortMetric === 'sim' ? 'similarity' : rankedSortMetric;
  const withMetric = results.filter(r => typeof r[key] === 'number');
  const withoutMetric = results.filter(r => typeof r[key] !== 'number');
  withMetric.sort((a, b) => b[key] - a[key]);
  return [...withMetric, ...withoutMetric];
}

function render(centerId, data, centerThumbUrl) {
  lastNeighborsRender = { centerId, data, centerThumbUrl };
  stage.innerHTML = '';
  const modeLabel = data.mode === 'face' ? 'FACE SIMILARITY' : 'CLIP (WHOLE-IMAGE) SIMILARITY';
  document.getElementById('hud-mode').textContent = modeLabel;

  const results = sortRankedResults(data.results.filter(r => r.assetId !== centerId));
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
      let poseLine = `pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
      if (typeof r.blur === 'number') poseLine += ` · sharp: ${r.blur.toFixed(0)}`;
      if (typeof r.vertFillPct === 'number') poseLine += ` · ${(r.vertFillPct * 100).toFixed(0)}% frame ht`;
      else if (typeof r.bboxRatio === 'number') poseLine += ` · ${(r.bboxRatio * 100).toFixed(0)}% frame`;
      poseHtml = `<div style="font-size:var(--fs-10);color:var(--dim);margin-top:2px;">${poseLine}</div>`;
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
        <div class="fname">${r.filename}${r.fromImmich ? ' <span style="color:#d4a544;font-size:var(--fs-9);">● Immich</span>' : ''}${linkedBadgeHtml}</div>
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

function wireFisheyeLensMouseMove() {
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
}

const POSE_LENS_RADIUS = 140;
const POSE_LENS_MAX_SCALE = 1.6;
function attachLensEffect(container, itemSelector, { radius = 140, maxScale = 1.6 } = {}) {
  let rafPending = false;
  let lastMouse = null;

  function apply(mx, my) {
    const items = container.querySelectorAll(itemSelector);
    items.forEach(item => {
      const rect = item.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dist = Math.hypot(mx - cx, my - cy);
      const base = item.dataset.baseTransform || '';
      if (dist < radius) {
        const t = 1 - (dist / radius);
        const eased = t * t * (3 - 2 * t);
        const scale = 1 + eased * (maxScale - 1);
        item.style.transform = `${base} scale(${scale})`;
        item.style.zIndex = Math.round(10 + eased * 50);
      } else {
        item.style.transform = base;
        item.style.zIndex = item.dataset.baseZ || 1;
      }
    });
    rafPending = false;
  }

  container.addEventListener('mousemove', (e) => {
    lastMouse = [e.clientX, e.clientY];
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(() => apply(...lastMouse));
    }
  });
  container.addEventListener('mouseleave', () => {
    container.querySelectorAll(itemSelector).forEach(item => {
      item.style.transform = item.dataset.baseTransform || '';
      item.style.zIndex = item.dataset.baseZ || 1;
    });
  });
}

function wireAttachLensEffectCall() {
attachLensEffect(document.getElementById('pose-list-view'), '.pose-list-item', { radius: POSE_LENS_RADIUS, maxScale: POSE_LENS_MAX_SCALE });
}

let posePickerPool = [];
let posePickerCells = [];
let posePickerDisplayed = new Array(9).fill(null);
function buildPosePickerCells() {
  const grid = document.getElementById('pose-picker-grid');
  grid.innerHTML = '';
  posePickerCells = [];
  posePickerDisplayed = new Array(9).fill(null);
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('div');
    cell.style.cursor = 'pointer';
    cell.style.visibility = 'hidden';
    cell.innerHTML = `<img loading="lazy" style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;border:2px solid #3a3a44;display:block;">`;
    grid.appendChild(cell);
    posePickerCells.push(cell);
  }
}

function setupPosePicker(dataResults, jobId, sourceType) {
  posePickerPool = (dataResults || [])
    .filter(r => r.passed && r.frameId && typeof r.pitch === 'number' && typeof r.yaw === 'number')
    .map(r => ({
      filename: r.origName || `frame_${r.frame}`,
      frame: r.frame,
      thumbUrl: `/api/framefile/${r.frameId}`,
      pitch: r.pitch,
      yaw: r.yaw,
      blur: typeof r.blur === 'number' ? r.blur : 0,
      bboxRatio: typeof r.bboxRatio === 'number' ? r.bboxRatio : null,
      vertFillPct: typeof r.vertFillPct === 'number' ? r.vertFillPct : null,
    }));

  const emptyEl = document.getElementById('pose-picker-empty');
  const controlsEl = document.getElementById('pose-picker-controls');
  if (!posePickerPool.length) {
    emptyEl.style.display = 'block';
    controlsEl.style.display = 'none';
    return;
  }
  emptyEl.style.display = 'none';
  controlsEl.style.display = 'flex';

  const pitchVals = posePickerPool.map(it => it.pitch);
  const yawVals = posePickerPool.map(it => it.yaw);
  const pitchSlider = document.getElementById('pose-picker-pitch');
  const yawSlider = document.getElementById('pose-picker-yaw');
  // fixed +/-90 range rather than clamped to the pool's own extent, so you
  // can dial in an arbitrary target pose (e.g. 45/45) even if nothing in
  // the current pool is actually that far off-angle yet - useful for
  // spotting coverage gaps, not just browsing what's already there
  pitchSlider.min = yawSlider.min = -90;
  pitchSlider.max = yawSlider.max = 90;
  document.getElementById('pose-picker-pitch-num').min = document.getElementById('pose-picker-yaw-num').min = -90;
  document.getElementById('pose-picker-pitch-num').max = document.getElementById('pose-picker-yaw-num').max = 90;
  // starting position centers on the pool's own centroid, not an arbitrary
  // 0 that might sit outside the actual pose range for this particular set
  pitchSlider.value = Math.round(pitchVals.reduce((s, v) => s + v, 0) / pitchVals.length);
  yawSlider.value = Math.round(yawVals.reduce((s, v) => s + v, 0) / yawVals.length);
  document.getElementById('pose-picker-pitch-num').value = pitchSlider.value;
  document.getElementById('pose-picker-yaw-num').value = yawSlider.value;

  buildPosePickerCells();
  renderPosePickerGrid();
}

function updatePosePickerCellVisual(cell, it, dist) {
  cell.style.visibility = 'visible';
  const isSelected = selectedFrames.has(it.frame);
  const img = cell.querySelector('img');
  if (img.dataset.frame !== String(it.frame)) {
    img.src = it.thumbUrl;
    img.dataset.frame = it.frame;
  }
  const scaleTxt = typeof it.vertFillPct === 'number' ? `, face ${(it.vertFillPct * 100).toFixed(0)}% frame ht`
    : (typeof it.bboxRatio === 'number' ? `, face ${(it.bboxRatio * 100).toFixed(0)}% of frame` : '');
  img.title = `pitch ${it.pitch.toFixed(1)}, yaw ${it.yaw.toFixed(1)} (Δ${dist.toFixed(1)} from target), sharpness ${it.blur.toFixed(0)}${scaleTxt} — click to ${isSelected ? 'remove from' : 'add to'} selection`;
  img.style.borderColor = isSelected ? 'var(--accent)' : '#3a3a44';
  cell.onclick = () => {
    if (selectedFrames.has(it.frame)) selectedFrames.delete(it.frame);
    else selectedFrames.add(it.frame);
    syncSelectionVisuals();
    updateSaveSelectedButton();
    const cb = document.querySelector(`#list-body-frames .frame-select-cb[data-frame="${it.frame}"]`);
    if (cb) cb.checked = selectedFrames.has(it.frame);
    renderPosePickerGrid();
  };
}

function renderPosePickerGrid() {
  if (!posePickerPool.length) return;
  const pitchTarget = parseFloat(document.getElementById('pose-picker-pitch').value);
  const yawTarget = parseFloat(document.getElementById('pose-picker-yaw').value);
  document.getElementById('pose-picker-pitch-num').value = pitchTarget;
  document.getElementById('pose-picker-yaw-num').value = yawTarget;

  const toleranceOn = document.getElementById('pose-picker-tolerance-enable').checked;
  const toleranceVal = parseFloat(document.getElementById('pose-picker-tolerance-val').value) || 5;

  let ranked = posePickerPool
    .map(it => ({ it, dist: Math.hypot(it.pitch - pitchTarget, it.yaw - yawTarget) }))
    .sort((a, b) => a.dist - b.dist);
  if (toleranceOn) {
    ranked = ranked.filter(r => r.dist <= toleranceVal);
  }
  ranked = ranked.slice(0, 9);
  const rankedFrames = ranked.map(r => r.it.frame);

  // items already on stage that are still in the new nearest-9 keep their
  // exact slot untouched; only slots whose occupant fell out of range get
  // refilled with whichever new items just entered
  const keepSlot = posePickerDisplayed.map(frame => frame !== null && rankedFrames.includes(frame));
  const toPlace = ranked.filter(r => !posePickerDisplayed.includes(r.it.frame));
  let placeIdx = 0;

  for (let i = 0; i < 9; i++) {
    if (keepSlot[i]) {
      const match = ranked.find(r => r.it.frame === posePickerDisplayed[i]);
      updatePosePickerCellVisual(posePickerCells[i], match.it, match.dist);
    } else if (placeIdx < toPlace.length) {
      const { it, dist } = toPlace[placeIdx++];
      posePickerDisplayed[i] = it.frame;
      updatePosePickerCellVisual(posePickerCells[i], it, dist);
    } else {
      posePickerDisplayed[i] = null;
      posePickerCells[i].style.visibility = 'hidden';
    }
  }

  document.getElementById('pose-picker-count').textContent = toleranceOn
    ? `${ranked.length} within ${toleranceVal}° of target (${posePickerPool.length} in pool)`
    : `${posePickerPool.length} in analyzed pool`;
}

function syncPosePickerFromSlider(axis) {
  document.getElementById(`pose-picker-${axis}-num`).value = document.getElementById(`pose-picker-${axis}`).value;
  renderPosePickerGrid();
}

function syncPosePickerFromNumber(axis) {
  const num = document.getElementById(`pose-picker-${axis}-num`);
  const slider = document.getElementById(`pose-picker-${axis}`);
  const clamped = Math.max(-90, Math.min(90, parseFloat(num.value) || 0));
  slider.value = clamped;
  renderPosePickerGrid();
}

function wirePosePickerControls() {
document.getElementById('pose-picker-pitch').addEventListener('input', () => syncPosePickerFromSlider('pitch'));
document.getElementById('pose-picker-yaw').addEventListener('input', () => syncPosePickerFromSlider('yaw'));
document.getElementById('pose-picker-pitch-num').addEventListener('input', () => syncPosePickerFromNumber('pitch'));
document.getElementById('pose-picker-yaw-num').addEventListener('input', () => syncPosePickerFromNumber('yaw'));

document.getElementById('pose-picker-tolerance-enable').addEventListener('change', (e) => {
  const input = document.getElementById('pose-picker-tolerance-val');
  input.disabled = !e.target.checked;
  input.style.color = e.target.checked ? 'var(--text)' : 'var(--dim)';
  renderPosePickerGrid();
});
document.getElementById('pose-picker-tolerance-val').addEventListener('input', renderPosePickerGrid);

document.getElementById('pose-picker-preview-btn').addEventListener('click', () => {
  const frames = posePickerDisplayed.filter(f => f !== null);
  if (!frames.length) return;
  modalFrameOverride = frames;
  document.getElementById('selection-modal-pose-layout').checked = true;
  document.getElementById('selection-modal-spread-wrap').style.display = 'flex';
  document.getElementById('selection-modal-overlay').style.display = 'flex';
  renderSelectionModal();
});
}

let scalePickerPool = [];
let scalePickerCells = [];
let scalePickerDisplayed = new Array(9).fill(null);
function buildScalePickerCells() {
  const grid = document.getElementById('scale-picker-grid');
  grid.innerHTML = '';
  scalePickerCells = [];
  scalePickerDisplayed = new Array(9).fill(null);
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('div');
    cell.style.cursor = 'pointer';
    cell.style.visibility = 'hidden';
    cell.innerHTML = `<img loading="lazy" style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;border:2px solid #3a3a44;display:block;">`;
    grid.appendChild(cell);
    scalePickerCells.push(cell);
  }
}

function setupScalePicker(dataResults, jobId, sourceType) {
  const minFacePx = parseFloat(document.getElementById('export-min-face')?.value) || 0;

  scalePickerPool = (dataResults || [])
    .filter(r => {
      if (!r.passed || !r.frameId) return false;
      const scaleVal = typeof r.vertFillPct === 'number' ? r.vertFillPct
        : (typeof r.bboxRatio === 'number' ? Math.sqrt(r.bboxRatio) : null);
      if (scaleVal === null) return false;
      // honor the same "too distant to be useful" floor Export Settings
      // uses, so this picker doesn't surface frames the export pipeline
      // would just skip anyway
      if (minFacePx > 0 && r.bbox) {
        const [x1, y1, x2, y2] = r.bbox;
        const faceH = y2 - y1;
        if (faceH < minFacePx) return false;
      }
      return true;
    })
    .map(r => ({
      filename: r.origName || `frame_${r.frame}`,
      frame: r.frame,
      thumbUrl: `/api/framefile/${r.frameId}`,
      scalePct: (typeof r.vertFillPct === 'number' ? r.vertFillPct : Math.sqrt(r.bboxRatio)) * 100,
      pitch: r.pitch, yaw: r.yaw, roll: r.roll,
      blur: typeof r.blur === 'number' ? r.blur : 0,
    }));

  const emptyEl = document.getElementById('scale-picker-empty');
  const controlsEl = document.getElementById('scale-picker-controls');
  if (!scalePickerPool.length) {
    emptyEl.style.display = 'block';
    controlsEl.style.display = 'none';
    return;
  }
  emptyEl.style.display = 'none';
  controlsEl.style.display = 'flex';

  const scaleVals = scalePickerPool.map(it => it.scalePct);
  const slider = document.getElementById('scale-picker-slider');
  const num = document.getElementById('scale-picker-num');
  // start centered on the pool's own average scale, same convention as the
  // pose picker's centroid start, rather than an arbitrary fixed default
  const startVal = Math.round(scaleVals.reduce((s, v) => s + v, 0) / scaleVals.length);
  slider.value = startVal;
  num.value = startVal;

  buildScalePickerCells();
  renderScalePickerGrid();
}

function updateScalePickerCellVisual(cell, it, dist) {
  cell.style.visibility = 'visible';
  const isSelected = selectedFrames.has(it.frame);
  const img = cell.querySelector('img');
  if (img.dataset.frame !== String(it.frame)) {
    img.src = it.thumbUrl;
    img.dataset.frame = it.frame;
  }
  img.title = `${it.scalePct.toFixed(0)}% frame ht (Δ${dist.toFixed(1)} from target), sharpness ${it.blur.toFixed(0)} — click to ${isSelected ? 'remove from' : 'add to'} selection`;
  img.style.borderColor = isSelected ? 'var(--accent)' : '#3a3a44';
  cell.onclick = () => {
    if (selectedFrames.has(it.frame)) selectedFrames.delete(it.frame);
    else selectedFrames.add(it.frame);
    syncSelectionVisuals();
    updateSaveSelectedButton();
    const cb = document.querySelector(`#list-body-frames .frame-select-cb[data-frame="${it.frame}"]`);
    if (cb) cb.checked = selectedFrames.has(it.frame);
    renderScalePickerGrid();
  };
}

function renderScalePickerGrid() {
  if (!scalePickerPool.length) return;
  const target = parseFloat(document.getElementById('scale-picker-slider').value);
  document.getElementById('scale-picker-num').value = target;

  const toleranceOn = document.getElementById('scale-picker-tolerance-enable').checked;
  const toleranceVal = parseFloat(document.getElementById('scale-picker-tolerance-val').value) || 10;

  let ranked = scalePickerPool
    .map(it => ({ it, dist: Math.abs(it.scalePct - target) }))
    .sort((a, b) => a.dist - b.dist);
  if (toleranceOn) {
    ranked = ranked.filter(r => r.dist <= toleranceVal);
  }
  ranked = ranked.slice(0, 9);
  const rankedFrames = ranked.map(r => r.it.frame);

  const keepSlot = scalePickerDisplayed.map(frame => frame !== null && rankedFrames.includes(frame));
  const toPlace = ranked.filter(r => !scalePickerDisplayed.includes(r.it.frame));
  let placeIdx = 0;

  for (let i = 0; i < 9; i++) {
    if (keepSlot[i]) {
      const match = ranked.find(r => r.it.frame === scalePickerDisplayed[i]);
      updateScalePickerCellVisual(scalePickerCells[i], match.it, match.dist);
    } else if (placeIdx < toPlace.length) {
      const { it, dist } = toPlace[placeIdx++];
      scalePickerDisplayed[i] = it.frame;
      updateScalePickerCellVisual(scalePickerCells[i], it, dist);
    } else {
      scalePickerDisplayed[i] = null;
      scalePickerCells[i].style.visibility = 'hidden';
    }
  }

  document.getElementById('scale-picker-count').textContent = toleranceOn
    ? `${ranked.length} within ${toleranceVal}% of target (${scalePickerPool.length} in pool)`
    : `${scalePickerPool.length} in analyzed pool`;
}

function wireScalePickerControls() {
document.getElementById('scale-picker-slider').addEventListener('input', () => {
  document.getElementById('scale-picker-num').value = document.getElementById('scale-picker-slider').value;
  renderScalePickerGrid();
});
document.getElementById('scale-picker-num').addEventListener('input', () => {
  const numEl = document.getElementById('scale-picker-num');
  const clamped = Math.max(0, Math.min(100, parseFloat(numEl.value) || 0));
  document.getElementById('scale-picker-slider').value = clamped;
  renderScalePickerGrid();
});
document.getElementById('scale-picker-tolerance-enable').addEventListener('change', (e) => {
  const input = document.getElementById('scale-picker-tolerance-val');
  input.disabled = !e.target.checked;
  input.style.color = e.target.checked ? 'var(--text)' : 'var(--dim)';
  renderScalePickerGrid();
});
document.getElementById('scale-picker-tolerance-val').addEventListener('input', renderScalePickerGrid);

document.getElementById('scale-picker-preview-btn').addEventListener('click', () => {
  const frames = scalePickerDisplayed.filter(f => f !== null);
  if (!frames.length) return;
  modalFrameOverride = frames;
  document.getElementById('selection-modal-pose-layout').checked = true;
  document.getElementById('selection-modal-spread-wrap').style.display = 'flex';
  document.getElementById('selection-modal-overlay').style.display = 'flex';
  renderSelectionModal();
});
}

const SQUEEZE_DEFAULTS = { sim: 65, yaw: 20, pitch: 20, roll: 20, blur: 65 };
let squeezeMinPct = SQUEEZE_DEFAULTS.sim;
let squeezeUserOverridden = false;
const squeezeSlider = document.getElementById('ring-squeeze-slider');
const squeezeVal = document.getElementById('ring-squeeze-val');
let sharpCutoffEnabled = false;
let sharpMinVal = 0;
const sharpEnableCb = document.getElementById('sharp-squeeze-enable');
const sharpControls = document.getElementById('sharp-squeeze-controls');
const sharpSlider = document.getElementById('sharp-squeeze-slider');
const sharpVal = document.getElementById('sharp-squeeze-val');
function wireSqueezeSharpSortControls() {
squeezeSlider.value = squeezeMinPct;

squeezeSlider.addEventListener('input', () => {
  squeezeMinPct = parseFloat(squeezeSlider.value);
  squeezeUserOverridden = true;
  if (lastVideoRingState) renderVideoRing();
});

// sharpness cutoff: independent min-blur-score filter, toggled on/off via
// its own checkbox so tightening blur doesn't force-tighten similarity too.








document.querySelectorAll('.ranked-sort-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    if (!cb.checked) return;
    rankedSortMetric = cb.value;
    if (lastNeighborsRender) {
      render(lastNeighborsRender.centerId, lastNeighborsRender.data, lastNeighborsRender.centerThumbUrl);
    }
  });
});

sharpEnableCb.addEventListener('change', () => {
  sharpCutoffEnabled = sharpEnableCb.checked;
  sharpControls.style.display = sharpCutoffEnabled ? 'flex' : 'none';
  if (lastVideoRingState) renderVideoRing();
});

// reflect whatever the checkbox's starting state is on page load (e.g. if
// the browser restored a checked checkbox from form autofill/back-forward
// cache) rather than assuming it starts unchecked.
sharpCutoffEnabled = sharpEnableCb.checked;
sharpControls.style.display = sharpCutoffEnabled ? 'flex' : 'none';

sharpSlider.addEventListener('input', () => {
  sharpMinVal = parseFloat(sharpSlider.value);
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
}

function metricValueForRing(r) {
  const raw = r.similarity;
  return typeof raw === 'number' ? raw : 0;
}

function applySqueeze(combined) {
  const cutoff = squeezeMinPct / 100;
  const simKept = combined.filter(r => {
    const sim = typeof r.similarity === 'number' ? r.similarity : (typeof r.sim === 'number' ? r.sim : 1);
    return sim >= cutoff;
  });
  squeezeVal.textContent = `${squeezeMinPct}% (${simKept.length}/${combined.length})`;

  if (!sharpCutoffEnabled) {
    sharpVal.textContent = `${sharpMinVal} (${simKept.length}/${combined.length})`;
    return simKept;
  }

  // sharpness cutoff only applies to items that actually carry a blur
  // score (video/folder analysis results); Immich-only nodes with no
  // blur field pass through untouched rather than being dropped.
  const kept = simKept.filter(r => {
    if (typeof r.blur !== 'number') return true;
    return r.blur >= sharpMinVal;
  });
  sharpVal.textContent = `${sharpMinVal} (${kept.length}/${combined.length})`;
  return kept;
}

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
  const metricLabel = { sim: 'Similarity', yaw: 'Yaw', pitch: 'Pitch', roll: 'Roll', blur: 'Sharpness' }[ringSortMetric];
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
    const unit = metric === 'blur' ? '' : '°';
    const label = metric === 'blur' ? 'sharp' : metric;
    item.innerHTML = `<img src="${thumb}" loading="lazy"><div class="plabel">${label}: ${r[metric].toFixed(1)}${unit}</div>`;
    if (typeof r.pitch === 'number') {
      // pitch up (nose up) raises the thumbnail, pitch down lowers it -
      // gives the strip a wavy "head bob" feel that mirrors the pose itself.
      const clamped = Math.max(-PITCH_CLAMP_DEG, Math.min(PITCH_CLAMP_DEG, r.pitch));
      const offsetPx = -clamped * PITCH_PX_PER_DEG;
      item.dataset.baseTransform = `translateY(${offsetPx}px)`;
      item.style.transform = item.dataset.baseTransform;
    } else {
      item.dataset.baseTransform = '';
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

