let lastNeighborsRender = null;
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
    let text = `${sim}${sim ? ' · ' : ''}pitch: ${centerResult.pitch.toFixed(1)} yaw: ${centerResult.yaw.toFixed(1)} roll: ${centerResult.roll.toFixed(1)}`;
    if (typeof centerResult.blur === 'number') text += ` · sharpness: ${centerResult.blur.toFixed(1)}`;
    if (typeof centerResult.vertFillPct === 'number') text += ` · face: ${(centerResult.vertFillPct * 100).toFixed(0)}% frame height`;
    else if (typeof centerResult.bboxRatio === 'number') text += ` · face: ${(centerResult.bboxRatio * 100).toFixed(0)}% of frame`;
    detailEl.textContent = text;
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
    const pctText = typeof r.similarity === 'number' ? `${(r.similarity * 100).toFixed(1)}%` : '';

    let poseText = '';
    if (r.pitch !== undefined && r.yaw !== undefined && r.roll !== undefined && r.pitch !== null) {
      poseText = `<br>pitch: ${r.pitch.toFixed(1)} yaw: ${r.yaw.toFixed(1)} roll: ${r.roll.toFixed(1)}`;
      if (typeof r.blur === 'number') poseText += ` · sharpness: ${r.blur.toFixed(0)}`;
      if (typeof r.vertFillPct === 'number') poseText += ` · face: ${(r.vertFillPct * 100).toFixed(0)}% frame height`;
      else if (typeof r.bboxRatio === 'number') poseText += ` · face: ${(r.bboxRatio * 100).toFixed(0)}% of frame`;
    }

    hoverCaption.innerHTML = `${r.filename}${pctText ? ` — ${pctText}` : ''}${poseText}`;
    hoverPanel.classList.add('active');
  }, 80);
}

function hideHoverPreview() {
  clearTimeout(hoverTimer);
  hoverPanel.classList.remove('active');
}

const selectedFrames = new Set();
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
    native: document.getElementById('export-native').checked,
  };
}

function wireMatchResSummaryBtn() {
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

  const nativeCb = document.getElementById('export-native');
  const widthInput = document.getElementById('export-width');
  const heightInput = document.getElementById('export-height');
  function syncNativeState() {
    const on = nativeCb.checked;
    // native mode still uses width/height as an aspect ratio for face/center
    // crop modes, so keep them enabled but dim them to signal the meaning
    // shifted from 'exact output size' to 'output aspect ratio'.
    [widthInput, heightInput].forEach(el => { el.style.opacity = on ? '0.55' : '1'; });
    document.querySelectorAll('.export-preset-btn').forEach(btn => { btn.style.opacity = on ? '0.4' : '1'; });
    const matchBtn = document.getElementById('match-source-res-btn');
    if (matchBtn) matchBtn.style.display = on ? 'none' : matchBtn.style.display;
  }
  nativeCb.addEventListener('change', () => { syncNativeState(); applyResolutionSummary(); });
  syncNativeState();
})();
}

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

function wireSelectAllDeselectAllBtns() {
document.querySelectorAll('.section-select-all-btn').forEach(btn => {
  btn.addEventListener('click', () => selectAllInSection(btn.dataset.target));
});
document.querySelectorAll('.section-deselect-all-btn').forEach(btn => {
  btn.addEventListener('click', () => deselectAllInSection(btn.dataset.target));
});
}

function findFrameResult(frame) {
  if (!lastVideoRingState) return null;
  return lastVideoRingState.baseResults.find(r => r.frame === frame)
      || extraImmichNodes.find(r => r.frame === frame)
      || null;
}

let modalFrameOverride = null;
function openSelectionModal() {
  modalFrameOverride = null;
  document.getElementById('selection-modal-undo-wrap').style.display = 'none';
  document.getElementById('selection-modal-overlay').style.display = 'flex';
  renderSelectionModal();
}

function closeSelectionModal() {
  document.getElementById('selection-modal-overlay').style.display = 'none';
  modalFrameOverride = null;
}

function refreshSelectionModalIfOpen() {
  const overlay = document.getElementById('selection-modal-overlay');
  if (overlay && overlay.style.display === 'flex') renderSelectionModal();
}

function renderSelectionModal() {
  const grid = document.getElementById('selection-modal-grid');
  const countEl = document.getElementById('selection-modal-count');
  const titleEl = document.getElementById('selection-modal-title');
  const realPreview = document.getElementById('selection-modal-real-preview').checked;
  const poseLayout = document.getElementById('selection-modal-pose-layout').checked;
  grid.innerHTML = '';

  const isPickerPreview = Array.isArray(modalFrameOverride);
  const assetItems = isPickerPreview ? [] : Array.from(selectedAssetIds).map(id => {
    const known = [...(lastVideoRingState ? lastVideoRingState.baseResults : []), ...extraImmichNodes]
      .find(r => r.assetId === id);
    const cached = assetPoseCache[id];
    return {
      kind: 'asset', assetId: id, filename: known ? known.filename : id, thumb: `/api/thumb/${id}`,
      pitch: cached ? cached.pitch : (known && typeof known.pitch === 'number' ? known.pitch : null),
      yaw: cached ? cached.yaw : (known && typeof known.yaw === 'number' ? known.yaw : null),
      blur: cached ? cached.blur : (known && typeof known.blur === 'number' ? known.blur : null),
      vertFillPct: cached ? cached.vertFillPct : (known && typeof known.vertFillPct === 'number' ? known.vertFillPct : null),
    };
  });
  const frameSource = isPickerPreview ? modalFrameOverride : Array.from(selectedFrames);
  const frameItems = frameSource.map(frame => {
    const r = findFrameResult(frame);
    return {
      kind: 'frame', frame, filename: r ? r.filename : `frame ${frame}`, thumb: r ? thumbUrlFor(r) : '',
      pitch: r && typeof r.pitch === 'number' ? r.pitch : null,
      yaw: r && typeof r.yaw === 'number' ? r.yaw : null,
    };
  });
  const items = [...assetItems, ...frameItems];
  if (titleEl) {
    titleEl.innerHTML = isPickerPreview
      ? `Pose Picker preview — <span id="selection-modal-count">${items.length}</span> nearest to target. Not part of your export selection yet.`
      : `Export selection — <span id="selection-modal-count">${items.length}</span> item(s). Double-click to remove.`;
  }
  if (!items.length) {
    grid.innerHTML = isPickerPreview
      ? '<div style="grid-column:1/-1;color:var(--dim);font-size:var(--fs-11);text-align:center;padding:20px;">Nothing on stage in the Pose Picker yet.</div>'
      : '<div style="grid-column:1/-1;color:var(--dim);font-size:var(--fs-11);text-align:center;padding:20px;">Nothing selected yet.</div>';
    return;
  }

  // real crop preview reuses the exact same crop_resize_export call the
  // actual export does - what you see here is genuinely what gets written,
  // not a generic square thumbnail unrelated to crop mode/margin/native
  const params = new URLSearchParams(getExportParams());
  const jobId = lastVideoRingState ? lastVideoRingState.jobId : null;

  function srcFor(it) {
    if (!realPreview) return it.thumb;
    if (it.kind === 'frame' && jobId) return `/api/export-preview/${jobId}/${it.frame}?${params.toString()}`;
    if (it.kind === 'asset') return `/api/export-preview-immich/${it.assetId}?${params.toString()}`;
    return it.thumb;
  }

  function isItemSelected(it) {
    return it.kind === 'asset' ? selectedAssetIds.has(it.assetId) : selectedFrames.has(it.frame);
  }

  function toggleItem(it) {
    if (it.kind === 'asset') {
      if (selectedAssetIds.has(it.assetId)) selectedAssetIds.delete(it.assetId);
      else selectedAssetIds.add(it.assetId);
      updateImmichSelectionBar();
    } else {
      if (selectedFrames.has(it.frame)) selectedFrames.delete(it.frame);
      else selectedFrames.add(it.frame);
      updateSaveSelectedButton();
    }
    const cb = document.querySelector(`#list-body-frames .frame-select-cb[data-frame="${it.frame}"], #list-body-immich .asset-select-cb[data-asset-id="${it.assetId}"]`);
    if (cb) cb.checked = isItemSelected(it);
    syncSelectionVisuals();
  }

  function removeItem(it) {
    if (it.kind === 'asset') {
      selectedAssetIds.delete(it.assetId);
      updateImmichSelectionBar();
    } else {
      selectedFrames.delete(it.frame);
      updateSaveSelectedButton();
    }
    document.querySelectorAll('.pose-list-item.selected, .node.export-selected').forEach(el => {
      if (el.dataset.filename === it.filename) el.classList.remove('selected', 'export-selected');
    });
    // a manual edit after a deselect-all makes that undo snapshot stale
    // (it no longer matches "everything before you cleared it") - hide it
    // rather than let it silently restore items you've since removed again
    document.getElementById('selection-modal-undo-wrap').style.display = 'none';
    renderSelectionModal();
  }

  // in picker-preview mode there's nothing to "remove" (these 9 are fixed
  // as the picker's current stage) - clicking just toggles export
  // selection instead, same as the picker's own live grid
  const onInteract = isPickerPreview
    ? (it) => { toggleItem(it); renderSelectionModal(); }
    : removeItem;

  if (poseLayout) {
    renderSelectionModalPoseScatter(grid, items, srcFor, onInteract, realPreview, isPickerPreview, isItemSelected);
  } else {
    renderSelectionModalGrid(grid, items, srcFor, onInteract, realPreview, isPickerPreview, isItemSelected);
  }
}

function renderSelectionModalGrid(grid, items, srcFor, onInteract, realPreview, isPickerPreview, isItemSelected) {
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = 'repeat(auto-fill,minmax(84px,1fr))';
  grid.style.position = 'static';
  grid.style.height = 'auto';
  grid.style.backgroundImage = 'none';

  items.forEach(it => {
    const selected = isPickerPreview && isItemSelected(it);
    const cell = document.createElement('div');
    cell.style.textAlign = 'center';
    cell.style.cursor = 'pointer';
    cell.innerHTML = `
      <img src="${srcFor(it)}" loading="lazy" title="${isPickerPreview ? `Click to ${selected ? 'remove from' : 'add to'} export selection` : 'Double-click to remove'}"
           style="width:100%;${realPreview ? '' : 'aspect-ratio:1;'}object-fit:${realPreview ? 'contain' : 'cover'};border-radius:6px;border:2px solid ${isPickerPreview ? (selected ? 'var(--accent)' : '#3a3a44') : 'var(--accent)'};display:block;background:#0a0a0d;">
      <div style="font-size:var(--fs-9);color:var(--dim);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${it.filename}</div>
    `;
    if (isPickerPreview) {
      cell.onclick = () => onInteract(it);
    } else {
      cell.ondblclick = () => onInteract(it);
    }
    // reuse the same rollover preview panel the ring/list views use - it's
    // position:fixed with its own z-index, so it just needs to sit above
    // the modal overlay (handled via z-index in style.css). Modal items
    // carry a `.thumb` field, not `.thumbUrl` (what previewUrlFor looks
    // for), and frame-kind items have no assetId for its fallback either
    // - so pass the already-computed srcFor() URL explicitly as thumbUrl
    // rather than relying on previewUrlFor's defaults.
    cell.addEventListener('mouseenter', () => showHoverPreview({ ...it, thumbUrl: srcFor(it) }));
    cell.addEventListener('mouseleave', hideHoverPreview);
    grid.appendChild(cell);
  });
}

const POSE_SCATTER_SPREAD_BASE = 42;
function applyPoseScatterSpread() {
  const spread = parseFloat(document.getElementById('selection-modal-spread').value) || 1;
  document.querySelectorAll('.pose-scatter-item').forEach(cell => {
    const yawRatio = parseFloat(cell.dataset.yawRatio) || 0;
    const pitchRatio = parseFloat(cell.dataset.pitchRatio) || 0;
    const leftPct = 50 + yawRatio * POSE_SCATTER_SPREAD_BASE * spread;
    const topPct = 50 + pitchRatio * POSE_SCATTER_SPREAD_BASE * spread;
    cell.style.left = `${Math.max(4, Math.min(96, leftPct))}%`;
    cell.style.top = `${Math.max(4, Math.min(96, topPct))}%`;
  });
}

function wirePoseScatterSpreadSlider() {
document.getElementById('selection-modal-spread').addEventListener('input', applyPoseScatterSpread);
}

function renderSelectionModalPoseScatter(grid, items, srcFor, onInteract, realPreview, isPickerPreview, isItemSelected) {
  const posed = items.filter(it => it.pitch !== null && it.yaw !== null);
  const unposed = items.filter(it => it.pitch === null || it.yaw === null);

  grid.style.display = 'block';
  grid.style.position = 'relative';
  grid.style.gridTemplateColumns = '';
  const stageHeight = 480;
  grid.style.height = unposed.length ? `${stageHeight + 110}px` : `${stageHeight}px`;
  grid.style.backgroundImage =
    'linear-gradient(#22222a 1px, transparent 1px), linear-gradient(90deg, #22222a 1px, transparent 1px)';
  grid.style.backgroundSize = '40px 40px';
  grid.style.backgroundPosition = 'center center';

  if (!posed.length) {
    const detectableCount = unposed.filter(it => it.kind === 'asset').length;
    grid.innerHTML = `
      <div style="color:var(--dim);font-size:var(--fs-11);text-align:center;padding:20px;">
        No pose data on the current selection — pitch/yaw only comes from the pose-analysis pipeline (analyze video/folder/Immich selection).<br><br>
        ${detectableCount ? `<button id="detect-pose-btn" style="padding:6px 12px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:var(--fs-11);">Detect pose for ${detectableCount} Immich item(s)</button>` : ''}
      </div>`;
    if (detectableCount) {
      document.getElementById('detect-pose-btn').addEventListener('click', () => detectPoseForItems(unposed.filter(it => it.kind === 'asset')));
    }
    return;
  }

  // find the item closest to this selection's own pose centroid, not
  // absolute (0,0) - if the whole set is consistently turned/tilted (e.g.
  // camera wasn't dead-on for this sequence), anchoring on absolute zero
  // picks an unrepresentative outlier and everything else huddles in one
  // corner relative to it. anchoring on the actual centroid keeps the
  // layout balanced around where the data genuinely sits.
  const meanPitch = posed.reduce((s, it) => s + it.pitch, 0) / posed.length;
  const meanYaw = posed.reduce((s, it) => s + it.yaw, 0) / posed.length;
  let anchor = posed[0];
  let bestScore = Infinity;
  posed.forEach(it => {
    const score = Math.abs(it.pitch - meanPitch) + Math.abs(it.yaw - meanYaw);
    if (score < bestScore) { bestScore = score; anchor = it; }
  });

  const deltaYaw = it => it.yaw - anchor.yaw;
  const deltaPitch = it => it.pitch - anchor.pitch;
  const yawMax = Math.max(15, ...posed.map(it => Math.abs(deltaYaw(it))));
  const pitchMax = Math.max(15, ...posed.map(it => Math.abs(deltaPitch(it))));
  const stage = document.createElement('div');
  stage.style.position = 'relative';
  stage.style.width = '100%';
  stage.style.height = `${stageHeight}px`;

  const spread = parseFloat(document.getElementById('selection-modal-spread').value) || 1;

  posed.forEach(it => {
    const isAnchor = it === anchor;
    // pitch is inverted here to match the pose-list strip's convention:
    // positive pitch (nose up) moves toward the top of the stage, negative
    // (nose down) moves toward the bottom - matches how up/down actually
    // reads visually, confirmed against a real sample (p-18 placed upper,
    // should be lower since -18 is nose-down)
    const yawRatio = isAnchor ? 0 : deltaYaw(it) / yawMax;
    const pitchRatio = isAnchor ? 0 : -deltaPitch(it) / pitchMax;
    const leftPct = 50 + yawRatio * POSE_SCATTER_SPREAD_BASE * spread;
    const topPct = 50 + pitchRatio * POSE_SCATTER_SPREAD_BASE * spread;
    const size = isAnchor ? 108 : 76;

    const cell = document.createElement('div');
    cell.className = 'pose-scatter-item';
    cell.style.position = 'absolute';
    // yawRatio/pitchRatio kept on the cell so the Spread slider can just
    // recompute left/top directly (no rebuild, no image reload/refetch -
    // matters a lot with real-crop-preview on, since that hits the network
    // per Immich asset)
    cell.dataset.yawRatio = yawRatio;
    cell.dataset.pitchRatio = pitchRatio;
    cell.style.left = `${Math.max(4, Math.min(96, leftPct))}%`;
    cell.style.top = `${Math.max(4, Math.min(96, topPct))}%`;
    cell.dataset.baseTransform = 'translate(-50%,-50%)';
    cell.dataset.baseZ = isAnchor ? '5' : '2';
    cell.style.transform = cell.dataset.baseTransform;
    cell.style.zIndex = cell.dataset.baseZ;
    cell.style.width = `${size}px`;
    cell.style.textAlign = 'center';
    cell.style.cursor = 'pointer';
    const selected = isPickerPreview && isItemSelected(it);
    const borderColor = isPickerPreview ? (selected ? 'var(--accent)' : '#3a3a44') : (isAnchor ? 'var(--accent)' : '#3a3a44');
    cell.innerHTML = `
      <img src="${srcFor(it)}" loading="lazy" title="pitch ${it.pitch.toFixed(1)}, yaw ${it.yaw.toFixed(1)}${isAnchor ? ' (closest to this selection\'s pose centroid, not necessarily true zero)' : ` — ${deltaPitch(it) >= 0 ? '+' : ''}${deltaPitch(it).toFixed(1)}p / ${deltaYaw(it) >= 0 ? '+' : ''}${deltaYaw(it).toFixed(1)}y from center`}${isPickerPreview ? ` — click to ${selected ? 'remove from' : 'add to'} export selection` : ' — double-click to remove'}"
           style="width:${size}px;height:${size}px;${realPreview ? 'object-fit:contain;background:#0a0a0d;' : 'object-fit:cover;'}border-radius:8px;
                  border:${isAnchor ? '3px' : '2px'} solid ${borderColor};
                  box-shadow:${isAnchor ? '0 0 20px rgba(124,196,255,0.35)' : 'none'};display:block;">
      <div style="font-size:var(--fs-8);color:var(--dim);margin-top:2px;">${isAnchor ? `center (p${it.pitch.toFixed(0)} y${it.yaw.toFixed(0)})` : `p${it.pitch.toFixed(0)} y${it.yaw.toFixed(0)}`}</div>
    `;
    if (isPickerPreview) {
      cell.onclick = () => onInteract(it);
    } else {
      cell.ondblclick = () => onInteract(it);
    }
    cell.addEventListener('mouseenter', () => showHoverPreview({ ...it, thumbUrl: srcFor(it) }));
    cell.addEventListener('mouseleave', hideHoverPreview);
    stage.appendChild(cell);
  });

  grid.appendChild(stage);
  // reuse the same dock-style magnify used on the pose-list strip - genuinely
  // overlapping thumbnails at similar pitch/yaw are otherwise impossible to
  // pick apart, same problem the strip already solved
  attachLensEffect(stage, '.pose-scatter-item', { radius: 90, maxScale: 1.8 });

  if (unposed.length) {
    const strip = document.createElement('div');
    strip.style.marginTop = '8px';
    strip.style.paddingTop = '8px';
    strip.style.borderTop = '1px solid #26262e';
    const detectableCount = unposed.filter(it => it.kind === 'asset').length;
    strip.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
        <span style="font-size:var(--fs-9);color:var(--dim);">No pose data (${unposed.length}):</span>
        ${detectableCount ? `<button id="detect-pose-btn" style="padding:3px 8px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:5px;cursor:pointer;font-size:var(--fs-9);">Detect pose (${detectableCount})</button>` : ''}
      </div>`;
    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.flexWrap = 'wrap';
    row.style.gap = '8px';
    unposed.forEach(it => {
      const cell = document.createElement('div');
      cell.style.width = '64px';
      cell.style.textAlign = 'center';
      cell.style.cursor = 'pointer';
      cell.innerHTML = `
        <img src="${srcFor(it)}" loading="lazy" title="${isPickerPreview ? 'Click to toggle export selection' : 'Double-click to remove'}"
             style="width:64px;height:64px;object-fit:cover;border-radius:6px;border:2px solid ${isPickerPreview && isItemSelected(it) ? 'var(--accent)' : '#3a3a44'};display:block;">
      `;
      if (isPickerPreview) {
        cell.onclick = () => onInteract(it);
      } else {
        cell.ondblclick = () => onInteract(it);
      }
      row.appendChild(cell);
    });
    strip.appendChild(row);
    grid.appendChild(strip);
    if (detectableCount) {
      document.getElementById('detect-pose-btn').addEventListener('click', () => detectPoseForItems(unposed.filter(it => it.kind === 'asset')));
    }
  }
}

async function detectPoseForItems(items) {
  const btn = document.getElementById('detect-pose-btn');
  if (btn) { btn.textContent = `Detecting 0/${items.length}…`; btn.disabled = true; }
  let done = 0;
  await Promise.all(items.map(async it => {
    try {
      const res = await fetch(`/api/asset-face-pose/${it.assetId}`);
      const data = await res.json();
      if (!data.error) {
        // this endpoint now returns the full metric set (pose + blur +
        // vertFillPct), same as a video/folder analysis frame would -
        // cache all of it, not just pose, so sharpness sort/filtering
        // works on on-demand-detected Immich items too.
        assetPoseCache[it.assetId] = {
          pitch: data.pitch, yaw: data.yaw,
          blur: data.blur, vertFillPct: data.vertFillPct,
        };
      }
    } catch (e) {
      console.warn('Pose detection failed for', it.assetId, e);
    } finally {
      done++;
      if (btn) btn.textContent = `Detecting ${done}/${items.length}…`;
    }
  }));
  renderSelectionModal();
}

function wireSelectionModalOpeners1() {
document.getElementById('selection-modal-real-preview').addEventListener('change', renderSelectionModal);
document.getElementById('selection-modal-pose-layout').addEventListener('change', (e) => {
  document.getElementById('selection-modal-spread-wrap').style.display = e.target.checked ? 'flex' : 'none';
  renderSelectionModal();
});
}

let lastDeselectedFrames = null;
let lastDeselectedAssetIds = null;
function wireSelectionModalOpeners2() {
document.getElementById('selection-modal-deselect-all').addEventListener('click', () => {
  const totalCleared = selectedFrames.size + selectedAssetIds.size;
  if (totalCleared === 0) return;
  lastDeselectedFrames = new Set(selectedFrames);
  lastDeselectedAssetIds = new Set(selectedAssetIds);
  selectedFrames.clear();
  selectedAssetIds.clear();
  syncSelectionVisuals();
  updateSaveSelectedButton();
  updateImmichSelectionBar();
  renderSelectionModal();

  const undoWrap = document.getElementById('selection-modal-undo-wrap');
  document.getElementById('selection-modal-undo-count').textContent = totalCleared;
  undoWrap.style.display = 'inline-flex';
});

document.getElementById('selection-modal-undo').addEventListener('click', () => {
  if (lastDeselectedFrames) lastDeselectedFrames.forEach(f => selectedFrames.add(f));
  if (lastDeselectedAssetIds) lastDeselectedAssetIds.forEach(a => selectedAssetIds.add(a));
  lastDeselectedFrames = null;
  lastDeselectedAssetIds = null;
  syncSelectionVisuals();
  updateSaveSelectedButton();
  updateImmichSelectionBar();
  renderSelectionModal();
  document.getElementById('selection-modal-undo-wrap').style.display = 'none';
});
}

function wireSelectionModalClose() {
document.getElementById('selection-modal-close').addEventListener('click', closeSelectionModal);
document.getElementById('selection-modal-overlay').addEventListener('click', (e) => {
  if (e.target.id === 'selection-modal-overlay') closeSelectionModal();
});
}

function wireSelectionModalOpenBtn() {
document.getElementById('immich-view-selected-btn').addEventListener('click', openSelectionModal);
}

function wireExportParamsForm() {
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
}

function wireImmichSearchModal() {
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
          <div class="fname">${p.name}${tight ? ' <span style="color:#d4a544;font-size:var(--fs-9);">● tight cluster</span>' : ''}</div>
          <div class="simbar-track"><div class="simbar-fill" style="width:${pct}%"></div></div>
        </div>
        <div class="simpct">${pct}%</div>
        <div style="font-size:var(--fs-10);color:var(--dim);margin-left:6px;">${p.faceCount}</div>
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
}

function wireImmichClearSelectedBtn() {
document.getElementById('immich-clear-selected-btn').addEventListener('click', () => {
  selectedAssetIds.clear();
  updateImmichSelectionBar();
  if (lastVideoRingState) renderVideoRing();
  else {
    document.querySelectorAll('.asset-select-cb').forEach(cb => { cb.checked = false; });
  }
});
}

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

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimer = null;
function wireSearchInputAndNeighbors() {
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
}

function wireMiscBlock1() {
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
}

function wireMiscBlock2() {
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
}

function wireSidebarResizeHandle() {
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
}

let lastVideoRingState = null;
let lastImmichAnalysisAssetIds = null;
const extraImmichNodes = [];
const assetPoseCache = {};
let ringSortMetric = 'sim';
function wireFindNeutralBtn() {
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
    <button type="button" id="use-as-reference-btn" style="display:block;width:100%;margin-top:6px;font-size:var(--fs-10);padding:5px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:4px;cursor:pointer;">Use as reference &amp; re-analyze</button>`;
  flashHighlightFrame(best);

  document.getElementById('use-as-reference-btn').onclick = () => {
    const btn = document.getElementById('use-as-reference-btn');
    const sourceType = lastVideoRingState.sourceType;
    // this readout describes the *old* anchor's neutral-pose stats, which
    // are invalidated by the re-analysis it's about to trigger - hide it
    // now rather than leaving a stale "Re-analyzing…" button behind once
    // the new job finishes (nothing else resets this div on completion).
    const hideReadout = () => { readout.style.display = 'none'; readout.innerHTML = ''; };
    if (sourceType === 'folder') {
      if (!lastFolderSource) {
        btn.textContent = 'Original folder/zip no longer available — reload it first';
        return;
      }
      hideReadout();
      startFolderAnalysis(lastFolderSource, best.frame);
    } else if (sourceType === 'immich') {
      if (!lastImmichAnalysisAssetIds) {
        btn.textContent = 'Original Immich selection no longer available — re-select and analyze again';
        return;
      }
      hideReadout();
      reanalyzeImmichSelection(lastImmichAnalysisAssetIds, best.frame);
    } else {
      if (!currentVideoFile) {
        btn.textContent = 'Original video no longer available — reload it first';
        return;
      }
      hideReadout();
      startVideoAnalysis(currentVideoFile, best.frame);
    }
  };
});
}

function bootstrap() {
  wireRenderPatch();
  wireMatchResSummaryBtn();
  wireSelectAllDeselectAllBtns();
  wirePoseScatterSpreadSlider();
  wireSelectionModalOpeners1();
  wireSelectionModalOpeners2();
  wireSelectionModalClose();
  wireSelectionModalOpenBtn();
  wireExportParamsForm();
  wireImmichSearchModal();
  wireImmichClearSelectedBtn();
  wireImmichAnalyzeSelectedBtn();
  wireFisheyeLensMouseMove();
  wireAttachLensEffectCall();
  init(); // original app init (unchanged)
  wireSearchInputAndNeighbors();
  wireMiscBlock1();
  wireMiscBlock2();
  wireSidebarResizeHandle();
  wireMatchSourceResBtn();
  wireDropzoneAndFileInput();
  wireFramePlaybackControls();
  wireStartAnalysisBtn();
  wirePosePickerControls();
  wireScalePickerControls();
  wireFolderInput();
  wireSqueezeSharpSortControls();
  wireFindNeutralBtn();
}

bootstrap();
