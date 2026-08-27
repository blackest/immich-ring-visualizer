async function loadNeighbors(assetId) {
  stage.innerHTML = '<div id="loading">loading neighbors…</div>';
  const res = await fetch(`/api/neighbors?assetId=${assetId}&limit=36`);
  const data = await res.json();
  render(assetId, data);
}

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
        cacheFormat: document.getElementById('cache-format-png').checked ? 'png' : 'jpg',
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

function wireImmichAnalyzeSelectedBtn() {
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
}

const dropZone = document.getElementById('video-drop-zone');
const fileInput = document.getElementById('video-file-input');
const previewContainer = document.getElementById('preview-container');
const previewCanvas = document.getElementById('preview-canvas');
const hiddenVideo = document.getElementById('hidden-video');
const frameCounter = document.getElementById('frame-counter');
let lastResolutionSummary = null;
function applyResolutionSummary(summary) {
  if (summary !== undefined) lastResolutionSummary = summary;
  summary = lastResolutionSummary;
  const note = document.getElementById('source-resolution-note');
  const matchBtn = document.getElementById('match-source-res-btn');
  if (!summary) {
    note.style.display = 'none';
    matchBtn.style.display = 'none';
    return;
  }
  const nativeOn = document.getElementById('export-native').checked;
  const exportW = parseInt(document.getElementById('export-width').value, 10);
  const exportH = parseInt(document.getElementById('export-height').value, 10);
  const downsampling = !nativeOn && (summary.modeWidth > exportW || summary.modeHeight > exportH);

  let text;
  if (summary.uniform) {
    text = `Source: all ${summary.totalCount} images are ${summary.modeWidth}×${summary.modeHeight}`;
  } else {
    text = `Source: mostly ${summary.modeWidth}×${summary.modeHeight} (${summary.modeCount}/${summary.totalCount}), range ${summary.minWidth}–${summary.maxWidth} × ${summary.minHeight}–${summary.maxHeight}`;
  }
  if (downsampling) {
    text += ` — <span style="color:#e0a94a;">exporting at ${exportW}×${exportH} throws away resolution — tick "native resolution" below to keep it</span>`;
  } else if (nativeOn) {
    text += ` — native resolution export is on, source pixels are kept`;
  }
  note.innerHTML = text;
  note.style.display = 'block';
  matchBtn.style.display = (!nativeOn && (downsampling || exportW !== summary.modeWidth || exportH !== summary.modeHeight)) ? 'inline-block' : 'none';
  matchBtn.dataset.w = summary.modeWidth;
  matchBtn.dataset.h = summary.modeHeight;
}

function wireMatchSourceResBtn() {
document.getElementById('match-source-res-btn').addEventListener('click', function () {
  document.getElementById('export-width').value = this.dataset.w;
  document.getElementById('export-height').value = this.dataset.h;
  this.style.display = 'none';
});
}

const videoStatus = document.getElementById('video-status');
let currentVideoFile = null;
let currentPreviewId = null;
let currentFrameIdx = 1;
let totalFrames = 1;
let fps = 0;
function wireDropzoneAndFileInput() {
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
}

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

const SHOT_SCALE_BANDS = [
  { max: 0.05, label: 'Extreme wide',      color: '#8a5fd4' },
  { max: 0.12, label: 'Full shot',         color: '#4a7fd4' },
  { max: 0.20, label: 'Cowboy/American',   color: '#4ad4c4' },
  { max: 0.25, label: 'Medium',            color: '#4ad46a' },
  { max: 0.35, label: 'Medium close-up',   color: '#d4c04a' },
  { max: 0.50, label: 'Close-up',          color: '#d4824a' },
  { max: Infinity, label: 'Extreme close-up', color: '#d4544a' },
];
function shotScaleFor(r) {
  const pct = typeof r.vertFillPct === 'number' ? r.vertFillPct
    : (typeof r.bboxRatio === 'number' ? Math.sqrt(r.bboxRatio) : null);
  if (pct === null) return null;
  const band = SHOT_SCALE_BANDS.find(b => pct <= b.max) || SHOT_SCALE_BANDS[SHOT_SCALE_BANDS.length - 1];
  return { pct, ...band };
}

function renderSimSparkline(results, threshold, blurThreshold, sourceType) {
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

  // blur (Laplacian variance) is unbounded, not 0-1 like similarity, so its
  // cutoff line needs its own scale to sit at a meaningful height - derived
  // from the actual blur values in this batch, not the trace itself (we
  // only show where the cutoff falls, not a full blur trace)
  const blurVals = sorted.map(r => typeof r.blur === 'number' ? r.blur : 0);
  const blurMax = Math.max(1, blurThreshold * 1.4, ...blurVals) * 1.05;
  const yForBlur = (blur) => padT + (1 - Math.max(0, blur) / blurMax) * plotH;

  const linePoints = sorted.map(r => `${xFor(r.frame).toFixed(1)},${yFor(r.sim).toFixed(1)}`).join(' ');
  const thresholdY = yFor(threshold).toFixed(1);
  const blurThresholdY = yForBlur(blurThreshold).toFixed(1);

  const dots = sorted.map(r => {
    const cx = xFor(r.frame).toFixed(1);
    const cy = yFor(r.sim).toFixed(1);
    const color = r.passed ? '#7cc4ff' : '#d9534f';
    return `<circle cx="${cx}" cy="${cy}" r="7" fill="transparent" data-frame="${r.frame}" class="spark-hit" style="cursor:pointer;"></circle>` +
           `<circle cx="${cx}" cy="${cy}" r="2" fill="${color}" style="pointer-events:none;"></circle>`;
  }).join('');

  // shot-scale strip: a thin false-color band under the main chart, one
  // segment per frame, colored by which fixed shot-scale bucket that
  // frame's face-fill % falls into - so a glance at the strip shows the
  // rough shape of the sequence (wide -> push in -> hold close -> ...)
  // without scrubbing through frame by frame. Frames with no scale data
  // (e.g. no bbox recorded) render as a neutral gray gap rather than being
  // skipped, so gaps in coverage are visible too, not silently hidden.
  const stripH = 14;
  const segW = Math.max(1, plotW / sorted.length);
  const stripSegs = sorted.map(r => {
    const scale = shotScaleFor(r);
    const x = xFor(r.frame).toFixed(1);
    const color = scale ? scale.color : '#2a2a32';
    const title = scale ? `${scale.label} (${(scale.pct * 100).toFixed(0)}%)` : 'no scale data';
    return `<rect x="${(x - segW / 2).toFixed(1)}" y="0" width="${segW.toFixed(1)}" height="${stripH}" fill="${color}" data-frame="${r.frame}" class="spark-hit strip-seg" style="cursor:pointer;"><title>frame ${r.frame} — ${title}</title></rect>`;
  }).join('');

  const legend = SHOT_SCALE_BANDS.map(b =>
    `<span style="display:inline-flex;align-items:center;gap:3px;margin-right:8px;"><span style="width:8px;height:8px;border-radius:2px;background:${b.color};display:inline-block;"></span>${b.label}</span>`
  ).join('');

  wrap.innerHTML = `
    <div style="font-size:var(--fs-10);color:var(--dim);margin-bottom:3px;display:flex;justify-content:space-between;">
      <span>Match confidence by frame — click a point to jump the preview</span>
      <span style="color:#d4c04a;">·· blur cutoff (${blurThreshold})</span>
    </div>
    <svg width="${W}" height="${H}" style="background:#0c0c10;border:1px solid #22222a;border-radius:6px;display:block;">
      <line x1="${padL}" y1="${thresholdY}" x2="${W - padR}" y2="${thresholdY}"
            stroke="#4a4a55" stroke-width="1" stroke-dasharray="3,3"></line>
      <line x1="${padL}" y1="${blurThresholdY}" x2="${W - padR}" y2="${blurThresholdY}"
            stroke="#d4c04a" stroke-width="1" stroke-dasharray="1,3" opacity="0.8"></line>
      <polyline points="${linePoints}" fill="none" stroke="#5a8fc4" stroke-width="1.5"></polyline>
      ${dots}
    </svg>
    <div style="font-size:var(--fs-9);color:var(--dim);margin-top:4px;">Shot scale by frame (face height % of frame)</div>
    <svg width="${W}" height="${stripH}" style="background:#0c0c10;border:1px solid #22222a;border-radius:4px;display:block;margin-top:2px;">
      ${stripSegs}
    </svg>
    <div style="font-size:var(--fs-8);color:var(--dim);margin-top:4px;line-height:1.6;">${legend}</div>
  `;

  wrap.querySelectorAll('.spark-hit').forEach(el => {
    el.addEventListener('click', () => {
      const frame = parseInt(el.dataset.frame, 10);
      if (sourceType === 'folder' || sourceType === 'immich') {
        // these "frame" numbers are just a sequential index over a batch of
        // stills (alphabetical filename order for folder jobs, or however
        // the Immich selection was ordered) - there's no actual video to
        // scrub, so seekToFrame()/syncAudioToFrame() are the wrong tool
        // here and silently no-op (they bail out on !currentPreviewId).
        // Show the already-exported still directly instead.
        const r = sorted.find(x => x.frame === frame);
        showStaticFramePreview(r);
      } else {
        stopFramePlayback();
        seekToFrame(frame);
        syncAudioToFrame(frame);
      }
      previewContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  });
}

function showStaticFramePreview(r) {
  if (!r) return;
  frameCounter.textContent = r.filename ? r.filename : `frame ${r.frame}`;
  if (!r.frameId) {
    videoStatus.textContent = `Frame ${r.frame} was rejected (no face / didn't pass thresholds) - no stored image to show.`;
    return;
  }
  const img = new Image();
  img.onload = () => {
    previewCanvas.width = img.naturalWidth;
    previewCanvas.height = img.naturalHeight;
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
  };
  img.onerror = () => {
    videoStatus.textContent = `Could not load stored image for frame ${r.frame}`;
  };
  img.src = `/api/framefile/${r.frameId}?t=${Date.now()}`;
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

function wireFramePlaybackControls() {
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
}

function wireStartAnalysisBtn() {
document.getElementById('btn-start-analysis').addEventListener('click', () => {
  if (currentVideoFile) {
    startVideoAnalysis(currentVideoFile, currentFrameIdx);
  }
});
}

async function startVideoAnalysis(videoFile, refFrameIdx) {
  selectedFrames.clear();
  videoStatus.textContent = 'Uploading for analysis…';
  const form = new FormData();
  form.append('video', videoFile);
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refFrame', refFrameIdx);
  form.append('cacheFormat', document.getElementById('cache-format-png').checked ? 'png' : 'jpg');
  const startSecVal = document.getElementById('analysis-start-sec').value;
  const endSecVal = document.getElementById('analysis-end-sec').value;
  if (startSecVal !== '') form.append('startSec', startSecVal);
  if (endSecVal !== '') form.append('endSec', endSecVal);

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

  applyResolutionSummary(data.resolutionSummary);
  setupPosePicker(data.results, jobId, sourceType);
  setupScalePicker(data.results, jobId, sourceType);

  statusEl.innerHTML = `
    Done — ${passed}/${data.frameCount} kept.
    <div id="sim-sparkline-wrap" style="margin-top:8px;"></div>
    <button id="export-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:var(--fs-11);">Save kept frames to disk</button>
    <button id="save-selected-btn" disabled style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:var(--fs-11);opacity:0.5;">Save 0 selected frames to disk</button>
    <button id="view-selected-btn" style="margin-top:6px;width:100%;padding:6px;background:#16161c;border:1px solid #2a2a32;color:var(--text);border-radius:6px;cursor:pointer;font-size:var(--fs-11);">View selected</button>
    ${sourceType === 'folder' || sourceType === 'immich' ? '' : `
    <button id="playback-btn" style="margin-top:6px;width:100%;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:var(--fs-11);">Build playback (rejected frames blanked)</button>
    <video id="playback-video" controls style="width:100%;margin-top:8px;display:none;border-radius:6px;"></video>
    <div id="playback-frame-controls" style="display:none;margin-top:6px;gap:6px;">
      <button id="playback-prev-frame" class="btn-seek" style="flex:1;">◀ -1 frame</button>
      <button id="playback-next-frame" class="btn-seek" style="flex:1;">+1 frame ▶</button>
    </div>
    `}
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #22222a;">
      <div id="crosscheck-header" style="display:flex;align-items:center;gap:6px;cursor:pointer;font-size:var(--fs-11);color:var(--dim);margin-bottom:8px;user-select:none;">
        <span class="chev" id="crosscheck-chev" style="display:inline-block;transition:transform .15s ease;font-size:var(--fs-10);width:10px;">▾</span>
        <span style="flex:1;">Cross-check vs Immich library</span>
      </div>
      <div style="display:flex;gap:6px;">
        <input id="crosscheck-frame-input" type="number" min="1" style="width:70px;background:#0c0c10;border:1px solid #2a2a32;color:var(--text);border-radius:6px;padding:4px 6px;font-size:var(--fs-11);" placeholder="frame #">
        <button id="crosscheck-btn" style="flex:1;padding:6px;background:#1a2a3a;border:1px solid #2a4a6a;color:var(--accent);border-radius:6px;cursor:pointer;font-size:var(--fs-11);">Check vs Immich library</button>
      </div>
      <div id="crosscheck-results" style="margin-top:8px;"></div>
    </div>
  `;
  renderSimSparkline(data.results, data.simThreshold, data.blurThreshold, sourceType);
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
        out.innerHTML = `<div style="font-size:var(--fs-11);color:#d9534f;">${result.error}</div>`;
      } else if (!result.results.length) {
        out.innerHTML = `<div style="font-size:var(--fs-11);color:var(--dim);">No faces in Immich library yet to compare against.</div>`;
      } else {
        out.innerHTML = `<div style="font-size:var(--fs-10);color:var(--dim);margin-bottom:4px;">Closest matches already in Immich:</div>` +
          result.results.slice(0, 8).map((r, i) => `
            <div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid #1a1a20;">
              <img src="/api/thumb/${r.assetId}" style="width:36px;height:36px;object-fit:cover;border-radius:4px;">
              <div style="flex:1;min-width:0;font-size:var(--fs-10);color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.filename}</div>
              <div style="font-size:var(--fs-10);color:var(--accent);">${(r.similarity * 100).toFixed(1)}%</div>
              <button class="cc-add-btn" data-idx="${i}" data-asset-id="${r.assetId}" style="font-size:var(--fs-9);padding:3px 6px;background:#2a1a3a;border:1px solid #4a2a6a;color:#d4a5ff;border-radius:4px;cursor:pointer;flex-shrink:0;">${extraImmichNodes.some(n => n.assetId === r.assetId) ? 'Remove ✓' : '+ Ring'}</button>
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
      out.innerHTML = `<div style="font-size:var(--fs-11);color:#d9534f;">Request failed: ${e}</div>`;
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
      blur: r.blur,
      bboxRatio: r.bboxRatio,
      vertFillPct: r.vertFillPct,
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
    jobId,
  };
  renderVideoRing();
  updateSaveSelectedButton();
}

const folderImagesInput = document.getElementById('folder-images-input');
const folderZipInput = document.getElementById('folder-zip-input');
const folderStatus = document.getElementById('folder-status');
let lastFolderSource = null;
function wireFolderInput() {
document.getElementById('folder-images-btn').addEventListener('click', () => folderImagesInput.click());
document.getElementById('folder-zip-btn').addEventListener('click', () => folderZipInput.click());

folderImagesInput.addEventListener('change', () => {
  if (folderImagesInput.files.length) startFolderAnalysis({ images: folderImagesInput.files });
});
folderZipInput.addEventListener('change', () => {
  if (folderZipInput.files.length) startFolderAnalysis({ zip: folderZipInput.files[0] });
});
}

async function startFolderAnalysis({ images, zip }, refIndexOverride) {
  selectedFrames.clear();
  const refIndex = refIndexOverride || (parseInt(document.getElementById('folder-ref-index').value, 10) || 1);
  document.getElementById('folder-ref-index').value = refIndex;
  lastFolderSource = { images, zip };
  const form = new FormData();
  form.append('simThreshold', document.getElementById('sim-threshold').value);
  form.append('blurThreshold', document.getElementById('blur-threshold').value);
  form.append('refIndex', refIndex);
  form.append('cacheFormat', document.getElementById('cache-format-png').checked ? 'png' : 'jpg');

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

