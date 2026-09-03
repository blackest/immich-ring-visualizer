// Character sheet panel -- turns a face into a shot-list of turnaround /
// dataset views via character_sheet.py + hidream_engine.py (ported from
// Phosphene, see PHOSPHENE_DECOUPLING_PLAN.md; no longer talks to
// Phosphene at all). Three sources feed the same result:
//   1. The currently-centered Immich face (lastNeighborsRender.centerId,
//      a real assetId) -> POST /api/phosphene/sheet-from-asset (JSON).
//   2. The currently-centered video-frame or folder-image anchor
//      (centerId === '__anchor__', details in lastVideoRingState) -> the
//      anchor thumbnail is fetch()ed as a blob and uploaded.
//   3. Any photo picked directly from disk via the file input.
// (2) and (3) both land on POST /api/phosphene/sheet-from-upload
// (multipart) because video frames and folder images are never written
// to disk in this app (see video_analysis.py's MemoryVideo).
//
// Generation now runs as a background job (routes/phosphene.py hands
// back {job_id, poll_url, shot_keys} immediately instead of blocking the
// request -- the 15-shot "extended" preset can run 2.5+ hours, way past
// any sane fetch timeout). This file polls GET poll_url every couple of
// seconds and renders a shot grid with live per-shot status, plus a
// re-roll button per finished shot.
//
// `lastNeighborsRender` / `lastVideoRingState` are selection-ui.js's
// globals - classic scripts share top-level scope on this page. This
// file stays a classic script too, on purpose: an inline onclick
// handler only works if the function declaring it lands on window.

const PHOSPHERE_POLL_MS = 2000;
let _phospherePollTimer = null;
let _phosphereCurrentTrigger = null;

// key -> { root, imgWrap, actions, imgEl, phEl, copyBtn, rerollBtn,
//          lastStatus, lastThumbnail }. Persists across polls of the
// SAME job so unchanged shots are left alone instead of torn down and
// rebuilt every tick (that rebuild-every-poll -- plus a fresh
// cache-busting timestamp on every <img> regardless of whether that
// shot's render actually changed -- was what made the grid visibly
// flash roughly once per poll). Cleared only when a brand new
// top-level generate/upload starts (_phosphereStart), NOT on a reroll
// poll -- a reroll job's status response only lists the one shot being
// re-rolled, and the rest of the grid should stay exactly as it was.
let _phosphereGridCells = new Map();
let _phosphereShotGridCollapsed = false;

function _phosphereEls() {
  return {
    btn: document.getElementById('phosphene-sheet-btn'),
    fileBtn: document.getElementById('phosphene-sheet-file-btn'),
    status: document.getElementById('phosphene-sheet-status'),
    logTail: document.getElementById('phosphene-log-tail'),
    img: document.getElementById('phosphene-sheet-img'),
    grid: document.getElementById('phosphene-shot-grid'),
    gridToggle: document.getElementById('phosphene-shot-grid-toggle'),
    gridWrap: document.getElementById('phosphene-shot-grid-wrap'),
    addToRingBtn: document.getElementById('phosphene-add-to-ring-btn'),
    nameInput: document.getElementById('phosphene-sheet-name'),
    presetSelect: document.getElementById('phosphene-preset-select'),
    styleSelect: document.getElementById('phosphene-style-select'),
    wardrobeInput: document.getElementById('phosphene-wardrobe-input'),
    hairColorInput: document.getElementById('phosphene-hair-color-input'),
    seedInput: document.getElementById('phosphene-seed-input'),
    identityLockCb: document.getElementById('phosphene-identity-lock-cb'),
    customPromptInput: document.getElementById('phosphene-custom-prompt-input'),
    settingsToggle: document.getElementById('phosphene-settings-toggle'),
    settingsMore: document.getElementById('phosphene-settings-more'),
  };
}

// "more settings" collapse -- wired once at load, see bottom of file.
function _phosphereInitSettingsToggle() {
  const { settingsToggle, settingsMore } = _phosphereEls();
  if (!settingsToggle) return;
  settingsToggle.addEventListener('click', () => {
    const open = settingsMore.style.display !== 'none';
    settingsMore.style.display = open ? 'none' : 'block';
    settingsToggle.textContent = (open ? '▸' : '▾') + ' more settings';
  });
}

// Shot grid collapse -- same pattern as "more settings" above, but the
// grid's own display:none/grid is already actively managed every poll
// tick by _phosphereStart/_phosphereRenderGrid (data-presence: is
// there anything to show at all), so the user's collapse preference
// lives on a separate wrapper element instead of fighting that -- see
// _phosphereApplyShotGridCollapseState.
function _phosphereInitShotGridToggle() {
  const { gridToggle } = _phosphereEls();
  if (!gridToggle) return;
  gridToggle.addEventListener('click', () => {
    _phosphereShotGridCollapsed = !_phosphereShotGridCollapsed;
    _phosphereApplyShotGridCollapseState();
  });
}

function _phosphereApplyShotGridCollapseState() {
  const { gridToggle, gridWrap } = _phosphereEls();
  if (!gridToggle || !gridWrap) return;
  gridToggle.style.display = 'block';
  gridWrap.style.display = _phosphereShotGridCollapsed ? 'none' : 'block';
  const count = _phosphereGridCells.size;
  gridToggle.textContent = (_phosphereShotGridCollapsed ? '▸' : '▾') +
    ` character sheet shots (${count})`;
}

function _phosphereSettings() {
  const { presetSelect, styleSelect, wardrobeInput, hairColorInput, seedInput,
          identityLockCb, customPromptInput } = _phosphereEls();
  const seedRaw = (seedInput.value || '').trim();
  return {
    preset: presetSelect.value,
    style: styleSelect.value,
    wardrobe: (wardrobeInput.value || '').trim(),
    hair_color: (hairColorInput.value || '').trim(),
    seed: seedRaw === '' ? -1 : parseInt(seedRaw, 10),
    identity_lock: identityLockCb.checked,
    custom_prompt: (customPromptInput.value || '').trim(),
  };
}

function _phosphereSetBusy(busy) {
  const { btn, fileBtn } = _phosphereEls();
  btn.disabled = busy;
  fileBtn.disabled = busy;
}

function _phosphereStart(message) {
  const { status, img, grid, gridToggle, gridWrap, logTail, addToRingBtn } = _phosphereEls();
  _phosphereSetBusy(true);
  status.textContent = message;
  img.style.display = 'none';
  img.src = '';
  grid.style.display = 'none';
  grid.innerHTML = '';
  _phosphereGridCells.clear();
  gridToggle.style.display = 'none';
  gridWrap.style.display = 'none';
  logTail.style.display = 'none';
  logTail.textContent = '';
  addToRingBtn.style.display = 'none';
  if (_phospherePollTimer) {
    clearInterval(_phospherePollTimer);
    _phospherePollTimer = null;
  }
}

function _phosphereShowError(payload, fallback) {
  const { status } = _phosphereEls();
  const msg = (payload && payload.error) ? payload.error : (fallback || 'unknown error');
  status.textContent = 'Error: ' + msg;
  _phosphereSetBusy(false);
}

const _PHOSPHERE_STATUS_LABEL = {
  queued: 'queued', rendering: 'rendering…', done: 'done',
  failed: 'failed', not_started: 'skipped',
};

function _phosphereBuildShotCell(trigger, shot) {
  const cell = document.createElement('div');
  cell.style.cssText = 'display:flex;flex-direction:column;gap:2px;';

  const imgWrap = document.createElement('div');
  imgWrap.style.cssText = 'position:relative;aspect-ratio:1;background:#0e0e12;' +
    'border:1px solid #2a2a32;border-radius:4px;overflow:hidden;';
  cell.appendChild(imgWrap);

  const label = document.createElement('div');
  label.style.cssText = 'font-size:var(--fs-9);color:var(--dim);text-align:center;' +
    'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
  label.textContent = shot.key;
  cell.appendChild(label);

  const actions = document.createElement('div');
  actions.style.cssText = 'display:flex;gap:2px;';
  cell.appendChild(actions);

  const entry = {
    root: cell, imgWrap, actions,
    imgEl: null, phEl: null, copyBtn: null, rerollBtn: null,
    lastStatus: null, lastThumbnail: undefined,
  };
  _phosphereApplyShotState(trigger, entry, shot);
  return entry;
}

// Patches one shot cell in place -- only touches the DOM for the parts
// that actually changed since the last poll of this same shot. This is
// the fix for the once-a-second flash: previously every poll replaced
// every <img> with a brand new one carrying a fresh Date.now() cache
// buster, so every image re-fetched from the network every tick
// regardless of whether that shot had actually rendered anything new.
// `shot.thumbnail` is the backend's resolved file path for that shot's
// latest render -- it only changes when a new PNG has actually landed
// (a fresh render or a re-roll), so using IT as the cache-bust value
// (instead of the current time) means the <img> only gets replaced,
// and only re-fetches, when there is genuinely something new to show.
function _phosphereApplyShotState(trigger, entry, shot) {
  const showsImage = shot.status === 'done' || shot.status === 'failed';
  const thumbChanged = shot.thumbnail !== entry.lastThumbnail;

  if (showsImage && (thumbChanged || !entry.imgEl)) {
    entry.imgWrap.innerHTML = '';
    const im = document.createElement('img');
    im.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
    im.src = `/api/phosphene/characters/${encodeURIComponent(trigger)}/shots/` +
             `${encodeURIComponent(shot.key)}?v=${encodeURIComponent(shot.thumbnail || shot.status)}`;
    im.alt = shot.key;
    entry.imgWrap.appendChild(im);
    entry.imgEl = im;
    entry.phEl = null;
  } else if (!showsImage && (shot.status !== entry.lastStatus || !entry.phEl)) {
    entry.imgWrap.innerHTML = '';
    const ph = document.createElement('div');
    ph.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;' +
      'justify-content:center;color:var(--dim);font-size:var(--fs-9);text-align:center;padding:4px;';
    ph.textContent = _PHOSPHERE_STATUS_LABEL[shot.status] || shot.status;
    entry.imgWrap.appendChild(ph);
    entry.phEl = ph;
    entry.imgEl = null;
  }
  entry.lastThumbnail = shot.thumbnail;
  entry.lastStatus = shot.status;

  // Prompts are resolved at job-start time (see sheet_jobs.py), so this
  // is available the moment a job starts -- not just once a shot
  // finishes rendering. Built once; refreshed only if a re-roll gave
  // this same shot key a different prompt override.
  if (shot.prompt && !entry.copyBtn) {
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.textContent = 'copy prompt';
    copy.title = shot.prompt;
    copy.style.cssText = 'flex:1;font-size:var(--fs-9);padding:2px;background:#16161c;' +
      'border:1px solid #2a2a32;color:var(--dim);border-radius:3px;cursor:pointer;';
    copy.onclick = () => _phosphereCopyPrompt(copy, shot.prompt);
    entry.actions.appendChild(copy);
    entry.copyBtn = copy;
  } else if (shot.prompt && entry.copyBtn && entry.copyBtn.title !== shot.prompt) {
    entry.copyBtn.title = shot.prompt;
    entry.copyBtn.onclick = () => _phosphereCopyPrompt(entry.copyBtn, shot.prompt);
  }

  // Reroll/retry -- appears once a shot has something to reroll, label
  // flips between the two only when status actually crosses that line.
  if (showsImage) {
    const wantLabel = shot.status === 'failed' ? 'retry' : 'reroll';
    if (!entry.rerollBtn) {
      const reroll = document.createElement('button');
      reroll.type = 'button';
      reroll.style.cssText = 'flex:1;font-size:var(--fs-9);padding:2px;background:#16161c;' +
        'border:1px solid #2a2a32;color:var(--dim);border-radius:3px;cursor:pointer;';
      entry.actions.appendChild(reroll);
      entry.rerollBtn = reroll;
    }
    if (entry.rerollBtn.textContent !== wantLabel) entry.rerollBtn.textContent = wantLabel;
    entry.rerollBtn.onclick = () => _phosphereReroll(trigger, shot.key);
  }
}

function _phosphereRenderGrid(trigger, shots) {
  const { grid } = _phosphereEls();
  grid.style.display = 'grid';
  shots.forEach(shot => {
    let entry = _phosphereGridCells.get(shot.key);
    if (!entry) {
      entry = _phosphereBuildShotCell(trigger, shot);
      _phosphereGridCells.set(shot.key, entry);
      grid.appendChild(entry.root);
    } else {
      _phosphereApplyShotState(trigger, entry, shot);
    }
  });
  // Reveal the toggle once there's anything to show, and keep its
  // label's shot count and the wrap's collapsed/expanded state current
  // on every poll tick (cheap -- just two style/text writes).
  _phosphereApplyShotGridCollapseState();
}

function _phosphereCopyPrompt(button, text) {
  const flash = (label) => {
    const original = button.textContent;
    button.textContent = label;
    setTimeout(() => { button.textContent = original; }, 1200);
  };
  const fallbackCopy = () => {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;top:-1000px;left:-1000px;';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      flash(ok ? 'copied!' : 'copy failed');
    } catch (err) {
      console.warn('[phosphere] fallback copy failed:', err);
      flash('copy failed');
    }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => flash('copied!'), fallbackCopy);
  } else {
    fallbackCopy();
  }
}

function _phospherePoll(jobId, trigger) {
  const { status, img, logTail, addToRingBtn } = _phosphereEls();
  fetch(`/api/phosphene/sheet-jobs/${jobId}`)
    .then(r => r.json())
    .then(job => {
      const doneCount = job.shots.filter(s => s.status === 'done').length;
      status.textContent =
        `${job.status === 'running' ? 'Generating' : job.status} — ` +
        `${doneCount}/${job.shots.length} shots`;

      if (job.log_tail && job.log_tail.length) {
        logTail.style.display = 'block';
        logTail.textContent = job.log_tail.join('\n');
        logTail.scrollTop = logTail.scrollHeight;
      }

      _phosphereRenderGrid(trigger, job.shots);
      // Shown as soon as anything has actually rendered, even mid-job
      // or after a job that ultimately failed partway through -- reads
      // sheet_views/ directly (see sheet_shot_image_paths), not
      // sheet.json, so it doesn't need the whole job to have finished.
      addToRingBtn.style.display = doneCount > 0 ? 'block' : 'none';

      if (job.status !== 'running') {
        clearInterval(_phospherePollTimer);
        _phospherePollTimer = null;
        _phosphereSetBusy(false);
        if (job.status === 'failed') {
          status.textContent = 'Error: ' + (job.error || 'generation failed');
        } else if (job.sheet_url) {
          img.src = job.sheet_url + '?t=' + Date.now();
          img.style.display = 'block';
        }
      }
    })
    .catch(err => {
      // A transient poll failure shouldn't kill the whole job -- the
      // background thread on the server keeps running regardless; just
      // skip this tick and try again on the next interval.
      console.warn('[phosphere] poll failed, retrying:', err);
    });
}

function _phosphereWatchJob(jobId, trigger) {
  _phosphereCurrentTrigger = trigger;
  if (_phospherePollTimer) clearInterval(_phospherePollTimer);
  _phospherePoll(jobId, trigger);
  _phospherePollTimer = setInterval(() => _phospherePoll(jobId, trigger), PHOSPHERE_POLL_MS);
}

function _phosphereSubmit(requestPromise, triggerHint) {
  _phosphereStart('Starting…');
  requestPromise
    .then(r => r.json().then(payload => ({ ok: r.ok, payload })))
    .then(({ ok, payload }) => {
      if (!ok || !payload || !payload.ok || !payload.job_id) {
        _phosphereShowError(payload);
        return;
      }
      const { status } = _phosphereEls();
      status.textContent = `Started — 0/${payload.shot_keys.length} shots`;
      _phosphereWatchJob(payload.job_id, payload.trigger || triggerHint);
    })
    .catch(err => {
      _phosphereShowError(null, 'request failed: ' + err);
    });
}

function phosphereAddToRing() {
  const trigger = _phosphereCurrentTrigger;
  if (!trigger) return;
  const { status, addToRingBtn } = _phosphereEls();
  addToRingBtn.disabled = true;
  addToRingBtn.textContent = 'Adding to ring…';
  fetch(`/api/phosphene/characters/${encodeURIComponent(trigger)}/sheet/add-to-ring`, {
    method: 'POST',
  })
    .then(r => r.json().then(payload => ({ ok: r.ok, payload })))
    .then(({ ok, payload }) => {
      addToRingBtn.disabled = false;
      addToRingBtn.textContent = 'Add sheet to ring';
      if (!ok || payload.error) {
        status.textContent = 'Error: ' + (payload && payload.error ? payload.error : 'add-to-ring failed');
        return;
      }
      // Same job shape/poller /api/analyze-folder uses -- see
      // routes/phosphene.py's add_sheet_to_ring for why 'folder' is the
      // right sourceType here (it IS one, just generated not uploaded).
      pollAnalysis(payload.jobId, `${trigger} (character sheet)`, 1, status, 'folder');
    })
    .catch(err => {
      addToRingBtn.disabled = false;
      addToRingBtn.textContent = 'Add sheet to ring';
      status.textContent = 'Error: add-to-ring request failed: ' + err;
    });
}

function _phosphereReroll(trigger, shotKey) {
  const { status } = _phosphereEls();
  _phosphereSetBusy(true);
  status.textContent = `Re-rolling ${shotKey}…`;
  fetch(`/api/phosphene/characters/${encodeURIComponent(trigger)}/sheet/reroll`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ shot_key: shotKey }),
  })
    .then(r => r.json().then(payload => ({ ok: r.ok, payload })))
    .then(({ ok, payload }) => {
      if (!ok || !payload || !payload.ok || !payload.job_id) {
        _phosphereShowError(payload);
        return;
      }
      _phosphereWatchJob(payload.job_id, trigger);
    })
    .catch(err => _phosphereShowError(null, 'reroll request failed: ' + err));
}

// Source 1: the currently-centered Immich face.
function phosphereGenerateSheet() {
  const { nameInput, status } = _phosphereEls();
  const centerId = (typeof lastNeighborsRender !== 'undefined' && lastNeighborsRender)
    ? lastNeighborsRender.centerId : null;
  const settings = _phosphereSettings();

  if (centerId && centerId !== '__anchor__') {
    const trigger = (nameInput.value || '').trim() || centerId;
    _phosphereSubmit(fetch('/api/phosphene/sheet-from-asset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asset_id: centerId, name: (nameInput.value || '').trim(),
                            ...settings }),
    }), trigger);
    return;
  }

  // Source 2: video-frame / folder-image anchor -- same sentinel
  // centerId ('__anchor__') either way; lastVideoRingState.sourceType
  // tells them apart, but both are fetched as a blob and uploaded the
  // same way since neither has a real local path.
  if (centerId === '__anchor__' && typeof lastVideoRingState !== 'undefined' && lastVideoRingState) {
    const { anchorUrl } = lastVideoRingState;
    status.textContent = 'Fetching the reference frame…';
    fetch(anchorUrl)
      .then(r => {
        if (!r.ok) throw new Error('could not fetch the reference frame (HTTP ' + r.status + ')');
        return r.blob();
      })
      .then(blob => {
        const form = new FormData();
        const trigger = (nameInput.value || '').trim() || ('frame-' + Date.now());
        form.append('file', blob, 'reference.jpg');
        form.append('trigger', trigger);
        Object.entries(settings).forEach(([k, v]) => form.append(k, v));
        _phosphereSubmit(fetch('/api/phosphene/sheet-from-upload', { method: 'POST', body: form }), trigger);
      })
      .catch(err => _phosphereShowError(null, String(err)));
    return;
  }

  status.textContent = 'Nothing is currently centered — pick a face, frame, or image first.';
}

// Source 3: a photo picked directly from disk, independent of the ring
// view entirely.
function phosphereGenerateFromFile(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const { nameInput } = _phosphereEls();
  const settings = _phosphereSettings();
  const form = new FormData();
  const trigger = (nameInput.value || '').trim() || ('upload-' + Date.now());
  form.append('file', file, file.name);
  form.append('trigger', trigger);
  Object.entries(settings).forEach(([k, v]) => form.append(k, v));
  _phosphereSubmit(fetch('/api/phosphene/sheet-from-upload', { method: 'POST', body: form }), trigger);
  input.value = ''; // allow re-picking the same file later
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    _phosphereInitSettingsToggle();
    _phosphereInitShotGridToggle();
  });
} else {
  _phosphereInitSettingsToggle();
  _phosphereInitShotGridToggle();
}
