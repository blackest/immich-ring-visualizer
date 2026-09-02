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

function _phosphereEls() {
  return {
    btn: document.getElementById('phosphene-sheet-btn'),
    fileBtn: document.getElementById('phosphene-sheet-file-btn'),
    status: document.getElementById('phosphene-sheet-status'),
    logTail: document.getElementById('phosphene-log-tail'),
    img: document.getElementById('phosphene-sheet-img'),
    grid: document.getElementById('phosphene-shot-grid'),
    nameInput: document.getElementById('phosphene-sheet-name'),
    presetSelect: document.getElementById('phosphene-preset-select'),
    styleSelect: document.getElementById('phosphene-style-select'),
    wardrobeInput: document.getElementById('phosphene-wardrobe-input'),
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

function _phosphereSettings() {
  const { presetSelect, styleSelect, wardrobeInput, seedInput,
          identityLockCb, customPromptInput } = _phosphereEls();
  const seedRaw = (seedInput.value || '').trim();
  return {
    preset: presetSelect.value,
    style: styleSelect.value,
    wardrobe: (wardrobeInput.value || '').trim(),
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
  const { status, img, grid, logTail } = _phosphereEls();
  _phosphereSetBusy(true);
  status.textContent = message;
  img.style.display = 'none';
  img.src = '';
  grid.style.display = 'none';
  grid.innerHTML = '';
  logTail.style.display = 'none';
  logTail.textContent = '';
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

function _phosphereRenderGrid(trigger, shots) {
  const { grid } = _phosphereEls();
  grid.style.display = 'grid';
  grid.innerHTML = '';
  shots.forEach(shot => {
    const cell = document.createElement('div');
    cell.style.cssText = 'display:flex;flex-direction:column;gap:2px;';

    const imgWrap = document.createElement('div');
    imgWrap.style.cssText = 'position:relative;aspect-ratio:1;background:#0e0e12;' +
      'border:1px solid #2a2a32;border-radius:4px;overflow:hidden;';

    if (shot.status === 'done' || shot.status === 'failed') {
      const im = document.createElement('img');
      im.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
      im.src = `/api/phosphene/characters/${encodeURIComponent(trigger)}/shots/` +
               `${encodeURIComponent(shot.key)}?t=${Date.now()}`;
      im.alt = shot.key;
      imgWrap.appendChild(im);
    } else {
      const ph = document.createElement('div');
      ph.style.cssText = 'width:100%;height:100%;display:flex;align-items:center;' +
        'justify-content:center;color:var(--dim);font-size:var(--fs-9);text-align:center;padding:4px;';
      ph.textContent = _PHOSPHERE_STATUS_LABEL[shot.status] || shot.status;
      imgWrap.appendChild(ph);
    }
    cell.appendChild(imgWrap);

    const label = document.createElement('div');
    label.style.cssText = 'font-size:var(--fs-9);color:var(--dim);text-align:center;' +
      'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
    label.textContent = shot.key;
    cell.appendChild(label);

    if (shot.status === 'done' || shot.status === 'failed') {
      const reroll = document.createElement('button');
      reroll.type = 'button';
      reroll.textContent = shot.status === 'failed' ? 'retry' : 'reroll';
      reroll.style.cssText = 'font-size:var(--fs-9);padding:2px;background:#16161c;' +
        'border:1px solid #2a2a32;color:var(--dim);border-radius:3px;cursor:pointer;';
      reroll.onclick = () => _phosphereReroll(trigger, shot.key);
      cell.appendChild(reroll);
    }

    grid.appendChild(cell);
  });
}

function _phospherePoll(jobId, trigger) {
  const { status, img, logTail } = _phosphereEls();
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
  document.addEventListener('DOMContentLoaded', _phosphereInitSettingsToggle);
} else {
  _phosphereInitSettingsToggle();
}
