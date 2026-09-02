// Character sheet (Phosphene) panel — turns a face into a 3-angle
// turnaround sheet via Phosphene's draft-character pipeline. Three
// sources feed the same result:
//   1. The currently-centered Immich face (lastNeighborsRender.centerId,
//      a real assetId) -> POST /api/phosphene/sheet-from-asset (JSON).
//   2. The currently-centered video-frame or folder-image anchor
//      (centerId === '__anchor__', details in lastVideoRingState) -> the
//      anchor thumbnail is fetch()ed as a blob and uploaded.
//   3. Any photo picked directly from disk via the file input.
// (2) and (3) both land on POST /api/phosphene/sheet-from-upload
// (multipart) because video frames and folder images are never written
// to disk in this app (see video_analysis.py's MemoryVideo) - there's
// no stable local path to hand Phosphene the way Immich's asset_id
// route can, so uploading the bytes directly is the one path that
// works for all of "video frame", "folder image", and "some other file
// on disk" alike.
//
// `lastNeighborsRender` / `lastVideoRingState` are selection-ui.js's
// globals - classic scripts share top-level scope on this page. This
// file stays a classic script too, on purpose: an inline onclick
// handler only works if the function declaring it lands on window,
// which is exactly what just broke in Phosphene itself after its
// module split (see fix/module-scope-onclick-handlers).

function _phosphereEls() {
  return {
    btn: document.getElementById('phosphene-sheet-btn'),
    fileBtn: document.getElementById('phosphene-sheet-file-btn'),
    status: document.getElementById('phosphene-sheet-status'),
    img: document.getElementById('phosphene-sheet-img'),
    nameInput: document.getElementById('phosphene-sheet-name'),
  };
}

function _phosphereSetBusy(busy) {
  const { btn, fileBtn } = _phosphereEls();
  btn.disabled = busy;
  fileBtn.disabled = busy;
}

function _phosphereStart(message) {
  const { status, img } = _phosphereEls();
  _phosphereSetBusy(true);
  status.textContent = message;
  img.style.display = 'none';
  img.src = '';
}

function _phosphereShowResult(payload) {
  const { status, img } = _phosphereEls();
  status.textContent = 'Done — trigger "' + payload.trigger + '".';
  // Cache-bust: a re-click on the same face overwrites Phosphene's
  // sheet.png in place, so without this the browser would just keep
  // showing its cached copy of the old one.
  img.src = payload.sheet_url + '?t=' + Date.now();
  img.style.display = 'block';
}

function _phosphereShowError(payload, fallback) {
  const { status } = _phosphereEls();
  const msg = (payload && payload.error) ? payload.error : (fallback || 'unknown error');
  status.textContent = 'Error: ' + msg;
}

function _phosphereSubmit(request) {
  _phosphereStart('Generating (3 views, rendered one after another — can take a few minutes)…');
  request
    .then(r => r.json().then(payload => ({ ok: r.ok, payload })))
    .then(({ ok, payload }) => {
      if (!ok || !payload || !payload.ok) {
        _phosphereShowError(payload);
        return;
      }
      _phosphereShowResult(payload);
    })
    .catch(err => {
      _phosphereShowError(null, 'request failed: ' + err);
    })
    .finally(() => _phosphereSetBusy(false));
}

// Source 1: the currently-centered Immich face.
function phosphereGenerateSheet() {
  const { nameInput, status } = _phosphereEls();
  const centerId = (typeof lastNeighborsRender !== 'undefined' && lastNeighborsRender)
    ? lastNeighborsRender.centerId : null;

  if (centerId && centerId !== '__anchor__') {
    _phosphereSubmit(fetch('/api/phosphene/sheet-from-asset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asset_id: centerId, name: (nameInput.value || '').trim() }),
    }));
    return;
  }

  // Source 2: video-frame / folder-image anchor — same sentinel
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
        form.append('name', (nameInput.value || '').trim());
        _phosphereSubmit(fetch('/api/phosphene/sheet-from-upload', { method: 'POST', body: form }));
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
  const form = new FormData();
  const trigger = (nameInput.value || '').trim() || ('upload-' + Date.now());
  form.append('file', file, file.name);
  form.append('trigger', trigger);
  form.append('name', (nameInput.value || '').trim());
  _phosphereSubmit(fetch('/api/phosphene/sheet-from-upload', { method: 'POST', body: form }));
  input.value = ''; // allow re-picking the same file later
}
