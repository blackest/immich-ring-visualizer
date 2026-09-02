// Character sheet (Phosphene) panel — turns the currently-centered
// Immich face into a 3-angle turnaround sheet via Phosphene's draft-
// character pipeline. `lastNeighborsRender` is selection-ui.js's global
// (classic scripts share top-level scope on this page); its `.centerId`
// is the Immich asset_id of whichever node is centered right now.
//
// This is a plain classic script (no type="module"), matching the rest
// of this app's JS — the reason to be deliberate about that: Phosphene
// itself just fixed a bug where its OWN inline onclick="fn()" handlers
// broke because a module split made their top-level functions private
// per-file. Staying a classic script here means phosphereGenerateSheet
// lands on window automatically, same as every other onclick handler
// on this page.
function phosphereGenerateSheet() {
  const btn = document.getElementById('phosphene-sheet-btn');
  const status = document.getElementById('phosphene-sheet-status');
  const img = document.getElementById('phosphene-sheet-img');
  const nameInput = document.getElementById('phosphene-sheet-name');

  const centerId = (typeof lastNeighborsRender !== 'undefined' && lastNeighborsRender)
    ? lastNeighborsRender.centerId : null;
  if (!centerId || centerId === '__anchor__') {
    status.textContent = 'No Immich photo is currently centered — pick a face first.';
    return;
  }

  img.style.display = 'none';
  img.src = '';
  btn.disabled = true;
  status.textContent = 'Generating (3 views, rendered one after another — can take a few minutes)…';

  fetch('/api/phosphene/sheet-from-asset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      asset_id: centerId,
      name: (nameInput.value || '').trim(),
    }),
  })
    .then(r => r.json().then(payload => ({ ok: r.ok, payload })))
    .then(({ ok, payload }) => {
      if (!ok || !payload || !payload.ok) {
        const msg = (payload && payload.error) ? payload.error : 'unknown error';
        status.textContent = 'Error: ' + msg;
        return;
      }
      status.textContent = 'Done — trigger "' + payload.trigger + '".';
      // Cache-bust: a re-click on the same face overwrites Phosphene's
      // sheet.png in place, so without this the browser would just
      // keep showing its cached copy of the old one.
      img.src = payload.sheet_url + '?t=' + Date.now();
      img.style.display = 'block';
    })
    .catch(err => {
      status.textContent = 'Request failed: ' + err;
    })
    .finally(() => {
      btn.disabled = false;
    });
}
