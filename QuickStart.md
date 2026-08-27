# Immich Ring Visualizer

A local Flask tool for finding, clustering, and exporting faces — from your Immich library, a local folder/zip, or a video clip — for building clean LoRA training sets. Pick a reference face, see matches ranked by similarity, filter by sharpness and pose, and export precisely cropped, training-ready images.

---

## Project structure

| File | Role |
|---|---|
| `ring_viz.py` | Flask backend: Immich/Postgres queries, InsightFace analysis, video/folder analysis jobs, export pipeline |
| `app.js` | Frontend: ring layout, sparklines, pose scatter, selection modal, sliders |
| `index.html` | Page structure and controls |
| `style.css` | Styling |

## Requirements & setup

```bash
pip install flask psycopg2-binary requests numpy opencv-python insightface
```

Edit the config block at the top of `ring_viz.py`:
- `PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD` / `PG_DB` — your Immich Postgres instance
- `IMMICH_BASE_URL` / `IMMICH_API_KEY`

```bash
python3 ring_viz.py
```

Open `http://localhost:5050`. Face detection uses InsightFace's `buffalo_l` model, preferring CoreML on Apple Silicon, then CUDA, then CPU.

---

## Core workflows

### 1. Folder / zip analysis
`POST /api/analyze-folder` — upload individual image files or a `.zip`. Runs the same pipeline as video analysis over stills: face detection, similarity vs. a reference, blur (Laplacian variance), pose (yaw/pitch/roll), and bbox/frame ratio.

### 2. Video analysis
`POST /api/preview-video` to load a clip and scrub frame-by-frame for a reference, then `POST /api/analyze-video` to run the full pass. Only frames that pass both the similarity and blur thresholds get an image cached to disk and are eligible for export. `/api/build-playback` can render an annotated copy of the source video marking which frames passed or failed and why (blurry / low match / no face) — useful for auditing a threshold choice against the actual footage.

### 3. Immich integration
- **`/api/person-clusters`** — ranks named Immich persons by how tightly their faces cluster (average cosine similarity to their own centroid). High `avgSim` + high `faceCount` is a strong LoRA source, though a very tight cluster can also mean near-duplicate stills rather than genuinely varied photos — worth an eyeball either way.
- **`/api/analyze-immich`** — runs the full analysis pipeline against a set of Immich asset IDs directly, no manual download/re-upload round trip.
- **`/api/neighbors`** — nearest faces to a given asset, via pgvector cosine distance on `face_search`, falling back to CLIP-based `smart_search` similarity if no face embedding exists.
- **`/api/immich-cross-check/<job>/<frame>`** — takes a passed video/folder frame's stored embedding and searches the Immich library for matching faces, so you can tell whether a person already has library coverage before hunting for more footage.

---

## Layout & views

- **Ring layout** — reference face centered, matches arranged by similarity/distance, with a lens/fisheye hover effect for detail.
- **Sim/blur sparkline** — a chart of similarity and blur scores across all analyzed frames, with the current thresholds drawn as reference lines.
- **Pose scatter** (selection modal) — plots selected items on a yaw/pitch plane relative to the selection's own pose centroid, so you can see angle coverage rather than just similarity. Items without pose data get a "Detect pose" button that fetches it on demand for Immich assets.
- **Selection modal** — a review grid (grid or pose-scatter layout) for everything currently marked for export, backed by `localStorage` so the selection survives a reload.

---

## Filtering

Two independent filters sit on top of the initial `simThreshold`/`blurThreshold` job settings:

- **Ring squeeze slider** — a live post-hoc minimum-similarity cutoff on what's shown in the ring, with a live "kept / total" count.
- **Sharpness cutoff slider** (toggled via a "Sharpness cutoff" checkbox) — an independent minimum-sharpness cutoff, so you can tighten blur filtering without also squeezing the similarity band. This has landed in the current build, not just planned.

---

## Export pipeline

`crop_resize_export()` in `ring_viz.py` handles all crop modes:

| Mode | Behavior |
|---|---|
| `face` | Crops centered on the detected bbox with a configurable `margin`; widens (zooms out) rather than upscaling if a tight crop would need more magnification than `max_upscale` allows |
| `contain` | Letterboxes the full image into the target box, no cropping |
| `stretch` | Naive resize to target dimensions |
| `native` | Skips the final resize — for `face`/`center` modes the crop still respects the target aspect ratio but keeps native pixels; `stretch`/`contain` just return the source untouched, since a fixed target box is meaningless without resizing |

Other parameters: `interp` (resize filter, auto-selects area vs. Lanczos by scale direction), `min_face_px` (skip faces too small to be useful), `pad_mode` (`none` / `black` / `edge` — how a widened crop is handled if it hits the frame edge).

Export endpoints:
- `/api/export-job/<job_id>` — export selected frames from a video/folder analysis job, optionally combined with selected Immich asset IDs in the same call.
- `/api/export-immich-assets` — export straight from an Immich selection (e.g. a person cluster), no analysis job involved.
- `/api/export-preview/...` and `/api/export-preview-immich/...` — render an exact preview of what a given frame/asset will look like with the current export settings, so the selection modal shows the real crop rather than a generic thumbnail.

---

## Reading the metrics

- **Similarity** — cosine similarity to the reference embedding (or cluster centroid). `1.0` = exact match.
- **Sharpness (`blur`)** — Laplacian variance of the face crop; higher = sharper.
- **Face %** (`bboxRatio`) — detected face area as a fraction of the frame.
- **Yaw / pitch / roll** — head pose, used for pose-scatter coverage checks.

---

## Typical LoRA dataset workflow

1. Find an anchor via **Person Clusters** (high `faceCount`, high `avgSim`) or a video/folder pass with a good reference face.
2. Use the **sharpness cutoff** and **ring squeeze** sliders independently to trim blurry or weak matches without over-constraining each other.
3. Check **pose scatter** in the selection modal to see which angles are under-represented.
4. **Cross-check** promising video/folder frames against Immich to avoid exporting near-duplicates of what's already in the library.
5. Export with `mode=face`, a sensible `margin`, and `native` resolution where you want to avoid interpolation artifacts on close-ups.

---

*Internal tool — not for external distribution.*
