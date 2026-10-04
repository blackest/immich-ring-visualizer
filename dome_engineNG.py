"""NG-only: head-pose dome engine -- pick one face per yaw/pitch cell.

Source-agnostic core shared by dome_pick.py (command line) and, later, the
Dome panel's route. Does not import from or call into any non-NG module
except detectionNG (the same InsightFace loader the rest of NG uses).

Frames come in as (index, BGR ndarray) pairs, so any source can feed it
(OpenCV from a file path today, PyAV from in-memory bytes in the app).

Cell convention: 7x7, written col.row, 4.4 = the seed ("neutral").
Columns step in yaw (default 15 deg, negative yaw = left of the grid),
rows step in pitch (default 10 deg, top row = chin up). Angles are measured
as offsets from the seed's own raw angles, because InsightFace's raw yaw /
pitch / roll carry an offset that differs from source to source.
"""

import time

import cv2
import numpy as np

from detectionNG import get_face_app_ng, get_blur_score_ng
from stateNG import _face_app_lock_ng

GRID = 7
CENTRE = 4

# Several videos can be pooled: a frame index is clip * CLIP_STRIDE + local frame.
# For a single video clip is 0, so the index is just the frame number.
CLIP_STRIDE = 10 ** 6

# Raw InsightFace angles carry a per-source offset, but a seed this far from 0
# is probably not front-on at all (e.g. a subject turned toward someone).
SEED_WARN_YAW = 25.0
SEED_WARN_PITCH = 20.0
NEUTRAL_TOL = 10.0  # how close the seed should be to a user-given --neutral


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frames_from_cv2(video, step=1, clip=0):
    """Yield (frame_index, BGR frame) for every step-th frame of a video file.
    clip offsets the index (clip * CLIP_STRIDE) so several videos can be pooled."""
    cap = cv2.VideoCapture(str(video))
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step == 0:
            yield clip * CLIP_STRIDE + idx, frame
    cap.release()


def detect_and_track(frames, progress=None):
    """Detect every face in every frame and link them into tracks.

    A face joins the previous frame's face when bbox IoU >= 0.3 and embedding
    similarity >= 0.4 (greedy, best pair first); otherwise it starts a new
    track. Returns one array per field, one row per detected face.
    progress(frame_index, faces_so_far, tracks_so_far) is called every 100 frames.
    """
    app = get_face_app_ng()
    rec = {k: [] for k in ("clip", "frame", "track", "yaw", "pitch", "roll", "blur", "det", "bbox", "emb")}
    prev, next_track, seen, last_clip = [], 0, 0, None
    for idx, frame in frames:
        clip, local = divmod(idx, CLIP_STRIDE)
        if clip != last_clip:  # new video: never link a face across a cut between clips
            prev, last_clip = [], clip
        with _face_app_lock_ng:
            faces = app.get(frame)
        cur = []
        for f in faces:
            x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
            cur.append((f, (x1, y1, x2, y2), get_blur_score_ng(frame[y1:y2, x1:x2])))
        pairs = sorted(
            ((iou(c[1], pb) + float(np.dot(c[0].normed_embedding, pe)), ci, pi)
             for ci, c in enumerate(cur) for pi, (_, pb, pe) in enumerate(prev)
             if iou(c[1], pb) >= 0.3 and float(np.dot(c[0].normed_embedding, pe)) >= 0.4),
            reverse=True)
        used_c, used_p, assign = set(), set(), {}
        for _, ci, pi in pairs:
            if ci not in used_c and pi not in used_p:
                used_c.add(ci)
                used_p.add(pi)
                assign[ci] = prev[pi][0]
        prev = []
        for ci, (f, box, blur) in enumerate(cur):
            tid = assign.get(ci)
            if tid is None:
                tid, next_track = next_track, next_track + 1
            pitch, yaw, roll = (float(v) for v in f.pose)
            rec["clip"].append(clip)
            rec["frame"].append(local)
            rec["track"].append(tid)
            rec["yaw"].append(yaw)
            rec["pitch"].append(pitch)
            rec["roll"].append(roll)
            rec["blur"].append(blur)
            rec["det"].append(float(f.det_score))
            rec["bbox"].append(box)
            rec["emb"].append(f.normed_embedding)
            prev.append((tid, box, f.normed_embedding))
        seen += 1
        if progress and seen % 100 == 0:
            progress(local, len(rec["yaw"]), next_track)
    return {k: np.array(v) for k, v in rec.items()}


def embedding_from_image(path, face_index=0):
    """Embedding of one face in a reference image. Faces are ranked by width, so
    face_index 0 is the largest. Returns (embedding, bbox, n_faces_found)."""
    img = cv2.imread(str(path))
    if img is None:
        from PIL import Image  # e.g. .webp builds of OpenCV without codec support
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    with _face_app_lock_ng:
        faces = sorted(get_face_app_ng().get(img), key=lambda f: -(f.bbox[2] - f.bbox[0]))
    if not faces:
        raise ValueError(f"no face found in {path}")
    if face_index >= len(faces):
        raise ValueError(f"{path} has {len(faces)} faces; --ref-face {face_index} is out of range")
    f = faces[face_index]
    return f.normed_embedding, tuple(int(v) for v in f.bbox), len(faces)


def build_dome(d, ident_thr=0.5, yaw_step=15.0, pitch_step=10.0, ref_emb=None, ref_thr=0.3, min_track_faces=8,
               seed_mode="mode", neutral=None):
    """From detect_and_track output, work out identity, seed and the dome.

    Identity = the longest track plus every track whose centroid similarity to
    it is >= ident_thr -- or, when ref_emb is given (needed in multi-person
    scenes, where several tracks tie for longest), every track of at least
    min_track_faces whose centroid similarity to the reference is >= ref_thr.
    neutral=(yaw, pitch), if given, is the raw angle that means "front-on" for this
    source (it overrides seed_mode): the seed is picked near it, and the seed
    warning only fires if no good face lies within NEUTRAL_TOL of it.
    ref_emb may be one embedding or a stack of them (several reference pictures
    of the same person, e.g. from different scenes); a track's score is its
    best match across them.
    Seed = of the sharp, large, confident identity faces,
    the medoid of the 40 nearest the identity's median pose. Winner per cell
    (stand-in rule): of the 6 faces nearest the cell's target pose, the largest.

    Returns a dict: identity (indices), seed, col/row/dy/dp/sim_seed (per face),
    picks {(col, row): (face_index, faces_in_cell)}, plus bookkeeping counts.
    """
    if not len(d["yaw"]):
        raise ValueError("no faces found")
    tr, emb = d["track"], d["emb"]
    tids = sorted(set(tr.tolist()), key=lambda t: -int((tr == t).sum()))
    cent = {}
    for t in tids:
        v = emb[tr == t].mean(axis=0)
        cent[t] = v / np.linalg.norm(v)
    if ref_emb is None:
        main_t = tids[0]
        ident = np.array([float(cent[t] @ cent[main_t]) >= ident_thr for t in tr])
    else:
        refs = np.atleast_2d(ref_emb)
        sims = {t: float((refs @ cent[t]).max()) for t in tids if int((tr == t).sum()) >= min_track_faces}
        keep = {t for t, v in sims.items() if v >= ref_thr}
        if not keep:
            best = max(sims.values()) if sims else float("nan")
            raise ValueError(f"no track matches the reference image (best centroid similarity {best:.2f} < {ref_thr})")
        main_t = max(keep, key=lambda t: sims[t])
        ident = np.array([t in keep for t in tr])
    I = np.where(ident)[0]

    size = np.array([b[2] - b[0] for b in d["bbox"]], dtype=float)
    yaw, pitch, blur, det = d["yaw"], d["pitch"], d["blur"], d["det"]

    good = I[(size[I] >= 0.6 * np.median(size[I])) & (blur[I] >= 0.7 * np.median(blur[I])) & (det[I] > 0.7)]
    if not len(good):
        good = I
    if neutral is not None:  # caller knows what raw angle front-on reads as for this source
        target = (float(neutral[0]), float(neutral[1]))
    elif seed_mode == "front":  # most camera-facing by raw angle (raw angles carry a per-source offset)
        target = (0.0, 0.0)
    else:  # "mode": the identity's most typical pose
        target = (np.median(yaw[I]), np.median(pitch[I]))
    pool, relaxed = good, False
    if neutral is not None:
        # an explicit neutral outranks the quality filter: use sharp faces near it if
        # there are any, else any identity face near it, else fall back to the nearest good ones
        within = lambda ix: ix[np.hypot(yaw[ix] - target[0], pitch[ix] - target[1]) <= NEUTRAL_TOL]
        if len(within(good)):
            pool = within(good)
        elif len(within(I)):
            pool, relaxed = within(I), True
    near = pool[np.argsort((yaw[pool] - target[0]) ** 2 + (pitch[pool] - target[1]) ** 2)[:40]]
    S_near = emb[near] @ emb[near].T
    seed = int(near[int(np.argmax(S_near.mean(axis=1)))])

    dy, dp = yaw - yaw[seed], pitch - pitch[seed]
    col = np.clip(CENTRE + np.rint(dy / yaw_step).astype(int), 1, GRID)
    row = np.clip(CENTRE - np.rint(dp / pitch_step).astype(int), 1, GRID)  # top row = chin up
    sim_seed = emb @ emb[seed]

    picks = {}
    for c in range(1, GRID + 1):
        for r in range(1, GRID + 1):
            m = I[(col[I] == c) & (row[I] == r)]
            if not len(m):
                continue
            if (c, r) == (CENTRE, CENTRE):
                picks[(c, r)] = (seed, len(m))
                continue
            ok = m[det[m] > 0.7]
            ok = ok if len(ok) else m
            dist = np.hypot(dy[ok] - (c - CENTRE) * yaw_step, dp[ok] - (CENTRE - r) * pitch_step)
            cand = ok[np.argsort(dist)[:6]]
            picks[(c, r)] = (int(cand[np.argmax(size[cand])]), len(m))

    warnings = []
    if neutral is not None and relaxed:
        warnings.append(f"no sharp, large face near --neutral {neutral[0]:.0f},{neutral[1]:.0f}; seed chosen without the "
                        f"quality filter (face {int(size[seed])} px, blur {blur[seed]:.0f}) -- check it by eye")
    if neutral is not None:
        off = float(np.hypot(yaw[seed] - neutral[0], pitch[seed] - neutral[1]))
        if off > NEUTRAL_TOL:
            warnings.append(f"no good identity face within {NEUTRAL_TOL:.0f} deg of --neutral {neutral[0]:.0f},{neutral[1]:.0f}; "
                            f"the seed is {off:.0f} deg away (raw yaw {yaw[seed]:.0f}, pitch {pitch[seed]:.0f})")
    elif abs(yaw[seed]) > SEED_WARN_YAW or abs(pitch[seed]) > SEED_WARN_PITCH:
        warnings.append(f"seed's raw pose (yaw {yaw[seed]:.0f}, pitch {pitch[seed]:.0f}) is far from camera-front, "
                        f"so cell {CENTRE}.{CENTRE} is probably not a front-on view")
    return dict(identity=I, seed=seed, main_track=main_t, n_tracks=len(tids), col=col, row=row,
                dy=dy, dp=dp, sim_seed=sim_seed, size=size, picks=picks, warnings=warnings,
                identity_mode="reference" if ref_emb is not None else "longest-track")


def crop_square(frame, box, size, margin=0.9):
    """Square crop centred on the face box, edge-padded if it runs off the frame."""
    x1, y1, x2, y2 = box
    cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, int(max(x2 - x1, y2 - y1) * margin)
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_REPLICATE)
    return cv2.resize(padded[cy:cy + 2 * half, cx:cx + 2 * half], (size, size), interpolation=cv2.INTER_AREA)


# ---- Immich person-group source -------------------------------------------
#
# A group has no time axis, so there is no tracking: every face goes in as one
# track (the group *is* the identity) and build_dome() treats them all as the
# same person. Read-only against Immich: SELECT in a READ ONLY transaction, GETs
# against the API, nothing written to the library.

def immich_group_faces(group_id, limit=None):
    """Rows (face_id, asset_id, img_w, img_h, x1, y1, x2, y2) for a personGroupId.
    Deleted faces are skipped (the app's own queries count them)."""
    from dbNG import get_conn_ng, release_conn_ng
    conn = get_conn_ng()
    try:
        cur = conn.cursor()
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("""
            SELECT af.id, af."assetId", af."imageWidth", af."imageHeight",
                   af."boundingBoxX1", af."boundingBoxY1", af."boundingBoxX2", af."boundingBoxY2"
            FROM asset_face af
            WHERE af."personGroupId" = %s AND af."deletedAt" IS NULL
            ORDER BY af.id
        """, (group_id,))
        rows = cur.fetchall()
        cur.close()
    finally:
        release_conn_ng(conn)
    return rows[:limit] if limit else rows


def fetch_immich_image(asset_id, original=False, timeout=60):
    """Decode an Immich asset (preview by default) into a BGR array, or None."""
    import requests
    from configNG import IMMICH_BASE_URL, IMMICH_API_KEY
    url = f"{IMMICH_BASE_URL}/api/assets/{asset_id}/" + ("original" if original else "thumbnail?size=preview")
    r = requests.get(url, headers={"x-api-key": IMMICH_API_KEY}, timeout=timeout)
    if r.status_code != 200:
        return None
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def detect_immich_group(group_id, limit=None, progress=None, min_overlap=0.2, dup_sim=0.995):
    """Measure every face of an Immich group. Same fields as detect_and_track()
    plus asset / face_id / img_wh (the fetched preview's size, for rescaling
    the box onto the original). All faces share track 0.

    The detected face is the one overlapping the database's stored box (>=
    min_overlap IoU); others are skipped. Near-duplicates (embedding sim >=
    dup_sim and pose within 1 deg of an earlier face) are dropped.
    Returns (data, stats) where stats counts what was skipped.
    """
    app = get_face_app_ng()
    faces = immich_group_faces(group_id, limit)
    rec = {k: [] for k in ("frame", "track", "yaw", "pitch", "roll", "blur", "det", "bbox", "emb", "asset", "face_id", "img_wh")}
    stats = dict(total=len(faces), no_image=0, no_face=0, no_overlap=0, duplicates=0)
    for n, (fid, aid, iw, ih, bx1, by1, bx2, by2) in enumerate(faces, 1):
        img = fetch_immich_image(aid)
        if img is None:
            stats["no_image"] += 1
            continue
        h, w = img.shape[:2]
        sx, sy = w / float(iw or w), h / float(ih or h)
        db_box = (bx1 * sx, by1 * sy, bx2 * sx, by2 * sy)
        with _face_app_lock_ng:
            dets = app.get(img)
        if not dets:
            stats["no_face"] += 1
            continue
        best = max(dets, key=lambda f: iou(f.bbox, db_box))
        if iou(best.bbox, db_box) < min_overlap:
            stats["no_overlap"] += 1
            continue
        x1, y1, x2, y2 = (max(0, int(v)) for v in best.bbox)
        pitch, yaw, roll = (float(v) for v in best.pose)
        rec["frame"].append(len(rec["yaw"]))
        rec["track"].append(0)
        rec["yaw"].append(yaw)
        rec["pitch"].append(pitch)
        rec["roll"].append(roll)
        rec["blur"].append(get_blur_score_ng(img[y1:y2, x1:x2]))
        rec["det"].append(float(best.det_score))
        rec["bbox"].append((x1, y1, x2, y2))
        rec["emb"].append(best.normed_embedding)
        rec["asset"].append(str(aid))
        rec["face_id"].append(str(fid))
        rec["img_wh"].append((w, h))
        if progress and n % 50 == 0:
            progress(n, len(rec["yaw"]), 0)
    d = {k: np.array(v) for k, v in rec.items()}
    n = len(d["yaw"])
    if n:
        S = d["emb"] @ d["emb"].T
        keep = np.ones(n, bool)
        for i in range(n):
            if not keep[i]:
                continue
            m = (S[i] >= dup_sim) & (abs(d["yaw"] - d["yaw"][i]) < 1) & (abs(d["pitch"] - d["pitch"][i]) < 1)
            m[: i + 1] = False
            keep &= ~m
        stats["duplicates"] = int((~keep).sum())
        d = {k: v[keep] for k, v in d.items()}
    return d, stats


def crop_immich(d, i, size):
    """Square crop of face i from the ORIGINAL image (the box was measured on the
    preview, so it is scaled up). Returns None if the original can't be fetched."""
    img = fetch_immich_image(d["asset"][i], original=True)
    if img is None:
        return None
    pw, ph = d["img_wh"][i]
    h, w = img.shape[:2]
    sx, sy = w / float(pw), h / float(ph)
    x1, y1, x2, y2 = d["bbox"][i]
    return crop_square(img, (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)), size)
