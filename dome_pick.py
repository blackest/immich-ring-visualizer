#!/usr/bin/env python3
"""dome_pick.py -- pick one face per head-pose cell from a video.

Command-line wrapper over dome_engineNG.py. Read-only on the source (one or more
video files, or an Immich person group read through the DB/API); writes only into --out. See dome_engineNG.py for how identity, the seed ("neutral", cell
4.4) and the 7x7 dome are worked out.

--ref-image anchors identity to a face in a reference picture (needed when
several people share the scene). Repeat it, with pictures from different scenes,
and a track counts if it matches any; PATH:N picks the Nth-largest face.
--ref-thr sets the track-similarity cut-off; --seed-mode front picks the seed
by raw camera-facing angle instead of the most typical pose. --neutral YAW,PITCH
overrides both, for when you know what raw angle front-on reads as. With several videos the crops are
named <col>.<row>_c<clip>_f<frame>.png.

Output in --out: <col>.<row>_f<frame>.png crops, dome.csv, dome_sheet.png.
Cells are written col.row (4.4 = seed); top row = chin up, left = negative yaw.

Usage
  venv/bin/python dome_pick.py VIDEO --out exportsNG/dome_NAME [--step 2]
  venv/bin/python dome_pick.py CLIP1 CLIP2 ... --ref-image FRAME.png --out exportsNG/dome_NAME
  venv/bin/python dome_pick.py --immich-group GROUP_ID --out exportsNG/dome_NAME [--limit N]
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from dome_engineNG import (GRID, build_dome, crop_immich, crop_square, detect_and_track,
                           detect_immich_group, embedding_from_image, frames_from_cv2)

TILE = 170


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path, nargs="*", help="video file(s) (omit when using --immich-group)")
    ap.add_argument("--ref-image", action="append", metavar="PATH[:FACE]",
                    help="reference picture whose face defines the identity; repeat for several (best match counts). "
                         "FACE picks the Nth-largest face in that picture (default 0 = largest)")
    ap.add_argument("--neutral", metavar="YAW,PITCH",
                    help="raw angle (as InsightFace reads it) that means front-on for this source, e.g. -15,-3; "
                         "the seed is picked near it. Raw angles carry a per-source offset, so this beats guessing")
    ap.add_argument("--seed-mode", choices=("mode", "front"), default="mode",
                    help="mode: seed = the identity's most typical pose (default); front: the most camera-facing by raw angle")
    ap.add_argument("--ref-thr", type=float, default=0.3, help="min track-centroid similarity to the reference (default 0.3)")
    ap.add_argument("--immich-group", metavar="GROUP_ID", help="Immich personGroupId to use instead of a video")
    ap.add_argument("--limit", type=int, help="Immich only: measure at most N faces (for a quick test)")
    ap.add_argument("--out", type=Path, required=True, help="folder to create (must not already hold crops)")
    ap.add_argument("--step", type=int, default=2, help="analyse every Nth frame (default 2)")
    ap.add_argument("--ident-thr", type=float, default=0.5, help="track-centroid similarity to the main track to count as the same person")
    ap.add_argument("--yaw-step", type=float, default=15.0)
    ap.add_argument("--pitch-step", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=512, help="output crop size in px")
    args = ap.parse_args()

    if bool(args.videos) == bool(args.immich_group):
        raise SystemExit("give video file(s) or --immich-group, not both or neither")
    multi = len(args.videos) > 1
    neutral = None
    if args.neutral:
        try:
            ny, np_ = (float(v) for v in args.neutral.split(","))
            neutral = (ny, np_)
        except ValueError:
            raise SystemExit("--neutral takes two numbers, e.g. --neutral -15,-3")
    if args.out.exists() and any(args.out.glob("*.png")):
        raise SystemExit(f"{args.out} already contains crops; pick a new --out")

    t0 = time.time()
    progress = lambda i, faces, tracks: print(f"  at {i}  faces {faces}  tracks {tracks}  ({time.time() - t0:.0f}s)", flush=True)
    if args.immich_group:
        d, stats = detect_immich_group(args.immich_group, args.limit, progress)
        print(f"group {args.immich_group[:8]}: {stats}")
    else:
        import itertools
        d = detect_and_track(itertools.chain.from_iterable(
            frames_from_cv2(v, args.step, clip=k) for k, v in enumerate(args.videos)), progress)
    ref_emb = None
    try:
        if args.ref_image:
            embs = []
            for spec in args.ref_image:
                path, _, face = spec.rpartition(":")
                if not path or not face.isdigit():
                    path, face = spec, "0"
                e, rbox, nref = embedding_from_image(path, int(face))
                embs.append(e)
                print(f"reference: face {face} of {nref} in {Path(path).name}, box {rbox}")
            ref_emb = np.stack(embs)
        r = build_dome(d, args.ident_thr, args.yaw_step, args.pitch_step, ref_emb=ref_emb, ref_thr=args.ref_thr,
                       seed_mode=args.seed_mode, neutral=neutral)
    except ValueError as e:
        raise SystemExit(str(e))
    seed, I = r["seed"], r["identity"]
    print(f"{len(d['yaw'])} faces, {r['n_tracks']} tracks; identity = {len(I)} faces in "
          f"{len(set(d['track'][I].tolist()))} tracks (main track {r['main_track']}, identity by {r['identity_mode']})")
    for w_ in r["warnings"]:
        print(f"WARNING: {w_}")
    print(f"seed: {('clip %d frame %d' % (d['clip'][seed], d['frame'][seed])) if args.videos else 'asset ' + str(d['asset'][seed])[:8]}, raw yaw {d['yaw'][seed]:.1f} pitch {d['pitch'][seed]:.1f} roll {d['roll'][seed]:.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    caps = [cv2.VideoCapture(str(v)) for v in args.videos]

    def crop(i, size):
        if not caps:
            return crop_immich(d, i, size)
        cap = caps[int(d["clip"][i])]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(d["frame"][i]))
        ok, frame = cap.read()
        return crop_square(frame, d["bbox"][i], size) if ok else None

    sheet = np.full((GRID * (TILE + 18), GRID * TILE, 3), 22, np.uint8)
    rows = []
    for (c, rw), (i, n) in sorted(r["picks"].items(), key=lambda kv: (kv[0][1], kv[0][0])):
        full = crop(i, args.size)
        if full is None:
            continue
        label = int(d["frame"][i]) if args.videos else str(d["asset"][i])[:8]
        if not args.videos:
            name = f"{c}.{rw}_a{label}.png"
        elif multi:
            name = f"{c}.{rw}_c{int(d['clip'][i])}_f{label:05d}.png"
        else:
            name = f"{c}.{rw}_f{label:05d}.png"
        cv2.imwrite(str(args.out / name), full)
        ox, oy = (c - 1) * TILE, (rw - 1) * (TILE + 18)
        sheet[oy + 18:oy + 18 + TILE, ox:ox + TILE] = cv2.resize(full, (TILE, TILE), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, f"{c}.{rw} n={n} sim {r['sim_seed'][i]:.2f}", (ox + 4, oy + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (100, 255, 100) if (c, rw) == (4, 4) else (120, 220, 255), 1)
        rows.append([f"{c}.{rw}", name, label, n, round(float(r["dy"][i]), 1), round(float(r["dp"][i]), 1),
                     round(float(r["sim_seed"][i]), 3), round(float(d["blur"][i]), 1), int(r["size"][i])])
    cv2.imwrite(str(args.out / "dome_sheet.png"), sheet)
    with open(args.out / "dome.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "file", "frame_or_asset", "faces_in_cell", "yaw_offset", "pitch_offset", "sim_to_seed", "blur", "face_px"])
        w.writerows(rows)
    print(f"wrote {len(rows)} of {GRID * GRID} cells to {args.out}")


if __name__ == "__main__":
    main()
