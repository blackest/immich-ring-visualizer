
from flask import Flask, request, jsonify, Response, send_file, render_template
import psycopg2
from psycopg2 import pool as pg_pool
import requests
import os
import tempfile
import threading
import uuid
import numpy as np
import zipfile
from detection import get_blur_score, get_face_app, pick_best_face
from state import _analysis_jobs
from video_analysis import bbox_frame_ratio, summarize_resolutions, vert_fill_ratio, write_cache_frame


def _read_image(item):
    """item is either a filesystem path (str) -- used for sources that
    legitimately already live on disk, e.g. character-sheet shots under
    exports/<name>/character/ -- or an (orig_name, bytes) tuple for a
    source held only in memory (folder/zip upload, Immich download),
    which should never touch disk before export. Returns
    (frame_ndarray_or_None, orig_name)."""
    import cv2
    if isinstance(item, tuple):
        orig_name, data = item
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame, orig_name
    return cv2.imread(item), os.path.basename(item)


def run_folder_analysis(job_id, images, sim_threshold, blur_threshold, ref_index=1, cache_format="jpg", always_cache=False):
    """Same pipeline as run_video_analysis, but the 'frames' are a set of
    still images -- a folder/zip upload, an Immich selection, or a
    character sheet's rendered shots -- instead of decoded video frames.
    images is a pre-sorted list, each entry either a filesystem path (str)
    or an in-memory (orig_name, bytes) tuple -- see _read_image(). Frame
    numbering follows list order so the rest of the app (ring, pose strip,
    export, cross-check) can treat this exactly like a video analysis job
    with zero changes.

    always_cache: normally a frame that fails sim/blur never gets
    write_cache_frame()'d, so it has no viewable/exportable image in the
    ring (frameId stays None) -- fine for real photo/video curation,
    where most candidate frames are expected to fail and caching all of
    them would be wasteful. Character-sheet shots (routes/phosphene.py's
    add_sheet_to_ring) are the opposite case: a handful of already-
    expensive HiDream renders that already exist as real files on disk,
    where a side/three-quarter shot scoring under the similarity
    threshold is often the intended, successful result (side profiles
    inherently score lower against a front-facing reference -- see the
    README's training-profile notes), not noise. Set True there so every
    shot is always cached and viewable regardless of pass/fail,
    independent of whatever the thresholds are set to."""
    job = _analysis_jobs[job_id]
    try:
        face_app = get_face_app()

        ref_idx = max(1, min(ref_index, len(images))) - 1
        ref_frame, ref_name = _read_image(images[ref_idx])
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference image {ref_name}"
            return

        anchor_id = write_cache_frame(job_id, "anchor", ref_frame, "jpg")
        job["anchorFrameId"] = anchor_id

        ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference image {ref_name}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []
        for i, item in enumerate(images):
            frame_idx = i + 1
            frame, orig_name = _read_image(item)
            if frame is None:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name
                })
                job["results"] = results
                continue

            faces = face_app.get(frame)
            fh, fw = frame.shape[:2]
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name, "width": fw, "height": fh
                })
                job["results"] = results
                continue

            face = pick_best_face(faces, ref_embedding)
            sim_score = float(np.dot(ref_embedding, face.normed_embedding))

            # InsightFace face.pose is [pitch, yaw, roll] in degrees.
            pitch, yaw, roll = (float(p) for p in face.pose)

            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            blur_score = get_blur_score(crop)

            fail_reason = None
            if not (sim_score > sim_threshold):
                fail_reason = "sim"
            elif not (blur_score > blur_threshold):
                fail_reason = "blur"

            passed = (fail_reason is None)

            if passed or always_cache:
                frame_id = write_cache_frame(job_id, frame_idx, frame, cache_format)
                job.setdefault("frame_embeddings", {})[frame_idx] = face.normed_embedding.tolist()
            else:
                frame_id = f"{job_id}_{frame_idx:05d}"

            results.append({
                "frame": frame_idx, "sim": sim_score, "blur": blur_score,
                "passed": passed, "failReason": fail_reason, "hasFace": True,
                "frameId": frame_id if (passed or always_cache) else None,
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "bbox": [x1, y1, x2, y2], "origName": orig_name, "width": fw, "height": fh,
                "bboxRatio": bbox_frame_ratio([x1, y1, x2, y2], fw, fh),
                "vertFillPct": vert_fill_ratio([x1, y1, x2, y2], fw, fh)
            })
            job["results"] = results

        job["status"] = "done"
        job["resolutionSummary"] = summarize_resolutions(results)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)

