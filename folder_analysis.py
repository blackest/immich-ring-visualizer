
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

def run_folder_analysis(job_id, image_paths, sim_threshold, blur_threshold, ref_index=1, cache_format="jpg"):
    """Same pipeline as run_video_analysis, but the 'frames' are a set of
    still images from a folder/zip upload instead of decoded video frames.
    image_paths is a pre-sorted list; frame numbering follows that order so
    the rest of the app (ring, pose strip, export, cross-check) can treat
    this exactly like a video analysis job with zero changes."""
    import cv2

    job = _analysis_jobs[job_id]
    try:
        face_app = get_face_app()

        ref_idx = max(1, min(ref_index, len(image_paths))) - 1
        ref_frame = cv2.imread(image_paths[ref_idx])
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference image {os.path.basename(image_paths[ref_idx])}"
            return

        anchor_id = write_cache_frame(job_id, "anchor", ref_frame, "jpg")
        job["anchorFrameId"] = anchor_id

        ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference image {os.path.basename(image_paths[ref_idx])}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []
        for i, img_path in enumerate(image_paths):
            frame_idx = i + 1
            orig_name = os.path.basename(img_path)
            frame = cv2.imread(img_path)
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

            if passed:
                frame_id = write_cache_frame(job_id, frame_idx, frame, cache_format)
                job.setdefault("frame_embeddings", {})[frame_idx] = face.normed_embedding.tolist()
            else:
                frame_id = f"{job_id}_{frame_idx:05d}"

            results.append({
                "frame": frame_idx, "sim": sim_score, "blur": blur_score,
                "passed": passed, "failReason": fail_reason, "hasFace": True,
                "frameId": frame_id if passed else None,
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

