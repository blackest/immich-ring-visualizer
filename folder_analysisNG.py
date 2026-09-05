"""NG twin of folder_analysis.py -- in-memory image-set (folder/zip)
face-similarity analysis.

NG-only file: copied from folder_analysis.py as it exists today, wired to
detectionNG.py / stateNG.py / video_analysisNG.py instead of detection.py /
state.py / video_analysis.py, per the NG duplication rule in
APP_ARCHITECTURE_NOTES.md -- does not import from or call into
folder_analysis.py or any non-NG module.

Uses stateNG._face_app_lock_ng around every InsightFace call, same as
video_analysisNG.py -- InsightFace's FaceAnalysis wraps ONNXRuntime
sessions that aren't safe to call concurrently from multiple threads, and
a folder-analysis job runs in its own background thread just like a video
job, so it needs the same serialization.
"""

import os

import cv2
import numpy as np

from detectionNG import get_face_app_ng, get_blur_score_ng, pick_best_face_ng
from stateNG import _analysis_jobs_ng, _face_app_lock_ng
from video_analysisNG import bbox_frame_ratio_ng, summarize_resolutions_ng, vert_fill_ratio_ng, write_cache_frame_ng


def _read_image_ng(item):
    """item is either a filesystem path (str) or an (orig_name, bytes)
    tuple for a source held only in memory (folder/zip upload). Returns
    (frame_ndarray_or_None, orig_name)."""
    if isinstance(item, tuple):
        orig_name, data = item
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame, orig_name
    return cv2.imread(item), os.path.basename(item)


def run_folder_analysis_ng(job_id, images, sim_threshold, blur_threshold, ref_index=1, cache_format="jpg"):
    """Same pipeline as run_video_analysis_ng, but the 'frames' are a set
    of still images (a folder/zip upload) instead of decoded video
    frames. images is a pre-sorted list of (orig_name, bytes) tuples.
    Frame numbering follows list order so the rest of NG (ring, chart,
    export) can treat this exactly like a video analysis job."""
    job = _analysis_jobs_ng[job_id]
    try:
        face_app = get_face_app_ng()

        ref_idx = max(1, min(ref_index, len(images))) - 1
        ref_frame, ref_name = _read_image_ng(images[ref_idx])
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference image {ref_name}"
            return

        anchor_id = write_cache_frame_ng(job_id, "anchor", ref_frame, "jpg")
        job["anchorFrameId"] = anchor_id

        with _face_app_lock_ng:
            ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference image {ref_name}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []
        for i, item in enumerate(images):
            frame_idx = i + 1
            frame, orig_name = _read_image_ng(item)
            if frame is None:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name
                })
                job["results"] = results
                continue

            with _face_app_lock_ng:
                faces = face_app.get(frame)
            fh, fw = frame.shape[:2]
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "origName": orig_name, "width": fw, "height": fh
                })
                job["results"] = results
                continue

            face = pick_best_face_ng(faces, ref_embedding)
            sim_score = float(np.dot(ref_embedding, face.normed_embedding))

            pitch, yaw, roll = (float(p) for p in face.pose)

            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            blur_score = get_blur_score_ng(crop)

            fail_reason = None
            if not (sim_score > sim_threshold):
                fail_reason = "sim"
            elif not (blur_score > blur_threshold):
                fail_reason = "blur"

            passed = (fail_reason is None)

            if passed:
                frame_id = write_cache_frame_ng(job_id, frame_idx, frame, cache_format)
                job.setdefault("frame_embeddings", {})[frame_idx] = face.normed_embedding.tolist()
            else:
                frame_id = f"{job_id}_{frame_idx:05d}"

            results.append({
                "frame": frame_idx, "sim": sim_score, "blur": blur_score,
                "passed": passed, "failReason": fail_reason, "hasFace": True,
                "frameId": frame_id if passed else None,
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "bbox": [x1, y1, x2, y2], "origName": orig_name, "width": fw, "height": fh,
                "bboxRatio": bbox_frame_ratio_ng([x1, y1, x2, y2], fw, fh),
                "vertFillPct": vert_fill_ratio_ng([x1, y1, x2, y2], fw, fh)
            })
            job["results"] = results

        job["status"] = "done"
        job["resolutionSummary"] = summarize_resolutions_ng(results)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
