
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
from state import _analysis_jobs, _frame_cache, _frame_cache_lock

def run_video_analysis(job_id, video_bytes, sim_threshold, blur_threshold, ref_frame_idx=1, cache_format="jpg", start_sec=None, end_sec=None):
    job = _analysis_jobs[job_id]
    try:
        face_app = get_face_app()
        mv = MemoryVideo(video_bytes)

        # start_sec/end_sec (seconds into the clip) are the natural unit for
        # a person to specify - "skip to minute 54" - but iter_frames()
        # works in frame numbers, so convert here once fps is known rather
        # than making every caller do the arithmetic themselves.
        start_frame = int(start_sec * mv.fps) + 1 if start_sec else None
        end_frame = int(end_sec * mv.fps) + 1 if end_sec else None

        target_frame = max(1, ref_frame_idx)
        ref_frame = mv.seek_frame(target_frame)
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference frame {target_frame} from video"
            return

        # anchor thumbnail (single small file, not a per-frame cache issue)
        # kept as the one on-disk artifact for the "Frame N (Anchor)" UI
        # to load quickly without re-decoding from memory every time
        anchor_id = write_cache_frame(job_id, "anchor", ref_frame, "jpg")
        job["anchorFrameId"] = anchor_id

        ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference frame {target_frame}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []

        # start_frame/end_frame scope the analysis pass to a window of the
        # clip instead of always walking start-to-finish. iter_frames()
        # seeks straight to start_frame rather than decoding through
        # everything before it, so skipping "10 minutes of nobody in
        # frame yet" on a long clip costs nothing instead of costing a
        # full decode of the part you already know is unusable.
        for frame_idx, frame in mv.iter_frames(start_frame=start_frame, end_frame=end_frame):
            faces = face_app.get(frame)
            fh, fw = frame.shape[:2]
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "width": fw, "height": fh
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
                "bbox": [x1, y1, x2, y2], "width": fw, "height": fh,
                "bboxRatio": bbox_frame_ratio([x1, y1, x2, y2], fw, fh),
                "vertFillPct": vert_fill_ratio([x1, y1, x2, y2], fw, fh)
            })
            job["results"] = results

        job["status"] = "done"
        job["resolutionSummary"] = summarize_resolutions(results)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)

def summarize_resolutions(results):
    """Roll per-frame width/height into something worth glancing at: the
    dominant resolution plus whether anything deviates from it. Lets you
    catch 'these are actually 1024x1024, why are we exporting at 512' at
    a glance instead of after the fact."""
    from collections import Counter
    dims = [(r["width"], r["height"]) for r in results if r.get("width") and r.get("height")]
    if not dims:
        return None
    counts = Counter(dims)
    (mode_w, mode_h), mode_count = counts.most_common(1)[0]
    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]
    return {
        "modeWidth": mode_w, "modeHeight": mode_h,
        "modeCount": mode_count, "totalCount": len(dims),
        "uniform": len(counts) == 1,
        "minWidth": min(widths), "maxWidth": max(widths),
        "minHeight": min(heights), "maxHeight": max(heights),
    }

def bbox_frame_ratio(bbox, frame_w, frame_h):
    """Face bbox area as a fraction of the full frame area - a rough proxy
    for shot scale. Near 0 means a small face in a wide/distant shot, near
    1 means the face fills most of the frame (extreme close-up). Useful
    alongside pitch/yaw/blur for judging what kind of shot a candidate
    frame actually is, not just whether the face matched and was sharp.

    NOTE: this is area-based, so it's skewed by aspect ratio - a 1000x500
    frame and a 500x500 frame with the *same* face at the *same* pixel
    size report different ratios here, because the wider frame's extra
    width dilutes the denominator even though the face's on-screen scale
    hasn't changed. Kept for backward compatibility; prefer
    vert_fill_ratio() for comparing shot scale across frames of different
    aspect ratios."""
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    return float((bw * bh) / (frame_w * frame_h))

def vert_fill_ratio(bbox, frame_w, frame_h):
    """Face bbox height as a fraction of the frame's SHORTER dimension -
    an aspect-ratio-independent measure of shot scale. Normalizing by the
    shorter (usually vertical) dimension means a 1000x500 wide crop and a
    500x500 square crop containing the exact same face at the exact same
    pixel size report the same fill percentage, since only the
    constraining dimension is used rather than total frame area."""
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    bh = max(0, y2 - y1)
    short_side = min(frame_w, frame_h)
    if short_side <= 0:
        return 0.0
    return float(bh / short_side)

def write_cache_frame(job_id, frame_key, img, cache_format="jpg"):
    """Encodes an analysis-pipeline frame and holds it in RAM (never
    written to disk), returning the frame_id used to look it up again.
    frame_key is either a frame number (int) or the literal string
    "anchor" for the one reference-frame thumbnail each job has - both
    are stored under the same string-keyed cache so find_cache_frame()
    doesn't need to special-case which kind of frame_id it was asked for.
    PNG is lossless but ~10x the size and ~10x slower to encode than JPEG
    q88 (measured: ~2.2MB/71ms vs ~220KB/6ms per 1080p frame) - opt-in via
    the 'Cache as PNG' checkbox rather than a global default, since JPEG
    is the sensible default and PNG is for someone who specifically wants
    the cache itself at full fidelity (e.g. inspecting thumbnails closely
    before deciding what to export)."""
    import cv2
    suffix = "anchor" if frame_key == "anchor" else f"{frame_key:05d}"
    frame_id = f"{job_id}_{suffix}"
    if cache_format == "png":
        ok, buf = cv2.imencode(".png", img)
        mimetype = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        mimetype = "image/jpeg"
    if ok:
        with _frame_cache_lock:
            _frame_cache.setdefault(job_id, {})[suffix] = (buf.tobytes(), mimetype)
    return frame_id

def find_cache_frame(frame_id):
    """Locates a cached frame's bytes by frame_id ('jobid_00042' or
    'jobid_anchor'), regardless of which format it was cached in. Returns
    (bytes, mimetype) or (None, None) if not found (e.g. job/frame never
    cached, or the server restarted and the in-memory cache was lost -
    callers already treat a cache miss as "fall back to the true
    original", which is strictly better data anyway, so losing this cache
    on restart costs nothing beyond a slower re-read)."""
    try:
        job_id, suffix = frame_id.rsplit("_", 1)
    except ValueError:
        return None, None
    with _frame_cache_lock:
        entry = _frame_cache.get(job_id, {}).get(suffix)
    if entry is None:
        return None, None
    return entry

def clear_frame_cache(job_id):
    """Frees a job's in-memory frame cache. Call when a job is discarded
    (new analysis started, tab closed and job GC'd, etc.) so RAM doesn't
    accumulate indefinitely across a long session the way the old disk
    cache silently would have on the SSD."""
    with _frame_cache_lock:
        _frame_cache.pop(job_id, None)

class MemoryVideo:
    """Wraps a PyAV container opened directly from in-memory bytes - the
    uploaded video is never written to disk at all. Provides the two
    access patterns the app actually needs:

    - seek_frame(n): random access to an exact 1-based frame number, for
      picking a reference/anchor frame or re-reading one specific frame
      at export time. PyAV seeks by timestamp, not frame index, so this
      seeks near the target time then decodes forward to the exact frame
      - verified against cv2.VideoCapture's own frame numbering (pixel-
      identical results) before relying on it here.
    - iter_frames(): sequential decode of the whole stream in order, for
      the main per-frame analysis loop - this is the efficient path
      PyAV is built for, much cheaper than seeking frame-by-frame.

    fps/frame_count are read once at open time so callers don't need
    their own cv2.VideoCapture just to ask "how many frames is this and
    what's its rate" - previously a second full VideoCapture open."""

    def __init__(self, video_bytes):
        self._video_bytes = video_bytes  # keep the source bytes so a
                                          # fresh container can be reopened
                                          # for a new seek/iteration pass -
                                          # PyAV containers are single-pass
        self.fps, self.frame_count, self.width, self.height = self._probe()

    def _open_container(self):
        buf = io.BytesIO(self._video_bytes)
        container = av.open(buf)
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        return container, stream

    def _probe(self):
        container, stream = self._open_container()
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        frame_count = stream.frames or 0
        width, height = stream.width, stream.height
        container.close()
        return fps, frame_count, width, height

    def seek_frame(self, frame_idx_1based):
        """Returns the exact frame (1-based, matching cv2's numbering
        convention used throughout the rest of this app) as a BGR numpy
        array, or None if out of range."""
        container, stream = self._open_container()
        try:
            target_time = max(0, (frame_idx_1based - 1) / self.fps)
            container.seek(int(target_time / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                frame_time = float(frame.pts * stream.time_base)
                idx = round(frame_time * self.fps) + 1
                if idx >= frame_idx_1based:
                    return frame.to_ndarray(format="bgr24")
            return None
        finally:
            container.close()

    def iter_frames(self, start_frame=None, end_frame=None):
        """Yields (frame_idx_1based, bgr_ndarray) for frames in order,
        optionally scoped to [start_frame, end_frame] inclusive (both
        1-based, either end optional). Seeks straight to start_frame
        rather than decoding through everything before it - the same
        keyframe-seek-then-decode-forward approach seek_frame() uses, so
        a 90-minute clip scoped to minute 54-55 doesn't cost decoding the
        54 minutes before it. Stops as soon as end_frame is passed rather
        than decoding to EOF and discarding the rest."""
        container, stream = self._open_container()
        try:
            idx = 0
            if start_frame and start_frame > 1:
                target_time = max(0, (start_frame - 1) / self.fps)
                container.seek(int(target_time / stream.time_base), stream=stream)
                # seeking lands at the nearest keyframe at/before the
                # target, which is usually earlier than start_frame -
                # figure out where we actually landed by peeking the
                # first decoded frame's own timestamp, same idx math
                # seek_frame() uses, then continue from there.
                decoder = container.decode(stream)
                first = next(decoder, None)
                if first is None:
                    return
                frame_time = float(first.pts * stream.time_base)
                idx = round(frame_time * self.fps) + 1
                if idx >= start_frame and (end_frame is None or idx <= end_frame):
                    yield idx, first.to_ndarray(format="bgr24")
                for frame in decoder:
                    idx += 1
                    if idx < start_frame:
                        continue
                    if end_frame is not None and idx > end_frame:
                        return
                    yield idx, frame.to_ndarray(format="bgr24")
            else:
                for frame in container.decode(stream):
                    idx += 1
                    if end_frame is not None and idx > end_frame:
                        return
                    yield idx, frame.to_ndarray(format="bgr24")
        finally:
            container.close()

