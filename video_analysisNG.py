"""NG twin of video_analysis.py -- in-memory video decode + per-frame
face-similarity analysis.

NG-only file: copied from video_analysis.py as it exists today (PyAV
in-memory decode, no disk writes for the analysis pass itself), wired
to detectionNG.py / stateNG.py instead of detection.py / state.py, per
the NG duplication rule in APP_ARCHITECTURE_NOTES.md -- does not import
from or call into video_analysis.py, detection.py, or state.py.
"""

import io
import av
import numpy as np

from detectionNG import get_face_app_ng, get_blur_score_ng, pick_best_face_ng
from stateNG import _analysis_jobs_ng, _frame_cache_ng, _frame_cache_lock_ng, _face_app_lock_ng


def run_video_analysis_ng(job_id, video_bytes, sim_threshold, blur_threshold, ref_frame_idx=1, cache_format="jpg", start_sec=None, end_sec=None):
    job = _analysis_jobs_ng[job_id]
    try:
        face_app = get_face_app_ng()
        mv = MemoryVideo(video_bytes)

        start_frame = int(start_sec * mv.fps) + 1 if start_sec else None
        end_frame = int(end_sec * mv.fps) + 1 if end_sec else None

        target_frame = max(1, ref_frame_idx)
        ref_frame = mv.seek_frame(target_frame)
        if ref_frame is None:
            job["status"] = "error"
            job["error"] = f"Could not read reference frame {target_frame} from video"
            return

        anchor_id = write_cache_frame_ng(job_id, "anchor", ref_frame, "jpg")
        job["anchorFrameId"] = anchor_id

        with _face_app_lock_ng:
            ref_faces = face_app.get(ref_frame)
        if not ref_faces:
            job["status"] = "error"
            job["error"] = f"No face detected in reference frame {target_frame}"
            return
        ref_embedding = ref_faces[0].normed_embedding

        results = []

        for frame_idx, frame in mv.iter_frames(start_frame=start_frame, end_frame=end_frame):
            with _face_app_lock_ng:
                faces = face_app.get(frame)
            fh, fw = frame.shape[:2]
            if not faces:
                results.append({
                    "frame": frame_idx, "sim": 0.0, "blur": 0.0, "passed": False, "hasFace": False,
                    "yaw": None, "pitch": None, "roll": None, "width": fw, "height": fh
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
                "bbox": [x1, y1, x2, y2], "width": fw, "height": fh,
                "bboxRatio": bbox_frame_ratio_ng([x1, y1, x2, y2], fw, fh),
                "vertFillPct": vert_fill_ratio_ng([x1, y1, x2, y2], fw, fh)
            })
            job["results"] = results

        job["status"] = "done"
        job["resolutionSummary"] = summarize_resolutions_ng(results)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def summarize_resolutions_ng(results):
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


def bbox_frame_ratio_ng(bbox, frame_w, frame_h):
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    return float((bw * bh) / (frame_w * frame_h))


def vert_fill_ratio_ng(bbox, frame_w, frame_h):
    if not bbox or frame_w <= 0 or frame_h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    bh = max(0, y2 - y1)
    short_side = min(frame_w, frame_h)
    if short_side <= 0:
        return 0.0
    return float(bh / short_side)


def write_cache_frame_ng(job_id, frame_key, img, cache_format="jpg"):
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
        with _frame_cache_lock_ng:
            _frame_cache_ng.setdefault(job_id, {})[suffix] = (buf.tobytes(), mimetype)
    return frame_id


def find_cache_frame_ng(frame_id):
    try:
        job_id, suffix = frame_id.rsplit("_", 1)
    except ValueError:
        return None, None
    with _frame_cache_lock_ng:
        entry = _frame_cache_ng.get(job_id, {}).get(suffix)
    if entry is None:
        return None, None
    return entry


def clear_frame_cache_ng(job_id):
    with _frame_cache_lock_ng:
        _frame_cache_ng.pop(job_id, None)


class MemoryVideo:
    """Wraps a PyAV container opened directly from in-memory bytes -- the
    uploaded video is never written to disk. Copied from video_analysis.py's
    MemoryVideo (see that file for the fuller rationale comments)."""

    def __init__(self, video_bytes):
        self._video_bytes = video_bytes
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
        container, stream = self._open_container()
        try:
            idx = 0
            if start_frame and start_frame > 1:
                target_time = max(0, (start_frame - 1) / self.fps)
                container.seek(int(target_time / stream.time_base), stream=stream)
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
