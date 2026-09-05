"""NG twin of detection.py -- InsightFace face-detection helpers.

NG-only file: a separate lazy-loaded model instance from the current
app's detection.py, per the NG duplication rule in
APP_ARCHITECTURE_NOTES.md ("every file NG needs gets copied to an
NG-suffixed twin ... NG does not import from or call into the original
modules at all"). detection.py stays untouched.
"""

import numpy as np

_face_app_ng = None


def get_face_app_ng():
    """Lazy-load InsightFace, preferring Apple Silicon CoreML acceleration."""
    global _face_app_ng
    if _face_app_ng is None:
        from insightface.app import FaceAnalysis
        providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        _face_app_ng = FaceAnalysis(name='buffalo_l', providers=providers)
        _face_app_ng.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app_ng


def pick_best_face_ng(faces, ref_embedding):
    if len(faces) == 1:
        return faces[0]
    return max(faces, key=lambda f: float(np.dot(ref_embedding, f.normed_embedding)))


def get_blur_score_ng(crop):
    import cv2
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_largest_face_ng(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
