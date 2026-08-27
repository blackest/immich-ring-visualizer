
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

_face_app = None

def get_face_app():
    """Lazy-load InsightFace, preferring Apple Silicon CoreML acceleration."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        _face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app

def pick_best_face(faces, ref_embedding):
    if len(faces) == 1:
        return faces[0]
    return max(faces, key=lambda f: float(np.dot(ref_embedding, f.normed_embedding)))

def get_blur_score(crop):
    import cv2
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def pick_largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

