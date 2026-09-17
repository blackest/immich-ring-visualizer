"""NG twin of state.py -- in-memory job/cache stores for immichRingNG.

NG-only file: separate dicts from the current app's state.py, per the
NG duplication rule in APP_ARCHITECTURE_NOTES.md. Nothing here is
shared with the current app's _analysis_jobs / _preview_jobs /
_frame_cache -- state.py stays untouched.
"""

import threading

_analysis_jobs_ng = {}  # jobId -> {"status": ..., "results": [...], "videoBytes": ..., ...}

_split_jobs_ng = {}  # jobId -> {"status": ..., "results": [...], ...} -- no-analysis frame split, see video_analysisNG.run_video_split_ng

_hdmulti_jobs_ng = {}  # jobId -> {"status": ..., "error": ..., "pngPath": ..., "seed": ...} -- HiDream edit/multi-ref, see routes/hdmultiNG.py

_comfy_extracts_ng = {}  # extractId -> {"graph": {...}} -- a workflow extracted from a PNG's embedded metadata, see routes/comfyNG.py
_comfy_jobs_ng = {}  # jobId -> {"promptId": ..., "status": ..., "resultFilename": ...} -- a submitted ComfyUI run, see routes/comfyNG.py

_preview_jobs_ng = {}  # previewId -> {"videoBytes": bytes, "fps": float, "frames": int}

_frame_cache_ng = {}

_frame_cache_lock_ng = threading.Lock()

# InsightFace's FaceAnalysis wraps ONNXRuntime sessions that are not safe
# to call concurrently from multiple threads. Now that each project tab
# runs its analysis in its own background thread (see routes/videoNG.py),
# two projects analyzing at once would otherwise hit the same model
# object simultaneously and corrupt each other's results. This lock
# serializes actual inference calls; job orchestration (threads, per-job
# dicts, polling) stays fully parallel -- only the model call itself is
# single-file.
_face_app_lock_ng = threading.Lock()
