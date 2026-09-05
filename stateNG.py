"""NG twin of state.py -- in-memory job/cache stores for immichRingNG.

NG-only file: separate dicts from the current app's state.py, per the
NG duplication rule in APP_ARCHITECTURE_NOTES.md. Nothing here is
shared with the current app's _analysis_jobs / _preview_jobs /
_frame_cache -- state.py stays untouched.
"""

import threading

_analysis_jobs_ng = {}  # jobId -> {"status": ..., "results": [...], "videoBytes": ..., ...}

_preview_jobs_ng = {}  # previewId -> {"videoBytes": bytes, "fps": float, "frames": int}

_frame_cache_ng = {}

_frame_cache_lock_ng = threading.Lock()
