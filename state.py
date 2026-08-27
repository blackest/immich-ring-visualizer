
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

_analysis_jobs = {}  # jobId -> {"status": ..., "results": [...], "videoBytes": ..., ...}

_preview_jobs = {}  # previewId -> {"videoBytes": bytes, "fps": float, "frames": int}

_frame_cache = {}

_frame_cache_lock = threading.Lock()

