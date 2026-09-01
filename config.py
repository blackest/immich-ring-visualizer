
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

IMMICH_BASE_URL = "http://localhost:2283"

IMMICH_API_KEY = "L4mP37A5kNWHPME0024ms2SGep7KR8xP4oAB9UNGqOM"

PHOSPHENE_BASE_URL = os.environ.get("PHOSPHENE_BASE_URL", "http://127.0.0.1:8198")

FRAME_STORE = tempfile.mkdtemp(prefix="ringviz_frames_")

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

