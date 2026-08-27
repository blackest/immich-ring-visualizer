#!/usr/bin/env python3
"""
Immich Ring Visualizer
-----------------------
A small local tool: pick a reference face, see nearest matches arranged
in confidence rings around it. Click any thumbnail to recenter.

Run:
    pip install flask psycopg2-binary requests --break-system-packages
    python3 ring_viz.py

Then open http://localhost:5050
"""

from flask import Flask

from routes.video import video_bp
from routes.folder import folder_bp
from routes.export import export_bp
from routes.immich import immich_bp
from routes.main import main_bp

app = Flask(__name__)
app.register_blueprint(video_bp)
app.register_blueprint(folder_bp)
app.register_blueprint(export_bp)
app.register_blueprint(immich_bp)
app.register_blueprint(main_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
