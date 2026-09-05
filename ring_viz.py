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
from routes.phosphene import phosphene_bp
from routes.mainNG import mainNG_bp  # NG: new blueprint, no changes to existing ones
from routes.videoNG import videoNG_bp  # NG: new blueprint, no changes to existing ones
from routes.immichNG import immichNG_bp  # NG: new blueprint, no changes to existing ones

app = Flask(__name__)
app.register_blueprint(video_bp)
app.register_blueprint(folder_bp)
app.register_blueprint(export_bp)
app.register_blueprint(immich_bp)
app.register_blueprint(main_bp)
app.register_blueprint(phosphene_bp)
app.register_blueprint(mainNG_bp)  # NG: /immichRingNG, additive only
app.register_blueprint(videoNG_bp)  # NG: /api/ng/preview-video etc, additive only
app.register_blueprint(immichNG_bp)  # NG: /api/ng/find-by-filename etc, additive only
if __name__ == "__main__":
    # use_reloader on its own (without debug=True) restarts the process
    # when a .py file changes -- picks up backend fixes without a manual
    # restart -- without also turning on the in-browser interactive
    # debugger, which would be a real remote-code-execution surface given
    # this binds 0.0.0.0 and is reachable over Tailscale, not just
    # localhost. Note: a reload triggered mid-job (HiDream render, video
    # analysis) still kills it silently -- job state is in-memory only,
    # nothing here makes that safe.
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=True)
