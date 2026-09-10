"""immichRingNG -- settings-cog actions (venv maintenance).

NG-only file, additive. First (and so far only) action: update yt-dlp
in the project's own venv via pip, run as a subprocess so the running
Flask process doesn't need to import/reload pip machinery. Deliberately
a curated single endpoint rather than an open-ended "run any command"
one -- per the design note, this is meant to grow with a reviewed list
of specific actions, not become a shell.
"""

import subprocess
import sys

from flask import Blueprint, jsonify

settingsNG_bp = Blueprint("settingsNG", __name__)


@settingsNG_bp.route("/api/ng/settings/update-ytdlp", methods=["POST"])
def update_ytdlp_ng():
    """Run `pip install --upgrade yt-dlp` in the current venv (the same
    interpreter running this Flask process) and report back pass/fail
    plus the tail of pip's output for visibility."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": False,
            "error": "Timed out after 120s waiting for pip.",
        }), 504
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-15:])

    if result.returncode != 0:
        return jsonify({
            "ok": False,
            "error": f"pip exited with code {result.returncode}",
            "output": tail,
        }), 500

    return jsonify({"ok": True, "output": tail})
