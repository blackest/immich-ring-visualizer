"""immichRingNG -- secondary route for the new tabbed / three-axis layout.

NG-only file: does not import from or call into any original route
module. See APP_ARCHITECTURE_NOTES.md (repo root) for the full plan.
Original files stay read-only for this stage of development -- this is
a brand new blueprint, not a modification of routes/main.py.
"""

from flask import Blueprint, render_template

mainNG_bp = Blueprint("mainNG", __name__)


@mainNG_bp.route("/immichRingNG")
def index_ng():
    return render_template("indexNG.html")
