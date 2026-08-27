from flask import Blueprint, request, jsonify, Response, send_file
from ring_viz import app

main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def index():
    return render_template("index.html")

