from flask import Blueprint, request, jsonify, Response, send_file, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def index():
    return render_template("index.html")

