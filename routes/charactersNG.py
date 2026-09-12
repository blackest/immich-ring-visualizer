"""NG-only blueprint: cross-view character listing/persistence, backing
the Generate view's character-picker landing state (static/
characterPickerNG.js) and characterIONG.js's Save button.

Not generation-specific (unlike routes/generateNG.py) -- a character
here can be video/folder/Immich-analysis-only, never having touched
Generate at all, as long as a project was explicitly Saved onto it.
See character_sheetNG.py's list_characters_ng()/save_character_project_
doc_ng()/load_character_project_doc_ng() for the storage details.
"""

from flask import Blueprint, jsonify, request

import character_sheetNG as character_sheet

charactersNG_bp = Blueprint("charactersNG", __name__)


@charactersNG_bp.route("/api/ng/characters", methods=["GET"])
def list_characters_route_ng():
    """Feeds the Generate view's landing-state picker grid -- every
    known character (registered generation bundle and/or a Saved
    project doc), newest-saved first."""
    return jsonify({"characters": character_sheet.list_characters_ng()})


@charactersNG_bp.route("/api/ng/characters/<character_id>/doc", methods=["GET"])
def get_character_doc_route_ng(character_id):
    """Returns the Saved project doc (characterIONG.js's .character.json
    shape) for one character, so the picker grid can load it straight
    into a new tab without a file round-trip. 404 if nothing was ever
    Saved for this id (a generation-only character, e.g.)."""
    try:
        cid = character_sheet._safe_id_ng(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    doc = character_sheet.load_character_project_doc_ng(cid)
    if doc is None:
        return jsonify({"error": f"no saved project for character {cid!r}"}), 404
    return jsonify(doc)


@charactersNG_bp.route("/api/ng/characters/<character_id>/doc", methods=["POST"])
def save_character_doc_route_ng(character_id):
    """Mirrors characterIONG.js's client-side .character.json download
    onto exportsNG/<id>/character.json server-side, so the picker grid
    has something to list/load even before the user manually re-opens a
    downloaded file. Body: the same doc object save() already builds
    and downloads -- stored as-is, no server-side validation of its
    shape beyond being valid JSON."""
    try:
        cid = character_sheet._safe_id_ng(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    doc = request.get_json(silent=True)
    if not isinstance(doc, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    character_sheet.save_character_project_doc_ng(cid, doc)
    return jsonify({"ok": True}), 200


@charactersNG_bp.route("/api/ng/characters/<character_id>", methods=["DELETE"])
def delete_character_route_ng(character_id):
    """Removes a character (its whole exportsNG/<id>/ dir -- generation
    bundle and any Saved project doc) so it drops off the picker grid.
    Used to clear out test/junk characters."""
    try:
        cid = character_sheet._safe_id_ng(character_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    deleted = character_sheet.delete_character_ng(cid)
    return jsonify({"ok": deleted})
