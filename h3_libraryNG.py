"""H3 prompt library: read the ostris/minimax_h3_1k folder (NNNNNN.mp4 + NNNNNN.txt pairs) for browsing.

Input: a folder path. Output: a list of small records (id, look, shots, dialogue, music, a short summary) and,
on request, one full record with the three prompt fields. Nothing is written to disk; the index lives in memory
and is rebuilt when the folder's modified time changes.

A .txt file holds three labelled fields: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
"full" is those three joined with blank lines, which is the layout the H3 prompt box and the Enhanced H3 pane
already use.
"""

import os
import re

_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
_FIELD_RE = re.compile(r"^(%s):\s*" % "|".join(_FIELDS), re.M)
_ID_RE = re.compile(r"^\d{6}$")
_SHOT_RE = re.compile(r"\[Shot \d+\]")
_DIALOGUE_RE = re.compile(r"<d>")

# First match wins, so order matters. Matched against the start of shot 1, where the look is stated.
_LOOKS = (
    ("Game / 3D", re.compile(r"video-game|game render|game engine|low-poly|voxel|3d-render|cutscene|machinima|wargaming", re.I)),
    ("Animation", re.compile(r"animat|anime|cartoon|stop-motion|claymation|cel[- ]|pixar|cgi", re.I)),
    ("Documentary", re.compile(r"documentary|nature|newsreel|news ", re.I)),
    ("Vintage", re.compile(r"vhs|35mm|16mm|8mm|black-and-white|technicolor|silent-era|film noir|sepia|camcorder|minidv|shot-on-video|sitcom|soap opera", re.I)),
    ("Found footage", re.compile(r"found[- ]footage|dashcam|body ?cam|security|cctv|webcam|surveillance|doorbell|gopro|helmet|fpv|drone|bleachers|amateur|glitch", re.I)),
    ("Screen / phone", re.compile(r"screen recording|livestream|vlog|vertical|phone|gameplay|video game", re.I)),
    ("Live-action", re.compile(r"live-action|cinematic|film", re.I)),
)

_cache = {"key": None, "items": None, "by_id": None}


def parse_prompt_text(text):
    """Split a .txt file into its three fields. Missing fields come back as ''."""
    out = {f: "" for f in _FIELDS}
    marks = list(_FIELD_RE.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.end():end].strip()
    return out


def classify_look(description):
    head = _SHOT_RE.sub("", description[:260])
    for name, rx in _LOOKS:
        if rx.search(head):
            return name
    return "Other"


def full_prompt(fields):
    """The three fields in the layout the app's prompt boxes use."""
    return "\n\n".join(
        "%s: %s" % (f, fields[f]) for f in _FIELDS if fields.get(f)
    )


def _summary(description, limit=170):
    s = _SHOT_RE.sub("", description).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


def _record(clip_id, fields, with_text):
    desc = fields["integrated_multimodal_description"]
    music = fields["non_diegetic_music"]
    rec = {
        "id": clip_id,
        "look": classify_look(desc),
        "shots": max(1, len(_SHOT_RE.findall(desc))),
        "dialogue": bool(_DIALOGUE_RE.search(desc)),
        "music": bool(music) and music.strip().upper() != "N/A",
        "summary": _summary(desc),
    }
    if with_text:
        rec.update(fields)
        rec["full"] = full_prompt(fields)
    return rec


def _build(folder):
    items, by_id = [], {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext != ".txt" or not _ID_RE.match(stem):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                fields = parse_prompt_text(fh.read())
        except OSError:
            continue
        if not fields["integrated_multimodal_description"]:
            continue
        rec = _record(stem, fields, with_text=True)
        by_id[stem] = rec
        items.append(rec)
    return items, by_id


def _index(folder):
    key = (folder, os.stat(folder).st_mtime_ns)
    if _cache["key"] != key:
        items, by_id = _build(folder)
        _cache.update(key=key, items=items, by_id=by_id)
    return _cache["items"], _cache["by_id"]


def list_clips(folder, query="", look="", shots=0, dialogue=None, music=None, offset=0, limit=60):
    """Filtered page of short records plus the total match count and the look names with counts.

    query: every word must appear somewhere in the three fields (case-insensitive).
    dialogue / music: True keeps only clips that have it, False only clips that lack it, None doesn't filter.
    """
    items, _ = _index(folder)
    words = [w for w in query.lower().split() if w]
    looks = {}
    matched = []
    for rec in items:
        if words:
            hay = (rec["full"]).lower()
            if not all(w in hay for w in words):
                continue
        if shots and rec["shots"] != shots:
            continue
        if dialogue is not None and rec["dialogue"] != dialogue:
            continue
        if music is not None and rec["music"] != music:
            continue
        looks[rec["look"]] = looks.get(rec["look"], 0) + 1
        if look and rec["look"] != look:
            continue
        matched.append(rec)
    keys = ("id", "look", "shots", "dialogue", "music", "summary")
    page = [{k: r[k] for k in keys} for r in matched[offset:offset + limit]]
    return {"total": len(matched), "items": page, "looks": looks}


def get_clip(folder, clip_id):
    """One full record, or None when the id isn't in the folder."""
    if not _ID_RE.match(clip_id or ""):
        return None
    _, by_id = _index(folder)
    return by_id.get(clip_id)


def video_path(folder, clip_id):
    """Path of the clip's mp4, or None. The id is checked so nothing outside the folder can be asked for."""
    if not _ID_RE.match(clip_id or ""):
        return None
    p = os.path.join(folder, clip_id + ".mp4")
    return p if os.path.isfile(p) else None
