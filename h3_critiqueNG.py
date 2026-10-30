"""immichRingNG -- "Ask Gemma" about one window of a chained H3 render.

Gemma (a vision model, the same server the compiler uses) is shown one contact sheet: the previous
window's last frame, two anchor frames from window 1 that show how the character and place were first
established, and six frames across this window. It describes the anchor look, describes this window,
lists the differences and what in the prompt probably caused them, and writes a rewritten window prompt
that restates the look. Nothing renders and nothing is replaced: the reply fills the window's prompt box
in the Reroll panel (h3_rerollNG.py), where the person reads it, edits it and decides.

Found by testing it on windows with known faults: asked only "what is wrong", it called a clear
wardrobe change "fine"; made to describe both looks first and then compare, it found the change, and
it traced glowing eyes to the prompt's "reflecting the yellow light". Its own verdict was not
reliable (it also listed drift on a window that was fine), so there is none: the differences are hints
for the person, who may add a note of what they see, which Gemma checks against the pictures.
"""

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

import requests

import configNG
import h3_jobsNG as h3_jobs
import h3_rerollNG as h3_reroll
import h3_stitchNG as stitch

TIMEOUT_S = 300
TILE_WIDTH = 384
COLUMNS = 5
KINDS = ("identity", "wardrobe", "setting", "artifact", "motion", "other")
SECTION = "integrated_multimodal_description"

SYSTEM = """You review one 5-second window of an AI-generated video made by the MiniMax H3 model in a chain of windows.
You are shown ONE picture: a contact sheet. If there is a previous window, the top-left picture is its last frame. If \
there are ANCHOR frames, the next two pictures are from the start of the video and show how the character and place were first \
established. The remaining six pictures are in time order across THIS window. You are also given the prompt of window 1, the \
prompt used for this window, and a motion score. The person may add a note about what they see.

How this engine behaves (use it when you look for causes):
- Each window is rendered from its own prompt. Anything about a person's face, hair or clothes that is not written in THIS \
window's prompt is not remembered reliably: it drifts, often to a different person or outfit.
- Curly or textured hair tends to straighten; clothes change colour; a character called only 'the man' can become someone else.
- A shot with no action verb comes out almost static. Motion score: below 3 is nearly still, 5 to 8 is gentle, above 10 is lively.
- An extreme close-up of the eyes can render them glowing or unnatural, especially if the prompt mentions light on the eyes.
- A prompt has three sections: integrated_multimodal_description (the shots), overall_soundscape, non_diegetic_music. Keep that format.

Work in this order. First describe, from the pictures only, what the main character looks like in the ANCHOR pictures (hair, \
clothes, build) and the setting. Then describe the same things for THIS window, writing "not visible" where he is seen only from \
behind or not at all. Then list every difference between the two descriptions that matters to the story, with a kind: \
identity, wardrobe, setting, artifact (a visual glitch), motion or other, the picture it is seen in, and which words in the \
prompt (or which missing words) probably caused it. If the person added a note, check it against the pictures and say if you \
cannot see what they describe. Do not invent differences, and do not guess who a real person resembles.

Last, rewrite THIS window's prompt so it fixes what you found: restate the character's look from the anchor description \
(hair, clothes) inside this window's shots, remove wording that caused an artifact, and keep everything that works, including \
the shot structure and the sound and music sections. If nothing needs fixing, return the prompt unchanged.

Reply with JSON only: {"anchor_look": "...", "window_look": "...", "differences": [{"what": "...", "kind": "...", \
"seen_in": "...", "likely_cause": "..."}], "rewrite": "the full window prompt"}"""


class GemmaUnavailable(RuntimeError):
    pass


class GpuBusy(RuntimeError):
    pass


def _frame(clip, n, out, width=TILE_WIDTH):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"select=eq(n\\,{n}),scale={width}:-1",
                    "-frames:v", "1", str(out)], check=True, timeout=120)


def sheet_ng(clip, previous=None, anchor=None) -> bytes:
    """The JPEG contact sheet Gemma is shown. See the module docstring for the order of the tiles."""
    with tempfile.TemporaryDirectory(prefix="critique_") as tmp:
        tmp = Path(tmp)
        tiles = []

        def add(source, n):
            path = tmp / f"t{len(tiles)}.png"
            _frame(source, n, path)
            tiles.append(path)

        if previous:
            add(previous, stitch.probe_frames_ng(previous) - 1)
        if anchor:
            count = stitch.probe_frames_ng(anchor)
            add(anchor, round(count * 0.25))
            add(anchor, round(count * 0.6))
        count = stitch.probe_frames_ng(clip)
        for i in range(6):
            add(clip, round(i * (count - 1) / 5))
        rows = -(-len(tiles) // COLUMNS)
        sheet = tmp / "sheet.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", str(tmp / "t%d.png"),
                        "-vf", f"tile={COLUMNS}x{rows}", "-frames:v", "1", "-q:v", "3", str(sheet)],
                       check=True, timeout=120)
        return sheet.read_bytes()


def motion_ng(clip) -> float:
    """Mean change between frames (0 to 255) at 6 fps on a tiny grey copy: below about 3 is nearly still."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", "fps=6,scale=64:36,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, timeout=120).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 36, 64).astype(float)
    return round(float(np.abs(np.diff(frames, axis=0)).mean()), 2) if len(frames) > 1 else 0.0


def parse_reply_ng(text: str) -> dict:
    """Gemma's JSON (found inside any wrapping text) -> {"anchor_look", "window_look", "differences",
    "rewrite", "warning"}. A reply that is not usable raises ValueError."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise ValueError("Gemma's answer was not in the expected form; ask again")
    diffs = []
    for item in data.get("differences") or []:
        if isinstance(item, str):
            item = {"what": item}
        if isinstance(item, dict) and str(item.get("what") or "").strip():
            kind = str(item.get("kind") or "other").strip().lower()
            diffs.append({"what": str(item["what"]).strip(), "kind": kind if kind in KINDS else "other",
                          "seen_in": str(item.get("seen_in") or "").strip(),
                          "likely_cause": str(item.get("likely_cause") or "").strip()})
    rewrite, warning = data.get("rewrite"), None
    if not isinstance(rewrite, str) or SECTION not in rewrite:
        rewrite, warning = None, "Gemma's rewrite did not keep the prompt format, so it was left out"
    return {"anchor_look": str(data.get("anchor_look") or "").strip(),
            "window_look": str(data.get("window_look") or "").strip(),
            "differences": diffs, "rewrite": rewrite.strip() if rewrite else None, "warning": warning}


def _model_id(base):
    ids = [m.get("id", "") for m in requests.get(f"{base}/v1/models", timeout=10).json().get("data", [])]
    return next((i for i in ids if "gemma" in i.lower()), ids[0] if ids else "")


def critique_window_ng(job_id: str, window: int, note: str = "") -> dict:
    """Ask Gemma about `window` (1-based) of the job's current film. Raises LookupError / ValueError for
    a bad job or window, GpuBusy while something is on the GPU, GemmaUnavailable if the server is down."""
    src = h3_reroll.resolve_source_ng(job_id)
    man = h3_reroll.load_manifest_ng(src)
    count = len(man["windows"])
    if not 1 <= window <= count:
        raise ValueError(f"window must be 1 to {count} (got {window})")
    if h3_jobs.busy_ng() is not None:
        raise GpuBusy("a render is running; ask Gemma when it has finished (both need the GPU)")
    job_dir = Path(src.job_dir)
    clip = job_dir / man["windows"][window - 1]["clip"]
    previous = job_dir / man["windows"][window - 2]["clip"] if window > 1 else None
    anchor = job_dir / man["windows"][0]["clip"] if window > 1 else None
    motion = motion_ng(clip)
    sheet = base64.b64encode(sheet_ng(clip, previous, anchor)).decode()
    text = (f"Prompt of window 1:\n{man['windows'][0]['prompt']}\n\n"
            f"Prompt used for THIS window (window {window} of {count}):\n{man['windows'][window - 1]['prompt']}\n\n"
            f"Motion score for this window: {motion}\n\n"
            + (f"The person says: {note.strip()}\n\n" if note.strip() else "") + "Review this window.")
    base = configNG.get_gemma_review_base_url()
    try:
        body = {"model": _model_id(base), "temperature": 0.3, "max_tokens": 3000, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": text},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + sheet}}]}]}
        reply = requests.post(f"{base}/v1/chat/completions", json=body, timeout=TIMEOUT_S).json()
        content = reply["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        raise GemmaUnavailable("could not get an answer from the Gemma server at "
                               f"{base} (start it with ./h3ir_serveNG.sh gemma): {e}") from e
    out = parse_reply_ng(content)
    out.update(window=window, motion=motion, note_given=bool(note.strip()))
    return out
