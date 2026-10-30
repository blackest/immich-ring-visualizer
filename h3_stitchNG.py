"""immichRingNG -- join the 5 s window clips of a chained H3 render into one film, choosing how
each seam is made: a hard cut, a dissolve, a fade through black, or an audio lead (the next
window's sound starts a little before its picture, a "J-cut").

The runner already stitches a chain this way: a later window opens on the previous window's last
frame, so its first frame is dropped and the sound crossfades by that one frame. This file does
the same from the saved `_wN.mp4` clips, so any one window can be swapped for a re-render
(h3_rerollNG.py) without redoing the others. Picture and sound are kept in step by one rule: the
sound of the next clip overlaps what came before by exactly the frames removed from the head of
its picture plus any frames the picture dissolves over. ffmpeg is the only dependency.
"""

import os
import subprocess
from pathlib import Path

FPS = 24
JOIN_TYPES = ("cut", "dissolve", "fade", "audio_lead")
DEFAULT_SECONDS = 0.5
MIN_SECONDS = 0.25
MAX_SECONDS = 2.0
CRF = "16"


def normalise_joins_ng(joins, seams: int) -> list:
    """`joins`: [{"seam": 1.., "type", "seconds"}] -> one {"seam", "type", "seconds", "frames"} per
    seam, a hard cut wherever none is given. Seam n is the join after window n. Raises ValueError."""
    chosen = {}
    for item in joins or []:
        if not isinstance(item, dict):
            raise ValueError("each join must be {seam, type, seconds}")
        try:
            seam = int(item.get("seam"))
        except (TypeError, ValueError):
            raise ValueError(f"join seam must be a number (got {item.get('seam')!r})") from None
        if not 1 <= seam <= seams:
            raise ValueError(f"join seam must be 1 to {seams} (got {seam})")
        kind = str(item.get("type") or "cut").strip().lower()
        if kind not in JOIN_TYPES:
            raise ValueError(f"join type must be one of {', '.join(JOIN_TYPES)} (got {kind!r})")
        try:
            seconds = float(item.get("seconds", DEFAULT_SECONDS))
        except (TypeError, ValueError):
            raise ValueError(f"join seconds must be a number (got {item.get('seconds')!r})") from None
        if kind != "cut" and not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"join seconds must be {MIN_SECONDS} to {MAX_SECONDS} (got {seconds:g})")
        chosen[seam] = (kind, seconds)
    out = []
    for seam in range(1, seams + 1):
        kind, seconds = chosen.get(seam, ("cut", DEFAULT_SECONDS))
        out.append({"seam": seam, "type": kind, "seconds": seconds,
                    "frames": 0 if kind == "cut" else max(2, round(seconds * FPS))})
    return out


def build_graph_ng(frames: list, joins: list) -> tuple:
    """(filter_complex, video_label, audio_label, total_frames) for clips of `frames` frames each,
    joined by `joins` (normalised, one per seam)."""
    # The sound of a window is cut to its picture's length: the AAC in a saved clip ends in padding
    # (about 17 ms) that the runner's own stitch never sees, and left in it drifts every seam.
    parts = ["[0:v]fps=24,settb=1/24,setpts=PTS-STARTPTS[v0]",
             f"[0:a]atrim=end={frames[0] / FPS:.6f},asetpts=PTS-STARTPTS[a0]"]
    total = frames[0]
    for i in range(1, len(frames)):
        kind, k = joins[i - 1]["type"], joins[i - 1]["frames"]
        head = 1 + (k if kind == "audio_lead" else 0)  # frames cut from the front of this picture
        overlap = k if kind == "dissolve" else 0  # frames the picture dissolves over
        parts.append(f"[{i}:v]fps=24,settb=1/24,trim=start_frame={head},setpts=PTS-STARTPTS[n{i}]")
        parts.append(f"[{i}:a]atrim=end={frames[i] / FPS:.6f},asetpts=PTS-STARTPTS[m{i}]")
        sound = f"m{i}"
        if kind == "dissolve":
            parts.append(f"[v{i - 1}][n{i}]xfade=transition=fade:duration={overlap / FPS:.5f}"
                         f":offset={(total - overlap) / FPS:.5f}[x{i}]")
            parts.append(f"[x{i}]fps=24,settb=1/24[v{i}]")
        elif kind == "fade":
            half = max(1, k // 2)
            parts.append(f"[v{i - 1}]fade=t=out:s={total - half}:n={half}[o{i}]")
            parts.append(f"[n{i}]fade=t=in:s=0:n={half}[f{i}]")
            parts.append(f"[o{i}][f{i}]concat=n=2:v=1:a=0[c{i}]")
            parts.append(f"[c{i}]fps=24,settb=1/24[v{i}]")
            parts.append(f"[a{i - 1}]afade=t=out:st={(total - half) / FPS:.5f}:d={half / FPS:.5f}[b{i}]")
            parts.append(f"[m{i}]afade=t=in:st=0:d={half / FPS:.5f}[g{i}]")
            sound = f"g{i}"
        else:
            parts.append(f"[v{i - 1}][n{i}]concat=n=2:v=1:a=0[c{i}]")
            parts.append(f"[c{i}]fps=24,settb=1/24[v{i}]")
        acc = f"b{i}" if kind == "fade" else f"a{i - 1}"
        parts.append(f"[{acc}][{sound}]acrossfade=d={(head + overlap) / FPS:.5f}:c1=tri:c2=tri[a{i}]")
        total += frames[i] - head - overlap
    last = len(frames) - 1
    parts.append(f"[v{last}]format=yuv420p[vout]")
    return ";".join(parts), "vout", f"a{last}", total


def probe_frames_ng(clip, ffprobe: str = "ffprobe") -> int:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(clip)],
        capture_output=True, text=True, timeout=60)
    try:
        return int(out.stdout.strip().splitlines()[0].rstrip(","))
    except (ValueError, IndexError):
        raise ValueError(f"cannot read the frame count of {Path(clip).name}") from None


def stitch_windows_ng(clips: list, joins, out_path, *, trim_tail: int = 0,
                      ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", on_log=None) -> dict:
    """Join `clips` (paths, in window order) into `out_path` and return
    {"path", "frames", "seconds", "joins"}. `trim_tail` is how many frames to cut from the end,
    the way the runner trims a chain to the length that was asked for. The result replaces
    out_path only when ffmpeg succeeded."""
    clips = [str(c) for c in clips]
    if not clips:
        raise ValueError("no window clips to join")
    for clip in clips:
        if not Path(clip).is_file():
            raise FileNotFoundError(f"window clip not found: {clip}")
    joins = normalise_joins_ng(joins, len(clips) - 1)
    frames = [probe_frames_ng(c, ffprobe) for c in clips]
    graph, vlabel, alabel, total = build_graph_ng(frames, joins)
    total -= max(0, int(trim_tail))
    if total < 1:
        raise ValueError("nothing left after trimming the end")
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    cmd = [ffmpeg, "-v", "error", "-y"]
    for clip in clips:
        cmd += ["-i", clip]
    cmd += ["-filter_complex", graph, "-map", f"[{vlabel}]", "-map", f"[{alabel}]",
            "-c:v", "libx264", "-crf", CRF, "-preset", "medium", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", "-frames:v", str(total), "-t", f"{total / FPS:.5f}", str(tmp)]
    if on_log:
        on_log(f"[stitch] {len(clips)} windows, joins: "
               + ", ".join(f"{j['seam']}={j['type']}" for j in joins) + f" -> {total} frames")
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if run.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg could not join the windows: " + (run.stderr or "").strip()[-400:])
    os.replace(tmp, out_path)
    return {"path": str(out_path), "frames": total, "seconds": round(total / FPS, 3), "joins": joins}
