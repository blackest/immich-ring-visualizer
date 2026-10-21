"""Output-codec control for the LTX (Animate) engine.

ltx-2-mlx hardcodes its ffmpeg encode as `libx264 -pix_fmt yuv420p -crf 18`
(ltx_core_mlx/model/video_vae/video_vae.py, decode_and_stream). On LTX's
grainy diffusion output that runs ~80-100 Mbps -- a 15s 768x512 clip came out
at 98 MB. Phosphene solves this with a one-line runtime patch that reads
LTX_OUTPUT_PIX_FMT / LTX_OUTPUT_CRF from the environment plus an output-preset
setting (see its patch_ltx_codec.py and OUTPUT_PRESETS); this is the same
idea, ported. MIT-licensed upstream.

Two halves, both tiny:
  apply_codec_patch_ng()  idempotently edits the library's ffmpeg line
  codec_env_ng(preset)    the env vars the patched line reads at render time

With the env vars unset the patched line behaves exactly like upstream
(yuv420p, crf 18), so a stale/unpatched install is never worse than before.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from ltx_engineNG import LTX_REPO_DIR

OUTPUT_PRESETS: Dict[str, Dict[str, str]] = {
    "web": {
        "label": "Web / small files",
        "pix_fmt": "yuv420p",
        "crf": "23",
        "blurb": "Small files (a few MB per clip). Plays everywhere.",
    },
    "standard": {
        "label": "Standard",
        "pix_fmt": "yuv420p",
        "crf": "18",
        "blurb": "Near-visually-lossless, but LTX's grain makes files very large (~100 Mbps).",
    },
    "archival": {
        "label": "Lossless (production)",
        "pix_fmt": "yuv444p",
        "crf": "0",
        "blurb": "Mathematically lossless, full chroma. Huge files; for grading/compositing.",
    },
}
DEFAULT_OUTPUT_PRESET = "web"

_REL_TARGET = Path("ltx_core_mlx") / "model" / "video_vae" / "video_vae.py"

PATCH_OLD = 'cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])'
PATCH_MARKER = "LTX_OUTPUT_PIX_FMT"
PATCH_NEW = (
    '# PATCHED (immichring): output codec comes from the environment so the app\n'
    '        # can offer small-file presets. Unset -> upstream defaults. +faststart\n'
    '        # puts the moov atom first so thumbnails decode without a full download.\n'
    '        import os as _os\n'
    '        _pix = _os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv420p")\n'
    '        _crf = _os.environ.get("LTX_OUTPUT_CRF", "18")\n'
    '        cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, "-crf", _crf,\n'
    '                    "-movflags", "+faststart", output_path])'
)

APPLIED, ALREADY, MISSING, DRIFT = "applied", "already", "missing", "drift"


def codec_env_ng(preset: Optional[str]) -> Dict[str, str]:
    p = OUTPUT_PRESETS.get(preset or "", OUTPUT_PRESETS[DEFAULT_OUTPUT_PRESET])
    return {"LTX_OUTPUT_PIX_FMT": p["pix_fmt"], "LTX_OUTPUT_CRF": p["crf"]}


def _find_target_ng() -> Optional[Path]:
    """The file the ltx venv actually imports: the vendored source (editable
    install) first, then a regular site-packages copy."""
    cand = LTX_REPO_DIR / "packages" / "ltx-core-mlx" / "src" / _REL_TARGET
    if cand.is_file():
        return cand
    for sp in sorted((LTX_REPO_DIR / "env" / "lib").glob("python3*/site-packages")):
        cand = sp / _REL_TARGET
        if cand.is_file():
            return cand
    return None


def apply_codec_patch_ng(target: Optional[Path] = None) -> str:
    """Idempotent. Returns APPLIED / ALREADY / MISSING (no file) / DRIFT
    (upstream reworded the ffmpeg line -- PATCH_OLD needs updating)."""
    target = target or _find_target_ng()
    if target is None or not target.is_file():
        return MISSING
    text = target.read_text()
    if PATCH_MARKER in text:
        return ALREADY
    if PATCH_OLD not in text:
        return DRIFT
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text.replace(PATCH_OLD, PATCH_NEW))
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return APPLIED
