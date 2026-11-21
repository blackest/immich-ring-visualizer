"""immichRingNG -- "Sharp 2x" for a Klein picture: a 2x sharpen on the Neural Engine (see vendor/pipersr_x2).

The same PiperSR runner the H3 films use (h3_sharpNG.py), fed a still. It takes about a second, uses no GPU (a Klein
render can be running), and writes `<name>_sr2x.png` beside the picture, in the job's own folder, so the job's Delete takes
it away with the rest. Asking again for a picture that already has one just returns it. The picture itself is not touched.

Licence: PiperSR's code is AGPL-3.0 and its model is free for personal and academic use with attribution; commercial use
needs the author's licence. The runner prints the attribution line; the caller may log it.
"""

import subprocess
from pathlib import Path

import h3_sharpNG as h3_sharp

SUFFIX = "_sr2x"
TIMEOUT_S = 180


def sharp_path_ng(src: Path) -> Path:
    return src.with_name(src.stem + SUFFIX + ".png")


def sharpen_picture_ng(src: Path) -> Path:
    """The 2x PNG for `src`, made if it is not there yet. Raises FileNotFoundError / RuntimeError."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"no such picture: {src.name}")
    dest = sharp_path_ng(src)
    if dest.is_file():
        return dest
    python = h3_sharp.pipersr_python_ng()
    part = dest.with_name(dest.stem + ".partial.png")
    try:
        proc = subprocess.run([python, str(h3_sharp.SCRIPT), str(src), "-o", str(part)],
                              capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        part.unlink(missing_ok=True)
        raise RuntimeError("the sharpen took too long and was stopped") from None
    if proc.returncode != 0 or not part.is_file():
        part.unlink(missing_ok=True)
        lines = [l for l in (proc.stdout + proc.stderr).splitlines() if l.strip()]
        raise RuntimeError("the sharpen failed: " + (lines[-1].replace("[pipersr] error ", "") if lines else f"exit {proc.returncode}"))
    part.replace(dest)
    return dest
