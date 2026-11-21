"""immichRingNG -- "Sharp 2x" for a Klein picture: a 2x sharpen on the Neural Engine (see vendor/pipersr_x2).

The same PiperSR runner the H3 films use (h3_sharpNG.py), fed a still. It takes about a second, uses no GPU (a Klein
render can be running), and writes `<name>_sr2x<strength %>.png` beside the picture, in the job's own folder, so the job's Delete takes
it away with the rest. Asking again for a picture that already has one just returns it. The picture itself is not touched.

Licence: PiperSR's code is AGPL-3.0 and its model is free for personal and academic use with attribution; commercial use
needs the author's licence. The runner prints the attribution line; the caller may log it.
"""

import subprocess
from pathlib import Path

import h3_sharpNG as h3_sharp

SUFFIX = "_sr2x"
TIMEOUT_S = 180


def sharp_path_ng(src: Path, strength=h3_sharp.DEFAULT_STRENGTH) -> Path:
    """`<name>_sr2x100.png` for the default strength 1: each strength is its own file, so a second try at another strength is new."""
    return src.with_name(f"{src.stem}{SUFFIX}{int(round(float(strength) * 100))}.png")


def sharpen_picture_ng(src: Path, strength=h3_sharp.DEFAULT_STRENGTH) -> Path:
    """The 2x PNG for `src` at `strength` (0 plain resize to 1 pure PiperSR), made if it is not there yet.
    Raises ValueError / FileNotFoundError / RuntimeError."""
    strength = h3_sharp.clean_strength_ng(strength)
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"no such picture: {src.name}")
    dest = sharp_path_ng(src, strength)
    if dest.is_file():
        return dest
    python = h3_sharp.pipersr_python_ng()
    part = dest.with_name(dest.stem + ".partial.png")
    try:
        proc = subprocess.run([python, str(h3_sharp.SCRIPT), str(src), "-o", str(part), "--strength", f"{strength:g}"],
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
