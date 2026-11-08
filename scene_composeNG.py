"""immichRingNG -- the first frame for two speakers in one frame: two pictures side by side, as a vertical split-screen.

Each half is its speaker's picture centre-cropped to the half's shape (width/2 by height) and scaled to fit, so a
face stays a face and nothing is stretched. The composite is the clip's size; the H3 runner then sizes it as it does any
first frame. Pure PIL, no models.
"""

from __future__ import annotations

import io

from PIL import Image

SEAM_PX = 2          # a thin dark seam, so the two halves read as two frames


class ComposeError(ValueError):
    pass


def _open(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im.convert("RGB")
    except Exception:  # noqa: BLE001 -- Pillow raises many kinds for a bad file
        raise ComposeError("one of the pictures is not readable") from None


def fit_half(im: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop to width:height, then scale to exactly that size."""
    w, h = im.size
    want = width / height
    if w / h > want:                       # too wide: trim the sides
        new_w = int(round(h * want))
        box = ((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h)
    else:                                  # too tall: trim top and bottom
        new_h = int(round(w / want))
        box = (0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h)
    return im.crop(box).resize((width, height), Image.LANCZOS)


def split_screen_ng(left: bytes, right: bytes, width: int, height: int) -> bytes:
    """PNG bytes: `left` on the left half, `right` on the right, `width` x `height` overall."""
    if width < 64 or height < 64 or width % 2:
        raise ComposeError("the composite needs an even width of at least 64 px and a height of at least 64 px")
    half = width // 2
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(fit_half(_open(left), half, height), (0, 0))
    canvas.paste(fit_half(_open(right), half, height), (half, 0))
    if SEAM_PX:
        canvas.paste((0, 0, 0), (half - SEAM_PX // 2, 0, half + (SEAM_PX - SEAM_PX // 2), height))
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue()
