"""immichRingNG -- Klein outpaint: pad a picture with flat gray, and put the original pixels back afterwards.

The Outpaint LoRA (ausboss/FLUX-3-Image-Klein-Outpaint-LoRA) fills flat #808080 around a picture given to
Klein as its reference. The model keeps the picture within a pixel or two, not exactly, so the original is
pasted back over the result with a short soft edge. Pure PIL; no model code.
"""

from pathlib import Path

from PIL import Image, ImageFilter

GRAY = (128, 128, 128)
INSTRUCTION = ("Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, "
               "keeping the existing picture unchanged.")
MAX_PIXELS = 1024 * 1024
MIN_SIDE = 256
FEATHER = 6


def outpaint_prompt(scene: str) -> str:
    scene = (scene or "").strip()
    return INSTRUCTION + (f" Scene: {scene}" if scene else "")


def plan_ng(width: int, height: int, left: int, top: int, right: int, bottom: int) -> dict:
    """Where the picture sits on the padded canvas. Pads are pixels of the *original* picture. The canvas is
    rounded up to multiples of 16 on the right and bottom, and the whole plan is scaled down together when
    it would be over one megapixel. Returns {scale, source (w, h), canvas (w, h), offset (x, y)}."""
    pads = (left, top, right, bottom)
    if any(not isinstance(p, int) or p < 0 for p in pads):
        raise ValueError("padding is a whole number of pixels, zero or more")
    if not any(pads):
        raise ValueError("add padding on at least one side")
    full_w, full_h = width + left + right, height + top + bottom
    scale = min(1.0, (MAX_PIXELS / (full_w * full_h)) ** 0.5)
    src_w, src_h = max(1, round(width * scale)), max(1, round(height * scale))
    off_x, off_y = round(left * scale), round(top * scale)
    can_w = -(-(off_x + src_w + round(right * scale)) // 16) * 16
    can_h = -(-(off_y + src_h + round(bottom * scale)) // 16) * 16
    can_w, can_h = max(can_w, MIN_SIDE), max(can_h, MIN_SIDE)
    while can_w * can_h > MAX_PIXELS:  # rounding up can tip it over; trim the empty gray, not the picture
        if can_w >= can_h and can_w - 16 >= off_x + src_w:
            can_w -= 16
        elif can_h - 16 >= off_y + src_h:
            can_h -= 16
        else:
            raise ValueError("that padding does not fit in one megapixel; use less")
    return {"scale": scale, "source": (src_w, src_h), "canvas": (can_w, can_h), "offset": (off_x, off_y)}


def prepare_ng(image_path, left, top, right, bottom, canvas_path, source_path) -> dict:
    """Writes the gray-padded canvas (the reference Klein is given) and the scaled original (for the paste
    back). Returns the plan."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        plan = plan_ng(im.width, im.height, left, top, right, bottom)
        if plan["source"] != im.size:
            im = im.resize(plan["source"], Image.LANCZOS)
        canvas = Image.new("RGB", plan["canvas"], GRAY)
        canvas.paste(im, plan["offset"])
        canvas.save(canvas_path)
        im.save(source_path)
    return plan


def paste_original_ng(result_path, source_path, offset, feather: int = FEATHER) -> None:
    """Puts the original picture back over the model's result, in place. The inner edge fades over a few
    pixels so no seam shows; the outer edge of the canvas is untouched."""
    with Image.open(result_path) as res, Image.open(source_path) as src:
        res, src = res.convert("RGB"), src.convert("RGB")
        mask = Image.new("L", src.size, 0)
        inset = min(feather, src.width // 4, src.height // 4)
        mask.paste(255, (inset, inset, src.width - inset, src.height - inset))
        # edges that touch the canvas edge need no fade: keep the picture solid there
        keep = Image.new("L", src.size, 0)
        keep.paste(255, (0 if offset[0] == 0 else inset, 0 if offset[1] == 0 else inset,
                         src.width if offset[0] + src.width >= res.width else src.width - inset,
                         src.height if offset[1] + src.height >= res.height else src.height - inset))
        mask = keep.filter(ImageFilter.GaussianBlur(max(1, inset // 2))) if inset else keep
        res.paste(src, offset, mask)
        res.save(result_path)
