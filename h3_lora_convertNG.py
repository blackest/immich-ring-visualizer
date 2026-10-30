"""immichRingNG -- make a kohya-format H3 LoRA usable by the runner.

Character LoRAs trained with kohya-style trainers name their weights `lora_down` / `lora_up` and
keep each module's `alpha` in a separate scalar. The runner reads `lora_A` / `lora_B` and applies
no alpha, so such a file is listed "unusable" (h3_lora_libraryNG.py). This renames the weights
onto the runner's names and folds alpha / rank into lora_B, so strength 1.0 means "as trained".
It is the same conversion as Phosphene's own importer, written to stream tensor by tensor with
numpy (no MLX) so a multi-GB file needs little memory. The original is copied to `_originals/`
beside the library first, and the file is replaced only after the new one reads back as a valid
LoRA with the same number of modules.
"""

import json
import os
import re
import shutil
import struct
from pathlib import Path

import h3_loraNG

KOHYA_DOWN = ".lora_down.weight"
KOHYA_UP = ".lora_up.weight"
ORIGINALS_DIRNAME = "_originals"
# kohya module names -> the runner's checkpoint names. Underscores are both the trainer's
# separator and part of the names (`out_proj`), so the map is by known target, never `_` -> `.`.
_MODULE_RE = re.compile(
    r"^lora_unet_(?:(token_refiner)_)?blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)$")
_TAILS = {"attn_qkv_proj": "attn.qkv_proj", "attn_out_proj": "attn.out_proj",
          "mlp_fc1": "mlp.fc1", "mlp_fc2": "mlp.fc2"}


def is_kohya_ng(path) -> bool:
    try:
        _hlen, header = h3_loraNG._header(Path(path))
    except (OSError, ValueError):
        return False
    return any(k.endswith(KOHYA_DOWN) for k in header if k != "__metadata__")


def target_name_ng(module: str):
    m = _MODULE_RE.match(module)
    if not m:
        return None
    refiner, index, tail = m.groups()
    return f"{'token_refiner.' if refiner else ''}blocks.{int(index)}.{_TAILS[tail]}"


def _encode(arr, dtype: str) -> bytes:
    """float32 numpy -> the safetensors bytes of `dtype` (BF16 rounds to nearest even)."""
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype="<f4")
    if dtype == "BF16":
        u = arr.view("<u4").astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
    return arr.astype({"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype]).tobytes()


def _raw(fh, hlen: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    fh.seek(8 + hlen + start)
    return fh.read(end - start)


def convert_kohya_ng(path, on_log=None) -> dict:
    """Rewrite the kohya LoRA at `path` in place. Returns {"pairs", "dropped", "alpha_over_rank",
    "backup"}. Raises ValueError for a file that is not kohya-format or has no module the runner
    can take, and leaves the file untouched on any failure."""
    path = Path(path)
    hlen, header = h3_loraNG._header(path)
    meta = {str(k): str(v) for k, v in (header.get("__metadata__") or {}).items()}
    names = sorted(k[:-len(KOHYA_DOWN)] for k in header if k.endswith(KOHYA_DOWN))
    if not names:
        raise ValueError(f"{path.name} is not a kohya-format LoRA")
    plan, dropped = [], 0
    for name in names:
        target = target_name_ng(name)
        down, up = header.get(name + KOHYA_DOWN), header.get(name + KOHYA_UP)
        if target is None or up is None or down["dtype"] not in ("BF16", "F16", "F32", "F64") \
                or up["dtype"] not in ("BF16", "F16", "F32", "F64"):
            dropped += 1
            continue
        plan.append((name, target, down, up, header.get(name + ".alpha")))
    if not plan:
        raise ValueError(f"{path.name}: no module the H3 runner can take ({dropped} unknown module names)")
    size = path.stat().st_size
    free = shutil.disk_usage(path.parent).free
    if free < 2 * size + 512 * 1024 * 1024:
        raise RuntimeError(f"not enough free disk to convert {path.name} ({2 * size / 1e9:.1f} GB needed)")
    backup_dir = path.parent / ORIGINALS_DIRNAME
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / path.name
    if not backup.exists():
        shutil.copy2(path, backup)

    quotients, out_header, order, offset = [], {}, [], 0
    with open(path, "rb") as fh:
        for _name, target, down, up, alpha in plan:
            rank = int(down["shape"][0])
            a = float(h3_loraNG._tensor_f32(fh, hlen, alpha).reshape(-1)[0]) if alpha else float(rank)
            q = a / rank if rank else 1.0
            quotients.append(q)
            for suffix, info in ((".lora_A.weight", down), (".lora_B.weight", up)):
                n = info["data_offsets"][1] - info["data_offsets"][0]
                out_header[target + suffix] = {"dtype": info["dtype"], "shape": info["shape"],
                                               "data_offsets": [offset, offset + n]}
                offset += n
            order.append((target, down, up, q))
        unique = sorted(set(round(q, 6) for q in quotients))
        meta.update({
            "converted_from": "kohya", "converted_by": "immichringNG",
            "peft_scale_folded_into_B": f"{unique[0]:g}" if len(unique) == 1 else "per-module",
            "alpha_over_rank": ",".join(f"{q:g}" for q in unique),
            "converted_modules": str(len(plan)), "dropped_modules": str(dropped)})
        raw = json.dumps({**out_header, "__metadata__": meta}, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        tmp = path.with_name(path.stem + ".kohya-tmp.safetensors")
        try:
            with open(tmp, "wb") as out:
                out.write(struct.pack("<Q", len(raw)) + raw)
                for _target, down, up, q in order:
                    out.write(_raw(fh, hlen, down))
                    if abs(q - 1.0) <= 1e-6:
                        out.write(_raw(fh, hlen, up))
                    else:
                        out.write(_encode(h3_loraNG._tensor_f32(fh, hlen, up) * q, up["dtype"]))
            _hl, check = h3_loraNG._header(tmp)
            pairs = sum(1 for k in check if k.endswith(".lora_A.weight"))
            if pairs != len(plan) or any(k.endswith(KOHYA_DOWN) for k in check):
                raise RuntimeError(f"post-conversion check failed ({pairs} pairs, expected {len(plan)}); "
                                   "the original was left untouched")
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    if on_log:
        on_log(f"[lora] {path.name}: kohya -> runner layout, {len(plan)} modules, alpha/rank "
               f"{meta['alpha_over_rank']} folded into lora_B" + (f", {dropped} dropped" if dropped else ""))
    return {"pairs": len(plan), "dropped": dropped, "alpha_over_rank": unique, "backup": str(backup)}
