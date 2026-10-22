"""Re-lays a speed adapter's qkv rows into H3's per-head order.

H3's DiT stores the fused attn.qkv_proj rows per head, (heads, 3,
head_dim); every speed adapter we resolve (the v1.0 LightX2V repack, the
larryvrh checkpoints, TaoMate) has its qkv lora_B rows contiguous, (3,
heads, head_dim). The pinned runner only re-orders them under
--lora-qkv-permute, which is off by default and global to a run -- so
without this step a turbo render writes the q/k/v updates onto the wrong
rows (soft, smeared detail rather than an error).

Ported from Phosphene's h3_bake_runner_layout (MIT, v4.19.11): same
output path (<adapter dir>/runner-layout/<same name>) and same metadata
stamp, so the two apps share one baked file instead of each making
their own. Pure byte moves of whole rows -- no float maths -- written
through a .partial + rename; the source file is never touched.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

LAYOUT_DIRNAME = "runner-layout"
HEADS = 56
HEAD_DIM = 128
QKV_LORA_B_SUFFIX = ".attn.qkv_proj.lora_B.weight"
LAYOUT_KEY = "phosphene_qkv_rows"
LAYOUT_VALUE = "per-head (heads, 3, head_dim), re-ordered from (3, heads, head_dim)"


def runner_has_flag_ng(runner: Path | str, flag: str) -> bool:
    """Whether the runner script mentions `flag` -- a cheap source probe,
    so an old checkout is detected without launching a subprocess."""
    try:
        return flag in Path(runner).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def layout_file_ng(src: Path | str) -> Path:
    src = Path(src)
    return src.parent / LAYOUT_DIRNAME / src.name


def _header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as fh:
        size = int.from_bytes(fh.read(8), "little")
        if not 0 < size <= 100 * 1024 * 1024:
            raise ValueError(f"{path.name}: not a safetensors file")
        header = json.loads(fh.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: not a safetensors file")
    return size, header


def _current(src: Path, dst: Path) -> bool:
    """The re-ordered copy exists and was made from THIS source file."""
    try:
        meta = _header(dst)[1].get("__metadata__") or {}
        st = src.stat()
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return (meta.get(LAYOUT_KEY) == LAYOUT_VALUE
            and meta.get("phosphene_source_bytes") == str(st.st_size)
            and meta.get("phosphene_source_mtime_ns") == str(st.st_mtime_ns))


def rows_per_head_ng(blob: bytes, rows: int) -> bytes:
    """lora_B rows (3, heads, head_dim) -> (heads, 3, head_dim), as bytes."""
    if rows != 3 * HEADS * HEAD_DIM or len(blob) % rows:
        raise ValueError(f"qkv lora_B has {rows} rows, expected {3 * HEADS * HEAD_DIM}")
    block = (len(blob) // rows) * HEAD_DIM
    return b"".join(blob[(j * HEADS + h) * block:(j * HEADS + h + 1) * block]
                    for h in range(HEADS) for j in range(3))


def bake_runner_layout_ng(src: Path | str, on_log=None) -> Path:
    """Write (once) and return the per-head copy of a speed adapter.
    Re-made when the source's size or mtime changes. Raises with a plain
    reason on disk shortage or an unreadable file."""
    src = Path(src)
    dst = layout_file_ng(src)
    if _current(src, dst):
        return dst
    hlen, header = _header(src)
    st = src.stat()
    dst.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dst.parent).free
    if free < st.st_size + 512 * 1024 * 1024:
        raise RuntimeError(
            f"not enough free disk to prepare {src.name} for H3 "
            f"({st.st_size / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free)")
    meta = dict(header.get("__metadata__") or {})
    meta.update({LAYOUT_KEY: LAYOUT_VALUE, "phosphene_source": src.name,
                 "phosphene_source_bytes": str(st.st_size),
                 "phosphene_source_mtime_ns": str(st.st_mtime_ns)})
    raw = json.dumps({**header, "__metadata__": meta}, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)  # the format's 8-byte alignment pad
    tensors = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                     key=lambda kv: kv[1]["data_offsets"][0])
    moved = 0
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.partial")
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(len(raw).to_bytes(8, "little"))
            fout.write(raw)
            pos = 0
            for name, info in tensors:
                start, end = info["data_offsets"]
                if start != pos:
                    raise ValueError(f"{src.name}: tensor data is not contiguous at {name}")
                fin.seek(8 + hlen + start)
                blob = fin.read(end - start)
                if len(blob) != end - start:
                    raise ValueError(f"{src.name}: truncated at {name}")
                if name.endswith(QKV_LORA_B_SUFFIX):
                    blob = rows_per_head_ng(blob, int(info["shape"][0]))
                    moved += 1
                fout.write(blob)
                pos = end
        if not moved:
            raise ValueError(f"{src.name}: no attn.qkv_proj lora_B tensors found")
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if on_log:
        on_log(f"[h3] {src.name}: {moved} qkv row-blocks re-ordered to the "
               f"checkpoint's per-head layout -> {dst}")
    return dst
