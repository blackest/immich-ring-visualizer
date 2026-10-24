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
import threading
from pathlib import Path

LAYOUT_DIRNAME = "runner-layout"
HEADS = 56
HEAD_DIM = 128
QKV_LORA_A_SUFFIX = ".attn.qkv_proj.lora_A.weight"
QKV_LORA_B_SUFFIX = ".attn.qkv_proj.lora_B.weight"
BASE_QKV_LAYOUT = "per-head"
# Share of a file's trained qkv modules that must agree before its order is
# called; below this the file is "unclear" and re-ordered like any fused file.
LAYOUT_VOTE = 0.7
# Metadata a per-head file declares itself with (checked before the vote).
DECLARED_PER_HEAD = (("phosphene_qkv_rows", "per-head"),
                     ("qkv_layout", "per-head"),
                     ("qkv_layout", "interleaved"))
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


_bake_locks: dict[str, threading.Lock] = {}
_bake_locks_guard = threading.Lock()


def bake_runner_layout_ng(src: Path | str, on_log=None) -> Path:
    """Write (once) and return the per-head copy of an adapter.
    Re-made when the source's size or mtime changes. Raises with a plain
    reason on disk shortage or an unreadable file. One bake per destination
    at a time: two jobs reaching the same new LoRA would otherwise write
    the same .partial and the loser's rename would fail its render."""
    src = Path(src)
    dst = layout_file_ng(src)
    if _current(src, dst):
        return dst
    with _bake_locks_guard:
        lock = _bake_locks.setdefault(str(dst), threading.Lock())
    with lock:
        if _current(src, dst):
            return dst
        return _bake_locked_ng(src, dst, on_log)


def _prune_layout_dir_ng(directory: Path) -> None:
    """Drop re-ordered copies whose source left the library."""
    try:
        for copy in directory.glob("*.safetensors"):
            if not (directory.parent / copy.name).exists():
                copy.unlink(missing_ok=True)
    except OSError:
        pass


def _bake_locked_ng(src: Path, dst: Path, on_log=None) -> Path:
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
    _prune_layout_dir_ng(dst.parent)
    if on_log:
        on_log(f"[h3] {src.name}: {moved} qkv row-blocks re-ordered to the "
               f"checkpoint's per-head layout -> {dst}")
    return dst


def _tensor_f32(fh, hlen: int, info: dict):
    """One safetensors tensor as a float32 numpy array (BF16 widened exactly)."""
    import numpy as np
    start, end = info["data_offsets"]
    fh.seek(8 + hlen + start)
    raw = fh.read(end - start)
    dtype = info["dtype"]
    if dtype == "BF16":
        arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype in ("F16", "F32", "F64"):
        arr = np.frombuffer(raw, {"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype]).astype(np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return arr.reshape(info["shape"])


def _anova3_f(values, labels) -> float:
    """One-way ANOVA F of `values` over three groups (labels 0/1/2)."""
    groups = [values[labels == g] for g in range(3)]
    grand = values.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / 2
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(values) - 3)
    return float(between / within) if within > 0 else float("inf")


_layout_cache: dict[tuple, dict] = {}


def qkv_layout_ng(path: Path | str) -> dict:
    """Which order a LoRA's fused qkv rows are in: declared per-head, else measured.

    Per qkv module pair, takes the row norms of B.A (via B and the rank x rank
    Gram matrix A.A^T, so the full delta is never built) and compares an ANOVA
    F with q/k/v grouped per head against grouped contiguously; the module
    votes for the larger, and untrained (all-zero) modules abstain.
    Returns {"layout": "contiguous" | "per-head" | "unclear" | "none",
    "modules", "contiguous", "per_head", "declared"}. Ported from Phosphene's
    h3_lora_qkv_layout (MIT, v4.19.12). Cached per (path, size, mtime)."""
    import numpy as np
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _layout_cache:
        return _layout_cache[key]
    hlen, header = _header(path)
    meta = header.get("__metadata__") or {}
    declared = next((f"{k}={meta[k]}" for k, v in DECLARED_PER_HEAD
                     if str(meta.get(k) or "").strip().lower().startswith(v)), "")
    if declared:
        out = {"layout": "per-head", "modules": 0, "contiguous": 0, "per_head": 0,
               "declared": declared}
        _layout_cache[key] = out
        return out
    rows = 3 * HEADS * HEAD_DIM
    idx = np.arange(rows)
    per_head_labels = (idx // HEAD_DIM) % 3
    contiguous_labels = idx // (HEADS * HEAD_DIM)
    votes = {"contiguous": 0, "per_head": 0}
    with open(path, "rb") as fh:
        for name in sorted(k for k in header if k.endswith(QKV_LORA_B_SUFFIX)):
            a_info = header.get(name[: -len(QKV_LORA_B_SUFFIX)] + QKV_LORA_A_SUFFIX)
            b_info = header[name]
            if a_info is None or int(b_info["shape"][0]) != rows:
                continue
            b = _tensor_f32(fh, hlen, b_info).astype(np.float64)
            a = _tensor_f32(fh, hlen, a_info).astype(np.float64)
            norms = np.sqrt(np.clip(((b @ (a @ a.T)) * b).sum(axis=1), 0, None))
            if not norms.any():
                continue
            # A tie votes for the contiguous default, never for "leave alone".
            if _anova3_f(norms, per_head_labels) > _anova3_f(norms, contiguous_labels):
                votes["per_head"] += 1
            else:
                votes["contiguous"] += 1
    modules = votes["contiguous"] + votes["per_head"]
    if not modules:
        layout = "none"
    elif votes["contiguous"] >= LAYOUT_VOTE * modules:
        layout = "contiguous"
    elif votes["per_head"] >= LAYOUT_VOTE * modules:
        layout = "per-head"
    else:
        layout = "unclear"
    out = {"layout": layout, "modules": modules, **votes, "declared": ""}
    if len(_layout_cache) > 64:
        _layout_cache.clear()
    _layout_cache[key] = out
    return out


def lora_for_runner_ng(src: Path | str, runner: Path | str, on_log=None) -> tuple[Path, str]:
    """(the file the runner should read, what was done to its qkv rows).

    "re-ordered"  a contiguous-order file (measured, or unclear) -> its per-head copy
    "as-trained"  already in the base's per-head order (declared or measured)
    "none"        no trained qkv module (an MLP-only LoRA)
    "runner"      a runner without --lora-qkv-permute predates 2026-08-15 and
                  re-orders every --lora itself; a re-ordered copy would be
                  put back in the wrong order, so it gets the source file
    """
    src = Path(src)
    if not runner_has_flag_ng(runner, "--lora-qkv-permute"):
        return src, "runner"
    try:
        info = qkv_layout_ng(src)
    except (ValueError, KeyError) as e:
        info = {"layout": "unclear", "modules": 0, "contiguous": 0,
                "per_head": 0, "declared": "", "error": str(e)}
    if info["layout"] == "none":
        return src, "none"
    if info["layout"] == BASE_QKV_LAYOUT:
        return src, "as-trained"
    votes = (f"{info['contiguous']} of {info['modules']} qkv modules read contiguous"
             if info["modules"] else f"order not measured: {info.get('error')}")
    dst = bake_runner_layout_ng(src, on_log)
    if on_log:
        note = "qkv row order unclear" if info["layout"] == "unclear" else "qkv rows"
        on_log(f"[h3] {src.name}: {note} ({votes}) -- re-ordered to the base's per-head order")
    return dst, "re-ordered"
