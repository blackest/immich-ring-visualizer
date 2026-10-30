"""--reference sizing, clip sampling, the 2-frame text block and flag rules. No weights needed.

    ./.venv/bin/python tests/test_reference_encode.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from transformers import Qwen2VLImageProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_staged as gs
from minimax_h3_mlx.refmod_encode import load_references, sample_text_blocks, trim_reference_frames
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def picture(tmp: Path, name: str, size: tuple[int, int]) -> str:
    path = tmp / name
    Image.new("RGB", size, (90, 40, 40)).save(path)
    return str(path)


def refused(**kw) -> bool:
    base = dict(chain_context_frames=0, keyframe=None, extend_from=None, video_to_audio=None,
                audio_drive=None, audio_mux=None, last_frame=None, first_frame=None,
                draft_fps=None, playback_fps=None, draft_seconds=None, draft_cache_dir=None,
                save_stage_a=None, frames=124, extend_frames=None, refmod=None, reference=None,
                chain_windows=1, chain_total_frames=None)
    base.update(kw)

    class P(argparse.ArgumentParser):
        def error(self, message):
            raise SystemExit(message)

    try:
        return gs.media_mode_errors(argparse.Namespace(**base), P())
    except SystemExit:
        return True


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        big = picture(tmp, "big.png", (1536, 1024))
        small = picture(tmp, "small.png", (320, 192))
        wide = picture(tmp, "wide.png", (1000, 100))

        sizes = [r.size for r in load_references([big, small], 640, 384, 124, "match")]
        check("match scales a large picture down to the render's pixel area", sizes[0] == (608, 416), str(sizes[0]))
        check("match never upscales", sizes[1] == (320, 192), str(sizes[1]))
        check("max is a 2048 px short edge", load_references([small], 640, 384, 124, "max")[0].size == (3424, 2048))
        check("every side is a multiple of 32", all(v % 32 == 0 for sz in sizes for v in sz))
        try:
            load_references([wide], 640, 384, 124)
            check("a 10:1 picture is refused", False)
        except ValueError:
            check("a 10:1 picture is refused", True)

        clip = tmp / "c.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x384:rate=30",
                        "-t", "6", "-pix_fmt", "yuv420p", str(clip)], check=True)
        ref = load_references([str(clip)], 640, 384, 124)[0]
        check("a clip is read at 24 fps, cut to the window and trimmed to 17n+5",
              ref.kind == "video" and ref.frames.shape == (124, 384, 640, 3), str(ref.frames.shape))
        check("a clip at the canvas size keeps its size", ref.size == (640, 384))
        check("trim snaps down to 17n+5", [trim_reference_frames(n) for n in (5, 21, 22, 124, 130, 141)] == [5, 5, 22, 124, 124, 141])
        try:
            trim_reference_frames(4)
            check("under 5 frames refused", False)
        except ValueError:
            check("under 5 frames refused", True)

        blocks = sample_text_blocks(ref.frames)
        check("124 frames read at 2 fps make 6 pairs", len(blocks) == 6, str(len(blocks)))
        check("a pair is labelled with the mean of its timestamps", [b[0] for b in blocks][:3] == [0.25, 1.25, 2.25])
        check("0.25 renders as <0.2 seconds> (round half to even)", f"{blocks[0][0]:.1f}" == "0.2")
        check("pairs are 12 frames apart at 24 fps",
              np.array_equal(blocks[1][1], ref.frames[24]) and np.array_equal(blocks[0][2], ref.frames[12]))
        check("an odd sample count repeats the last frame", np.array_equal(blocks[-1][1], blocks[-1][2]))

        ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2, image_mean=[.5] * 3,
                                   image_std=[.5] * 3, min_pixels=32 * 32 * 4, max_pixels=32 * 32 * 16384)
        enc = SimpleNamespace(processor=SimpleNamespace(image_processor=ip))
        rng = np.random.default_rng(0)
        f0, f1 = (rng.integers(0, 255, (384, 640, 3), dtype=np.uint8) for _ in range(2))
        single = np.asarray(ip(images=[f0], return_tensors="np")["pixel_values"]).reshape(-1, 3, 2, 16, 16)
        check("a picture's patches repeat the frame across the temporal patch", np.array_equal(single[:, :, 0], single[:, :, 1]))
        same, grid = MiniMaxH3TextEncoder._video_block(enc, f0, f0)
        check("a block of one frame twice equals the picture path",
              np.array_equal(same, np.asarray(ip(images=[f0], return_tensors="np")["pixel_values"])) and grid.tolist() == [[1, 24, 40]])
        mixed, _ = MiniMaxH3TextEncoder._video_block(enc, f0, f1)
        other = np.asarray(ip(images=[f1], return_tensors="np")["pixel_values"]).reshape(-1, 3, 2, 16, 16)
        got = mixed.reshape(-1, 3, 2, 16, 16)
        check("a block takes slice 0 from the first frame and slice 1 from the second",
              np.array_equal(got[:, :, 0], single[:, :, 0]) and np.array_equal(got[:, :, 1], other[:, :, 1]))

        check("--reference alone is a reference render", refused(reference=[big]) == "reference")
        check("a missing picture is refused", refused(reference=[str(tmp / "nope.png")]) is True)
        for flag, value in (("first_frame", Path(big)), ("last_frame", Path(big)), ("keyframe", [big + "@1"]),
                            ("chain_windows", 2), ("chain_context_frames", 17), ("refmod", [big])):
            check(f"--reference refuses {flag}", refused(reference=[big], **{flag: value}) is True)
        check("--reference refuses a draft cache", refused(reference=[big], draft_cache_dir=tmp) is True)


if __name__ == "__main__":
    main()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall ok")
    sys.exit(1 if FAILURES else 0)
