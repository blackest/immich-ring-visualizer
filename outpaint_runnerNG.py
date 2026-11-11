"""The "Make it vertical" AI fill, run inside the LTX install's own Python (it imports ltx_pipelines_mlx).

One JSON request on stdin, progress lines on stdout, and a final "OUTPAINT_RESULT {json}" line.
The recipe is Lightricks' In/Outpainting IC-LoRA workflow as written up in
vendor/vertical_outpaint/vertical_outpaint.py (MIT, from Phosphene v5.0.16); the model calls here are ours:

  reference  source centred on a tall canvas, the new bands flat green (the colour the LoRA was trained on)
  stage 1    half resolution, 8 distilled steps, IC-LoRA at 1.0 with the reference as the guide
  blend      stage-1 frames pyramid-blended with the real source so the middle is the real picture again
  stage 2    the blend upsampled x2 and refined for 2 steps with the reference re-applied
  final      scaled to the output, blended with the source once more, the middle pasted back untouched

Request (gemma_max_length optional, default 256): {"model", "gemma", "lora", "source", "output", "work_dir", "src_w", "src_h", "out_w", "out_h",
          "frames", "fps", "prompt", "seed", "crf", "keep_audio"}
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "vertical_outpaint"))

RESULT_PREFIX = "OUTPAINT_RESULT "


def say(line: str) -> None:
    print(line, flush=True)


def shorten_prompt_encode(limit: int) -> None:
    """Gemma pads every prompt to 1024 tokens before encoding; on this Mac that one big GPU job is where macOS's
    watchdog kills the render. Phosphene's own fallback after a kill is 256 tokens (its LTX_GEMMA_FALLBACK_MAX_LENGTH);
    both encode paths go through tokenize(), so the cap goes there. The fill's prompt is short, so nothing is cut."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    original = GemmaLanguageModel.tokenize

    def tokenize(self, text, max_length=1024):
        return original(self, text, min(max_length, limit))

    GemmaLanguageModel.tokenize = tokenize


def build_pipe(req: dict):
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
    import vertical_outpaint as vo
    shorten_prompt_encode(int(req.get("gemma_max_length") or 256))
    strength = float(req.get("lora_strength") or vo.OUTPAINT_LORA_STRENGTH)
    return ICLoraPipeline(model_dir=str(req["model"]), lora_paths=[(str(req["lora"]), strength)],
                          gemma_model_id=str(req["gemma"]), low_memory=True)


def run_stage(req: dict) -> None:
    """Child mode: ONE model call in a fresh process, then exit. macOS's GPU watchdog killed stage 2 every time when it
    ran in the same warm process as stage 1 (Phosphene's notes say the same: a fresh process is what helps)."""
    import vertical_outpaint as vo
    pipe = build_pipe(req)
    common = dict(prompt=str(req.get("prompt") or ""), output_path=req["stage_out"],
                  video_conditioning=[(req["ref"], 1.0)], height=int(req["ch"]), width=int(req["cw"]),
                  num_frames=int(req["n"]), frame_rate=float(req["fps"]), seed=int(req["stage_seed"]))
    if req["stage"] == 1:
        pipe.generate_and_save(stage1_steps=vo.STAGE1_STEPS, skip_stage_2=True, **common)
    else:
        pipe.generate_and_save(source_video=req["source_video"], refine_steps=vo.REFINE_STEPS, **common)


ATTEMPTS = 3   # macOS's GPU watchdog kills a stage at random, usually while it reads the prompt; a fresh process is the cure


def child(req: dict, **extra) -> None:
    """Run one stage as a child of this process, passing its output lines on; retry it in a fresh process if killed."""
    import subprocess
    last = ""
    for attempt in range(1, ATTEMPTS + 1):
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        proc.stdin.write(json.dumps({**req, **extra}) + "\n")
        proc.stdin.close()
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-8:]
            say(line)
        if not proc.wait():
            return
        last = " | ".join(l for l in tail if l.strip())[-300:]
        if "Impacting Interactivity" not in last and "kIOGPU" not in last:
            break                                               # not the watchdog: retrying would not help
        say(f"Outpaint: macOS stopped stage {extra.get('stage')} (attempt {attempt}/{ATTEMPTS})"
            + ("; starting it again" if attempt < ATTEMPTS else ""))
    raise RuntimeError(f"stage {extra.get('stage')} failed: " + last)


def run(req: dict) -> dict:
    import subprocess

    import vertical_outpaint as vo

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    started = time.monotonic()
    source, output = str(req["source"]), str(req["output"])
    work = Path(req["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    src_w, src_h = int(req["src_w"]), int(req["src_h"])
    out_w, out_h = int(req["out_w"]), int(req["out_h"])
    cw, ch = vo.canvas_size(out_w, out_h)
    fps = float(req.get("fps") or 24.0)
    total = int(req["frames"])
    seed = int(req["seed"])

    windows = vo.windows(total)
    say(f"Outpaint: {src_w}x{src_h} -> canvas {cw}x{ch} -> {out_w}x{out_h}, {total} frames in "
        f"{len(windows)} window(s)")
    parts = []
    for wi, (start, count) in enumerate(windows):
        n = vo.render_frames(count)
        ref = str(work / f"w{wi}_ref.mp4")
        r = subprocess.run(vo.reference_args(source, ref, src_w, src_h, cw, ch, n, start_frame=start,
                                             fps=fps, ffmpeg=ffmpeg),
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            raise RuntimeError("could not build the reference clip: " + r.stderr[-300:])
        geometry = dict(ref=ref, cw=cw, ch=ch, n=n, fps=fps, stage_seed=seed + wi)
        say(f"Outpaint: window {wi + 1}/{len(windows)} stage 1 (frames {start}..{start + count - 1}, {n} rendered)")
        s1 = str(work / f"w{wi}_s1.mp4")
        child(req, stage=1, stage_out=s1, **geometry)
        s1_blend = str(work / f"w{wi}_s1_blend.mp4")
        vo.composite_video(s1, ref, s1_blend, cw // 2, ch // 2, src_w, src_h, n, fps=fps, lossless=True,
                           seam_inset=3, keep_inset=20, ffmpeg=ffmpeg, source_is_canvas=True)
        say(f"Outpaint: window {wi + 1}/{len(windows)} stage 2")
        s2 = str(work / f"w{wi}_s2.mp4")
        child(req, stage=2, stage_out=s2, source_video=s1_blend, **geometry)
        part = str(work / f"w{wi}_final.mp4")
        vo.composite_video(s2, source, part, out_w, out_h, src_w, src_h, count, start_frame=start, fps=fps,
                           lossless=True, ffmpeg=ffmpeg)
        parts.append(part)

    say("Outpaint: joining and encoding")
    vo.join_parts(parts, source, output, fps=fps, crf=req.get("crf", 18), audio=bool(req.get("keep_audio", True)),
                  ffmpeg=ffmpeg, steady=(out_w, out_h, src_w, src_h))
    return {"output": output, "seconds": round(time.monotonic() - started, 1), "windows": len(windows)}


def main() -> None:
    req = json.loads(sys.stdin.readline())
    if req.get("stage"):
        run_stage(req)
        return
    try:
        result = run(req)
    except Exception as exc:                                      # noqa: BLE001
        say("Outpaint failed: " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
        say(traceback.format_exc())
        sys.exit(1)
    finally:
        if not req.get("keep_work"):
            shutil.rmtree(req.get("work_dir", ""), ignore_errors=True)
    say(RESULT_PREFIX + json.dumps(result))


if __name__ == "__main__":
    main()
