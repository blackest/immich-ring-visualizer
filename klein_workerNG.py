"""immichRingNG -- FLUX 3 Klein worker. Run only with the FLUX 3 worker's own python.

klein_engineNG.py starts this as a subprocess with the worker's venv python (mflux and MLX live
there, never in the app) and writes one JSON request on stdin:
    {"model": <snapshot dir>, "variant": "distilled"|"base", "prompt", "seeds": [..], "width",
     "height", "output_dir", "refs": [paths],
     optional: "steps", "guidance" (base only), "match_ref" (edit at the first reference's size),
     "low_ram" (default true), "vae_tile" (pixels), "loras": [[path, scale], ..]}
For each seed it renders one PNG through mflux's own command (text to image, or edit when there are
reference pictures) and prints `KLEIN_RESULT {json}` so the app can pick the files up.
"""

import gc
import json
import sys
import time
from pathlib import Path

DISTILLED_STEPS = 16
BASE_STEPS = 50
BASE_GUIDANCE = "3.5"


def cli_args(req: dict, seed: int, output: Path) -> list:
    base = req["variant"] == "base"
    steps = req.get("steps") or (BASE_STEPS if base else DISTILLED_STEPS)
    args = ["mflux-generate-flux3", "--model", str(req["model"]),
            "--base-model", "flux3-image-klein" + ("-base" if base else ""),
            "--prompt", req["prompt"], "--seed", str(seed), "--steps", str(steps),
            "--output", str(output), "--metadata"]
    if not (req["refs"] and req.get("match_ref")):  # an edit's own default is its first reference's size
        args += ["--width", str(req["width"]), "--height", str(req["height"])]
    if req.get("low_ram", True):
        args.append("--low-ram")
    elif req.get("vae_tile"):
        args += ["--vae-tile-size", str(req["vae_tile"])]
    if base:
        args += ["--guidance", str(req.get("guidance") or BASE_GUIDANCE)]
    for path, scale in req.get("loras", []):
        args += ["--lora", str(path), str(scale)]
    if req["refs"]:
        args += ["--image-paths", *req["refs"]]
    return args


def main() -> None:
    req = json.loads(sys.stdin.readline())
    import mlx.core as mx
    if req["refs"]:
        from mflux.models.flux3.cli.flux3_edit_generate import main as render
    else:
        from mflux.models.flux3.cli.flux3_generate import main as render
    from PIL import Image
    for seed in req["seeds"]:
        output = Path(req["output_dir"]) / f"klein_{time.time_ns()}_{seed}.png"
        mx.clear_cache()
        started = time.monotonic()
        print(f"Klein: rendering seed {seed} ({'edit' if req['refs'] else 'text to image'})", flush=True)
        sys.argv = cli_args(req, seed, output)
        render()
        if not output.is_file():
            raise RuntimeError("the worker stopped before writing a complete image")
        with Image.open(output) as im:
            im.verify()
        with Image.open(output) as im:
            width, height = im.size
        print("KLEIN_RESULT " + json.dumps({
            "png_path": str(output), "seed": seed, "width": width, "height": height,
            "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # the parent shows the last log lines
        print(f"Klein: {exc}", flush=True)
        sys.exit(1)
