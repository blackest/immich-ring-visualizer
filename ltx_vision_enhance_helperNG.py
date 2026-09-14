"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
never imported in-process -- same subprocess boundary as ltx_engineNG.py's
other Gemma helpers) to rewrite an I2V prompt WITH the reference image
actually visible to Gemma.

ltx_engineNG.enhance_ltx_prompt_ng / the `ltx-2-mlx enhance` CLI it fronts
are both text-only: they load Gemma through `mlx_lm`, which has no image
input at all (confirmed in ltx_core_mlx's own GemmaLanguageModel.enhance_i2v
docstring -- "Unlike the reference implementation, this does not pass the
image to Gemma"). This script loads the exact same `gemma-3-12b-it-4bit`
checkpoint through `mlx_vlm` instead, which does have a vision tower
(Gemma3ForConditionalGeneration), and hands it the real reference image
alongside the user's raw prompt -- so the "Analyze the Image" instruction
already baked into LTX's own i2v system prompt (reused verbatim below, not
copied by hand, so it can't drift from the original) has actual pixels to
work from instead of being silently ignored.

The reference image arrives on stdin as base64, not a --image path, and
stays in memory the whole way through: mlx_vlm's own load_image() accepts
a "data:image/...;base64,..." string directly, so there's no reason to
round-trip these bytes through the filesystem -- see the no-disk-cache
principle in this repo's memory notes (only real exports belong on disk).
"""

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", required=True, help="Path to the Gemma checkpoint")
    ap.add_argument("--prompt", required=True, help="Raw user prompt to enhance")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    image_b64 = sys.stdin.read().strip()
    if not image_b64:
        print("reference image (base64 on stdin) is required", file=sys.stderr)
        return 1

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    # Reuse LTX's own i2v system prompt file rather than hand-copying its
    # text here -- it already says "Given an image (first frame)... Analyze
    # the Image", written for exactly this call shape; the text-only path
    # just never gave it an image to analyze.
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    system_prompt = GemmaLanguageModel().default_gemma_i2v_system_prompt

    model, processor = load(args.gemma)
    mx.random.seed(args.seed)

    # Declared subtype is irrelevant to mlx_vlm's data-URI parsing (it
    # just checks the "data:image/" prefix and decodes what follows the
    # comma; PIL sniffs the real format from the bytes).
    image_uri = "data:image/jpeg;base64," + image_b64

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Raw Input Prompt: {args.prompt}."},
    ]
    formatted = apply_chat_template(processor, model.config, messages, num_images=1)
    out = generate(
        model, processor, formatted, image=[image_uri],
        max_tokens=args.max_tokens, verbose=False, temperature=0.7,
    )
    text = (getattr(out, "text", out) or "").strip()
    if not text:
        print("vision prompt enhancement produced empty output", file=sys.stderr)
        return 1
    print(f"\nEnhanced: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
