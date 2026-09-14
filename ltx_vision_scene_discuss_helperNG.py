"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
same subprocess boundary as ltx_vision_chat_helperNG.py) for the "what should
the next shot be" storyboard chat, when the current turn has an attached
image.

Sibling of ltx_scene_discuss_helperNG.py (text-only, mlx_lm's
GemmaLanguageModel), split out for the same reason ltx_vision_chat_helperNG.py
was split from ltx_scene_chat_helperNG.py: vision needs mlx_vlm's
Gemma3ForConditionalGeneration instead. Same JSON-options reply contract as
the text-only helper -- ltx_engineNG._parse_scene_discuss_json_ng parses
either one's output identically.

Images arrive on stdin as base64 (alongside the turns) and stay in memory
the whole way through -- see ltx_vision_chat_helperNG.py's docstring for why,
and for why only the CURRENT turn's image(s) can ever be visible to Gemma.
"""

import argparse
import json
import sys

SYSTEM_PROMPT = (
    "You are a filmmaking collaborator storyboarding one shot at a time "
    "for an image-to-video render (LTX-2.5). The user describes the "
    "previous shot (if any) and a rough idea for what comes next, may "
    "attach a reference image (the frame the next shot will animate from), "
    "and may reply across several messages asking you to refine your "
    "suggestions (e.g. \"make B punchier\", \"combine A and C\"). "
    "Every reply MUST propose exactly 3 distinct options for the next "
    "shot, labeled A, B, C. Each option needs: a concise motion prompt "
    "(present tense, describing camera/character motion and any "
    "dialogue with a delivery note, suitable as direct input to an "
    "image-to-video model) and an estimated duration in seconds (a "
    "decimal between 0.5 and 30, accounting for dialogue pace and any "
    "lead-in/trailing action beats). "
    "Reply with ONLY a single JSON object, no markdown fences, no text "
    "outside the JSON, in exactly this shape: "
    '{"discussion": "<1-3 sentences of natural-language commentary>", '
    '"options": [{"label": "A", "prompt": "...", "seconds": 4.5}, '
    '{"label": "B", "prompt": "...", "seconds": 3.0}, '
    '{"label": "C", "prompt": "...", "seconds": 6.5}]}'
)


def _data_uri(b64: str) -> str:
    return "data:image/jpeg;base64," + b64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", required=True, help="Path to the Gemma checkpoint")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    payload = json.load(sys.stdin)
    turns = payload.get("turns") if isinstance(payload, dict) else None
    if not isinstance(turns, list) or not turns:
        print("conversation turns (payload.turns, a JSON list) are required", file=sys.stderr)
        return 1
    images = payload.get("images") or []
    if not isinstance(images, list):
        images = []

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(args.gemma)
    mx.random.seed(args.seed)

    image_uris = [_data_uri(b64) for b64 in images if isinstance(b64, str) and b64]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + turns
    formatted = apply_chat_template(
        processor, model.config, messages, num_images=len(image_uris))
    out = generate(
        model, processor, formatted, image=image_uris or None,
        max_tokens=args.max_tokens, verbose=False, temperature=0.7,
    )
    text = (getattr(out, "text", out) or "").strip()
    if not text:
        print("vision scene discussion produced empty output", file=sys.stderr)
        return 1
    print(f"\nReply: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
