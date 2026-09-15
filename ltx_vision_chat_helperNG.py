"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
same subprocess boundary as ltx_scene_chat_helperNG.py) for a free-form chat
against the enhance Gemma checkpoint that can also see an attached image.

Sibling of ltx_scene_chat_helperNG.py, split out rather than branching
inside it, because the two load Gemma through entirely different libraries:
that one uses GemmaLanguageModel (mlx_lm, text-only). This one uses
mlx_vlm's Gemma3ForConditionalGeneration, the same vision-capable path
ltx_vision_enhance_helperNG.py uses for image-aware prompt enhancement.

Images arrive on stdin as base64 (alongside the turns, not as --image
paths) and stay in memory the whole way through: mlx_vlm's own
load_image() accepts a "data:image/...;base64,..." string directly and
decodes it into a BytesIO under the hood, so there's never a reason to
round-trip the bytes through the filesystem for a value that starts and
ends its life as base64 text anyway -- see the no-disk-cache principle in
this repo's memory notes (only real exports belong on disk).

mlx_vlm's apply_chat_template only places image tokens at the LAST
non-system/assistant message in the list (see its own prompt_utils.py --
"find the last user message to place image/audio tokens"), stripping any
image reference from earlier turns. So only the CURRENT turn's attached
image(s) are ever actually visible to Gemma -- an image attached earlier
in the conversation and referenced again later is not re-shown; the text
around it survives, the pixels don't. ltx_engineNG.chat_with_gemma_ng only
ever passes images from the last user turn for exactly this reason.
"""

import argparse
import json
import sys


def _data_uri(b64: str) -> str:
    # The declared subtype is irrelevant to mlx_vlm's own data-URI parsing
    # (utils.load_image just checks the "data:image/" prefix, splits on
    # the comma, and base64-decodes -- PIL sniffs the real format from the
    # decoded bytes) so "jpeg" here is just a placeholder, not a claim
    # about the actual encoding.
    return "data:image/jpeg;base64," + b64


SYSTEM_PROMPT = (
    "You are Gemma, chatting with a filmmaker who is using you elsewhere in "
    "this app to storyboard shots for an image-to-video render. Right now "
    "they just want to talk -- about the project, an idea, or anything else "
    "on their mind, sometimes with a reference image attached. Reply "
    "naturally and conversationally, in plain text. Do not force the "
    "conversation toward shot options, JSON, or any fixed format unless "
    "they specifically ask for one."
    " You have a friend, Rachel, who can look things up in the real world "
    "for you -- current events, live data, anything your training can't "
    "know or might have wrong. If you genuinely need that, write a line "
    "starting with exactly 'ASK_RACHEL:' followed by your question, and "
    "nothing else in your reply. Only do this when you actually need "
    "outside information -- not for creative or storyboarding questions "
    "you can already answer yourself."
)


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
        print("vision chat produced empty output", file=sys.stderr)
        return 1
    print(f"\nReply: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
