"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
same subprocess boundary as ltx_scene_discuss_helperNG.py) for a free-form,
unconstrained chat against the enhance Gemma checkpoint.

Unlike ltx_scene_discuss_helperNG.py, replies are plain text on any topic --
no forced 3-option JSON shape. The caller (see ltx_engineNG.chat_with_gemma_ng)
owns the running conversation and resends the full turn history each call,
since this subprocess has no memory between invocations.
"""

import argparse
import json
import sys

SYSTEM_PROMPT = (
    "You are Gemma, chatting with a filmmaker who is using you elsewhere in "
    "this app to storyboard shots for an image-to-video render. Right now "
    "they just want to talk -- about the project, an idea, or anything else "
    "on their mind. Reply naturally and conversationally, in plain text. Do "
    "not force the conversation toward shot options, JSON, or any fixed "
    "format unless they specifically ask for one."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", required=True, help="Path to the Gemma checkpoint")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    turns = json.load(sys.stdin)
    if not isinstance(turns, list) or not turns:
        print("conversation turns (a JSON list on stdin) are required", file=sys.stderr)
        return 1

    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    model = GemmaLanguageModel()
    model.load(args.gemma)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + turns
    # _enhance is the shared chat-template + mlx_generate primitive that
    # enhance_t2v/enhance_i2v both call internally -- there's no public
    # multi-turn chat method on this library, so this reaches into it
    # directly rather than through either of those single-shot wrappers.
    reply = model._enhance(messages, max_new_tokens=768, seed=args.seed)
    print(f"\nReply: {reply.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
