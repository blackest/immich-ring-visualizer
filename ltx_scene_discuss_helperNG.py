"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
same subprocess boundary as ltx_scene_timing_helperNG.py) for a multi-turn
"what should the next shot be" chat against the enhance Gemma checkpoint.

Every reply proposes exactly 3 labeled shot options (motion prompt +
duration) as JSON, plus a short discussion note -- the caller (see
ltx_engineNG.discuss_next_scene_ng) owns the running conversation and
resends the full turn history each call, since this subprocess has no
memory between invocations.
"""

import argparse
import json
import sys

SYSTEM_PROMPT = (
    "You are a filmmaking collaborator storyboarding one shot at a time "
    "for an image-to-video render (LTX-2.5). The user describes the "
    "previous shot (if any) and a rough idea for what comes next, and may "
    "reply across several messages asking you to refine your suggestions "
    "(e.g. \"make B punchier\", \"combine A and C\"). "
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
