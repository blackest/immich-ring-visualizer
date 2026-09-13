"""Standalone script run inside ltx-2-mlx's own venv (via its env/bin/python,
never imported in-process -- same subprocess boundary as ltx_engineNG.py's
calls into the `ltx-2-mlx` binary) to ask the enhance Gemma checkpoint how
long a shot needs, in seconds.

The `ltx-2-mlx enhance` CLI subcommand only exposes Gemma's prompt-rewrite
templates (enhance_t2v/enhance_i2v) with no way to swap in a different
question. Their underlying methods do accept a `system_prompt` override,
though, so this calls that library directly with a timing-specific system
prompt instead of going through the CLI. One input (a shot description) ->
one output (a seconds estimate on stdout), matching this repo's other
engine-call shapes.
"""

import argparse
import sys

SYSTEM_PROMPT = (
    "You are a film editor timing a single shot for an image-to-video "
    "render. You will be given a shot description, which may include "
    "action beats (e.g. a character's look or gesture) and a quoted line "
    "of dialogue with a delivery note (e.g. slow drawl, hesitant, rapid, "
    "snapped). Estimate how many seconds the shot needs in total: time "
    "for any lead-in action before dialogue, the dialogue itself read at "
    "the pace its delivery note implies, and any beat after it (a cut, a "
    "reaction, etc). Respond with ONLY a single decimal number of seconds "
    "between 0.5 and 30 -- no words, no units, no punctuation, nothing else."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", required=True, help="Path to the Gemma checkpoint")
    ap.add_argument("--prompt", required=True, help="Shot description to time")
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    model = GemmaLanguageModel()
    model.load(args.gemma)
    result = model.enhance_i2v(
        args.prompt,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=16,
        seed=args.seed,
    )
    print(f"\nSeconds: {result.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
