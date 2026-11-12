"""The `ltx-2-mlx` command with Gemma's prompt padding capped, run inside the LTX install's own Python.

Gemma pads every prompt to 1024 tokens before encoding; on this Mac that one big GPU job is where macOS's watchdog
kills the render ("Impacting Interactivity") before the picture even starts. Phosphene's own fallback after a kill is
256 tokens, and the plain command has no setting for it, so this runs the same command with tokenize() capped.
Arguments pass straight through; LTX2_GEMMA_MAX_LENGTH (default 256) sets the cap, 0 leaves it alone.
Same cap as outpaint_runnerNG.shorten_prompt_encode; that file stays separate because it only imports inside its own process.
"""
from __future__ import annotations

import os
import sys

DEFAULT_CAP = 256


def cap_prompt_encode(limit: int) -> None:
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
    original = GemmaLanguageModel.tokenize

    def tokenize(self, text, max_length=1024):
        return original(self, text, min(max_length, limit))

    GemmaLanguageModel.tokenize = tokenize


def main() -> int:
    try:
        limit = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", DEFAULT_CAP))
    except ValueError:
        limit = DEFAULT_CAP
    if limit > 0:
        cap_prompt_encode(limit)
    from ltx_pipelines_mlx.cli import main as cli_main
    sys.argv[0] = "ltx-2-mlx"
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
