"""immichRingNG -- "Write with the H3 writer": one window of an H3 prompt written by LightX2V's MiniMax-H3 prompt rewriter.

A 9B Qwen fine-tuned to turn a short idea into H3's three-field prompt (see vendor/h3_writer). The converted model is the one
Phosphene's app sets up ("Write it for me"); it is read in place, never copied. One call writes ONE window (a single
continuous 5 s shot); the writer cannot see an earlier clip, so for an extension the established cast and look lines are
added afterwards by h3_extend_lookNG, and the speaker tags are made to match the pane's. The model runs in a short-lived
child process (about 20 s, 11 GB) and must not overlap a render.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import h3_continueNG as cont

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "vendor" / "h3_writer" / "h3_writer_run.py"
PYTHON = ROOT / "vendor" / "minimax_h3_mlx" / ".venv" / "bin" / "python"
MODEL_DIRS = ("h3-prompt-rewriter-9b-mlx-q8", "h3-prompt-rewriter-9b-mlx-q4")
MODEL_ROOTS = (Path("/Volumes/AI/PhospheneNative/models"),)
SENTINEL = "@@REWRITER@@ "
TIMEOUT_S = 300
HEAD = "integrated_multimodal_description:"


class WriterError(Exception):
    pass


def model_dir_ng() -> Optional[Path]:
    """The converted writer (marker file present = complete): RINGVIZ_H3_WRITER_MODEL, then Phosphene's models folder."""
    env = os.environ.get("RINGVIZ_H3_WRITER_MODEL")
    cands = [Path(env)] if env else [root / name for root in MODEL_ROOTS for name in MODEL_DIRS]
    return next((c for c in cands if (c / "phosphene_rewriter.json").is_file()), None)


def clean_reply_ng(text: str) -> str:
    """The window only: anything before integrated_multimodal_description (the model sometimes opens with a reference-picture
    alignment paragraph) is dropped. WriterError if the reply has no description."""
    at = (text or "").find(HEAD)
    if at < 0:
        raise WriterError("the writer's reply had no integrated_multimodal_description")
    return text[at:].strip()


def with_speaker_tags_ng(prompt: str, speaker: str) -> str:
    """The writer tags the first speaker (S1); the pane's speaker tag (for example S2) replaces the tags, so a voice keeps one id
    across windows."""
    speaker = (speaker or "").strip().upper()
    if not re.fullmatch(r"S\d", speaker):
        return prompt
    return re.sub(r"\(S\d\)", f"({speaker})", prompt)


def request_ng(intent: str, seconds: float) -> dict:
    return {"prompt": intent.strip(), "duration": max(1, min(15, round(seconds))), "one_shot": True, "task": "T2VA"}


def write_window_ng(intent: str, seconds: float = 5.0, speaker: str = "", runner=subprocess.run) -> str:
    """One H3 window for `intent`. Raises WriterError (not set up, failed, or no usable reply)."""
    if not intent.strip():
        raise WriterError("describe what happens first")
    model = model_dir_ng()
    if model is None:
        raise WriterError("the H3 prompt writer is not set up -- use \"Write it for me\" once in Phosphene's H3 form to install it")
    if not PYTHON.is_file() or not WORKER.is_file():
        raise WriterError(f"missing {PYTHON.name if not PYTHON.is_file() else WORKER.name}")
    try:
        proc = runner([str(PYTHON), str(WORKER), "--model", str(model)], input=json.dumps(request_ng(intent, seconds)),
                      capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise WriterError("the H3 prompt writer took too long and was stopped") from None
    line = next((l for l in reversed((proc.stdout or "").splitlines()) if l.startswith(SENTINEL)), None)
    if line is None:
        raise WriterError("the H3 prompt writer returned nothing: " + (proc.stderr or "")[-200:].strip())
    reply = json.loads(line[len(SENTINEL):])
    if reply.get("error"):
        raise WriterError("the H3 prompt writer failed: " + reply["error"])
    return with_speaker_tags_ng(clean_reply_ng(reply.get("rewrite", "")), speaker)
