"""immichRingNG -- carry the earlier clip's cast and look into an extension's brief.

When "Extend this video" is on, the brief writer sees only the new sentence: it cannot see the clip being extended, so it
invents its own people and style ("live-action", a "luminous" fairy, a long beard) and the cast and look lines that
h3_continueNG then copies into every window would copy that. This reads the extended film's own window prompts (from its
folder, so it works after a restart), asks the same narrow Gemma call h3_continueNG uses what cast and look they
established, and adds those two lines to the start of every new window. Nothing is rendered; the person still reads and
edits every window.

If Gemma is down or the earlier prompts cannot be read, the brief is left as the compiler wrote it, with a note.
"""

import json
from pathlib import Path
from typing import Callable

import h3_continueNG as cont
import h3_rerollNG as reroll

MAX_PROMPTS = 4   # the earlier film's last windows are enough to read the established look from


def _metrics_prompt_ng(src) -> str:
    """The prompt the runner recorded beside the film (a finished job read back from disk keeps no prompt in its settings)."""
    try:
        path = next(Path(src.job_dir).glob("*.metrics.json"))
        return str(json.loads(path.read_text(encoding="utf-8")).get("prompt") or "")
    except (StopIteration, OSError, ValueError, AttributeError):
        return ""


def previous_prompts_ng(job_id: str) -> list:
    """The window prompts of the film at `job_id`, newest windows last. LookupError / ValueError if it cannot be read."""
    src = reroll.resolve_source_ng(job_id)
    try:
        prompts = [w.get("prompt", "") for w in reroll.load_manifest_ng(src)["windows"]]
    except ValueError:
        prompts = [getattr(src, "prompt", "") or _metrics_prompt_ng(src)]   # a single-window render has no manifest
    prompts = [p for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("the earlier clip has no saved prompt to read the look from")
    return prompts[-MAX_PROMPTS:]


def established_ng(job_id: str, intent: str = "", ask: Callable[[str], str] = cont.ask_gemma_ng) -> dict:
    """{"cast": str, "look": str} the earlier film established. ContinueError if Gemma gives nothing usable."""
    windows = [{"prompt": p, "continues": False} for p in previous_prompts_ng(job_id)]
    reply = cont.parse_reply_ng(ask(cont.build_request_ng(windows, intent, False)))
    if not (reply["cast"] or reply["look"]):
        raise cont.ContinueError("the reply had no cast or look")
    return {"cast": reply["cast"], "look": reply["look"]}


def intent_with_ng(intent: str, est: dict) -> str:
    """The sentence with the established style and cast stated up front, briefly, so the brief writer keeps them."""
    parts = []
    if est.get("look"):
        parts.append(f"Style: {est['look'].rstrip('.')}.")
    if est.get("cast"):
        parts.append(f"The same people as in the earlier clip: {est['cast'].rstrip('.')}.")
    return (" ".join(parts) + " " + intent).strip()


def add_lines_ng(windows: list, est: dict) -> list:
    """Every new window starts with the established cast and look lines (one already there is skipped)."""
    out = []
    for w in windows:
        w = dict(w)
        w["prompt"] = cont._with_lines(w["prompt"], [est.get("cast", ""), est.get("look", "")])
        out.append(w)
    return out
