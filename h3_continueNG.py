"""immichRingNG -- write the chained windows a brief leaves empty, instead of repeating the shot before.

The compiler writes shots, not windows. A 20 s clip with two shots has four 5 s windows, and
`routes/enhancedh3NG._split_windows` used to give each window no shot starts in the previous shot's text
word for word: the same action twice, the same spoken line twice, and, in windows that only ever saw a later
shot, no description of who the person is (the face and clothes then drift).

This module makes one narrow Gemma call (text only; no agent, no tools) that returns:

* a cast line -- who must stay in the scene and how they look, copied from window 1 (Phosphene's tip for
  Flex applies to every H3 mode: the model drops people the prompt does not name), put at the start of every
  later window;
* for each window that carries on, a new beat in the style of the shot before it, a varied sound line, and,
  only when the brief had speech and the person allows it, a new short line in the same voice.

If Gemma is down or answers badly the windows are left as they were (the old copy), with a note.
Nothing here renders; the person still reads and edits every window prompt.
"""

import json
import re
from typing import Callable, Optional

import requests

import configNG

TIMEOUT_S = 180
SECTION = "integrated_multimodal_description"
_SPEECH_RE = re.compile(r"says in ([^:<]*?\(S\d+\)):\s*<d>\[([^\]]+)\]")
_SHOT1_RE = re.compile(r"(integrated_multimodal_description:\s*\[Shot 1\]\s*)")
_SOUND_RE = re.compile(r"(overall_soundscape:\s*)(.*?)(?=\n\n|\Z)", re.S)

SYSTEM = """You help write a chained AI video made of 5-second windows. Each window is rendered from its own prompt, so \
anything not written in a window's prompt is forgotten: people change face, hair and clothes, and a prompt that is a copy \
of the previous one makes the action happen twice.

You are given the text of the shots the brief already has and which windows have no shot of their own. Reply with ONE JSON \
object and nothing else:
{"cast": "one sentence, at most 60 words: every person who must stay in the scene, with a name if the intent gave one, \
and their visible look exactly as the first shot describes them: face, hair, build, glasses or other worn things (shape and \
colour), and clothes (material, cut, how they fit, any armour or accessories); no action",
 "windows": [{"window": <number>, "beat": "...", "soundscape": "...", "dialogue": "..."}]}

For each window you are asked to write:
- beat: 40 to 70 words of camera and action in the same style as the shots you were given. It continues straight on from \
the previous window's last moment: a NEW action or camera move, never a repeat of an earlier one, no cut, no new person. \
Do not describe the cast again; the cast line is added for you.
- soundscape: one or two sentences of ambient sound for this beat, varied from the other windows; do not describe the voice. \
Use only sounds made by things already in the scene (the place, the weather, the cast's own movement): never name an animal, \
object or person that is not already there, because the video model will draw it.
- dialogue: if you are told speech is allowed, a short new line (at most 15 words) that follows what was already said, in \
the same language, and does not repeat an earlier line. Otherwise an empty string."""


class ContinueError(Exception):
    """Gemma did not give a usable answer."""


def speech_in_ng(text: str) -> Optional[tuple]:
    """(voice phrase, language) of the first spoken line in `text`, e.g. ("a low, gravelly voice (S1)", "English"),
    or None. This is what a new line has to be written in so the same voice speaks it."""
    m = _SPEECH_RE.search(text or "")
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def _description_ng(prompt: str) -> str:
    body = prompt.split(f"{SECTION}:", 1)[-1]
    return re.split(r"\n\noverall_soundscape:", body, maxsplit=1)[0].strip()


def build_request_ng(windows: list, intent: str, speech_ok: bool) -> str:
    """The user message: the intent, every window's shot text so far, and the windows to write."""
    lines = [f"Intent: {intent.strip() or '(none given)'}", "", "Windows so far (a window marked EMPTY has no shot of its own):"]
    for i, w in enumerate(windows, 1):
        lines.append(f"Window {i}: " + ("EMPTY" if w.get("continues") else _description_ng(w["prompt"])))
    todo = [i for i, w in enumerate(windows, 1) if w.get("continues")]
    lines += ["", "Write the beat and soundscape for window(s): " + ", ".join(map(str, todo)) + ".",
              "Speech is " + ("allowed: write a new dialogue line for each of them." if speech_ok
                              else "NOT allowed: leave dialogue empty.")]
    return "\n".join(lines)


def parse_reply_ng(content: str) -> dict:
    """{"cast": str, "windows": {number: {"beat", "soundscape", "dialogue"}}} from Gemma's text; ContinueError if it
    has no JSON object or no usable window."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        raise ContinueError("no JSON in the reply")
    try:
        raw = json.loads(m.group(0))
    except ValueError as e:
        raise ContinueError(f"bad JSON in the reply: {e}") from e
    out = {}
    for item in raw.get("windows") or []:
        try:
            n = int(item["window"])
        except (KeyError, TypeError, ValueError):
            continue
        beat = str(item.get("beat") or "").strip()
        if beat:
            out[n] = {"beat": beat, "soundscape": str(item.get("soundscape") or "").strip(),
                      "dialogue": str(item.get("dialogue") or "").strip()}
    if not out:
        raise ContinueError("the reply had no window beats")
    return {"cast": str(raw.get("cast") or "").strip(), "windows": out}


def _bare_line(text: str) -> str:
    """A spoken line without a language tag or quotes of its own (the tag is added from the first window's)."""
    return re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", text).strip().strip('"\u201c\u201d')


def _with_cast(prompt: str, cast: str) -> str:
    """`cast` as the first sentence of Shot 1, once."""
    if not cast or cast in prompt:
        return prompt
    return _SHOT1_RE.sub(lambda m: m.group(1) + cast.rstrip() + " ", prompt, count=1)


def apply_ng(windows: list, reply: dict, speech: Optional[tuple], speech_ok: bool) -> list:
    """New window list: windows that carry on get Gemma's beat (and sound, and a new spoken line when `speech`
    is known and `speech_ok`); every window after the first gets the cast line. Window 1 is never touched."""
    out = []
    for i, w in enumerate(windows, 1):
        w = dict(w)
        item = reply["windows"].get(i)
        if i > 1 and w.get("continues") and item:
            beat = item["beat"].rstrip()
            if speech_ok and speech and item["dialogue"]:
                voice, lang = speech
                beat = f'{beat} The speaker says in {voice}: <d>[{lang}] {_bare_line(item["dialogue"])}</d>'
            head, _, tail = w["prompt"].partition(f"{SECTION}:")
            _desc, sep, rest = tail.partition("\n\n")
            if item["soundscape"] and rest:
                rest = _SOUND_RE.sub(lambda m: m.group(1) + item["soundscape"], rest, count=1)
            w["prompt"] = f"{head}{SECTION}: [Shot 1] {beat}" + (f"{sep}{rest}" if sep else "")
            w["generated"] = True
        if i > 1:
            w["prompt"] = _with_cast(w["prompt"], reply["cast"])
        out.append(w)
    return out


def ask_gemma_ng(user_text: str) -> str:
    """Gemma's text reply (the same standalone server the compiler and Ask Gemma use). ContinueError if it is down."""
    base = configNG.get_gemma_review_base_url()
    try:
        ids = [m.get("id", "") for m in requests.get(f"{base}/v1/models", timeout=10).json().get("data", [])]
        model = next((i for i in ids if "gemma" in i.lower()), ids[0] if ids else "")
        body = {"model": model, "temperature": 0.7, "max_tokens": 2000, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]}
        reply = requests.post(f"{base}/v1/chat/completions", json=body, timeout=TIMEOUT_S).json()
        return reply["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        raise ContinueError(f"could not reach Gemma at {base}: {e}") from e


def continue_windows_ng(windows: list, intent: str, improvise: bool = True,
                        ask: Callable[[str], str] = ask_gemma_ng) -> tuple:
    """(windows, note). Windows are returned unchanged, with a note saying why, when there is nothing to write or
    Gemma fails. Dialogue is improvised only if the first window already has a spoken line."""
    if len(windows) < 2 or not any(w.get("continues") for w in windows):
        return windows, None
    speech = speech_in_ng(" ".join(w["prompt"] for w in windows))
    speech_ok = bool(improvise and speech)
    try:
        reply = parse_reply_ng(ask(build_request_ng(windows, intent, speech_ok)))
    except ContinueError as e:
        return windows, f"Windows with no shot of their own repeat the shot before: Gemma could not write them ({e})."
    return apply_ng(windows, reply, speech, speech_ok), None
