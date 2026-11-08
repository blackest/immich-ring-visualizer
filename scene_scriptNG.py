"""immichRingNG -- the Scene script: who says what, in order.

    KNIGHT [on the battlements]: The wall has stood a long time, and so have I.
    WOMAN [in the great hall]: Then come in out of the wind.
    KNIGHT: Some nights I count the stones.

A line that starts with `&` joins the line before it into ONE shot with both speakers in the same frame (a phone
call, a split-screen): the two characters' pictures are placed side by side and one audio track carries both voices in
turn. The split-screen test (2026-11-08) showed each person's mouth working for their own voice and still for the
other's. A two-speaker shot is still one 5 s window, so together the two lines are about 12 words:

    KNIGHT [in the tower]: Where are you now?
    & WOMAN [in the kitchen]: Right here, the pot is boiling.

An optional `[setting]` after the name says where THAT clip happens, so a scene can move indoors to outdoors. It is
free text that goes into that clip's prompt; if the speaker has a saved picture ("look") with the same name it also
picks that picture as the clip's first frame. A line without a tag stays wherever the previous line of the same
speaker was (their first setting, if none yet), so you only tag the moves.

One line is one clip (a fixed 5 s window, about 12 words), so a speaker's line that is too long is refused here with
a plain sentence, not cut off after the render. Names are what the form's character list is keyed by; they are
compared case-insensitively and may hold letters, digits, spaces, `_` and `-`.

Pure text handling: no files, no models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

MAX_LINES = 12
MAX_WORDS = 18          # a 5 s window holds about 12 at a natural pace; more only fits a fast reader
MAX_NAME = 30
MAX_SETTING = 120
MAX_SHOT_WORDS = 14       # two speakers share one 5 s window
_JOIN_RE = re.compile(r"^\s*&\s*(.*)$")
_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9 _-]{0,%d}?)\s*(?:\[([^\]\n]{1,%d})\])?\s*:\s*(.*\S)\s*$" % (MAX_NAME - 1, MAX_SETTING))


class SceneScriptError(ValueError):
    pass


@dataclass(frozen=True)
class Line:
    index: int           # 0-based position in the scene
    speaker: str         # as typed first for this character (display), compare with .key
    text: str
    setting: str = ""    # where this clip happens ("" = the speaker's previous setting, resolved by the caller)
    joined: bool = False  # True: shares the previous line's shot (a second speaker in the same frame)

    @property
    def key(self) -> str:
        return self.speaker.casefold()

    @property
    def words(self) -> int:
        return len(self.text.split())


def parse_script_ng(script: str, known: Optional[list] = None) -> list:
    """Script text -> [Line]. A line that starts `NAME:` begins an utterance; any other non-blank line continues the
    previous speaker's utterance, so a long speech can be wrapped. `known` (the form's character names, if given)
    must contain every speaker."""
    out: list = []
    first_spelling: dict = {}   # casefolded name -> first spelling
    for n, raw in enumerate((script or "").replace("\r\n", "\n").split("\n"), 1):
        if not raw.strip():
            continue
        joined = False
        jm = _JOIN_RE.match(raw)
        if jm:
            if not out:
                raise SceneScriptError(f"line {n}: '&' joins a line to the one before it; there is none yet")
            joined, raw = True, jm.group(1)
        m = _LINE_RE.match(raw)
        if m:
            name, setting, text = m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip()
            first_spelling.setdefault(name.casefold(), name)
            out.append([first_spelling[name.casefold()], text, setting, joined])
        elif joined:
            raise SceneScriptError(f"line {n}: after '&' write  NAME: What they say")
        elif out:
            out[-1][1] += " " + raw.strip()
        else:
            raise SceneScriptError(f"line {n}: start with a speaker, like  KNIGHT: What they say")
    if not out:
        raise SceneScriptError("write at least one line, like  KNIGHT: What they say")
    if len(out) > MAX_LINES:
        raise SceneScriptError(f"a scene is at most {MAX_LINES} lines for now (this has {len(out)})")
    lines = [Line(i, spk, txt, setting, joined) for i, (spk, txt, setting, joined) in enumerate(out)]
    _check_shots(lines)
    for ln in lines:
        if ln.words > MAX_WORDS:
            raise SceneScriptError(
                f"{ln.speaker}'s line {ln.index + 1} is {ln.words} words; a clip holds about 12. "
                f"Split it into two lines (the same speaker can follow themself).")
    if known is not None:
        have = {k.casefold() for k in known}
        for ln in lines:
            if ln.key not in have:
                raise SceneScriptError(f"{ln.speaker!r} has no face and voice set up; "
                                       f"add them under Characters or fix the spelling")
    return lines


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def sentences_ng(text: str) -> list:
    """A line's text split into sentences at . ! ? (the places the voice takes a breath). The text is first made to
    end with a full stop if it does not, so every sentence has a pause after it."""
    t = " ".join((text or "").split())
    if t and t[-1] not in ".!?":
        t += "."
    return [s for s in _SENT_RE.split(t) if s]


def speakable_text_ng(lines: list) -> tuple:
    """For ONE speaker's lines: (the single paragraph to speak in one pass, [sentences per line]). The paragraph is
    every line's sentences joined with spaces; the counts let the cutter regroup the cut sentences into lines."""
    sents = [sentences_ng(ln.text) for ln in lines]
    return " ".join(s for group in sents for s in group), [len(g) for g in sents]


def speakers_ng(lines: list) -> list:
    """Distinct speakers in order of first appearance, as typed."""
    seen, out = set(), []
    for ln in lines:
        if ln.key not in seen:
            seen.add(ln.key)
            out.append(ln.speaker)
    return out


def lines_of_ng(lines: list, speaker_key: str) -> list:
    return [ln for ln in lines if ln.key == speaker_key.casefold()]


def shots_ng(lines: list) -> list:
    """The lines grouped into shots, in order: [[Line]] with one line for an ordinary shot and two for a shared frame."""
    shots: list = []
    for ln in lines:
        if ln.joined and shots:
            shots[-1].append(ln)
        else:
            shots.append([ln])
    return shots


def _check_shots(lines: list) -> None:
    for shot in shots_ng(lines):
        if len(shot) == 1:
            continue
        if len(shot) > 2:
            raise SceneScriptError("a shared frame holds two speakers at most; start a new shot for the third")
        if shot[0].key == shot[1].key:
            raise SceneScriptError(f"{shot[0].speaker} follows themself with '&'; a shared frame needs two different speakers")
        words = sum(ln.words for ln in shot)
        if words > MAX_SHOT_WORDS:
            raise SceneScriptError(
                f"the shared frame with {shot[0].speaker} and {shot[1].speaker} is {words} words; two speakers share one "
                f"5 second window, so about {MAX_SHOT_WORDS - 2} words in all. Shorten both lines or split the shot.")


def resolve_settings_ng(lines: list) -> list:
    """The same lines with every setting filled in: an untagged line keeps its speaker's last setting ("" if the
    speaker has never been given one). Tags are matched as typed; case only matters for display."""
    last: dict = {}
    out = []
    for ln in lines:
        setting = ln.setting or last.get(ln.key, "")
        last[ln.key] = setting
        out.append(Line(ln.index, ln.speaker, ln.text, setting, ln.joined))
    return out


def moves_ng(lines: list) -> list:
    """Indices of lines whose setting differs from the line before it (the cuts that change location), after
    resolve_settings_ng. Used to reset the exposure hold: indoors and outdoors must not be pushed to one level."""
    resolved = resolve_settings_ng(lines)
    return [i for i in range(1, len(resolved)) if resolved[i].setting.casefold() != resolved[i - 1].setting.casefold()]
