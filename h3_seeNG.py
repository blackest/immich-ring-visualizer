"""immichRingNG -- let Gemma look at the last frame of the clip being extended and say what it sees, as plain text for the
H3 prompt writer (which is blind: Phosphene's conversion drops its vision tower).

One narrow call: the frame goes in, one short paragraph comes out (who is there, hair, clothes, place, light, style). The
paragraph is stated in the sentence the writer gets, in front of the story. If Gemma is down or the frame cannot be read,
SeeError is raised and the caller carries on without it.
"""

import base64
import subprocess
from pathlib import Path
from typing import Callable, Optional

import requests

import configNG

TIMEOUT_S = 120
SYSTEM = ("You describe one video frame for a writer who cannot see it. In 2 to 4 plain sentences say who or what is in "
          "the frame (hair, beard, clothes, species and colouring of any animal), where it is, the time of day and light, "
          "and the picture's style (for example painted, photographic, 3D). Describe only what is visible. No story, no "
          "opinions, no lists.")
USER = "Describe this frame."


class SeeError(Exception):
    pass


def last_frame_jpeg_ng(clip: Path) -> bytes:
    """The last frame of `clip` as a JPEG (ffmpeg seeks to just before the end)."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.2", "-i", str(clip), "-frames:v", "1", "-vf",
                           "scale=768:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
                          capture_output=True, timeout=60)
    if proc.returncode or not proc.stdout:
        raise SeeError("could not read the last frame of the earlier clip")
    return proc.stdout


def ask_gemma_with_picture_ng(jpeg: bytes) -> str:
    base = configNG.get_gemma_review_base_url()
    try:
        ids = [m.get("id", "") for m in requests.get(f"{base}/v1/models", timeout=10).json().get("data", [])]
        model = next((i for i in ids if "gemma" in i.lower()), ids[0] if ids else "")
        body = {"model": model, "temperature": 0.2, "max_tokens": 300, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": USER},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]}
        reply = requests.post(f"{base}/v1/chat/completions", json=body, timeout=TIMEOUT_S).json()
        return reply["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        raise SeeError(f"could not reach Gemma at {base}: {e}") from e


def clean_ng(text: str) -> str:
    """One paragraph, no markdown, bounded."""
    text = " ".join((text or "").replace("*", "").replace("#", "").split())
    if len(text) < 20:
        raise SeeError("Gemma's description was empty")
    return text[:700]


def seen_ng(clip: Path, ask: Callable[[bytes], str] = ask_gemma_with_picture_ng,
            frame: Callable[[Path], bytes] = last_frame_jpeg_ng) -> str:
    """What the last frame of `clip` shows, as one short paragraph. SeeError on any failure."""
    return clean_ng(ask(frame(Path(clip))))


def intent_with_seen_ng(intent: str, seen: Optional[str]) -> str:
    """The sentence with the seen description stated up front."""
    return f"The last frame of the earlier clip shows: {seen.rstrip('.')}. {intent}".strip() if seen else intent
