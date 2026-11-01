"""Klein "Improve prompt": rewrite a short request as the full caption FLUX 3 Klein is trained on.

BFL's reference code runs a separate 18.8 GB upsampler model; mflux does not port it. This is one
narrow call to a local Ollama model instead: prompt in, caption out, nothing else. The rewrite is
shown to the user and only used if they accept it. The system prompt is editable in the UI.
"""

import requests

from configNG import get_ollama_base_url

TIMEOUT_S = 180
MAX_PROMPT = 4000

DEFAULT_SYSTEM = (
    "You rewrite a short picture request as one complete caption for an image model.\n"
    "Rules:\n"
    "- Describe the whole finished picture: subject, setting, composition, lighting, palette, style.\n"
    "- Keep every fact the user gave. Do not change who or what is in it. Add detail, never contradict.\n"
    "- Describe style by its attributes (palette, stroke, texture), not by naming an artist.\n"
    "- Any text that must appear in the picture goes in single quotes exactly as given.\n"
    "- Keep wording such as 'reference 1' or 'the person from reference 2' exactly as written.\n"
    "- Keep every <tag> like <vase_1> exactly as written, in the same place.\n"
    "- One paragraph, plain prose, under 120 words. No lists, no preamble, no quotes around it.\n"
    "Reply with the caption only."
)


def improve_prompt_ng(prompt: str, model: str, system: str | None = None, post=requests.post) -> str:
    """Return the rewritten caption. Raises ValueError for bad input, RuntimeError if Ollama fails."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write a prompt first.")
    if len(prompt) > MAX_PROMPT:
        raise ValueError("The prompt is too long to improve.")
    if not (model or "").strip():
        raise ValueError("Choose a model.")
    body = {
        "model": model.strip(),
        "stream": False,
        "messages": [{"role": "system", "content": (system or "").strip() or DEFAULT_SYSTEM},
                     {"role": "user", "content": prompt}],
    }
    try:
        r = post(f"{get_ollama_base_url()}/api/chat", json=body, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach Ollama: {e}") from e
    if r.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {r.status_code}.")
    text = ((r.json().get("message") or {}).get("content") or "").strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        raise RuntimeError("The model returned nothing.")
    # A rewrite that drops a tag or reference the user wrote would silently lose their layout.
    for token in _tokens(prompt):
        if token not in text:
            raise RuntimeError(f"The rewrite dropped '{token}', so it was not used. Try again or edit the system prompt.")
    return text


def _tokens(prompt: str) -> list[str]:
    import re
    return re.findall(r"<[a-z0-9_]+>", prompt)
