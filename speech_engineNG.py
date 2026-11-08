"""Speech engine: a script and a voice in, one WAV out.

Qwen3-TTS 12 Hz 1.7B (Apache-2.0, the Qwen team) through mlx-audio, run as a SUBPROCESS in its own
venv (`speech-mlx/.venv`), never imported here -- same shape as h3_engineNG.py/music_engineNG.py.
The runner is the copy in this repository (vendor/speech_runner/speech_run.py, see its VENDORED.md),
so a Phosphene update cannot change what this app runs. Only the weights and the venv are shared with
Phosphene.

Three ways to pick a voice (the runner has two checkpoints of the same model):
  preset     "customvoice": nine named fictional speakers + a free-text delivery instruction
  character  "base": the voice sample saved beside a character LoRA (`<name>.voice.<ext>` in the
             shared loras folder) -- a clone
  clip       "base": a short clip you give it -- a clone
A clone needs the caller to confirm the voice is their own or that they have the speaker's permission
(Phosphene's acceptable-use rule); the confirmation is written into the output's .json record.

One module, one job: no queue and no routes here (speech_jobsNG.py / routes/speechNG.py own those).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional


class SpeechJobCancelled(RuntimeError):
    """The speech subprocess was killed (cancel button or the watchdog timeout)."""


class SpeechRequestError(ValueError):
    """A request that cannot be queued as asked; the message is plain words for the UI."""


# --- where things live -------------------------------------------------------------------------

def _resolve_speech_repo_dir_ng() -> Path:
    """LTX_SPEECH_ROOT, then the clone on the AI drive, then Phosphene's own install."""
    env_p = os.environ.get("LTX_SPEECH_ROOT")
    if env_p:
        return Path(env_p).expanduser()
    for cand in (Path("/Volumes/AI/PhospheneNative/speech-mlx"),
                 Path("/Volumes/AI/Pinkio/api/phosphene.git/speech-mlx"),
                 Path.home() / "speech-mlx"):
        if (cand / ".venv" / "bin" / "python").exists():
            return cand
    return Path.home() / "speech-mlx"


def _resolve_speech_pack_ng() -> Path:
    """LTX_SPEECH_MODELS (the `speech` pack folder holding customvoice/ and base/), then the shared
    mlx_models tree."""
    env_p = os.environ.get("LTX_SPEECH_MODELS")
    if env_p:
        return Path(env_p).expanduser()
    for cand in (Path("/Volumes/AI/Pinkio/drive/drives/peers/d1785024377531/mlx_models/speech"),
                 Path("/Volumes/AI/PhospheneNative/models/speech")):
        if (cand / "customvoice" / "config.json").exists():
            return cand
    return Path.home() / "mlx_models" / "speech"


SPEECH_REPO_DIR = _resolve_speech_repo_dir_ng()
SPEECH_PACK = _resolve_speech_pack_ng()
SPEECH_PYTHON = SPEECH_REPO_DIR / ".venv" / "bin" / "python"
SPEECH_RUNNER = Path(__file__).resolve().parent / "vendor" / "speech_runner" / "speech_run.py"
# Ours, not Phosphene's: a voice made from a written description (Qwen3-TTS VoiceDesign). See its docstring.
SPEECH_DESIGN_RUNNER = Path(__file__).resolve().parent / "vendor" / "speech_runner" / "speech_design_run.py"
DESIGN_MIN_CHARS, DESIGN_MAX_CHARS = 8, 300
# Character voice samples sit beside the character LoRAs: <name>.voice.<ext>.
SPEECH_VOICES_DIR = SPEECH_PACK.parent / "loras"

VOICE_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mov")
SPEECH_MAX_CHARS = 2000
SPEECH_TIMEOUT_S = 1800.0
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")

# The model's own nine speakers (Qwen3-TTS CustomVoice). Fictional voices; no real person is imitated.
PRESETS = (
    {"id": "ryan", "label": "Ryan", "sub": "dynamic male voice, strong rhythm", "native": "English"},
    {"id": "aiden", "label": "Aiden", "sub": "sunny American male, clear midrange", "native": "English"},
    {"id": "serena", "label": "Serena", "sub": "warm, gentle young female", "native": "Chinese"},
    {"id": "vivian", "label": "Vivian", "sub": "bright, slightly edgy young female", "native": "Chinese"},
    {"id": "uncle_fu", "label": "Uncle Fu", "sub": "seasoned male, low and mellow", "native": "Chinese"},
    {"id": "dylan", "label": "Dylan", "sub": "youthful male, clear and natural", "native": "Chinese (Beijing)"},
    {"id": "eric", "label": "Eric", "sub": "lively male, slightly husky", "native": "Chinese (Sichuan)"},
    {"id": "ono_anna", "label": "Ono Anna", "sub": "playful female, light and nimble", "native": "Japanese"},
    {"id": "sohee", "label": "Sohee", "sub": "warm female, rich emotion", "native": "Korean"},
)
PRESET_IDS = frozenset(p["id"] for p in PRESETS)

# One-click delivery styles: the instruction text the model reads for a preset voice.
STYLES = (
    {"id": "natural", "label": "Natural", "instruct": ""},
    {"id": "warm", "label": "Warm", "instruct": "Warm and friendly, smiling while speaking."},
    {"id": "excited", "label": "Excited", "instruct": "Very excited and energetic, fast and bright."},
    {"id": "deadpan", "label": "Deadpan", "instruct": "Dry and deadpan, flat delivery, completely serious."},
    {"id": "dramatic", "label": "Dramatic", "instruct": "Grand and theatrical, like a ringmaster announcing the show."},
    {"id": "whisper", "label": "Whisper", "instruct": "Whispering softly, close to the microphone, secretive."},
    {"id": "sad", "label": "Sad", "instruct": "Sad and quiet, slow, on the edge of tears."},
    {"id": "angry", "label": "Angry", "instruct": "Angry and sharp, forceful, clipped words."},
)
STYLE_BY_ID = {s["id"]: s for s in STYLES}

LANGUAGES = ("english", "chinese", "japanese", "korean", "german", "french", "russian",
             "portuguese", "spanish", "italian")


def list_voice_samples_ng() -> list:
    """Every `<name>.voice.<ext>` in the shared loras folder: the saved voices of your characters."""
    out = []
    if not SPEECH_VOICES_DIR.is_dir():
        return out
    for p in sorted(SPEECH_VOICES_DIR.iterdir()):
        stem, sep, ext = p.name.partition(".voice.")
        if not sep or f".{ext.lower()}" not in VOICE_EXTS or not _NAME_RE.match(stem):
            continue
        out.append({"id": stem, "label": stem, "path": str(p), "bytes": p.stat().st_size})
    return out


def resolve_voice_sample_ng(name: str) -> Optional[Path]:
    """A character's voice sample by name, never by a client-supplied path."""
    for v in list_voice_samples_ng():
        if v["id"] == name:
            return Path(v["path"])
    return None


def speech_health_ng() -> dict:
    """What the Speech view needs to know before it offers a Generate button."""
    missing = []
    if not SPEECH_PYTHON.exists():
        missing.append(f"speech environment not found at {SPEECH_PYTHON}")
    if not SPEECH_RUNNER.is_file():
        missing.append(f"vendored runner missing at {SPEECH_RUNNER}")
    preset_ok = (SPEECH_PACK / "customvoice" / "config.json").is_file()
    clone_ok = (SPEECH_PACK / "base" / "config.json").is_file()
    if not preset_ok:
        missing.append(f"preset voice model not found in {SPEECH_PACK / 'customvoice'}")
    if not clone_ok:
        missing.append(f"clone voice model not found in {SPEECH_PACK / 'base'}")
    # VoiceDesign is optional: it never makes the engine "not ready", the Speech view just hides the option without it.
    design_ok = (SPEECH_PACK / "voicedesign" / "config.json").is_file() and SPEECH_DESIGN_RUNNER.is_file()
    return {"ok": not missing, "preset_ok": preset_ok, "clone_ok": clone_ok, "design_ok": design_ok,
            "missing": missing, "pack": str(SPEECH_PACK), "max_chars": SPEECH_MAX_CHARS}


# --- one request -> one wav --------------------------------------------------------------------

def _ffmpeg_dir() -> Optional[str]:
    """The runner needs ffmpeg (reference conversion, Speed). System first, then Phosphene's own."""
    found = shutil.which("ffmpeg")
    if found:
        return str(Path(found).parent)
    for cand in (Path.home() / "Library/Application Support/Phosphene/runtime/bin",
                 Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        if (cand / "ffmpeg").exists():
            return str(cand)
    return None


def _clean_env_ng() -> dict:
    """Decided, not inherited: no PyTorch MPS switches or stray Python env, offline weights."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("Malloc")
           and k not in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
                         "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    ff = _ffmpeg_dir()
    if ff:
        env["PATH"] = ff + os.pathsep + env.get("PATH", "")
    return env


def instruction_for(style_id: str, free_text: str) -> str:
    """The delivery instruction: the free-text box if filled, else the chosen one-click style."""
    free = (free_text or "").strip()
    if free:
        return free[:300]
    return STYLE_BY_ID.get(style_id or "natural", STYLE_BY_ID["natural"])["instruct"]


def validate_request_ng(kind: str, text: str, voice: str, language: str, speed: float,
                        has_ref_audio: bool, voice_consent: bool, design_text: str = "") -> None:
    """Refuse what cannot work, in plain words, before anything is queued."""
    if kind not in ("preset", "character", "clip", "design"):
        raise SpeechRequestError("Pick a voice type: preset, character, clip or a described voice.")
    if not (text or "").strip():
        raise SpeechRequestError("The script is empty. Write at least one line.")
    if len(text) > SPEECH_MAX_CHARS:
        raise SpeechRequestError(f"The script is longer than {SPEECH_MAX_CHARS} characters. Split it into shorter takes.")
    if language not in LANGUAGES:
        raise SpeechRequestError(f"Language must be one of: {', '.join(LANGUAGES)}.")
    if not 0.5 <= speed <= 2.0:
        raise SpeechRequestError("Speed must be between 0.5 and 2.0.")
    if kind == "preset":
        if voice not in PRESET_IDS:
            raise SpeechRequestError("Pick one of the preset voices.")
    elif kind == "design":
        n = len(" ".join((design_text or "").split()))
        if not DESIGN_MIN_CHARS <= n <= DESIGN_MAX_CHARS:
            raise SpeechRequestError(f"Describe the voice in {DESIGN_MIN_CHARS} to {DESIGN_MAX_CHARS} characters, "
                                     f"for example: a warm, slow, older male voice with a slight rasp.")
    else:
        if not has_ref_audio:
            raise SpeechRequestError("The voice clip is missing. Pick the character again, or add the clip again.")
        if voice_consent is not True:
            raise SpeechRequestError("Cloning a voice needs your confirmation that the voice is your own "
                                     "or that you have the speaker's permission.")


_SPEECH_SUBPROCESS_LOCK = threading.Lock()


def generate_speech_ng(text: str, output_dir: Path, kind: str, voice: str = "", style: str = "",
                       ref_audio_path: Optional[str] = None, ref_text: str = "",
                       language: str = "english", speed: float = 1.0, seed: Optional[int] = None,
                       voice_consent: bool = False, voice_label: str = "", design_text: str = "",
                       on_log: Optional[Callable[[str], None]] = None,
                       on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: the script in, `speech.wav` (+ `speech.wav.json`) out. `style` is the
    already-resolved delivery instruction (see instruction_for). Raises SpeechRequestError for a bad
    request, SpeechJobCancelled if the process is killed, RuntimeError for any other failure."""
    validate_request_ng(kind, text, voice, language, speed,
                        bool(ref_audio_path and Path(ref_audio_path).is_file()), voice_consent, design_text)
    health = speech_health_ng()
    if not health["ok"]:
        raise RuntimeError("The speech engine is not ready: " + "; ".join(health["missing"]))
    if kind == "design" and not health["design_ok"]:
        raise RuntimeError("The voice-design model is not installed (it belongs in " + str(SPEECH_PACK / "voicedesign") + ").")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_file = output_dir / "script.txt"
    text_file.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
    out = output_dir / "speech.wav"

    if kind == "design":
        cmd = [str(SPEECH_PYTHON), str(SPEECH_DESIGN_RUNNER), "--pack", str(SPEECH_PACK),
               "--design", " ".join(design_text.split()), "--language", language, "--speed", f"{speed:.2f}",
               "--seed", str(seed if seed is not None else -1), "--text-file", str(text_file), "--out", str(out)]
    else:
        cmd = [str(SPEECH_PYTHON), str(SPEECH_RUNNER), "--pack", str(SPEECH_PACK),
               "--kind", "preset" if kind == "preset" else "clone",
               "--language", language, "--speed", f"{speed:.2f}",
               "--seed", str(seed if seed is not None else -1),
               "--text-file", str(text_file), "--out", str(out)]
    if kind == "design":
        pass
    elif kind == "preset":
        cmd += ["--voice", voice]
        if style:
            cmd += ["--style", style]
    else:
        cmd += ["--ref-audio", str(ref_audio_path)]
        if ref_text.strip():
            cmd += ["--ref-text", ref_text.strip()[:600]]

    timeout_s = float(os.environ.get("RINGVIZ_SPEECH_TIMEOUT_S", SPEECH_TIMEOUT_S))
    started = time.time()
    done_info: dict = {}
    error_line = ""
    tail: list = []
    timed_out = {"v": False}
    with _SPEECH_SUBPROCESS_LOCK:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=_clean_env_ng(), cwd=str(SPEECH_REPO_DIR), start_new_session=True)
        if on_proc_start:
            on_proc_start(proc)

        def _kill_on_timeout():
            timed_out["v"] = True
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        watchdog = threading.Timer(timeout_s, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()
        try:
            for raw in proc.stdout or []:
                line = raw.rstrip()
                if not line:
                    continue
                tail.append(line)
                if len(tail) > 40:
                    tail.pop(0)
                m = re.match(r"^\[speech\] (done|error)\s*(.*)$", line)
                if m and m.group(1) == "done":
                    try:
                        done_info = json.loads(m.group(2))
                    except ValueError:
                        pass
                elif m:
                    error_line = m.group(2)
                if on_log:
                    on_log(f"[speech] {line}" if not line.startswith("[speech]") else line)
            rc = proc.wait()
        finally:
            watchdog.cancel()

    if timed_out["v"]:
        raise SpeechJobCancelled(f"Speech exceeded its {timeout_s:.0f}s deadline and was killed.")
    if rc != 0:
        if rc in (-15, 143, -9, 137, 130):
            raise SpeechJobCancelled(f"Speech job cancelled (rc={rc})")
        if rc == 3:
            raise RuntimeError(error_line or "Not enough free memory for speech. Close other apps and try again.")
        raise RuntimeError(error_line or f"The speech engine exited with code {rc}: " + " | ".join(tail[-4:]))
    if not out.is_file():
        raise RuntimeError("The speech engine finished but wrote no audio. " + (error_line or ""))

    record = {
        "engine": "speech", "model": ("Qwen3-TTS 12Hz 1.7B VoiceDesign (bf16 MLX)" if kind == "design"
                                      else "Qwen3-TTS 12Hz 1.7B (8-bit MLX)"), "voice_kind": kind,
        "voice": voice_label or voice or "", "style": style, "language": language, "speed": speed,
        "design": " ".join(design_text.split()) if kind == "design" else "",
        "seed": done_info.get("seed", seed), "text": text, "ref_text": ref_text if kind in ("character", "clip") else "",
        "audio_seconds": done_info.get("audio_seconds"), "wall_seconds": round(time.time() - started, 2),
        "runner": done_info,
    }
    if kind in ("character", "clip"):
        record["voice_consent"] = {"confirmed": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "statement": "The voice is the requester's own, or they have the speaker's permission."}
    (output_dir / "speech.wav.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"path": str(out), "audio_seconds": done_info.get("audio_seconds"), "record": record}
