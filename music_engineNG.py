"""NG engine for MiniMax-free music generation via YuE2 (style + lyrics ->
one song), a peer of ltx_engineNG.py / h3_engineNG.py.

Talks to the standalone yue2-mlx install (vanch007/mlx-Yue port,
package name "lyra") as a subprocess, its own venv (Python 3.12,
mlx 0.32.2 -- a real pin conflict with LTX/H3's Python 3.11/mlx 0.31.1
lane, not a version to "catch up"; see the conversation this file grew
out of), never imported in-process. Uses the standalone repo's OWN
`lyra generate` console-script (installed at <repo>/.venv/bin/lyra),
not phosphene's own scripts/music/yue2_run.py -- that script is "owned
by the panel" (phosphene's own wrapper, not part of the upstream
checkout), and depending on it would reach across the "two apps stay
separate, share only the model store" line this repo's engines
otherwise hold (ltx-2-mlx's own binary, minimax-h3-mlx's own
generate_staged.py -- always the standalone repo's own entrypoint).

One input (style + lyrics + duration) -> one output (a .flac path +
result dict). `lyra generate` takes a JSON request (style/lyrics/cot/
seed/cfg_scale) plus a `semantic_sampling.max_tokens` override for
duration -- see _build_request_ng's docstring for where the token-per-
second math comes from. Writes the request alongside the rendered song
under the job's own output_dir, both real provenance, not scratch.

Also supports `lyra cover` (generate_cover_ng): transcribe an existing
track's melody/chords into an ABC score (native MLX SheetSage2 --
lyra-mlx/src/lyra/transcription/pipeline.py, not the PyTorch/MPS path
some ComfyUI node used, which was measured falling back to full
emulation for every quantization op it needed -- ~1.5s/token instead
of the MLX-native run this wraps), then render a fresh take over that
score with your own style/lyrics. Unlike generate, the first cover
call needs network access once: SheetSage2 + MERT-v2-FullSong aren't
part of the generate-only weights pack already on this machine, so
they're fetched to MUSIC_HF_CACHE_DIR on first use (see
_resolve_music_cover_models_ng) -- not passed --offline for that
reason, everything else here always is.

Deliberately out of scope here (keep this file small -- add a sibling
module later if any of these turn out to be needed): supplied/edited
ABC scores (--abc) on the plain generate path, the plan/render-plan/
replay two-step workflow, batch requests, 4-bit precision (no ar-4bit.
safetensors on this machine's pack -- see _resolve_music_precision_ng).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

MusicMode = Literal["full", "melody", "off"]
MusicPrecision = Literal["bf16", "8bit"]


class MusicJobCancelled(RuntimeError):
    """Raised when the music subprocess is killed by its watchdog timeout.
    No cancel button exists yet, so the only source here is the timeout."""


def _resolve_music_repo_dir_ng() -> Path:
    """LTX_MUSIC_ROOT env var (same name phosphene's own mlx_ltx_panel.py
    reads), then conventional install locations."""
    env_p = os.environ.get("LTX_MUSIC_ROOT")
    if env_p:
        return Path(env_p).expanduser()
    home = Path.home()
    for cand in (home / "yue2-mlx",
                 home / "AI" / "yue2-mlx",
                 Path("/Volumes/AI/yue2-mlx"),
                 Path("/Volumes/AI/Pinkio/api/phosphene.git/yue2-mlx")):
        if cand.exists():
            return cand
    return home / "yue2-mlx"


def _resolve_music_models_root_ng() -> Path:
    """LTX_MUSIC_MODELS env var (same name phosphene reads), then the
    conventional shared mlx_models/yue2 tree -- same drive
    LTX_WEIGHTS_ROOT/H3_MODELS_ROOT already live under."""
    env_p = os.environ.get("LTX_MUSIC_MODELS")
    if env_p:
        return Path(env_p).expanduser()
    cand = Path("/Volumes/AI/Pinkio/drive/drives/peers/d1785024377531/mlx_models/yue2")
    if cand.exists():
        return cand
    return Path.home() / "mlx_models" / "yue2"


MUSIC_REPO_DIR = _resolve_music_repo_dir_ng()
MUSIC_MODELS_ROOT = _resolve_music_models_root_ng()
MUSIC_DEFAULT_BIN = MUSIC_REPO_DIR / ".venv" / "bin" / "lyra"
MUSIC_MODEL_DIR = MUSIC_MODELS_ROOT / "generator"
MUSIC_VAE_DIR = MUSIC_MODELS_ROOT / "vae"

# Docs' own fitted timing model was measured on 8-bit AR, and that's the
# precision file actually on disk here (alongside bf16) -- see
# _resolve_music_precision_ng. 4-bit is a real choice upstream but no
# ar-4bit.safetensors exists in this pack, so it's left out entirely
# rather than offered and failing at render time.
MUSIC_DEFAULT_PRECISION: MusicPrecision = "8bit"

# yue2_run.py's own constants (docs/MUSIC_ENGINE.md, scripts/music/
# yue2_run.py) -- duplicated here, not imported, for the same reason
# ltx_engineNG.py/h3_engineNG.py duplicate their own venv-only helpers'
# constants: this file only ever reaches the yue2-mlx venv as a
# subprocess, never in-process.
MUSIC_TOKENS_PER_SECOND = 25
MUSIC_MIN_SECONDS = 8.0
MUSIC_MAX_SECONDS = 360.0
# YuE2 has no instrumental switch of its own; its lyrics protocol reads
# bare section tags with nothing under them as instrumental passages.
# Validated by ear in phosphene's own bake-off (yue2_run.py's comment).
MUSIC_INSTRUMENTAL_LYRICS = "[Intro]\n\n[Instrumental]\n\n[Instrumental]\n\n[Outro]"
MUSIC_INSTRUMENTAL_STYLE = "instrumental, no vocals"

MUSIC_MODES: tuple = ("full", "melody", "off")

# `lyra cover`'s own task choices (yue2-mlx/src/lyra/cli.py) -- picks
# BOTH the transcription target (what SheetSage2 extracts) and the
# render mode, which cover() derives as "full" if task=="full" else
# "melody" (see _cover_mode_for_task_ng). "off" isn't a cover task --
# a cover's whole point is rendering over a transcribed score.
MUSIC_COVER_TASKS: tuple = ("full", "melody-full", "melody-vocal")
MUSIC_DEFAULT_COVER_TASK: str = "melody-full"

# HF repo IDs transcribe() downloads on first use (yue2-mlx's own
# defaults, lyra/transcription/pipeline.py) -- separate from the
# generate-only weights pack already on this machine (MUSIC_MODEL_DIR/
# MUSIC_VAE_DIR), so the first cover call needs network access once.
# Cached under the shared models root (not the yue2-mlx checkout
# itself, which git operations could wipe) so it survives like every
# other weight here.
MUSIC_TRANSCRIPTION_MODEL = "m-a-p/SheetSage2"
MUSIC_BASE_MODEL = "m-a-p/MERT-v2-FullSong"
MUSIC_HF_CACHE_DIR = MUSIC_MODELS_ROOT / "hf-cache"


def _cover_mode_for_task_ng(task: str) -> str:
    """Mirrors lyra.commands.cover's own derivation exactly -- the
    request's `cot` MUST equal this or the upstream command raises."""
    return "full" if task == "full" else "melody"


def _validate_music_duration_ng(duration_s: float) -> None:
    if not (MUSIC_MIN_SECONDS <= duration_s <= MUSIC_MAX_SECONDS):
        raise ValueError(
            f"duration_s must be between {MUSIC_MIN_SECONDS} and "
            f"{MUSIC_MAX_SECONDS} (got {duration_s})")


def _clean_subprocess_env_ng() -> dict:
    """Same Malloc*-stripping helper as ltx_engineNG.py/h3_engineNG.py."""
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("Malloc"):
            del env[key]
    return env


@dataclass
class MusicConfig:
    """How to run yue2-mlx's `lyra generate` for one song."""
    precision: MusicPrecision = MUSIC_DEFAULT_PRECISION
    binary_path: str = ""    # default: MUSIC_DEFAULT_BIN
    model_dir: str = ""      # default: MUSIC_MODEL_DIR
    vae_dir: str = ""        # default: MUSIC_VAE_DIR
    timeout_s: float = 900.0  # per-call watchdog; override via RINGVIZ_MUSIC_TIMEOUT_S
    # Cover adds a transcription pass on top of generate's own render,
    # plus a possible first-use HF download of the transcription/base
    # models -- more headroom than plain generate's timeout_s.
    cover_timeout_s: float = 1500.0  # override via RINGVIZ_MUSIC_COVER_TIMEOUT_S


def _resolve_music_binary_ng(config: MusicConfig) -> Optional[str]:
    p = Path(config.binary_path) if config.binary_path else MUSIC_DEFAULT_BIN
    return str(p) if p.is_file() and os.access(p, os.X_OK) else None


def _resolve_music_vae_dir_ng(config: MusicConfig) -> Optional[str]:
    p = Path(config.vae_dir) if config.vae_dir else MUSIC_VAE_DIR
    return str(p) if (p / "model.safetensors").is_file() and (p / "config.json").is_file() else None


def _resolve_music_model_dir_ng(config: MusicConfig) -> Optional[str]:
    """The precision picked (config.precision) must have its own AR
    weight file in the generator dir -- YuE2Pipeline itself refuses to
    load otherwise (see yue2-mlx/src/lyra/pipeline.py: "Convert the
    requested AR precision first")."""
    p = Path(config.model_dir) if config.model_dir else MUSIC_MODEL_DIR
    ok = (p / "conversion.json").is_file() and (p / f"ar-{config.precision}.safetensors").is_file()
    return str(p) if ok else None


def music_health_ng(precision: MusicPrecision = MUSIC_DEFAULT_PRECISION) -> dict:
    """Cheap up-front check the routes layer can surface to the UI/status
    endpoint without launching a subprocess."""
    cfg = MusicConfig(precision=precision)
    binary = _resolve_music_binary_ng(cfg)
    model_dir = _resolve_music_model_dir_ng(cfg)
    vae_dir = _resolve_music_vae_dir_ng(cfg)
    return {
        "repo_dir": str(MUSIC_REPO_DIR),
        "models_root": str(MUSIC_MODELS_ROOT),
        "precision": precision,
        "binary_ok": binary is not None,
        "binary_path": binary or str(MUSIC_DEFAULT_BIN),
        "model_ok": model_dir is not None,
        "model_path": model_dir or str(MUSIC_MODEL_DIR),
        "vae_ok": vae_dir is not None,
        "vae_path": vae_dir or str(MUSIC_VAE_DIR),
        "ready": all((binary, model_dir, vae_dir)),
    }


_MUSIC_SUBPROCESS_LOCK = threading.Lock()


def _run_lyra_subprocess_ng(cmd: list, timeout_s: float, timeout_env_var: str,
                             on_log: Optional[Callable[[str], None]],
                             on_proc_start: Optional[Callable[[subprocess.Popen], None]]) -> list:
    """Shared subprocess-run-with-watchdog body for generate_music_ng and
    generate_cover_ng -- same GPU/venv, same _MUSIC_SUBPROCESS_LOCK,
    same kill/timeout/cancelled-rc handling either way; only the
    command line and what's done with the result differ between them.
    Returns the captured stdout lines (caller parses whatever final-
    line JSON summary or output path it expects); raises
    MusicJobCancelled/RuntimeError on timeout/failure."""
    timeout_s = float(os.environ.get(timeout_env_var, timeout_s))
    with _MUSIC_SUBPROCESS_LOCK:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_clean_subprocess_env_ng(),
            cwd=str(MUSIC_REPO_DIR),
            start_new_session=True,
        )
        if on_proc_start:
            on_proc_start(proc)

        timed_out = {"v": False}
        tail_lines: list = []

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
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    tail_lines.append(line)
                    if on_log:
                        on_log(f"[music] {line}")
            rc = proc.wait()
        finally:
            watchdog.cancel()

    if timed_out["v"]:
        raise MusicJobCancelled(
            f"Music job exceeded its {timeout_s:.0f}s deadline and was "
            f"killed (likely a hung render). Raise {timeout_env_var} "
            f"if this machine legitimately needs longer.")
    if rc != 0:
        if rc in (-15, 143, -9, 137, 130):
            raise MusicJobCancelled(f"Music job cancelled (rc={rc})")
        raise RuntimeError(f"Music job failed with rc={rc}: {' '.join(tail_lines[-5:])}")
    return tail_lines


def _build_request_ng(style: str, lyrics: str, duration_s: float, seed: int,
                       mode: MusicMode, cfg_scale: Optional[float]) -> dict:
    """seconds -> semantic_sampling.max_tokens, TOKENS_PER_SECOND per
    second of audio (yue2_run.py's own conversion -- the codec's frame
    rate, not a phosphene-specific choice). min_tokens mirrors
    yue2_run.py's own min(200, max_tokens) floor."""
    max_tokens = max(1, round(duration_s * MUSIC_TOKENS_PER_SECOND))
    request = {
        "style": style,
        "lyrics": lyrics,
        "cot": mode,
        "seed": seed,
        "semantic_sampling": {"max_tokens": max_tokens, "min_tokens": min(200, max_tokens)},
    }
    if cfg_scale is not None:
        request["cfg_scale"] = float(cfg_scale)
    return request


def generate_music_ng(style: str, lyrics: str, duration_s: float,
                       output_dir: Path, seed: Optional[int],
                       config: MusicConfig,
                       mode: MusicMode = "full", instrumental: bool = False,
                       cfg_scale: Optional[float] = None,
                       on_log: Optional[Callable[[str], None]] = None,
                       on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: style + lyrics + duration in, one .flac out.
    style and/or lyrics may be empty but not both (same rule
    yue2_run.py enforces -- "give it something to work with"), unless
    instrumental is set, which fills in a bare-section-tags lyrics
    skeleton and an "instrumental, no vocals" style cue instead (see
    MUSIC_INSTRUMENTAL_LYRICS/STYLE)."""
    style = (style or "").strip()
    lyrics = (lyrics or "").replace("\r\n", "\n").strip("\n")
    if mode not in MUSIC_MODES:
        raise ValueError(f"mode must be one of {MUSIC_MODES} (got {mode!r})")
    if instrumental:
        lyrics = MUSIC_INSTRUMENTAL_LYRICS
        if "instrumental" not in style.lower():
            style = f"{style}, {MUSIC_INSTRUMENTAL_STYLE}" if style else MUSIC_INSTRUMENTAL_STYLE
    if not style and not lyrics:
        raise ValueError("Give it something to work with -- style, lyrics, or both.")
    _validate_music_duration_ng(duration_s)

    binary = _resolve_music_binary_ng(config)
    if not binary:
        raise FileNotFoundError(
            f"lyra binary not found at {config.binary_path or MUSIC_DEFAULT_BIN}")
    model_dir = _resolve_music_model_dir_ng(config)
    if not model_dir:
        raise FileNotFoundError(
            f"YuE2 {config.precision} generator not found under "
            f"{config.model_dir or MUSIC_MODEL_DIR}")
    vae_dir = _resolve_music_vae_dir_ng(config)
    if not vae_dir:
        raise FileNotFoundError(
            f"YuE2 VAE not found at {config.vae_dir or MUSIC_VAE_DIR}")

    seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    request = _build_request_ng(style, lyrics, duration_s, seed, mode, cfg_scale)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    request_path = output_dir / f"music_{ts}.request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    # `lyra generate --output DIR` refuses a non-empty DIR (never mixes
    # recordings) -- a fresh subdir per call, sibling to the request
    # file above, not output_dir itself.
    song_dir = output_dir / f"music_{ts}"

    cmd = [
        binary, "generate",
        "--request", str(request_path),
        "--model", model_dir,
        "--vae", vae_dir,
        "--precision", config.precision,
        "--offline",
        "--output", str(song_dir),
    ]

    if on_log:
        on_log(f"[music] launching {duration_s:.0f}s cap, mode={mode}, "
               f"precision={config.precision}, seed={seed}"
               + (" (instrumental)" if instrumental else ""))

    tail_lines = _run_lyra_subprocess_ng(
        cmd, timeout_s=config.timeout_s, timeout_env_var="RINGVIZ_MUSIC_TIMEOUT_S",
        on_log=on_log, on_proc_start=on_proc_start)

    audio_path = song_dir / "audio.flac"
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"Music gen finished but no/empty audio at {audio_path}")

    # `lyra generate`'s own final stdout line is a JSON summary (_save in
    # yue2-mlx/src/lyra/cli.py) -- best-effort parse for the real
    # audio_seconds/truncated info; the file on disk is the ground truth
    # either way, so a parse miss here never fails the render.
    summary = {}
    for line in reversed(tail_lines):
        try:
            summary = json.loads(line)
            break
        except ValueError:
            continue

    return {
        "audio_path": str(audio_path),
        "request_path": str(request_path),
        "seed": seed,
        "engine": f"yue2-{config.precision}",
        "mode": mode,
        "instrumental": instrumental,
        "duration_cap_s": duration_s,
        "audio_seconds": summary.get("audio_seconds"),
        "truncated": summary.get("truncated"),
    }


def generate_cover_ng(audio_path: str, output_dir: Path, seed: Optional[int],
                       config: MusicConfig,
                       task: str = MUSIC_DEFAULT_COVER_TASK,
                       style: str = "", lyrics: str = "",
                       duration_s: Optional[float] = None,
                       cfg_scale: Optional[float] = None,
                       transcription_model: str = MUSIC_TRANSCRIPTION_MODEL,
                       base_model: str = MUSIC_BASE_MODEL,
                       on_log: Optional[Callable[[str], None]] = None,
                       on_proc_start: Optional[Callable[[subprocess.Popen], None]] = None) -> dict:
    """One subprocess call: an existing track in, a transcribed ABC
    score AND a fresh song rendered over it out. `task` picks what
    SheetSage2 transcribes (melody+chords / melody only / melody+vocal
    line) and, via _cover_mode_for_task_ng, which render mode the
    request must use -- lyra cover enforces this pairing itself.
    style/lyrics are YOUR words/style rendered over the SOURCE track's
    melodic backbone -- both may be empty (an instrumental re-render of
    the transcribed score), unlike generate_music_ng there's no
    "give it something to work with" requirement here since the
    transcription itself is always something to work with.

    duration_s, when given, caps the render the same way
    generate_music_ng does (semantic_sampling.max_tokens); omitted,
    the render follows the transcribed score's own natural length up
    to lyra's own 360s ceiling.

    NOT passed --offline: transcription_model/base_model aren't part
    of the generate-only pack already on disk (see MUSIC_HF_CACHE_DIR
    above) -- the first call needs network access to fetch them.
    Subsequent calls reuse the same on-disk cache and don't re-fetch,
    even without --offline."""
    if task not in MUSIC_COVER_TASKS:
        raise ValueError(f"task must be one of {MUSIC_COVER_TASKS} (got {task!r})")
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"source track not found at {audio_path}")
    if duration_s is not None:
        _validate_music_duration_ng(duration_s)

    binary = _resolve_music_binary_ng(config)
    if not binary:
        raise FileNotFoundError(
            f"lyra binary not found at {config.binary_path or MUSIC_DEFAULT_BIN}")
    model_dir = _resolve_music_model_dir_ng(config)
    if not model_dir:
        raise FileNotFoundError(
            f"YuE2 {config.precision} generator not found under "
            f"{config.model_dir or MUSIC_MODEL_DIR}")
    vae_dir = _resolve_music_vae_dir_ng(config)
    if not vae_dir:
        raise FileNotFoundError(
            f"YuE2 VAE not found at {config.vae_dir or MUSIC_VAE_DIR}")

    style = (style or "").strip()
    lyrics = (lyrics or "").replace("\r\n", "\n").strip("\n")
    seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    mode = _cover_mode_for_task_ng(task)
    request: dict = {"style": style, "lyrics": lyrics, "cot": mode, "seed": seed}
    if duration_s is not None:
        max_tokens = max(1, round(duration_s * MUSIC_TOKENS_PER_SECOND))
        request["semantic_sampling"] = {"max_tokens": max_tokens, "min_tokens": min(200, max_tokens)}
    if cfg_scale is not None:
        request["cfg_scale"] = float(cfg_scale)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    request_path = output_dir / f"cover_{ts}.request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    # cover's own --output must be absent/empty too, same rule as
    # generate's --output -- see generate_music_ng's own comment.
    cover_dir = output_dir / f"cover_{ts}"
    MUSIC_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        binary, "cover",
        "--audio", str(source),
        "--request", str(request_path),
        "--task", task,
        "--transcription-model", transcription_model,
        "--base-model", base_model,
        "--cache-dir", str(MUSIC_HF_CACHE_DIR),
        "--model", model_dir,
        "--vae", vae_dir,
        "--precision", config.precision,
        "--output", str(cover_dir),
    ]

    if on_log:
        on_log(f"[music] launching cover, task={task}"
               + (f", {duration_s:.0f}s cap" if duration_s is not None else "")
               + f", seed={seed}")

    tail_lines = _run_lyra_subprocess_ng(
        cmd, timeout_s=config.cover_timeout_s, timeout_env_var="RINGVIZ_MUSIC_COVER_TIMEOUT_S",
        on_log=on_log, on_proc_start=on_proc_start)

    audio_out = cover_dir / "song" / "audio.flac"
    if not audio_out.exists() or audio_out.stat().st_size == 0:
        raise RuntimeError(f"Cover finished but no/empty audio at {audio_out}")
    score_path = cover_dir / "transcription" / "score.abc"

    summary = {}
    for line in reversed(tail_lines):
        try:
            summary = json.loads(line)
            break
        except ValueError:
            continue

    return {
        "audio_path": str(audio_out),
        # The actual sheet music -- an ABC score transcribed from the
        # source track, independent of whether the render above used
        # it for melody+chords (task="full") or melody only.
        "score_path": str(score_path) if score_path.is_file() else None,
        "request_path": str(request_path),
        "seed": seed,
        "engine": f"yue2-{config.precision}",
        "task": task,
        "mode": mode,
        "duration_cap_s": duration_s,
        "audio_seconds": summary.get("audio_seconds"),
        "truncated": summary.get("truncated"),
    }
