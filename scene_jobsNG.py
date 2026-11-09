"""NG "Scene" jobs: a script of lines said by different characters, rendered as one joined clip with sound.

    KNIGHT [on the battlements]: The wall has stood a long time, and so have I.
    WOMAN [in the great hall]: Then come in out of the wind.

One scene job chains the pieces we already have:

  1. voicing   -- ONE Speech pass per character over all of that character's lines (one voice, one delivery; cloning
                  each line separately measured as three different voices), cut back into one piece per line at its
                  pauses (scene_voiceNG) and brought to one speech level.
  2. rendering -- one H3 audio-drive clip per line (h3_jobsNG), each with that speaker's face (LoRA and/or the picture
                  for that line's setting), the line's setting in its prompt, and that line's audio. Fixed 5 s windows.
  (A line starting with `&` shares the previous line's shot: two speakers in ONE frame, as a split-screen first frame
   made from their two pictures and one audio track carrying both voices in turn; see scene_scriptNG.)
  (Optional, between 1 and 2: one 5 s ambience bed per setting, H3's own sound of the empty place, mixed under the
   dialogue of every clip in that setting; a bed that fails leaves that setting dry, never stops the scene.)
  3. joining   -- the clips trimmed to their speech and cut together in script order, with an exposure hold on each
                  clip (H3 clips brighten as they play, which stepped 16-20 levels at every cut) and, optionally, the
                  full soft-finish look. A failed finish never loses a render: that clip is joined unfinished.

Shot/reverse-shot: one speaker in frame per clip. Two speakers in ONE frame (phone call, split-screen) is a later shot
type. Scenes run one at a time (FIFO); each stage is an ordinary job in its own queue so cancel, the Job Log and memory
handling are the existing ones. A render that is cancelled or fails keeps the clips that finished.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import finish_engineNG as finish
import h3_engineNG as h3
import h3_jobsNG as h3_jobs
import h3_lora_libraryNG as h3_lora_lib
import h3_refmod_libraryNG as h3_refmod_lib
import numpy as np

import scene_ambienceNG as ambience
import scene_composeNG as compose
import scene_scriptNG as script_mod
import scene_voiceNG as voice_mod
import speech_engineNG as speech
import speech_jobsNG as speech_jobs
import talk_jobsNG as talk
from configNG import SCENEGEN_DIR

CLIP_S = talk.TALK_CLIP_S
TAIL_S = 0.35               # kept after the last word of a line before the cut
SHARED_GAP_S = 0.25         # breath between the two voices in a shared frame
FADE_S = 0.02               # audio fade at each cut, so a join never clicks
MAX_JOBS_KEPT = 10
QUALITIES = talk.TALK_QUALITIES
SceneCancelled = talk.TalkJobCancelled


class SceneRequestError(ValueError):
    pass


@dataclass
class SceneCharacter:
    name: str
    subject: str = ""
    lora_name: str = ""
    lora_path: str = ""
    lora_strength: float = 1.0
    refmod_name: str = ""                            # a RefMod is the whole character: no LoRA, no pictures
    refmod_entry: dict = field(default_factory=dict, repr=False)
    looks: dict = field(default_factory=dict)        # casefolded look name -> (bytes, ext, display name)
    default_look: str = ""                           # casefolded name of the look used when a line names none
    voice: dict = field(default_factory=dict)        # kind, voice, style_id, style_text, character_voice, ref_text,
                                                      # consent, clip_bytes, clip_ext, language, speed

    def look_for(self, setting: str):
        """The (bytes, ext) first-frame picture for a line's setting: the look with that name, else the default
        look, else the first one, else None (the face then comes from the LoRA alone)."""
        for key in (setting.casefold(), self.default_look):
            if key and key in self.looks:
                return self.looks[key][:2]
        return next(iter(self.looks.values()))[:2] if self.looks else None


@dataclass
class SceneClip:
    index: int
    speaker: str
    setting: str
    text: str
    line_indexes: list = field(default_factory=list)   # the script lines in this shot (two for a shared frame)
    shared: bool = False
    audio_seconds: Optional[float] = None
    audio_path: Optional[str] = None
    h3_job_id: Optional[str] = None
    mp4_path: Optional[str] = None
    final_path: Optional[str] = None
    status: str = "waiting"            # waiting | rendering | done | failed
    error: Optional[str] = None
    finish_error: Optional[str] = None


@dataclass
class SceneJobNG:
    job_id: str
    script: str
    lines: list
    characters: dict                   # casefolded name -> SceneCharacter
    quality: str
    width: int
    height: int
    fast: bool
    finish: bool
    finish_params: dict
    match_exposure: bool
    ambience: bool
    seed: Optional[int]
    timeout_s: Optional[float]
    job_dir: Path
    clips: list = field(default_factory=list)
    queued_at: float = field(default_factory=time.time)
    dispatched_at: Optional[float] = None
    finished_at: Optional[float] = None
    stage: str = "queued"              # queued | voicing | rendering | joining | done | failed | cancelled
    error: Optional[str] = None
    error_type: Optional[str] = None
    cancelled: bool = False
    speech_job_ids: dict = field(default_factory=dict)   # speaker key -> speech job id
    beds: dict = field(default_factory=dict)             # casefolded setting -> path of its ambience wav
    bed_errors: dict = field(default_factory=dict)       # casefolded setting -> why that setting has no bed
    output_path: Optional[str] = None


_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_QUEUE: "queue.Queue[str]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


# ---- request -----------------------------------------------------------------------------------------------------

def _build_character(raw: dict) -> SceneCharacter:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise SceneRequestError("every character needs a name")
    ch = SceneCharacter(name=name, subject=str(raw.get("subject") or "").strip()[: talk.TALK_MAX_SUBJECT])
    lora = str(raw.get("lora") or "").strip()
    refmod = str(raw.get("refmod") or "").strip()
    if refmod:
        if lora or raw.get("looks"):
            raise SceneRequestError(f"{name}: a RefMod is the whole character; leave the LoRA and the pictures empty")
        entry = h3_refmod_lib.resolve_h3_refmod_ng(refmod)
        if entry is None:
            raise SceneRequestError(f"{name}: unknown or unusable RefMod {refmod!r}")
        ch.refmod_name = refmod
        ch.refmod_entry = {"name": refmod, "path": entry["path"], "strength": 1.0,
                           "images": entry["images"], "videos": entry["videos"]}
    if lora:
        path = h3_lora_lib.resolve_h3_lora_path_ng(lora)
        if not path:
            raise SceneRequestError(f"{name}: unknown or unusable LoRA {lora!r}")
        strength = raw.get("lora_strength", 1.0)
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 2.0:
            raise SceneRequestError(f"{name}: LoRA strength must be between 0 and 2")
        ch.lora_name, ch.lora_path, ch.lora_strength = lora, path, float(strength)
    for look in raw.get("looks") or []:
        lname = str(look.get("name") or "").strip()
        if not lname or not look.get("bytes"):
            raise SceneRequestError(f"{name}: each look needs a name and a picture")
        ch.looks[lname.casefold()] = (look["bytes"], str(look.get("ext") or ".jpg").lower(), lname)
    dl = str(raw.get("default_look") or "").strip().casefold()
    ch.default_look = dl if dl in ch.looks else ""
    if not (ch.lora_name or ch.looks or ch.refmod_name):
        raise SceneRequestError(f"{name}: give them a RefMod, a LoRA or at least one picture, otherwise there is no face")
    ch.voice = dict(raw.get("voice") or {})
    return ch


def _validate_voice(ch: SceneCharacter, speakable: str) -> None:
    v = ch.voice
    kind = str(v.get("kind") or "preset")
    v["kind"] = kind
    has_ref = False
    if kind == "character":
        if speech.resolve_voice_sample_ng(str(v.get("character_voice") or "")) is None:
            raise SceneRequestError(f"{ch.name}: that character has no saved voice sample")
        has_ref = True
    elif kind == "clip":
        if not v.get("clip_bytes"):
            raise SceneRequestError(f"{ch.name}: add a voice clip")
        has_ref = True
    try:
        speech.validate_request_ng(kind, speakable, str(v.get("voice") or ""), str(v.get("language") or "english"),
                                   float(v.get("speed") or 1.0), has_ref_audio=has_ref,
                                   voice_consent=bool(v.get("consent")))
    except speech.SpeechRequestError as e:
        raise SceneRequestError(f"{ch.name}: {e}") from e


def start_scene_job_ng(script: str, characters: list, quality: str = "draft", fast: bool = True,
                       finish_pass: bool = False, finish_params: Optional[dict] = None,
                       match_exposure: bool = True, seed: Optional[int] = None,
                       ambience_pass: bool = False) -> SceneJobNG:
    """Validate everything that can be checked now (a bad request fails the POST, not an hour in) and enqueue."""
    if quality not in QUALITIES:
        raise SceneRequestError(f"quality must be one of {', '.join(QUALITIES)}")
    chars = {}
    for raw in characters or []:
        ch = _build_character(raw)
        if ch.name.casefold() in chars:
            raise SceneRequestError(f"{ch.name} is listed twice")
        chars[ch.name.casefold()] = ch
    if not chars:
        raise SceneRequestError("add at least one character")
    try:
        lines = script_mod.resolve_settings_ng(script_mod.parse_script_ng(script, known=[c.name for c in chars.values()]))
    except script_mod.SceneScriptError as e:
        raise SceneRequestError(str(e)) from e
    for spk in script_mod.speakers_ng(lines):
        speakable, _ = script_mod.speakable_text_ng(script_mod.lines_of_ng(lines, spk))
        _validate_voice(chars[spk.casefold()], speakable)
    shots = script_mod.shots_ng(lines)
    for shot in shots:
        if len(shot) == 2:
            for ln in shot:
                if chars[ln.key].refmod_name:
                    raise SceneRequestError(
                        f"{ln.speaker} is a RefMod character, and a RefMod cannot share a split-screen frame: "
                        f"give each their own shot")
                if not chars[ln.key].looks:
                    raise SceneRequestError(
                        f"{ln.speaker} shares a frame with {shot[0].speaker if ln is shot[1] else shot[1].speaker}: "
                        f"a shared frame is made from both characters' pictures, so {ln.speaker} needs one")
    if fast and h3._resolve_h3_tristep_ng() is None:
        raise SceneRequestError("Fast needs the 3-step adapter (TaoMate) in the H3 turbo-lora folder, and it is not there")
    try:
        finish_clean = finish.normalize_params_ng(finish_params) if finish_pass else {}
    except finish.FinishRequestError as e:
        raise SceneRequestError(str(e)) from e
    if finish_pass or match_exposure:
        health = finish.finish_health_ng()
        if not health["ok"]:
            raise SceneRequestError("the finish step is not available: " + "; ".join(health["problems"]))
    width, height = h3.resolve_h3_quality_ng(quality)
    _ensure_worker_started()
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(SCENEGEN_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "script.txt").write_text(script, encoding="utf-8")
    job = SceneJobNG(
        job_id=job_id, script=script, lines=lines, characters=chars, quality=quality, width=width, height=height,
        fast=bool(fast), finish=bool(finish_pass), finish_params=finish_clean, match_exposure=bool(match_exposure),
        ambience=bool(ambience_pass),
        seed=seed, timeout_s=h3.resolve_h3_quality_timeout_ng(quality, CLIP_S), job_dir=job_dir,
        clips=[_clip_for(i, shot) for i, shot in enumerate(shots)])
    with _JOBS_LOCK:
        _JOBS[job_id] = job
        _prune_locked()
    _QUEUE.put(job_id)
    return job


def _clip_for(index: int, shot: list) -> SceneClip:
    return SceneClip(index, " + ".join(ln.speaker for ln in shot), " | ".join(dict.fromkeys(ln.setting for ln in shot if ln.setting)),
                     " / ".join(ln.text for ln in shot), line_indexes=[ln.index for ln in shot], shared=len(shot) == 2)


# ---- the run -----------------------------------------------------------------------------------------------------

def _prune_locked() -> None:
    if len(_JOBS) <= MAX_JOBS_KEPT:
        return
    done = sorted((j for j in _JOBS.values() if j.finished_at is not None), key=lambda j: j.finished_at)
    while len(_JOBS) > MAX_JOBS_KEPT and done:
        _JOBS.pop(done.pop(0).job_id, None)          # the record only; the scene's folder stays on disk


def _wait_for(job: SceneJobNG, done, cancel_stage) -> None:
    talk._wait_for(job, done, cancel_stage)


def _voice_stage(job: SceneJobNG) -> dict:
    """{line index: (samples, sample rate)} for every line, one Speech pass per speaker."""
    pieces: dict = {}
    for spk in script_mod.speakers_ng(job.lines):
        ch = job.characters[spk.casefold()]
        mine = script_mod.lines_of_ng(job.lines, spk)
        speakable, counts = script_mod.speakable_text_ng(mine)
        v = ch.voice
        sj = speech_jobs.start_speech_job_ng(
            speakable, v["kind"], voice=str(v.get("voice") or ""), character=str(v.get("character_voice") or ""),
            style_id=str(v.get("style") or "natural"), style_text=str(v.get("style_text") or ""),
            language=str(v.get("language") or "english"), speed=float(v.get("speed") or 1.0), seed=job.seed,
            ref_text=str(v.get("ref_text") or ""), voice_consent=bool(v.get("consent")),
            clip_bytes=v.get("clip_bytes"), clip_ext=str(v.get("clip_ext") or ""))
        job.speech_job_ids[spk.casefold()] = sj.job_id
        _wait_for(job, lambda sj=sj: sj.finished_at is not None, lambda sj=sj: speech_jobs.cancel_speech_job_ng(sj.job_id))
        if sj.error:
            raise RuntimeError(f"{ch.name}'s voice: {sj.error}")
        x, sr = voice_mod.read_wav_ng(sj.audio_path)
        try:
            cut = voice_mod.split_take_ng(x, sr, len(mine), max_s=CLIP_S, group_sizes=counts)
        except voice_mod.SceneAudioError as e:
            raise RuntimeError(f"{ch.name}: {e}") from e
        for ln, seg in zip(mine, cut):
            pieces[ln.index] = (seg, sr)
    ordered = sorted(pieces)
    leveled = voice_mod.match_loudness_ng([pieces[i][0] for i in ordered])
    for i, seg in zip(ordered, leveled):
        pieces[i] = (seg, pieces[i][1])
    for clip in job.clips:                      # a shared frame must fit one window: find out before any render
        if clip.shared:
            seg, sr = _shot_audio(job, clip, pieces)
            if len(seg) / sr > CLIP_S:
                raise RuntimeError(f"the shared frame {clip.speaker} runs {len(seg) / sr:.1f} s but a clip is "
                                   f"{CLIP_S:.0f} s -- shorten the two lines or give each their own shot")
    return pieces


def _scene_prompt(ch: SceneCharacter, setting: str) -> str:
    return talk.build_prompt_ng(ch.subject, f"Setting: {setting}." if setting else "")


def _shared_prompt(chs: list, lines: list) -> str:
    """The split-screen prompt, in the dataset's own pattern: left half then right half, each with a tagged speaker and
    a spoken line, so the model can tell whose mouth works for which voice."""
    def half(ch, ln, tag):
        place = f", {ln.setting}" if ln.setting else ""
        return f"{ch.subject or 'a person'} ({tag}){place}"
    return (f"Live-action, cinematic, a vertical split-screen composite held as one continuous shot: on the left half "
            f"{half(chs[0], lines[0], 'S1')}, says: <d>[English] {lines[0].text}</d> On the right half "
            f"{half(chs[1], lines[1], 'S2')}, answers: <d>[English] {lines[1].text}</d> The camera holds steady on both "
            f"halves with the seam between them fixed, each face large enough to see.")


def _shot_audio(job: SceneJobNG, clip: SceneClip, pieces: dict) -> tuple:
    """(samples, rate) for a shot: one piece, or the two voices in order with a breath between."""
    segs = [pieces[i][0] for i in clip.line_indexes]
    sr = pieces[clip.line_indexes[0]][1]
    if len(segs) == 1:
        return segs[0], sr
    gap = np.zeros(int(SHARED_GAP_S * sr), dtype=segs[0].dtype)
    return np.concatenate([segs[0], gap, segs[1]]), sr


def _render_stage(job: SceneJobNG, pieces: dict) -> None:
    by_index = {ln.index: ln for ln in job.lines}
    for clip in job.clips:
        seg, sr = _shot_audio(job, clip, pieces)
        wav_path = job.job_dir / f"shot{clip.index + 1:02d}_{clip.speaker.casefold().replace(' ', '_').replace('+', '_')}.wav"
        voice_mod.write_wav_ng(wav_path, seg, sr)
        clip.audio_path, clip.audio_seconds = str(wav_path), round(len(seg) / sr, 2)
        lines = [by_index[i] for i in clip.line_indexes]
        chs = [job.characters[ln.key] for ln in lines]
        if clip.shared:
            picks = [ch.look_for(ln.setting) for ch, ln in zip(chs, lines)]
            first, ext = compose.split_screen_ng(picks[0][0], picks[1][0], job.width, job.height), ".png"
            prompt = _shared_prompt(chs, lines)
            loras = [{"name": ch.lora_name, "path": ch.lora_path, "strength": ch.lora_strength}
                     for ch in {c.name: c for c in chs}.values() if ch.lora_name]
        else:
            ch, look = chs[0], chs[0].look_for(lines[0].setting)
            first, ext = (look[0], look[1]) if look else (None, ".jpg")
            prompt = _scene_prompt(ch, lines[0].setting)
            loras = [{"name": ch.lora_name, "path": ch.lora_path, "strength": ch.lora_strength}] if ch.lora_name else []
        clip.status = "rendering"
        hj = h3_jobs.start_h3_job_ng(
            first, ext, prompt, CLIP_S,
            seed=None if job.seed is None else job.seed + clip.index, model="h3ref" if not clip.shared and chs[0].refmod_name else "h3q8",
            refmods=[chs[0].refmod_entry] if not clip.shared and chs[0].refmod_name else [],
            width=job.width, height=job.height,
            steps=talk.TALK_STEPS, tristep=job.fast, loras=loras,
            timeout_s=job.timeout_s, audio_bytes=wav_path.read_bytes(), audio_ext=".wav")
        clip.h3_job_id = hj.job_id
        try:
            _wait_for(job, lambda hj=hj: hj.finished_at is not None, lambda hj=hj: h3_jobs.cancel_h3_job_ng(hj.job_id))
        except SceneCancelled:
            clip.status = "failed"
            clip.error = "cancelled"
            raise
        if hj.error:
            clip.status, clip.error = "failed", hj.error
            raise RuntimeError(f"shot {clip.index + 1} ({clip.speaker}): {hj.error}")
        clip.mp4_path, clip.status = hj.mp4_path, "done"


def _bed_key(setting: str) -> str:
    return setting.split(" | ")[0].strip().casefold()


def _ambience_stage(job: SceneJobNG) -> None:
    """One bed per distinct non-empty setting (a shared frame uses its first setting). A failure only leaves that
    setting dry."""
    wanted: dict = {}
    for clip in job.clips:
        key = _bed_key(clip.setting)
        if key and key not in wanted:
            wanted[key] = clip.setting.split(" | ")[0].strip()
    for n, (key, setting) in enumerate(wanted.items(), 1):
        job.stage = "ambience"
        try:
            hj = h3_jobs.start_h3_job_ng(
                None, ".jpg", ambience.ambience_prompt_ng(setting), CLIP_S,
                seed=None if job.seed is None else job.seed + 100 + n, model="h3q8", width=job.width, height=job.height,
                steps=talk.TALK_STEPS, tristep=job.fast, loras=[], timeout_s=job.timeout_s)
            _wait_for(job, lambda hj=hj: hj.finished_at is not None, lambda hj=hj: h3_jobs.cancel_h3_job_ng(hj.job_id))
            if hj.error:
                raise RuntimeError(hj.error)
            job.beds[key] = ambience.extract_audio_ng(hj.mp4_path, str(job.job_dir / f"ambience_{n}.wav"))
        except SceneCancelled:
            raise
        except Exception as e:  # noqa: BLE001 -- the scene goes on without this bed
            job.bed_errors[key] = str(e)


def _finish_clip(job: SceneJobNG, clip: SceneClip) -> str:
    """The clip to join: held to its own starting exposure (and given the full look when asked). If that step fails
    the render is kept and joined as it came."""
    if not (job.finish or job.match_exposure):
        return clip.mp4_path
    params = dict(job.finish_params) if job.finish else {
        "clarity_strength": 0.0, "tone_strength": 0.0, "saturation": 1.0, "hold": 1.0}     # exposure hold only
    out = job.job_dir / f"shot{clip.index + 1:02d}_held.mp4"
    try:
        return finish.finish_video_ng(clip.mp4_path, str(out), params, should_cancel=lambda: job.cancelled)["path"]
    except finish.FinishCancelled:
        raise SceneCancelled("cancelled")
    except Exception as e:  # noqa: BLE001 -- the render is safe; say why the pass did not happen
        clip.finish_error = str(e)
        return clip.mp4_path


def join_clips_ng(parts: list, out_path: str) -> None:
    """parts = [(mp4 path, seconds to keep)]. Hard cuts in order, audio faded 20 ms at every join."""
    if not parts:
        raise ValueError("nothing to join")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for path, _ in parts:
        cmd += ["-i", str(path)]
    chains, labels = [], ""
    for i, (_, keep) in enumerate(parts):
        fade_out = max(0.0, keep - FADE_S)
        chains.append(f"[{i}:v]trim=0:{keep:.3f},setpts=PTS-STARTPTS[v{i}]")
        chains.append(f"[{i}:a]atrim=0:{keep:.3f},asetpts=PTS-STARTPTS,"
                      f"afade=t=in:d={FADE_S},afade=t=out:st={fade_out:.3f}:d={FADE_S}[a{i}]")
        labels += f"[v{i}][a{i}]"
    chains.append(f"{labels}concat=n={len(parts)}:v=1:a=1[v][a]")
    cmd += ["-filter_complex", ";".join(chains), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("joining the clips failed: " + (r.stderr.strip().splitlines() or ["ffmpeg error"])[-1])


def _join_stage(job: SceneJobNG) -> None:
    parts = []
    for clip in job.clips:
        clip.final_path = _finish_clip(job, clip)
        keep = min(CLIP_S, (clip.audio_seconds or CLIP_S) + TAIL_S)
        bed = job.beds.get(_bed_key(clip.setting))
        if bed:
            try:
                clip.final_path = ambience.mix_ambience_ng(
                    clip.final_path, bed, str(job.job_dir / f"shot{clip.index + 1:02d}_ambience.mp4"), keep)
            except Exception as e:  # noqa: BLE001 -- the dry clip is still good
                job.bed_errors[_bed_key(clip.setting)] = f"mixing: {e}"
        parts.append((clip.final_path, keep))
    out = job.job_dir / "scene.mp4"
    join_clips_ng(parts, str(out))
    job.output_path = str(out)
    (job.job_dir / "scene.json").write_text(json.dumps({
        "script": job.script, "quality": job.quality, "fast": job.fast, "finish": job.finish,
        "match_exposure": job.match_exposure, "ambience": job.ambience, "bed_errors": job.bed_errors, "seed": job.seed,
        "clips": [{"index": c.index, "speaker": c.speaker, "setting": c.setting, "text": c.text, "shared": c.shared,
                   "lines": c.line_indexes, "audio_seconds": c.audio_seconds, "h3_job_id": c.h3_job_id,
                   "finish_error": c.finish_error}
                  for c in job.clips]}, indent=2), encoding="utf-8")


def _run_scene(job: SceneJobNG) -> None:
    job.stage = "voicing"
    pieces = _voice_stage(job)
    if job.ambience:
        _ambience_stage(job)
    job.stage = "rendering"
    _render_stage(job, pieces)
    job.stage = "joining"
    _join_stage(job)


def _worker_loop() -> None:
    while True:
        job = _JOBS.get(_QUEUE.get())
        if job is None:
            continue
        if job.cancelled:
            job.stage, job.error, job.error_type = "cancelled", "cancelled before it started", "SceneCancelled"
            job.finished_at = time.time()
            continue
        job.dispatched_at = time.time()
        try:
            _run_scene(job)
            job.stage = "done"
        except SceneCancelled:
            job.stage, job.error, job.error_type = "cancelled", "cancelled", "SceneCancelled"
        except Exception as e:  # noqa: BLE001 -- job.error is the report
            job.stage, job.error, job.error_type = "failed", str(e), type(e).__name__
        finally:
            job.finished_at = time.time()


def _ensure_worker_started() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        threading.Thread(target=_worker_loop, name="scene-job-worker-ng", daemon=True).start()
        _WORKER_STARTED = True


# ---- status --------------------------------------------------------------------------------------------------------

def get_job_ng(job_id: str) -> Optional[SceneJobNG]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def cancel_scene_job_ng(job_id: str) -> bool:
    job = get_job_ng(job_id)
    if job is None or job.finished_at is not None:
        return False
    job.cancelled = True
    return True


def job_status_ng(job_id: str) -> dict:
    job = get_job_ng(job_id)
    if job is None:
        raise LookupError(f"no such job {job_id!r}")
    status = {"queued": "queued", "done": "completed", "failed": "failed", "cancelled": "failed"}.get(job.stage, "rendering")
    clips = []
    for c in job.clips:
        clips.append({
            "index": c.index, "speaker": c.speaker, "setting": c.setting, "text": c.text, "shared": c.shared,
            "status": c.status,
            "audio_seconds": c.audio_seconds, "error": c.error, "finish_error": c.finish_error,
            "h3_job_id": c.h3_job_id,
            "progress": talk._render_progress(c.h3_job_id) if c.status == "rendering" else {},
            "video_url": f"/api/ng/h3/jobs/{c.h3_job_id}/video" if c.status == "done" and c.h3_job_id else None})
    return {
        "job_id": job.job_id, "status": status, "stage": job.stage, "lines": len(job.lines), "shots": len(job.clips),
        "quality": job.quality,
        "fast": job.fast, "finish": job.finish, "match_exposure": job.match_exposure, "ambience": job.ambience,
        "bed_errors": job.bed_errors, "seed": job.seed,
        "clips": clips, "error": job.error, "error_type": job.error_type, "cancelled": bool(job.cancelled),
        "queued_at": job.queued_at, "dispatched_at": job.dispatched_at, "finished_at": job.finished_at,
        "video_url": f"/api/ng/scene/jobs/{job.job_id}/video" if job.stage == "done" else None,
        "download_url": f"/api/ng/scene/jobs/{job.job_id}/download" if job.stage == "done" else None,
    }


def list_jobs_ng() -> list:
    with _JOBS_LOCK:
        ids = sorted(_JOBS, key=lambda jid: _JOBS[jid].queued_at or 0)
    out = []
    for jid in ids:
        try:
            out.append(job_status_ng(jid))
        except LookupError:
            continue
    return out
