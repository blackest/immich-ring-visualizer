"""Writing MiniMax-H3 output to files.

Deliberately dependency-free: stereo WAV goes through the standard library's ``wave`` module and
video is piped as raw RGB frames into the ``ffmpeg`` binary, so the port needs no imageio,
soundfile or torchvision. If ``ffmpeg`` is absent the frames can still be written as a PNG
sequence.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write ``(channels, samples)`` float audio in ``[-1, 1]`` as 16-bit PCM."""
    path = Path(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None]
    clipped = np.clip(audio, -1.0, 1.0)
    # Interleave channels, which is what WAV expects.
    pcm = (clipped.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(audio.shape[0])
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return path


def save_mp4(
    path: str | Path,
    video: np.ndarray,
    fps: float,
    audio: np.ndarray | None = None,
    sample_rate: int = 32000,
    crf: int = 18,
    audio_tempo: float = 1.0,
    audio_stretch_script: str | Path | None = None,
    audio_output_frames: int | None = None,
) -> Path:
    """Encode ``(frames, height, width, 3)`` uint8 video, muxing audio when given.

    ``audio_tempo`` below 1.0 stretches the generated audio to cover a clip written at a reduced
    frame rate. ``atempo`` preserves pitch, so speech stays intelligible where a plain resample
    would drop it an octave — but it is a time-stretch artefact, not a model output, and long
    stretches audibly smear transients.

    Raises if ``ffmpeg`` is not on PATH; use :func:`save_frames` in that case.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; use save_frames() instead.")

    path = Path(path)
    video = np.ascontiguousarray(video, dtype=np.uint8)
    frames, height, width, _ = video.shape

    audio_path = None
    if audio is not None:
        audio_path = path.with_suffix(".wav")
        if audio_stretch_script is not None and abs(audio_tempo - 1.0) > 1e-6:
            if audio_output_frames is None:
                raise ValueError("audio_output_frames is required for Apple time stretching")
            source_path = path.with_name(f"{path.stem}_source.wav")
            save_wav(source_path, audio, sample_rate)
            audio_path.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    "/usr/bin/swift",
                    str(audio_stretch_script),
                    str(source_path),
                    str(audio_path),
                    str(audio_tempo),
                    str(audio_output_frames),
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Apple audio time stretch failed: {result.stderr.decode()[:500]}"
                )
        else:
            save_wav(audio_path, audio, sample_rate)

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
    ]
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
        filters = []
        if audio_stretch_script is None and abs(audio_tempo - 1.0) > 1e-6:
            # atempo is only defined on [0.5, 100]; chain factors for anything slower.
            factors, remaining = [], float(audio_tempo)
            while remaining < 0.5:
                factors.append(0.5)
                remaining /= 0.5
            factors.append(remaining)
            filters += [f"atempo={f:.6f}" for f in factors]
        # The picture decides the length, stated explicitly. `-shortest` used to do this and
        # dropped the LAST video frame on every clip (73 -> 72, 124 -> 123): libx264 still holds
        # frames in its lookahead when the audio stream ends, and the muxer stops there. Pad the
        # audio with silence (a stretched or crossfaded waveform can end a few ms early) and cut
        # it at exactly frames / fps, so both streams end on the picture's last frame.
        filters += ["apad", f"atrim=duration={frames / float(fps):.6f}"]
        cmd += ["-filter:a", ",".join(filters), "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-frames:v", str(frames),
        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path),
    ]

    process = subprocess.run(cmd, input=video.tobytes(), capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {process.stderr.decode()[:500]}")
    return path


def probe_video_frames(path: str | Path) -> int | None:
    """Frames in the first video stream of an ENCODED file, or ``None`` when unknown.

    Counts packets rather than trusting container metadata, so it reports what a player will
    actually show. ``None`` (no ffprobe, unreadable file) means "not verified", never zero.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return int(result.stdout.strip().splitlines()[0].strip().rstrip(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def save_frames(directory: str | Path, video: np.ndarray, limit: int | None = None) -> Path:
    """Write frames as ``frame_00000.png`` — the fallback when ffmpeg is unavailable."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = len(video) if limit is None else min(limit, len(video))
    for index in range(count):
        Image.fromarray(video[index]).save(directory / f"frame_{index:05d}.png")
    return directory


# -- reading media back in (continuation, lip-sync, video-to-audio) ---------------------------


def _require(tool: str) -> str:
    found = shutil.which(tool)
    if found is None:
        raise RuntimeError(f"{tool} not found on PATH; it is needed to read input media.")
    return found


def probe_media(path: str | Path) -> dict:
    """``{"width", "height", "frames", "fps", "duration", "has_audio"}`` of a media file.

    ``frames`` counts decoded packets of the first video stream (what a player shows); a file
    with no video stream reports 0 frames and its audio duration.
    """
    ffprobe = _require("ffprobe")
    import json as _json

    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-count_packets",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path}: {result.stderr.strip()[:300]}")
    data = _json.loads(result.stdout or "{}")
    info = {"width": 0, "height": 0, "frames": 0, "fps": 0.0, "duration": 0.0, "has_audio": False}
    try:
        info["duration"] = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        pass
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video" and not info["width"]:
            info["width"] = int(stream.get("width") or 0)
            info["height"] = int(stream.get("height") or 0)
            try:
                info["frames"] = int(stream.get("nb_read_packets") or stream.get("nb_frames") or 0)
            except (TypeError, ValueError):
                info["frames"] = 0
            num, _, den = str(stream.get("avg_frame_rate") or "0/1").partition("/")
            try:
                info["fps"] = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def read_video_frames(
    path: str | Path,
    width: int,
    height: int,
    fps: float = 24.0,
    max_frames: int | None = None,
) -> np.ndarray:
    """Decode a clip to ``(F, height, width, 3)`` uint8 at ``fps``, cover-scaled onto the canvas.

    Cover = scale until both axes are filled, then centre-crop, so the picture keeps its
    proportions (the same rule the panel uses to put a reference image on an H3 canvas). The
    timeline is resampled to ``fps`` with ffmpeg's ``fps`` filter, which drops or repeats frames
    rather than blending them, so every frame is a real frame of the source.
    """
    ffmpeg = _require("ffmpeg")
    vf = (
        f"fps={fps:g},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vf", vf, "-an"]
    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    # Frames are read straight into one preallocated array (grown only if the probe
    # under-counted), so a long clip costs its frames once, not a pipe buffer plus a copy.
    info = probe_media(path)
    guess = int(round(float(info.get("duration") or 0.0) * fps)) + 2
    if max_frames is not None:
        guess = min(guess, int(max_frames)) if guess > 2 else int(max_frames)
    frames = np.empty((max(1, guess), height, width, 3), dtype=np.uint8)
    count = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                if count == len(frames):
                    frames = np.concatenate([frames, np.empty_like(frames[: max(24, count // 4)])])
                view = memoryview(frames[count]).cast("B")
                got = 0
                while got < len(view):
                    n = proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < len(view):
                    break
                count += 1
        finally:
            proc.stdout.close()
            returncode = proc.wait()
        if returncode != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg could not decode {path}: {err.read().decode()[:300]}")
    if count < 1:
        raise RuntimeError(f"{path} decoded to no video frames")
    return frames[:count]  # a view: the probe over-allocates by a frame or two at most


def read_audio(
    path: str | Path,
    sample_rate: int = 32000,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Decode any media's first audio stream to ``(2, samples)`` float32 in [-1, 1].

    Mono is duplicated to both channels; a file with no audio stream returns an empty
    ``(2, 0)`` array (the caller decides whether silence is acceptable).
    """
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-v", "error"]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.6f}"]
    cmd += ["-i", str(path)]
    if duration_seconds is not None:
        cmd += ["-t", f"{max(0.0, duration_seconds):.6f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(int(sample_rate)), "-f", "f32le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode()
        if "does not contain any stream" in stderr or "Output file #0 does not contain" in stderr \
                or "matches no streams" in stderr:
            return np.zeros((2, 0), dtype=np.float32)
        raise RuntimeError(f"ffmpeg could not decode audio from {path}: {stderr[:300]}")
    data = np.frombuffer(result.stdout, dtype="<f4")
    if data.size % 2:
        data = data[:-1]
    return np.ascontiguousarray(data.reshape(-1, 2).T.astype(np.float32))


def mux_audio_onto_video(
    source: str | Path, audio: np.ndarray, sample_rate: int, output: str | Path
) -> Path:
    """Copy ``source``'s video stream untouched and give it ``audio`` as its only soundtrack.

    The picture is never re-encoded (``-c:v copy``), so a video-to-audio render returns the
    exact pixels it was given. The audio is padded or cut to the video's own duration.
    """
    ffmpeg = _require("ffmpeg")
    output = Path(output)
    wav = output.with_suffix(".wav")
    save_wav(wav, audio, sample_rate)
    duration = probe_media(source)["duration"]
    # `apad` alone never ends, so it only rides with the trim that bounds it.
    audio_filter = (
        ["-filter:a", f"apad,atrim=duration={duration:.6f}"] if duration > 0 else ["-shortest"]
    )
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        *audio_filter, "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {result.stderr.decode()[:500]}")
    return output
