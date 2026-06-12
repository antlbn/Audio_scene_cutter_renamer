"""Trim standardized in-memory AAC audio using the strongest clapper hit."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import clapper
from . import preprocessor


@dataclass(slots=True)
class CutterResult:
    """Trimmed standardized audio derived from a clapper detection result."""

    clapper_result: clapper.ClapperResult
    best_hit: clapper.ClapperHit
    standardized_audio: preprocessor.StandardizedAudio
    duration_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float


def _select_best_hit(result: clapper.ClapperResult) -> clapper.ClapperHit:
    if not result.best_scores:
        print("[cutter] no timestamps were found in ClapperResult.best_scores", file=sys.stderr)
        raise ValueError("No timestamps were found in ClapperResult.best_scores")

    return min(result.best_scores, key=lambda hit: float(hit.timestamp))


def _build_ffmpeg_trim_command(audio: preprocessor.StandardizedAudio, start_seconds: float, duration_seconds: float) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        "pipe:0",
        "-ac",
        str(audio.channels),
        "-ar",
        str(audio.sample_rate),
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-f",
        "adts",
        "pipe:1",
    ]


def _encode_trimmed_audio(
    audio: preprocessor.StandardizedAudio,
    start_seconds: float,
    duration_seconds: float,
) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim audio in memory")

    ffmpeg_cmd = _build_ffmpeg_trim_command(audio, start_seconds, duration_seconds)
    proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, input=audio.payload)
    return proc.stdout


def cut_audio_by_clapper(
    audio_source: preprocessor.StandardizedAudio | str | Path,
    clapper_result: clapper.ClapperResult,
    config: clapper.ClapperConfig | None = None,
) -> CutterResult:
    """Trim the source audio from the start up to the strongest clapper hit + post_hit_seconds."""

    config = config or clapper.load_config()
    if isinstance(audio_source, (str, Path)):
        standardized_audio = preprocessor.standardize_audio(audio_source)
    else:
        standardized_audio = audio_source

    best_hit = _select_best_hit(clapper_result)

    source_duration_seconds = standardized_audio.duration_seconds
    pre_hit_offset = config.clip.pre_hit_seconds
    post_hit_offset = config.clip.post_hit_seconds
    clip_start_seconds = max(0.0, float(best_hit.timestamp) - pre_hit_offset)
    clip_end_seconds = min(
        float(best_hit.timestamp) + post_hit_offset,
        source_duration_seconds,
    )
    if clip_end_seconds <= 0:
        raise ValueError(
            f"Best clapper timestamp ({best_hit.timestamp:.2f}s) + "
            f"post_hit_seconds ({post_hit_offset:.2f}s) results in a non-positive clip end"
        )

    clip_duration_seconds = clip_end_seconds - clip_start_seconds
    if clip_duration_seconds <= 0:
        raise ValueError("Computed clip duration is empty")

    payload = _encode_trimmed_audio(standardized_audio, clip_start_seconds, clip_duration_seconds)
    clipped_audio = preprocessor.StandardizedAudio(
        file_name=standardized_audio.file_name,
        payload=payload,
        codec=standardized_audio.codec,
        sample_rate=standardized_audio.sample_rate,
        channels=standardized_audio.channels,
        duration_seconds=clip_duration_seconds,
    )

    return CutterResult(
        clapper_result=clapper_result,
        best_hit=best_hit,
        standardized_audio=clipped_audio,
        duration_seconds=clip_duration_seconds,
        clip_start_seconds=round(clip_start_seconds, 6),
        clip_end_seconds=round(clip_end_seconds, 6),
    )


# Backwards-compatible aliases for earlier drafts of the module name.
ClapperClipResult = CutterResult
clip_audio_by_clapper = cut_audio_by_clapper
