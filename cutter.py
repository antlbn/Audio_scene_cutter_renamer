"""Trim an audio file in memory using the strongest clapper hit."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import clapper


WHISPER_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class CutterResult:
    """In-memory trimmed audio derived from a clapper detection result."""

    clapper_result: clapper.ClapperResult
    best_hit: clapper.ClapperHit
    audio: torch.Tensor
    sample_rate: int
    duration_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float


def _select_best_hit(result: clapper.ClapperResult) -> clapper.ClapperHit:
    if not result.best_scores:
        print("[cutter] no timestamps were found in ClapperResult.best_scores", file=sys.stderr)
        raise ValueError("No timestamps were found in ClapperResult.best_scores")

    return max(result.best_scores, key=lambda hit: (float(hit.score), -float(hit.timestamp)))


def _load_source_duration_seconds(audio_path: str | Path) -> float:
    waveform, sample_rate = clapper._load_audio(audio_path)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for {audio_path}: {sample_rate}")
    return float(len(waveform)) / float(sample_rate)


def _build_ffmpeg_trim_command(audio_path: str | Path, start_seconds: float, duration_seconds: float) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(WHISPER_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]


def _decode_ffmpeg_pcm(stdout: bytes) -> torch.Tensor:
    if not stdout:
        raise ValueError("ffmpeg returned no audio data for the requested clip")

    samples = np.frombuffer(stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError("ffmpeg returned an empty audio clip")
    return torch.from_numpy(samples)


def cut_audio_by_clapper(
    audio_path: str | Path,
    clapper_result: clapper.ClapperResult,
    config: clapper.ClapperConfig | None = None,
) -> CutterResult:
    """Trim the source audio after the strongest clapper hit and keep it in memory."""

    config = config or clapper.load_config()
    best_hit = _select_best_hit(clapper_result)

    source_duration_seconds = _load_source_duration_seconds(audio_path)
    clip_start_seconds = max(0.0, float(best_hit.timestamp))
    if clip_start_seconds >= source_duration_seconds:
        raise ValueError("Best clapper timestamp is at or beyond the source audio duration")

    clip_end_seconds = min(
        source_duration_seconds,
        clip_start_seconds + float(config.clip.post_hit_seconds),
    )
    clip_duration_seconds = clip_end_seconds - clip_start_seconds
    if clip_duration_seconds <= 0:
        raise ValueError("Computed clip duration is empty")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to trim audio in memory")

    ffmpeg_cmd = _build_ffmpeg_trim_command(audio_path, clip_start_seconds, clip_duration_seconds)
    proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    audio = _decode_ffmpeg_pcm(proc.stdout)
    duration_seconds = float(audio.numel()) / float(WHISPER_SAMPLE_RATE)

    return CutterResult(
        clapper_result=clapper_result,
        best_hit=best_hit,
        audio=audio,
        sample_rate=WHISPER_SAMPLE_RATE,
        duration_seconds=duration_seconds,
        clip_start_seconds=round(clip_start_seconds, 6),
        clip_end_seconds=round(clip_end_seconds, 6),
    )


# Backwards-compatible aliases for earlier drafts of the module name.
ClapperClipResult = CutterResult
clip_audio_by_clapper = cut_audio_by_clapper
