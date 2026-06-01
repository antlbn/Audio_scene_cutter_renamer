"""Normalize source audio into an in-memory AAC container."""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import yaml



DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_CODEC = "aac"
DEFAULT_BITRATE = "320k"


@dataclass(slots=True)
class PreprocessorConfig:
    """Settings for audio standardization."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    codec: str = DEFAULT_CODEC
    bitrate: str = DEFAULT_BITRATE


@dataclass(slots=True)
class StandardizedAudio:
    """In-memory AAC audio used as the shared pipeline contract."""

    file_name: str
    payload: bytes
    codec: str
    sample_rate: int
    channels: int
    duration_seconds: float


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def _merge_config_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> PreprocessorConfig:
    """Load the preprocessor section from config.yaml."""

    config_file = Path(config_path) if config_path is not None else _default_config_path()
    raw: dict[str, Any] = {}

    if config_file.exists():
        loaded = yaml.safe_load(config_file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
        raw = loaded.get("preprocessor", loaded)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid preprocessor section in {config_file}: expected a mapping")

    defaults = {
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "channels": DEFAULT_CHANNELS,
        "codec": DEFAULT_CODEC,
        "bitrate": DEFAULT_BITRATE,
    }
    merged = _merge_config_dict(defaults, raw)
    return PreprocessorConfig(
        sample_rate=int(merged["sample_rate"]),
        channels=max(1, int(merged["channels"])),
        codec=str(merged["codec"]),
        bitrate=str(merged["bitrate"]),
    )


def _load_audio_waveform(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load source audio as mono float32 samples."""

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.float()
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise

        ffmpeg_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-f",
            "wav",
            "-",
        ]
        proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        import soundfile as sf

        decoded, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(decoded.T.copy())

    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError(f"Audio file is empty or invalid: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), sample_rate


def _encode_aac_bytes(
    samples: np.ndarray,
    sample_rate: int,
    *,
    channels: int,
    bitrate: str,
) -> bytes:
    input_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-i",
        "pipe:0",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "adts",
        "pipe:1",
    ]
    proc = subprocess.run(input_cmd, check=True, capture_output=True, input=samples.tobytes())
    return proc.stdout


def standardize_audio(audio_path: str | Path, config: PreprocessorConfig | None = None) -> StandardizedAudio:
    """Convert an input file into a standard in-memory AAC payload."""

    config = config or load_config()
    audio_path = Path(audio_path)
    waveform, source_rate = _load_audio_waveform(audio_path)
    if source_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=source_rate,
            new_freq=config.sample_rate,
        )
        source_rate = config.sample_rate

    if waveform.numel() == 0:
        raise ValueError(f"Audio source is empty: {audio_path}")

    samples = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    payload = _encode_aac_bytes(
        samples,
        source_rate,
        channels=config.channels,
        bitrate=config.bitrate,
    )
    duration_seconds = float(waveform.numel()) / float(source_rate)
    return StandardizedAudio(
        file_name=audio_path.name,
        payload=payload,
        codec=config.codec,
        sample_rate=source_rate,
        channels=config.channels,
        duration_seconds=duration_seconds,
    )


def decode_standardized_audio(
    audio: StandardizedAudio,
    *,
    target_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Decode AAC bytes back into a mono float32 tensor."""

    sample_rate = target_sample_rate or audio.sample_rate
    ffmpeg_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, input=audio.payload)
    samples = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError("StandardizedAudio decoded to an empty waveform")
    return torch.from_numpy(samples), sample_rate
