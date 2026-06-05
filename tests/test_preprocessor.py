from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cinema_clapboard_app import preprocessor



def test_load_config_fills_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("preprocessor: {}\n")

    config = preprocessor.load_config(config_path)

    assert config.sample_rate == preprocessor.DEFAULT_SAMPLE_RATE
    assert config.channels == preprocessor.DEFAULT_CHANNELS
    assert config.codec == preprocessor.DEFAULT_CODEC
    assert config.bitrate == preprocessor.DEFAULT_BITRATE


def test_standardize_audio_returns_in_memory_aac(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"ignored")

    monkeypatch.setattr(
        preprocessor,
        "_load_audio_waveform",
        lambda path: (torch.tensor([0.0, 0.5, -0.5, 0.0], dtype=torch.float32), 48_000),
    )
    monkeypatch.setattr(
        preprocessor.torchaudio.functional,
        "resample",
        lambda waveform, orig_freq, new_freq: torch.tensor([0.25, -0.25], dtype=torch.float32),
    )

    recorded: dict[str, bytes] = {}

    def fake_run(cmd, check, capture_output, input):
        recorded["cmd"] = cmd
        recorded["input"] = input
        return SimpleNamespace(stdout=b"aac-bytes")

    monkeypatch.setattr(preprocessor.subprocess, "run", fake_run)

    standardized = preprocessor.standardize_audio(audio_path)

    assert standardized.file_name == "input.wav"
    assert standardized.payload == b"aac-bytes"
    assert standardized.codec == "aac"
    assert standardized.sample_rate == 16_000
    assert standardized.channels == 1
    assert standardized.duration_seconds == pytest.approx(2 / 16_000)
    assert recorded["cmd"][0] == "ffmpeg"
    assert recorded["input"] == np.array([0.25, -0.25], dtype=np.float32).tobytes()


def test_decode_standardized_audio_returns_tensor(monkeypatch):
    audio = preprocessor.StandardizedAudio(
        file_name="input.aac",
        payload=b"aac-bytes",
        codec="aac",
        sample_rate=16_000,
        channels=1,
        duration_seconds=0.25,
    )

    def fake_run(cmd, check, capture_output, input):
        return SimpleNamespace(stdout=np.array([0.1, 0.2, 0.3], dtype="<f4").tobytes())

    monkeypatch.setattr(preprocessor.subprocess, "run", fake_run)

    waveform, sample_rate = preprocessor.decode_standardized_audio(audio)

    assert sample_rate == 16_000
    assert waveform.dtype == torch.float32
    assert waveform.tolist() == pytest.approx([0.1, 0.2, 0.3])
