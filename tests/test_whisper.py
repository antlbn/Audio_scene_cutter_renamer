from __future__ import annotations

from pathlib import Path

import torch

import clapper
import cutter
import preprocessor
import whisper


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, audio_input, return_timestamps, generate_kwargs):
        self.calls.append(
            {
                "audio_input": audio_input,
                "return_timestamps": return_timestamps,
                "generate_kwargs": generate_kwargs,
            }
        )
        return {
            "text": "привет",
            "chunks": [
                {
                    "timestamp": [0.0, 1.5],
                    "text": " привет",
                }
            ],
        }


def test_load_whisper_config_fills_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
whisper:
  language: fr
""".strip()
    )

    config = whisper.load_config(config_path)

    assert config.model_id == whisper.DEFAULT_MODEL_ID
    assert config.cache_dir == whisper.DEFAULT_CACHE_DIR
    assert config.language == "fr"
    assert config.task == whisper.DEFAULT_TASK
    assert config.return_timestamps is True


def test_transcribe_audio_uses_standardized_audio(monkeypatch):
    fake_pipeline = FakePipeline()
    runtime = whisper.WhisperRuntime(
        asr_pipeline=fake_pipeline,
        device=torch.device("cpu"),
    )
    config = whisper.WhisperConfig(language="ru", task="transcribe", return_timestamps=True)
    source = preprocessor.StandardizedAudio(
        file_name="input.aac",
        payload=b"source-aac",
        codec="aac",
        sample_rate=16_000,
        channels=1,
        duration_seconds=3.0,
    )

    monkeypatch.setattr(
        preprocessor,
        "decode_standardized_audio",
        lambda audio, target_sample_rate: (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32), target_sample_rate),
    )

    result = whisper.transcribe_audio(source, config=config, runtime=runtime)

    assert result.file_name == "input.aac"
    assert result.detection.text == "привет"
    assert result.detection.model_id == whisper.DEFAULT_MODEL_ID
    assert result.chunks[0].timestamp == (0.0, 1.5)
    assert result.chunks[0].text == " привет"

    call = fake_pipeline.calls[0]
    assert call["return_timestamps"] is True
    assert call["generate_kwargs"] == {"task": "transcribe", "language": "ru"}
    assert call["audio_input"]["sampling_rate"] == 16_000


def test_transcribe_audio_uses_cutter_result(monkeypatch):
    fake_pipeline = FakePipeline()
    runtime = whisper.WhisperRuntime(
        asr_pipeline=fake_pipeline,
        device=torch.device("cpu"),
    )
    config = whisper.WhisperConfig(language="ru", task="transcribe", return_timestamps=True)
    standardized = preprocessor.StandardizedAudio(
        file_name="input.aac",
        payload=b"source-aac",
        codec="aac",
        sample_rate=16_000,
        channels=1,
        duration_seconds=3.0,
    )
    source = cutter.CutterResult(
        clapper_result=clapper.ClapperResult(
            file_name="input.aac",
            best_scores=[],
            num_points=0,
            threshold=0.3,
        ),
        best_hit=clapper.ClapperHit(
            timestamp=0.0,
            score=0.9,
            text_key="clap",
            top_matches=[],
        ),
        standardized_audio=standardized,
        duration_seconds=1.0,
        clip_start_seconds=0.0,
        clip_end_seconds=1.0,
    )

    monkeypatch.setattr(
        preprocessor,
        "decode_standardized_audio",
        lambda audio, target_sample_rate: (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32), target_sample_rate),
    )

    result = whisper.transcribe_audio(source, config=config, runtime=runtime)

    assert result.file_name == "input.aac"
    assert result.detection.text == "привет"
