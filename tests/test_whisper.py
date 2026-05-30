from __future__ import annotations

from pathlib import Path

import torch

import clapper
import cutter
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


def test_transcribe_audio_uses_cutter_result_in_memory():
    fake_pipeline = FakePipeline()
    runtime = whisper.WhisperRuntime(
        asr_pipeline=fake_pipeline,
        device=torch.device("cpu"),
    )
    config = whisper.WhisperConfig(language="ru", task="transcribe", return_timestamps=True)
    source = cutter.CutterResult(
        clapper_result=clapper.ClapperResult(
            file_name="input.wav",
            best_scores=[],
            num_points=0,
            threshold=0.3,
        ),
        best_hit=clapper.ClapperHit(
            timestamp=1.0,
            score=0.9,
            text_key="clap",
            top_matches=[],
        ),
        audio=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        sample_rate=16_000,
        duration_seconds=0.0001875,
        clip_start_seconds=1.0,
        clip_end_seconds=3.0,
    )

    result = whisper.transcribe_audio(source, config=config, runtime=runtime)

    assert result.file_name == "input.wav"
    assert result.text == "привет"
    assert result.model_id == whisper.DEFAULT_MODEL_ID
    assert result.chunks[0].timestamp == (0.0, 1.5)
    assert result.chunks[0].text == " привет"

    call = fake_pipeline.calls[0]
    assert call["return_timestamps"] is True
    assert call["generate_kwargs"] == {"task": "transcribe", "language": "ru"}
    assert call["audio_input"]["sampling_rate"] == 16_000
