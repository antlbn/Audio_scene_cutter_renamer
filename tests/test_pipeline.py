from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import pipeline
from pipeline import PipelineResult


@pytest.fixture
def fake_std_audio():
    mock_audio = MagicMock()
    mock_audio.file_name = "test.wav"
    mock_audio.codec = "aac"
    mock_audio.sample_rate = 16000
    mock_audio.channels = 1
    mock_audio.duration_seconds = 10.0
    return mock_audio


@pytest.fixture
def fake_clapper_result():
    # Setup clapper hits
    hit = MagicMock()
    hit.timestamp = 2.5
    hit.score = 0.85
    hit.text_key = "film slate clap"

    res = MagicMock()
    res.best_scores = [hit]
    return res


@pytest.fixture
def fake_whisper_result():
    res = MagicMock()
    res.detection.text = "сцена один дубль четыре"
    res.detection.model_id = "openai/whisper-small"
    res.language = "ru"
    res.task = "transcribe"
    return res


@pytest.fixture
def fake_scene_parser_result():
    st = MagicMock()
    st.sequence = "1"
    st.shot = "3"
    st.take = 4
    st.announcement = "сцена один дубль четыре"

    res = MagicMock()
    res.scene_take = st
    res.model = "google/gemini-3.1-flash-lite"
    return res


def test_run_pipeline_with_clapper_hits(
    tmp_path,
    fake_std_audio,
    fake_clapper_result,
    fake_whisper_result,
    fake_scene_parser_result,
):
    dummy_file = tmp_path / "audio.wav"
    dummy_file.write_text("dummy content")

    mock_cut_res = MagicMock()
    mock_cut_res.standardized_audio = fake_std_audio
    mock_cut_res.clip_start_seconds = 2.5
    mock_cut_res.clip_end_seconds = 4.5

    with patch("preprocessor.standardize_audio", return_value=fake_std_audio), \
         patch("clapper.analyze_audio", return_value=fake_clapper_result), \
         patch("cutter.cut_audio_by_clapper", return_value=mock_cut_res), \
         patch("whisper.transcribe_audio", return_value=fake_whisper_result), \
         patch("scene_parser.parse_scene", return_value=fake_scene_parser_result):

        result = pipeline.run_pipeline(dummy_file)

        # Assert correct propagation of fields
        assert result.input_name == "audio.wav"
        assert result.new_name == "audio.wav"
        assert result.preprocessor_codec == "aac"
        assert result.preprocessor_sample_rate == 16000
        assert result.preprocessor_duration_seconds == 10.0
        assert result.clapper_hits == 1
        assert result.clapper_best_timestamp == 2.5
        assert result.clapper_best_score == 0.85
        assert result.clapper_best_text_key == "film slate clap"
        assert result.cutter_clip_start == 2.5
        assert result.cutter_clip_end == 4.5
        assert result.whisper_text == "сцена один дубль четыре"
        assert result.llm_sequence == "1"
        assert result.llm_shot == "3"
        assert result.llm_take == 4


def test_run_pipeline_without_clapper_hits(
    tmp_path,
    fake_std_audio,
    fake_whisper_result,
    fake_scene_parser_result,
):
    dummy_file = tmp_path / "audio.wav"
    dummy_file.write_text("dummy content")

    fake_clapper_no_hits = MagicMock()
    fake_clapper_no_hits.best_scores = []

    with patch("preprocessor.standardize_audio", return_value=fake_std_audio), \
         patch("clapper.analyze_audio", return_value=fake_clapper_no_hits), \
         patch("cutter.cut_audio_by_clapper") as mock_cutter, \
         patch("whisper.transcribe_audio", return_value=fake_whisper_result), \
         patch("scene_parser.parse_scene", return_value=fake_scene_parser_result):

        result = pipeline.run_pipeline(dummy_file)

        # Verify cutter was skipped since there are no claps
        mock_cutter.assert_not_called()

        assert result.clapper_hits == 0
        assert result.clapper_best_timestamp is None
        assert result.cutter_clip_start is None
        assert result.cutter_clip_end is None
        assert result.whisper_text == "сцена один дубль четыре"
        assert result.llm_sequence == "1"
        assert result.llm_shot == "3"
        assert result.llm_take == 4


def test_pipeline_cli_output(tmp_path, capsys):
    dummy_file = tmp_path / "audio.wav"
    dummy_file.write_text("dummy content")

    mock_result = PipelineResult(
        input_name="audio.wav",
        new_name="audio.wav",
        preprocessor_codec="aac",
        preprocessor_sample_rate=16000,
        preprocessor_duration_seconds=10.0,
        clapper_hits=1,
        clapper_best_timestamp=2.5,
        clapper_best_score=0.85,
        clapper_best_text_key="film slate clap",
        cutter_clip_start=2.5,
        cutter_clip_end=4.5,
        whisper_text="сцена один дубль четыре",
        whisper_model="openai/whisper-small",
        whisper_language="ru",
        whisper_task="transcribe",
        llm_model="google/gemini-3.1-flash-lite",
        llm_sequence="1",
        llm_shot="3",
        llm_take=4,
        llm_announcement="сцена один дубль четыре"
    )

    with patch("pipeline.run_pipeline", return_value=mock_result):
        # 1. Plain stdout format
        exit_code = pipeline.main([str(dummy_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "🎬 Scene Naming Pipeline Result" in captured.out
        assert "📁 Input File:       audio.wav" in captured.out
        assert "🎬 Sequence:       1" in captured.out
        assert "🎞  Shot:          3" in captured.out
        assert "🎥 Take:          4" in captured.out

        # 2. JSON format
        exit_code = pipeline.main([str(dummy_file), "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["input_name"] == "audio.wav"
        assert parsed["new_name"] == "audio.wav"
        assert parsed["llm_sequence"] == "1"
        assert parsed["llm_shot"] == "3"
        assert parsed["llm_take"] == 4

        # 3. Save format
        exit_code = pipeline.main([str(dummy_file), "--save"])
        assert exit_code == 0
        output_json_file = tmp_path / "audio_result.json"
        assert output_json_file.exists()
        saved_data = json.loads(output_json_file.read_text())
        assert saved_data["input_name"] == "audio.wav"
        assert saved_data["new_name"] == "audio.wav"
        assert saved_data["llm_sequence"] == "1"
        assert saved_data["llm_shot"] == "3"
        assert saved_data["llm_take"] == 4
