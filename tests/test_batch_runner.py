from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cinema_clapboard_app import batch_runner
from cinema_clapboard_app.pipeline import PipelineResult


@pytest.fixture
def fake_success_result():
    return PipelineResult(
        input_name="test.wav",
        new_name="Sequence_1_shot_2_dubl_3.wav",
        preprocessor_codec="aac",
        preprocessor_sample_rate=16000,
        preprocessor_duration_seconds=5.0,
        clapper_hits=1,
        clapper_best_timestamp=1.5,
        clapper_best_score=0.9,
        clapper_best_text_key="clap",
        whisper_text="сцена один дубль три",
        whisper_model="small",
        whisper_language="ru",
        whisper_task="transcribe",
        llm_model="gemini",
        llm_sequence="1",
        llm_shot="2",
        llm_take=3,
        llm_announcement="сцена один дубль три",
    )


@pytest.fixture
def fake_clapper_fail_result():
    return PipelineResult(
        input_name="test.wav",
        new_name="test.wav",
        preprocessor_codec="aac",
        preprocessor_sample_rate=16000,
        preprocessor_duration_seconds=5.0,
        clapper_hits=0,
        clapper_best_timestamp=None,
        clapper_best_score=None,
        clapper_best_text_key=None,
        whisper_text=None,
        whisper_model=None,
        whisper_language=None,
        whisper_task=None,
        llm_model=None,
        llm_sequence=None,
        llm_shot=None,
        llm_take=None,
        llm_announcement=None,
    )


@pytest.fixture
def fake_llm_fail_result():
    return PipelineResult(
        input_name="test.wav",
        new_name="test.wav",
        preprocessor_codec="aac",
        preprocessor_sample_rate=16000,
        preprocessor_duration_seconds=5.0,
        clapper_hits=1,
        clapper_best_timestamp=1.5,
        clapper_best_score=0.9,
        clapper_best_text_key="clap",
        whisper_text="что-то непонятное",
        whisper_model="small",
        whisper_language="ru",
        whisper_task="transcribe",
        llm_model="gemini",
        llm_sequence=None,  # Missing field
        llm_shot="2",
        llm_take=3,
        llm_announcement="что-то непонятное",
    )


def test_run_single_file_success(tmp_path, fake_success_result):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("fake wav content")

    with patch("cinema_clapboard_app.pipeline.run_pipeline", return_value=fake_success_result), \
         patch("cinema_clapboard_app.presentator_cli.present") as mock_present:
        ok, res = batch_runner._run_single(
            audio_file,
            config_path=Path("config.yaml"),
            rename=True,
            json_output=False,
            save_json=False,
        )
        assert ok is True
        assert res == fake_success_result
        mock_present.assert_called_once()


def test_run_single_file_clapper_fail(tmp_path, fake_clapper_fail_result):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("fake wav content")

    with patch("cinema_clapboard_app.pipeline.run_pipeline", return_value=fake_clapper_fail_result), \
         patch("cinema_clapboard_app.presentator_cli.present") as mock_present:
        ok, res = batch_runner._run_single(
            audio_file,
            config_path=Path("config.yaml"),
            rename=True,
            json_output=False,
            save_json=False,
        )
        assert ok is True
        assert res == fake_clapper_fail_result
        mock_present.assert_called_once()


def test_batch_runner_cli_exit_codes_single_file(tmp_path, fake_success_result, fake_clapper_fail_result, fake_llm_fail_result):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("fake wav content")

    # Success scenario
    args = argparse.Namespace(
        path=audio_file,
        config=Path("config.yaml"),
        rename=True,
        json=False,
        save=False,
    )
    with patch("cinema_clapboard_app.pipeline.run_pipeline", return_value=fake_success_result):
        exit_code = batch_runner.run(args)
        assert exit_code == 0

    # Clapper fail scenario
    with patch("cinema_clapboard_app.pipeline.run_pipeline", return_value=fake_clapper_fail_result):
        exit_code = batch_runner.run(args)
        assert exit_code == 1

    # LLM fail scenario
    with patch("cinema_clapboard_app.pipeline.run_pipeline", return_value=fake_llm_fail_result):
        exit_code = batch_runner.run(args)
        assert exit_code == 1


def test_run_batch_manual_check_files_listing(tmp_path, fake_success_result, fake_clapper_fail_result, fake_llm_fail_result, capsys):
    dir_path = tmp_path / "recordings"
    dir_path.mkdir()
    
    file1 = dir_path / "file1.wav"
    file2 = dir_path / "file2.wav"
    file3 = dir_path / "file3.wav"
    file1.write_text("data")
    file2.write_text("data")
    file3.write_text("data")

    # Mock pipeline returns: file1 -> success, file2 -> clapper fail, file3 -> LLM fail
    def side_effect(path, **kwargs):
        if Path(path).name == "file1.wav":
            return fake_success_result
        elif Path(path).name == "file2.wav":
            return fake_clapper_fail_result
        else:
            return fake_llm_fail_result

    with patch("cinema_clapboard_app.pipeline.run_pipeline", side_effect=side_effect), \
         patch("cinema_clapboard_app.batch_runner._select_files_tui", return_value=[file1, file2, file3]), \
         patch("builtins.input", return_value="y"):

        exit_code = batch_runner._run_batch(
            dir_path,
            config_path=Path("config.yaml"),
            rename=True,
            json_output=False,
            save_json=False,
        )
        assert exit_code == 0  # No hard errors occurred, only warnings

        captured = capsys.readouterr()
        # Verify manual check warning checklist
        assert "Итог: ✅ 1 успешно / ⚠️ 2 на ручную проверку" in captured.out
        assert "Список файлов для ручной проверки:" in captured.out
        assert "file2.wav" in captured.out
        assert "file3.wav" in captured.out
