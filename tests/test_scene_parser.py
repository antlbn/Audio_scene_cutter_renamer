from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import scene_parser
from scene_parser import SceneTake, SceneParserConfig, SceneParseResult


def test_load_scene_parser_config_fills_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scene_parser:
  model: "google/gemini-3.1-flash-lite"
  temperature: 0.1
""".strip()
    )

    config = scene_parser.load_config(config_path)

    assert config.model == "google/gemini-3.1-flash-lite"
    assert config.api_base == scene_parser.DEFAULT_API_BASE
    assert config.api_key_env == scene_parser.DEFAULT_API_KEY_ENV
    assert config.prompt_path == scene_parser.DEFAULT_PROMPT_PATH
    assert config.temperature == 0.1
    assert config.max_tokens == scene_parser.DEFAULT_MAX_TOKENS
    assert config.sequence_description == scene_parser.DEFAULT_SEQUENCE_DESCRIPTION
    assert config.shot_description == scene_parser.DEFAULT_SHOT_DESCRIPTION
    assert config.take_description == scene_parser.DEFAULT_TAKE_DESCRIPTION


def test_scene_parser_cli(capsys):
    mock_result = SceneParseResult(
        scene_take=SceneTake(sequence="12", shot="3", take=5, announcement="сцена двенадцать дубль пять"),
        model="test-model",
        raw_llm_response="{}"
    )

    with patch("scene_parser.parse_scene", return_value=mock_result) as mock_parse:
        # Test human-readable output
        exit_code = scene_parser.main(["сцена двенадцать дубль пять"])
        assert exit_code == 0
        mock_parse.assert_called_once()
        captured = capsys.readouterr()
        assert "Sequence: 12" in captured.out
        assert "Shot: 3" in captured.out
        assert "Take: 5" in captured.out

        # Test JSON output
        mock_parse.reset_mock()
        exit_code = scene_parser.main(["сцена двенадцать дубль пять", "--json"])
        assert exit_code == 0
        mock_parse.assert_called_once()
        captured = capsys.readouterr()
        parsed_out = json.loads(captured.out.strip())
        assert parsed_out["sequence"] == "12"
        assert parsed_out["shot"] == "3"
        assert parsed_out["take"] == 5
        assert parsed_out["announcement"] == "сцена двенадцать дубль пять"


def test_parse_scene_with_mocked_llm(tmp_path: Path, monkeypatch):
    # Set up dummy environment variable for OpenRouter API key
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_12345")

    # Create a dummy prompt file
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "scene_parser.md"
    prompt_file.write_text("Prompt template. Input: {text}. Schema: {schema}")

    config = SceneParserConfig(
        model="google/gemini-2.0-flash-001",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        prompt_path=str(prompt_file),
        temperature=0.0,
        max_tokens=128
    )

    whisper_text = "сцена четыре дубль один"

    # Mock the OpenAI client and response
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "sequence": "3",
        "shot": "4",
        "take": 1,
        "announcement": "сцена четыре дубль один"
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    # Patch OpenAI client creation
    with patch("scene_parser.OpenAI", return_value=mock_client) as mock_openai_cls:
        result = scene_parser.parse_scene(whisper_text, config=config)

        # Verify client is created correctly
        mock_openai_cls.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="fake_key_12345"
        )

        # Verify chat completion called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "google/gemini-2.0-flash-001"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["response_format"] == {"type": "json_object"}

        # Verify parsed output
        assert result.scene_take.sequence == "3"
        assert result.scene_take.shot == "4"
        assert result.scene_take.take == 1
        assert result.scene_take.announcement == whisper_text
        assert result.model == "google/gemini-2.0-flash-001"


def test_parse_scene_populates_missing_raw_announcement(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_key_12345")
    prompt_file = tmp_path / "scene_parser.md"
    prompt_file.write_text("Prompt template: {text} {schema}")

    config = SceneParserConfig(
        prompt_path=str(prompt_file)
    )

    whisper_text = "сцена пять дубль три"

    # LLM doesn't return announcement in this JSON
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "sequence": "2",
        "shot": "5",
        "take": 3
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("scene_parser.OpenAI", return_value=mock_client):
        result = scene_parser.parse_scene(whisper_text, config=config)
        assert result.scene_take.sequence == "2"
        assert result.scene_take.shot == "5"
        assert result.scene_take.take == 3
        # Should be auto-populated with original whisper_text
        assert result.scene_take.announcement == whisper_text


def test_scene_take_pydantic_validation():
    # Happy path
    valid_data = {
        "sequence": "3",
        "shot": "3A",
        "take": 2,
        "announcement": "sequence three shot three a take two"
    }
    st = SceneTake.model_validate(valid_data)
    assert st.sequence == "3"
    assert st.shot == "3A"
    assert st.take == 2
    assert st.announcement == "sequence three shot three a take two"

    # Missing required announcement field
    invalid_data = {
        "sequence": "3",
        "shot": "3A",
        "take": 2,
        # missing announcement
    }
    with pytest.raises(ValidationError):
        SceneTake.model_validate(invalid_data)

    # Invalid take type
    invalid_take = {
        "announcement": "test",
        "take": "not-an-int",
    }
    with pytest.raises(ValidationError):
        SceneTake.model_validate(invalid_take)
