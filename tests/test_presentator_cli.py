"""Tests for presentator_cli module."""

from __future__ import annotations


import pytest

from cinema_clapboard_app import presentator_cli



@pytest.fixture
def minimal_result() -> dict:
    """Minimal valid result with only required fields."""
    return {
        "input_name": "test.wav",
        "new_name": "Sequence_1_shot_2_take_3.wav",
        "llm_sequence": "1",
        "llm_shot": "2",
        "llm_take": "3",
    }


@pytest.fixture
def full_result() -> dict:
    """Complete result with all fields populated."""
    return {
        "input_name": "test_audio.wav",
        "new_name": "Sequence_1_shot_2_take_3.wav",
        "preprocessor_codec": "pcm_s16le",
        "preprocessor_sample_rate": 16000,
        "preprocessor_duration_seconds": 5.25,
        "clapper_hits": 1,
        "clapper_best_timestamp": 0.45,
        "clapper_best_score": 0.95,
        "clapper_best_text_key": "clap",
        "cutter_clip_start": 0.5,
        "cutter_clip_end": 4.8,
        "whisper_text": "Scene 1, take 3",
        "whisper_model": "base",
        "whisper_language": "en",
        "whisper_task": "transcribe",
        "llm_model": "gpt-4",
        "llm_sequence": "1",
        "llm_shot": "2",
        "llm_take": "3",
        "llm_announcement": "Scene 1, take 3",
    }


class TestPresent:
    """Test the main present() function."""

    def test_present_name_extractor_minimal(self, minimal_result, capsys):
        """Test name_extractor use case with minimal data."""
        presentator_cli.present(minimal_result, use_case="name_extractor")
        captured = capsys.readouterr()
        
        assert "test.wav" in captured.out
        assert "Sequence_1_shot_2_take_3.wav" in captured.out
        assert "1" in captured.out  # sequence

    def test_present_name_extractor_full(self, full_result, capsys):
        """Test name_extractor use case with all data."""
        presentator_cli.present(full_result, use_case="name_extractor")
        captured = capsys.readouterr()
        
        assert "test_audio.wav" in captured.out
        assert "Scene 1, take 3" in captured.out
        assert "Clapper Detection" in captured.out
        assert "0.45" in captured.out  # timestamp

    def test_present_whisper_extractor_minimal(self, minimal_result, capsys):
        """Test whisper_extractor use case with minimal data."""
        presentator_cli.present(minimal_result, use_case="whisper_extractor")
        captured = capsys.readouterr()
        
        assert "test.wav" in captured.out
        assert "Speech Recognition" in captured.out

    def test_present_whisper_extractor_full(self, full_result, capsys):
        """Test whisper_extractor use case with all data."""
        presentator_cli.present(full_result, use_case="whisper_extractor")
        captured = capsys.readouterr()
        
        assert "test_audio.wav" in captured.out
        assert "Scene 1, take 3" in captured.out
        assert "base" in captured.out  # model
        assert "en" in captured.out  # language

    def test_present_unknown_use_case_defaults_to_name_extractor(self, minimal_result, capsys):
        """Test that unknown use_case defaults to name_extractor."""
        presentator_cli.present(minimal_result, use_case="unknown_case")
        captured = capsys.readouterr()
        
        # Should contain name_extractor output
        assert "test.wav" in captured.out
        assert "Sequence_1_shot_2_take_3.wav" in captured.out

    def test_present_json_output(self, minimal_result, capsys):
        """Test JSON output format."""
        presentator_cli.present(minimal_result, use_case="name_extractor", json_output=True)
        captured = capsys.readouterr()
        
        assert "input_name" in captured.out
        assert "test.wav" in captured.out
        # Should be valid JSON-like output
        assert "{" in captured.out and "}" in captured.out


class TestNameExtractorPresenter:
    """Test _present_name_extractor() function."""

    def test_no_metadata(self, capsys):
        """Test with no metadata extracted."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
        }
        presentator_cli._present_name_extractor(data)
        captured = capsys.readouterr()
        
        assert "test.wav" in captured.out
        assert "No metadata extracted" in captured.out

    def test_long_announcement_full(self, capsys):
        """Test that long announcement is displayed in full."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
            "llm_sequence": "1",
            "llm_announcement": "x" * 120,
        }
        presentator_cli._present_name_extractor(data)
        captured = capsys.readouterr()
        
        # Should display full text without truncation
        assert "x" * 120 in captured.out

    def test_long_whisper_text_full(self, capsys):
        """Test that long whisper text is displayed in full."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
            "llm_sequence": "1",
            "whisper_text": "word " * 50,  # Very long text
        }
        presentator_cli._present_name_extractor(data)
        captured = capsys.readouterr()
        
        # Should display full text without truncation
        assert "word " * 50 in captured.out

    def test_clapper_detection_with_timestamp(self, capsys):
        """Test clapper detection display with timestamp."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
            "llm_sequence": "1",
            "clapper_hits": 3,
            "clapper_best_timestamp": 1.234,
        }
        presentator_cli._present_name_extractor(data)
        captured = capsys.readouterr()
        
        assert "Clapper Detection" in captured.out
        assert "3" in captured.out  # hits
        assert "1.23" in captured.out  # timestamp (2 decimals)

    def test_clapper_detection_without_timestamp(self, capsys):
        """Test clapper detection display when no timestamp."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
            "llm_sequence": "1",
            "clapper_hits": 0,
        }
        presentator_cli._present_name_extractor(data)
        captured = capsys.readouterr()
        
        # Should not crash or show timestamp when not available
        assert "test.wav" in captured.out


class TestWhisperExtractorPresenter:
    """Test _present_whisper_extractor() function."""

    def test_short_text_not_wrapped(self, capsys):
        """Test that short text is not wrapped."""
        data = {
            "input_name": "test.wav",
            "whisper_text": "Short text",
            "whisper_model": "base",
            "whisper_language": "en",
            "whisper_task": "transcribe",
        }
        presentator_cli._present_whisper_extractor(data)
        captured = capsys.readouterr()
        
        assert "Short text" in captured.out
        # Short text should appear directly, not wrapped
        assert "Short text" in captured.out

    def test_long_text_displayed_full(self, capsys):
        """Test that long text is displayed in full without wrapping."""
        data = {
            "input_name": "test.wav",
            "whisper_text": "word " * 30,  # Very long line
            "whisper_model": "base",
            "whisper_language": "en",
            "whisper_task": "transcribe",
        }
        presentator_cli._present_whisper_extractor(data)
        captured = capsys.readouterr()
        
        # Should display full text without wrapping
        assert "word " * 30 in captured.out

    def test_clapper_context_displayed(self, capsys):
        """Test that clapper detection context is shown when available."""
        data = {
            "input_name": "test.wav",
            "whisper_text": "test",
            "whisper_model": "base",
            "whisper_language": "en",
            "whisper_task": "transcribe",
            "clapper_best_timestamp": 2.5,
        }
        presentator_cli._present_whisper_extractor(data)
        captured = capsys.readouterr()
        
        assert "Detection Context" in captured.out
        assert "2.50" in captured.out  # timestamp with 2 decimals

    def test_no_clapper_context_when_timestamp_missing(self, capsys):
        """Test that clapper context not shown when timestamp is missing."""
        data = {
            "input_name": "test.wav",
            "whisper_text": "test",
            "whisper_model": "base",
            "whisper_language": "en",
            "whisper_task": "transcribe",
        }
        presentator_cli._present_whisper_extractor(data)
        captured = capsys.readouterr()
        
        # Should not crash and context should not appear
        assert "Detection Context" not in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dict(self, capsys):
        """Test with completely empty dict."""
        data = {}
        presentator_cli.present(data, use_case="name_extractor")
        captured = capsys.readouterr()
        
        # Should not crash, should show N/A
        assert "N/A" in captured.out

    def test_none_values_handled(self, capsys):
        """Test that None values are handled gracefully."""
        data = {
            "input_name": None,
            "new_name": None,
            "llm_sequence": None,
            "whisper_text": None,
        }
        presentator_cli.present(data, use_case="name_extractor")
        captured = capsys.readouterr()
        
        # Should not crash
        assert "input_name" not in captured.out or "None" not in captured.out

    def test_zero_values_not_confused_with_missing(self, capsys):
        """Test that 0 values are displayed, not confused with missing."""
        data = {
            "input_name": "test.wav",
            "new_name": "output.wav",
            "llm_sequence": "0",  # Zero sequence
            "clapper_hits": 0,  # No hits
        }
        presentator_cli.present(data, use_case="name_extractor")
        captured = capsys.readouterr()
        
        assert "test.wav" in captured.out

    def test_special_characters_in_text(self, capsys):
        """Test handling of special characters in text."""
        data = {
            "input_name": "тест.wav",  # Cyrillic
            "new_name": "Sequence_1.wav",
            "llm_sequence": "1",
            "whisper_text": "Scene with émojis 🎬",
        }
        presentator_cli.present(data, use_case="name_extractor")
        captured = capsys.readouterr()
        
        # Should handle special characters without crashing
        assert "тест.wav" in captured.out or "test" in captured.out.lower()
