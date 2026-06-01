from __future__ import annotations

from pathlib import Path

import settings


def test_save_whisper_settings_updates_existing_keys_and_keeps_comments(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
clapper:
  audio_model:
    model_name: laion/clap-htsat-fused

whisper:
  # Local Whisper checkpoint used for speech-to-text.
  model_id: openai/whisper-small
  # Default transcription language from the notebook experiments.
  language: french
  task: transcribe

renamer:
  naming_template: "Sequence_{sequence}_shot_{shot}_take_{take}"
""".lstrip(),
        encoding="utf-8",
    )

    settings.save_whisper_settings(
        config_path,
        language="ru",
        task="translate",
        model_id="openai/whisper-medium",
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "# Local Whisper checkpoint used for speech-to-text." in updated
    assert "model_id: openai/whisper-medium" in updated
    assert "language: ru" in updated
    assert "task: translate" in updated
    assert "renamer:" in updated


def test_save_whisper_settings_creates_missing_whisper_section(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("clapper:\n  text_search: {}\n", encoding="utf-8")

    settings.save_whisper_settings(
        config_path,
        language="fr",
        task="transcribe",
        model_id="openai/whisper-small",
    )

    updated = settings.load_config_data(config_path)
    assert updated["whisper"]["language"] == "fr"
    assert updated["whisper"]["task"] == "transcribe"
    assert updated["whisper"]["model_id"] == "openai/whisper-small"
