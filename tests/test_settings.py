from __future__ import annotations

import sys
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
  suffix: ""
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
    assert "suffix: \"\"" in updated


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


def test_save_whisper_settings_preserves_inline_comment_and_nested_section(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
whisper:
  language: fr
scene_parser: # inline comment
  model: google/gemini-3.1-flash-lite
""".lstrip(),
        encoding="utf-8",
    )

    settings.save_whisper_settings(
        config_path,
        language="en",
        task="translate",
        model_id="openai/whisper-small",
    )

    updated = settings.load_config_data(config_path)
    assert updated["whisper"]["language"] == "en"
    assert updated["whisper"]["task"] == "translate"
    assert updated["whisper"]["model_id"] == "openai/whisper-small"
    assert updated["scene_parser"]["model"] == "google/gemini-3.1-flash-lite"
    updated_text = config_path.read_text(encoding="utf-8")
    assert "scene_parser: # inline comment" in updated_text


def test_empty_whisper_section_is_treated_as_defaults(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("whisper:\n", encoding="utf-8")
    calls = []

    class FakePrompt:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class FakeInquirer:
        def select(self, **kwargs):
            calls.append(kwargs)
            return FakePrompt(settings.MENU_DONE)

    monkeypatch.setitem(sys.modules, "InquirerPy", type("FakeModule", (), {"inquirer": FakeInquirer()}))

    assert settings.run_settings_ui(config_path) == 0
    updated = settings.load_config_data(config_path)
    assert updated["whisper"]["language"] == "french"
    assert updated["whisper"]["task"] == "transcribe"
    assert updated["whisper"]["model_id"] == "openai/whisper-small"


def test_build_settings_menu_choices_shows_current_values():
    choices = settings.build_settings_menu_choices(
        language="french",
        task="transcribe",
        model_id="openai/whisper-small",
    )

    assert choices == [
        {"name": "Whisper language: french", "value": settings.MENU_LANGUAGE},
        {"name": "Whisper mode: transcribe", "value": settings.MENU_TASK},
        {"name": "Whisper model: openai/whisper-small", "value": settings.MENU_MODEL},
        {"name": "Rename rendering", "value": settings.MENU_RENAMER},
        {"name": "LLM API Key Manager", "value": settings.MENU_LLM_KEY_MANAGER},
        {"name": "Delete downloaded Whisper weights", "value": settings.MENU_DELETE_WEIGHTS},
        {"name": "Done / Save", "value": settings.MENU_DONE},
    ]


def test_save_renamer_settings_updates_existing_keys_and_keeps_comments(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
whisper:
  model_id: openai/whisper-small

renamer:
  # The naming template for renamed files.
  naming_template: "Sequence_{sequence}_shot_{shot}_take_{take}"
  suffix: ""
""".lstrip(),
        encoding="utf-8",
    )

    settings.save_renamer_settings(
        config_path,
        naming_template="seq-{sequence}-shot-{shot}-take-{take}",
        suffix="_AB",
        extract_from_source=False,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "seq-{sequence}-shot-{shot}-take-{take}" in updated
    assert 'suffix: "_AB"' in updated
    assert "# The naming template for renamed files." in updated


def test_select_menu_item_uses_clean_prompt(monkeypatch):
    calls = []

    class FakePrompt:
        def execute(self):
            return "done"

    class FakeInquirer:
        def select(self, **kwargs):
            calls.append(kwargs)
            return FakePrompt()

    monkeypatch.setattr(settings, "clear_terminal", lambda: calls.append({"clear": True}))

    result = settings.select_menu_item(
        FakeInquirer(),
        message="Settings",
        choices=[{"name": "Done / Save", "value": "done"}],
        default="done",
    )

    assert result == "done"
    assert calls[0] == {"clear": True}
    assert calls[1]["qmark"] == ""
    assert calls[1]["amark"] == ""
    assert calls[1]["pointer"] == ">"


def test_list_cached_whisper_models_reads_hugging_face_cache_dirs(tmp_path: Path):
    cache_dir = tmp_path / "whisper"
    model_dir = cache_dir / "models--openai--whisper-small"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"1234")
    (cache_dir / ".locks" / "models--openai--whisper-small").mkdir(parents=True)
    (cache_dir / "unrelated").mkdir()

    models = settings.list_cached_whisper_models(cache_dir)

    assert models == [
        settings.CachedWhisperModel(
            path=model_dir,
            size_bytes=4,
        )
    ]


def test_delete_cached_whisper_model_removes_model_and_lock_dirs(tmp_path: Path):
    cache_dir = tmp_path / "whisper"
    model_dir = cache_dir / "models--openai--whisper-small"
    lock_dir = cache_dir / ".locks" / "models--openai--whisper-small"
    model_dir.mkdir(parents=True)
    lock_dir.mkdir(parents=True)

    model = settings.CachedWhisperModel(
        path=model_dir,
        size_bytes=0,
    )

    settings.delete_cached_whisper_model(model, cache_dir)

    assert not model_dir.exists()
    assert not lock_dir.exists()


def test_delete_cached_whisper_model_refuses_outside_cache(tmp_path: Path):
    cache_dir = tmp_path / "whisper"
    outside_dir = tmp_path / "outside"
    cache_dir.mkdir()
    outside_dir.mkdir()

    model = settings.CachedWhisperModel(
        path=outside_dir,
        size_bytes=0,
    )

    try:
        settings.delete_cached_whisper_model(model, cache_dir)
    except ValueError as exc:
        assert "outside cache directory" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_llm_key_manager_helpers(tmp_path: Path):
    env_file = tmp_path / ".env"

    # Test mask_key
    assert settings.mask_key("") == "<Not Set>"
    assert settings.mask_key("your_openrouter_key_here") == "<Not Set>"
    assert settings.mask_key("12345") == "****"
    assert settings.mask_key("sk-or-v1-abcdef12345678") == "sk-or-v1...5678"

    # Test read_api_key_from_env on non-existent file
    assert settings.read_api_key_from_env(env_file) == ""

    # Test write_api_key_to_env
    settings.write_api_key_to_env(env_file, "my-secret-key")
    assert settings.read_api_key_from_env(env_file) == "my-secret-key"

    # Test updating existing key and keeping other env vars
    env_file.write_text(
        """
OTHER_VAR=123
OPENROUTER_API_KEY="old-key"
ANOTHER_VAR="abc"
""".lstrip(),
        encoding="utf-8",
    )

    settings.write_api_key_to_env(env_file, "new-key")
    content = env_file.read_text(encoding="utf-8")
    assert 'OPENROUTER_API_KEY="new-key"' in content
    assert "OTHER_VAR=123" in content
    assert 'ANOTHER_VAR="abc"' in content
    assert settings.read_api_key_from_env(env_file) == "new-key"
