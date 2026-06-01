"""Interactive project settings editor for the terminal wrapper."""

from __future__ import annotations

import argparse
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

WHISPER_MODEL_CHOICES = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
]

WHISPER_TASK_CHOICES = ["transcribe", "translate"]

WHISPER_LANGUAGE_CHOICES = [
    "french",
    "english",
    "russian",
    "ru",
    "fr",
    "en",
]

MENU_LANGUAGE = "language"
MENU_TASK = "task"
MENU_MODEL = "model"
MENU_DONE = "done"


def clear_terminal() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def build_yaml_parser() -> Any:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def new_config_mapping() -> MutableMapping[str, Any]:
    from ruamel.yaml.comments import CommentedMap

    return CommentedMap()


def load_config_data(config_path: str | Path) -> MutableMapping[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        return new_config_mapping()

    yaml = build_yaml_parser()
    data = yaml.load(config_file.read_text(encoding="utf-8")) or new_config_mapping()
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
    return data


def _choice_with_current(choices: list[str], current: str) -> list[str]:
    if current and current not in choices:
        return [current, *choices]
    return choices


def build_settings_menu_choices(*, language: str, task: str, model_id: str) -> list[dict[str, str]]:
    return [
        {"name": f"Whisper language: {language}", "value": MENU_LANGUAGE},
        {"name": f"Whisper mode: {task}", "value": MENU_TASK},
        {"name": f"Whisper model: {model_id}", "value": MENU_MODEL},
        {"name": "Done / Save", "value": MENU_DONE},
    ]


def select_menu_item(inquirer: Any, *, message: str, choices: list[Any], default: str) -> Any:
    clear_terminal()
    return inquirer.select(
        message=message,
        choices=choices,
        default=default,
        qmark="",
        amark="",
        pointer=">",
    ).execute()


def save_whisper_settings(
    config_path: str | Path,
    *,
    language: str,
    task: str,
    model_id: str,
) -> None:
    config_file = Path(config_path)
    yaml = build_yaml_parser()
    data = load_config_data(config_file)
    whisper_config = data.get("whisper")

    if whisper_config is None:
        whisper_config = new_config_mapping()
        data["whisper"] = whisper_config
    elif not isinstance(whisper_config, MutableMapping):
        raise ValueError(f"Invalid whisper section in {config_file}: expected a mapping")

    whisper_config["language"] = language
    whisper_config["task"] = task
    whisper_config["model_id"] = model_id

    with config_file.open("w", encoding="utf-8") as file:
        yaml.dump(data, file)


def run_settings_ui(config_path: str | Path) -> int:
    try:
        from InquirerPy import inquirer
    except ImportError:
        print(
            "InquirerPy is not installed. Run `uv sync` or `uv pip install InquirerPy` first.",
            file=sys.stderr,
        )
        return 1

    config_file = Path(config_path)
    data = load_config_data(config_file)
    whisper_config = data.get("whisper", {})
    if whisper_config is None:
        whisper_config = {}
    elif not isinstance(whisper_config, dict):
        raise ValueError(f"Invalid whisper section in {config_file}: expected a mapping")

    current_language = str(whisper_config.get("language", "french"))
    current_task = str(whisper_config.get("task", "transcribe"))
    current_model = str(whisper_config.get("model_id", "openai/whisper-small"))

    language = current_language
    task = current_task
    model_id = current_model

    while True:
        action = select_menu_item(
            inquirer,
            message="Settings",
            choices=build_settings_menu_choices(language=language, task=task, model_id=model_id),
            default=MENU_DONE,
        )

        if action == MENU_DONE:
            break
        if action == MENU_LANGUAGE:
            language = select_menu_item(
                inquirer,
                message="Whisper audio language",
                choices=_choice_with_current(WHISPER_LANGUAGE_CHOICES, language),
                default=language,
            )
        elif action == MENU_TASK:
            task = select_menu_item(
                inquirer,
                message="Whisper mode",
                choices=_choice_with_current(WHISPER_TASK_CHOICES, task),
                default=task,
            )
        elif action == MENU_MODEL:
            model_id = select_menu_item(
                inquirer,
                message="Whisper local model",
                choices=_choice_with_current(WHISPER_MODEL_CHOICES, model_id),
                default=model_id,
            )

    save_whisper_settings(
        config_file,
        language=language,
        task=task,
        model_id=model_id,
    )

    clear_terminal()
    print(f"Settings saved to {config_file}")
    print(f"Whisper language: {language}")
    print(f"Whisper mode: {task}")
    print(f"Whisper model: {model_id}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit cinema_clapboard settings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return run_settings_ui(args.config)
    except KeyboardInterrupt:
        print("\nSettings unchanged.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Settings failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
