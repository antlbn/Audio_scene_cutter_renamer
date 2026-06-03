"""Interactive project settings editor for the terminal wrapper."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config_utils import resolve_path

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
MENU_RENAMER = "renamer"
MENU_DELETE_WEIGHTS = "delete_weights"
MENU_DONE = "done"
MENU_BACK = "back"


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


@dataclass(frozen=True)
class CachedWhisperModel:
    path: Path
    size_bytes: int


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def build_settings_menu_choices(*, language: str, task: str, model_id: str) -> list[dict[str, str]]:
    return [
        {"name": f"Whisper language: {language}", "value": MENU_LANGUAGE},
        {"name": f"Whisper mode: {task}", "value": MENU_TASK},
        {"name": f"Whisper model: {model_id}", "value": MENU_MODEL},
        {"name": "Rename rendering", "value": MENU_RENAMER},
        {"name": "Delete downloaded Whisper weights", "value": MENU_DELETE_WEIGHTS},
        {"name": "Done / Save", "value": MENU_DONE},
    ]


def select_menu_item(inquirer: Any, *, message: str, choices: list[Any], default: Any) -> Any:
    clear_terminal()
    return inquirer.select(
        message=message,
        choices=choices,
        default=default,
        qmark="",
        amark="",
        pointer=">",
    ).execute()


def select_text_item(inquirer: Any, *, message: str, default: str) -> str:
    clear_terminal()
    return str(
        inquirer.text(
            message=message,
            default=default,
            qmark="",
            amark="",
        ).execute()
    )


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def list_cached_whisper_models(cache_dir: str | Path) -> list[CachedWhisperModel]:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []

    models = []
    for child in sorted(cache_path.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("models--"):
            continue
        models.append(CachedWhisperModel(path=child, size_bytes=directory_size(child)))
    return models


def delete_cached_whisper_model(model: CachedWhisperModel, cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir).resolve()
    model_path = model.path.resolve()
    if cache_path not in model_path.parents:
        raise ValueError(f"Refusing to delete path outside cache directory: {model.path}")

    shutil.rmtree(model_path)
    lock_path = cache_path / ".locks" / model.path.name
    if lock_path.exists():
        shutil.rmtree(lock_path)


def run_delete_whisper_weights_ui(inquirer: Any, cache_dir: str | Path) -> None:
    models = list_cached_whisper_models(cache_dir)
    if not models:
        select_menu_item(
            inquirer,
            message=f"No downloaded Whisper weights found in {cache_dir}",
            choices=[{"name": "Back", "value": MENU_BACK}],
            default=MENU_BACK,
        )
        return

    choices = [
        {
            "name": f"{model.path.name} ({format_size(model.size_bytes)})",
            "value": index,
        }
        for index, model in enumerate(models)
    ]
    choices.append({"name": "Back", "value": MENU_BACK})

    selected = select_menu_item(
        inquirer,
        message="Delete downloaded Whisper weights",
        choices=choices,
        default=MENU_BACK,
    )
    if selected == MENU_BACK:
        return

    model = models[int(selected)]
    confirmation = select_menu_item(
        inquirer,
        message=f"Delete {model.path.name}?",
        choices=[
            {"name": "Cancel", "value": False},
            {"name": "Delete", "value": True},
        ],
        default=False,
    )
    if confirmation:
        delete_cached_whisper_model(model, cache_dir)


def build_renamer_menu_choices(
    *,
    naming_template: str,
    suffix: str,
    extract_from_source: bool,
    extract_pattern: str,
    extract_format: str,
) -> list[dict[str, str]]:
    suffix_display = suffix if suffix else "<off>"
    extract_status = "ON" if extract_from_source else "OFF"
    return [
        {"name": f"Naming template: {naming_template}", "value": "template"},
        {"name": f"Suffix: {suffix_display}", "value": "suffix"},
        {"name": f"Extract from source: {extract_status}", "value": "extract_from_source"},
        {"name": f"Extract pattern: {extract_pattern}", "value": "extract_pattern"},
        {"name": f"Extract format: {extract_format}", "value": "extract_format"},
        {"name": "Back", "value": MENU_BACK},
    ]


def save_renamer_settings(
    config_path: str | Path,
    *,
    naming_template: str,
    suffix: str,
    extract_from_source: bool,
    extract_pattern: str,
    extract_format: str,
) -> None:
    config_file = Path(config_path)
    yaml = build_yaml_parser()
    data = load_config_data(config_file)
    renamer_config = data.get("renamer")

    if renamer_config is None:
        renamer_config = new_config_mapping()
        data["renamer"] = renamer_config
    elif not isinstance(renamer_config, MutableMapping):
        raise ValueError(f"Invalid renamer section in {config_file}: expected a mapping")

    renamer_config["naming_template"] = naming_template
    renamer_config["suffix"] = suffix
    renamer_config["extract_from_source"] = extract_from_source
    renamer_config["extract_pattern"] = extract_pattern
    renamer_config["extract_format"] = extract_format

    with config_file.open("w", encoding="utf-8") as file:
        yaml.dump(data, file)


def run_renamer_settings_ui(inquirer: Any, config_path: Path, data: MutableMapping[str, Any]) -> None:
    renamer_config = data.get("renamer", {})
    if renamer_config is None:
        renamer_config = {}
    elif not isinstance(renamer_config, dict):
        raise ValueError(f"Invalid renamer section in {config_path}: expected a mapping")

    naming_template = str(renamer_config.get("naming_template", "Sequence_{sequence}_shot_{shot}_take_{take}"))
    suffix = str(renamer_config.get("suffix", ""))
    extract_from_source = bool(renamer_config.get("extract_from_source", False))
    extract_pattern = str(renamer_config.get("extract_pattern", "_([Tt]r\\d+)"))
    extract_format = str(renamer_config.get("extract_format", "_{match}"))

    while True:
        action = select_menu_item(
            inquirer,
            message="Rename rendering",
            choices=build_renamer_menu_choices(
                naming_template=naming_template,
                suffix=suffix,
                extract_from_source=extract_from_source,
                extract_pattern=extract_pattern,
                extract_format=extract_format,
            ),
            default=MENU_BACK,
        )

        if action == MENU_BACK:
            break
        if action == "template":
            naming_template = select_text_item(
                inquirer,
                message="Naming template",
                default=naming_template,
            )
        elif action == "suffix":
            suffix = select_text_item(
                inquirer,
                message="Suffix",
                default=suffix,
            )
        elif action == "extract_from_source":
            extract_from_source = select_menu_item(
                inquirer,
                message="Extract from source filename",
                choices=[
                    {"name": "ON", "value": True},
                    {"name": "OFF", "value": False},
                ],
                default=extract_from_source,
            )
        elif action == "extract_pattern":
            extract_pattern = select_text_item(
                inquirer,
                message="Regex pattern to extract (e.g., '_([Tt]r\\\\d+)' for ZOOM)",
                default=extract_pattern,
            )
        elif action == "extract_format":
            extract_format = select_text_item(
                inquirer,
                message="Format for extracted value (use {match} as placeholder, e.g., '_{match}')",
                default=extract_format,
            )

    save_renamer_settings(
        config_path,
        naming_template=naming_template,
        suffix=suffix,
        extract_from_source=extract_from_source,
        extract_pattern=extract_pattern,
        extract_format=extract_format,
    )


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
    current_cache_dir = resolve_path(str(whisper_config.get("cache_dir", ".cache/whisper")), config_file.parent)

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
        elif action == MENU_RENAMER:
            run_renamer_settings_ui(inquirer, config_file, data)
        elif action == MENU_DELETE_WEIGHTS:
            run_delete_whisper_weights_ui(inquirer, current_cache_dir)

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
