"""Interactive project settings editor for the terminal wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

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


def load_config_data(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        return {}

    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
    return data


def _choice_with_current(choices: list[str], current: str) -> list[str]:
    if current and current not in choices:
        return [current, *choices]
    return choices


def _replace_or_append_whisper_key(lines: list[str], key: str, value: str) -> list[str]:
    section_start = None
    section_end = len(lines)

    for index, line in enumerate(lines):
        if line.startswith("whisper:"):
            section_start = index
            break

    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["whisper:", f"  {key}: {value}"])
        return lines

    for index in range(section_start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "\t")) and line.rstrip().endswith(":"):
            section_end = index
            break

    prefix = f"  {key}:"
    for index in range(section_start + 1, section_end):
        stripped = lines[index].lstrip()
        if stripped.startswith(f"{key}:"):
            indent = lines[index][: len(lines[index]) - len(stripped)] or "  "
            comment = ""
            if " #" in lines[index]:
                comment = " #" + lines[index].split(" #", 1)[1]
            lines[index] = f"{indent}{key}: {value}{comment}"
            return lines

    insert_at = section_end
    lines.insert(insert_at, f"{prefix} {value}")
    return lines


def save_whisper_settings(
    config_path: str | Path,
    *,
    language: str,
    task: str,
    model_id: str,
) -> None:
    config_file = Path(config_path)
    lines = config_file.read_text(encoding="utf-8").splitlines() if config_file.exists() else []

    for key, value in (
        ("language", language),
        ("task", task),
        ("model_id", model_id),
    ):
        lines = _replace_or_append_whisper_key(lines, key, value)

    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if whisper_config and not isinstance(whisper_config, dict):
        raise ValueError(f"Invalid whisper section in {config_file}: expected a mapping")

    current_language = str(whisper_config.get("language", "french"))
    current_task = str(whisper_config.get("task", "transcribe"))
    current_model = str(whisper_config.get("model_id", "openai/whisper-small"))

    language = inquirer.select(
        message="Whisper audio language:",
        choices=_choice_with_current(WHISPER_LANGUAGE_CHOICES, current_language),
        default=current_language,
    ).execute()
    task = inquirer.select(
        message="Whisper mode:",
        choices=_choice_with_current(WHISPER_TASK_CHOICES, current_task),
        default=current_task,
    ).execute()
    model_id = inquirer.select(
        message="Whisper local model:",
        choices=_choice_with_current(WHISPER_MODEL_CHOICES, current_model),
        default=current_model,
    ).execute()

    save_whisper_settings(
        config_file,
        language=language,
        task=task,
        model_id=model_id,
    )

    print(f"Settings saved to {config_file}")
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
