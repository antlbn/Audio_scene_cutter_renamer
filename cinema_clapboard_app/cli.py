"""Thin CLI entry point layer for cinema_clapboard to avoid startup latency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config_utils import get_user_config_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Единый оркестратор cinema_clapboard.\n"
            "Принимает один файл или директорию.\n"
            "При директории — показывает TUI для выбора файлов."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Путь к аудио-файлу или директории с аудио-файлами. (Опционально, если используется --settings)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=get_user_config_path(),
        help="Путь к config.yaml.",
    )
    parser.add_argument(
        "--settings",
        "-settings",
        action="store_true",
        help="Открыть интерактивный UI для настройки программы.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести сырой JSON вместо человекочитаемого резюме.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить результат в {имя_файла}_result.json рядом с файлом.",
    )
    parser.add_argument(
        "--rename",
        "-rename",
        action="store_true",
        help="Переименовать исходный файл по схеме sequence/shot/take.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.settings:
        from . import settings
        settings.main(["--config", str(args.config)])
        return 0

    if args.path is None:
        print("Ошибка: путь не указан. Укажите файл/папку или используйте --settings", file=sys.stderr)
        return 1

    from . import batch_runner
    return batch_runner.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
