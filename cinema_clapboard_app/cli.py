"""Thin CLI entry point layer for cinema_clapboard to avoid startup latency."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config_utils import get_user_config_path


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: whisper
# ─────────────────────────────────────────────────────────────────────────────


def _build_whisper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinema-clapboard whisper",
        description=(
            "Прогнать аудиофайл через Whisper.\n"
            "Препроцессор перекодирует файл в нужный формат автоматически."
        ),
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Путь к аудиофайлу.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=get_user_config_path(),
        help="Путь к config.yaml.",
    )
    parser.add_argument(
        "--language",
        "-language",
        type=str,
        default=None,
        metavar="CODE",
        help="Форсить язык источника (например: fr, de, ru). "
             "Передаётся в Whisper как language hint.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Форсить режим translate — Whisper переводит в English.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести сырой JSON вместо человекочитаемого резюме.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить результат в {stem}_whisper.json рядом с файлом.",
    )
    parser.add_argument(
        "--no-chunks",
        dest="no_chunks",
        action="store_true",
        help="Не показывать список временны́х чанков в выводе.",
    )
    return parser


def _whisper_main(argv: list[str]) -> int:
    """Тонкий оркестратор: парсит аргументы и делегирует в whisper.py."""
    parser = _build_whisper_parser()
    args = parser.parse_args(argv)

    audio_path = args.audio_file
    if not audio_path.exists():
        print(f"Ошибка: файл не найден: {audio_path}", file=sys.stderr)
        return 1

    # Импортируем тяжёлые зависимости только здесь — после парсинга аргументов.
    from . import whisper as whisper_mod

    config = whisper_mod.load_config(args.config)

    if args.language is not None:
        config.language = args.language
    if args.translate:
        config.task = "translate"

    try:
        result = whisper_mod.transcribe_audio(audio_path, config=config)
    except Exception as exc:
        print(f"❌ Whisper завершился с ошибкой: {exc}", file=sys.stderr)
        return 1

    # ── Вывод результата ────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(whisper_mod.render_result(result, show_chunks=not args.no_chunks))

    # ── Сохранение JSON ─────────────────────────────────────────────────────
    if args.save:
        output_path = audio_path.parent / f"{audio_path.stem}_whisper.json"
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n💾 Сохранено: {output_path}", file=sys.stderr)

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Основной парсер (полный пайплайн)
# ─────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Единый оркестратор cinema_clapboard.\n"
            "Принимает один файл или директорию.\n"
            "При директории — показывает TUI для выбора файлов.\n"
            "\n"
            "Subcommands:\n"
            "  whisper <file>   Транскрибировать файл через Whisper (без CLAP/LLM)."
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
        "--doctor",
        action="store_true",
        help="Показать сведения об окружении, PyTorch и доступности CUDA.",
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
    raw = list(argv) if argv is not None else sys.argv[1:]

    # ── Subcommand routing ───────────────────────────────────────────────────
    # Если первый позиционный аргумент — известный subcommand, делегируем.
    if raw and raw[0] == "whisper":
        return _whisper_main(raw[1:])

    # ── Полный пайплайн (существующее поведение) ─────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args(raw)

    if args.settings:
        from . import settings
        settings.main(["--config", str(args.config)])
        return 0

    if args.doctor:
        import torch

        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        return 0

    if args.path is None:
        print("Ошибка: путь не указан. Укажите файл/папку или используйте --settings/--doctor", file=sys.stderr)
        return 1

    from . import batch_runner
    return batch_runner.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
