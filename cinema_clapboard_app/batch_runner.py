"""Единый CLI-оркестратор для cinema_clapboard.

Обрабатывает два use case:
  - Один файл:     сразу запускает пайплайн без TUI.
  - Директория:    показывает TUI (checkbox-список файлов),
                   предварительно выводит текущие настройки,
                   спрашивает подтверждение, затем обрабатывает по очереди.

В обоих случаях CLAP и Whisper загружаются один раз на весь процесс
за счёт _RUNTIME_CACHE внутри clapper.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from . import pipeline
from . import presentator_cli
from .config_utils import get_user_config_path

LOGGER = logging.getLogger("batch_runner")

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".flac", ".aac",
    ".m4a", ".ogg", ".aif", ".aiff",
    ".bwf", ".rf64",
})

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────


def _scan_audio_files(directory: Path) -> list[Path]:
    """Рекурсивно найти все аудио-файлы в директории, отсортировать."""
    found = [
        p for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return found


def _format_duration(seconds: float) -> str:
    """Форматировать секунды в MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _format_size(size_bytes: int) -> str:
    """Форматировать байты в человекочитаемый размер."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024 or unit == "GB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


def _file_size_str(path: Path) -> str:
    try:
        return _format_size(path.stat().st_size)
    except OSError:
        return "?"


def _load_renamer_info(config_path: Path) -> dict[str, str | bool]:
    """Загрузить renamer-секцию из config.yaml для отображения настроек."""
    if not config_path.exists():
        return {}
    try:
        loaded = yaml.safe_load(config_path.read_text()) or {}
        return loaded.get("renamer", {}) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _print_settings_block(
    *,
    rename: bool,
    save_json: bool,
    config_path: Path,
) -> None:
    """Вывести блок текущих настроек перед TUI."""
    renamer = _load_renamer_info(config_path)
    template = renamer.get("naming_template", "Sequence_{sequence}_shot_{shot}_take_{take}")
    suffix = renamer.get("suffix", "")
    extract = renamer.get("extract_from_source", False)

    rename_status = "ВКЛ ✓" if rename else "ВЫКЛ ✗"
    json_status = "ВКЛ ✓" if save_json else "ВЫКЛ ✗"

    print()
    print("─" * 56)
    print("  Настройки запуска:")
    print(f"  • Переименование:   {rename_status}")
    if rename:
        print(f"    Шаблон:          {template}")
        if suffix:
            print(f"    Суффикс:         {suffix}")
        if extract:
            pattern = renamer.get("extract_pattern", "")
            fmt = renamer.get("extract_format", "")
            print(f"    Извлечь из имени: {pattern!r} → {fmt!r}")
    print(f"  • Сохранять JSON:   {json_status}")
    print("─" * 56)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Обработка одного файла
# ─────────────────────────────────────────────────────────────────────────────


def _run_single(
    audio_path: Path,
    *,
    config_path: Path,
    rename: bool,
    json_output: bool,
    save_json: bool,
) -> bool:
    """Запустить пайплайн на одном файле. Вернуть True при успехе."""
    try:
        result = pipeline.run_pipeline(
            audio_path,
            config_path=config_path,
            rename=rename,
        )

        presentator_cli.present(
            result.model_dump(),
            use_case="name_extractor",
            json_output=json_output,
        )

        if save_json:
            parent_dir = audio_path.parent
            result_json_name = f"{Path(result.new_name).stem}_result.json"
            output_path = parent_dir / result_json_name
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"\n💾 Результат сохранён: {output_path}", file=sys.stderr)

        return True

    except Exception as exc:
        LOGGER.exception("Ошибка при обработке %s: %s", audio_path.name, exc)
        print(f"❌ {audio_path.name} — ошибка: {exc}", file=sys.stderr)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Batch-режим: TUI + цикл
# ─────────────────────────────────────────────────────────────────────────────


def _select_files_tui(files: list[Path], base_dir: Path) -> list[Path] | None:
    """Показать InquirerPy checkbox-список. Вернуть выбранные файлы или None при отмене."""
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        print(
            "InquirerPy не установлен. Запусти `uv sync` или `uv pip install InquirerPy`.",
            file=sys.stderr,
        )
        return None

    choices = []
    for f in files:
        try:
            rel = f.relative_to(base_dir)
        except ValueError:
            rel = f
        size_str = _file_size_str(f)
        label = f"{rel}    ({size_str})"
        choices.append(Choice(value=f, name=label, enabled=True))

    try:
        selected: list[Path] = inquirer.checkbox(
            message="Выберите файлы для обработки:",
            choices=choices,
            instruction="(Space — снять/поставить · A — все · I — инвертировать · Enter — запустить)",
            qmark="",
            amark="",
            pointer="❯",
            enabled_symbol="◉",
            disabled_symbol="◯",
        ).execute()
    except KeyboardInterrupt:
        return None

    return selected


def _run_batch(
    directory: Path,
    *,
    config_path: Path,
    rename: bool,
    json_output: bool,
    save_json: bool,
) -> int:
    """Batch-режим: сканировать директорию, TUI, обработать выбранное. Вернуть exit code."""
    files = _scan_audio_files(directory)

    print(f"\n🎬  cinema_clapboard — batch mode", flush=True)
    print(f"📁  {directory.resolve()}", flush=True)

    if not files:
        print(
            f"\n⚠️  Аудио-файлов не найдено в {directory}\n"
            f"   Поддерживаемые форматы: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )
        return 1

    print(f"   Найдено файлов: {len(files)}")

    _print_settings_block(rename=rename, save_json=save_json, config_path=config_path)

    selected = _select_files_tui(files, base_dir=directory)

    if not selected:
        print("\nОтменено.", file=sys.stderr)
        return 130

    print(f"\nВыбрано: {len(selected)} из {len(files)} файлов.")

    # Подтверждение y/N
    try:
        answer = input("Начать обработку? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nОтменено.", file=sys.stderr)
        return 130

    if answer not in ("y", "yes", "д", "да"):
        print("Отменено.", file=sys.stderr)
        return 130

    print()

    # ── Цикл обработки ──────────────────────────────────────────────────────
    total = len(selected)
    success_count = 0
    error_count = 0

    for index, audio_path in enumerate(selected, start=1):
        print(f"[{index}/{total}] 🔄 {audio_path.name}")
        ok = _run_single(
            audio_path,
            config_path=config_path,
            rename=rename,
            json_output=json_output,
            save_json=save_json,
        )
        if ok:
            success_count += 1
        else:
            error_count += 1
            print(f"       ↳ пропущен, продолжаем\n", file=sys.stderr)

    # ── Финальная сводка ────────────────────────────────────────────────────
    print()
    print("─" * 56)
    print(f"  Итог: ✅ {success_count} успешно" + (f" / ❌ {error_count} ошибка(и)" if error_count else ""))
    print("─" * 56)

    return 0 if error_count == 0 else 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.settings:
        from . import settings
        # settings.py main() doesn't return anything or return 0, we can just call it
        settings.main()
        return 0

    path: Path | None = args.path
    config_path: Path = args.config

    if path is None:
        print("Ошибка: путь не указан. Укажите файл/папку или используйте --settings", file=sys.stderr)
        return 1

    if not path.exists():
        print(f"Ошибка: путь не существует: {path}", file=sys.stderr)
        return 1

    if path.is_file():
        # ── Один файл: сразу пайплайн ───────────────────────────────────────
        ok = _run_single(
            path,
            config_path=config_path,
            rename=args.rename,
            json_output=args.json,
            save_json=args.save,
        )
        return 0 if ok else 1

    if path.is_dir():
        # ── Директория: batch-режим с TUI ───────────────────────────────────
        return _run_batch(
            path,
            config_path=config_path,
            rename=args.rename,
            json_output=args.json,
            save_json=args.save,
        )

    print(f"Ошибка: {path} — не файл и не директория.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
