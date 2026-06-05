# Audio Scene Cutter & Renamer — `cinema_clapboard`

Утилита для автоматической обработки аудиофайлов со съёмочной площадки:
распознавание хлопушки (CLAP), транскрибация речи (Whisper), парсинг сцены/дубля через LLM.

---

## Установка

### Шаг 1 — Установить uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> После установки перезапусти терминал, чтобы `uv` появился в PATH.

---

### Шаг 2 — Установить cinema-clapboard

```bash
uv tool install git+https://github.com/ВАШ_НИК/Audio_scene_cutter_renamer
```

Это единственная команда, которая нужна пользователю. `uv` сам:
- скачает Python нужной версии (≥ 3.12) если его нет
- создаст изолированное окружение
- поставит все зависимости включая CLAP, Whisper, PyTorch
- зарегистрирует команду `cinema-clapboard` в PATH

> **Windows:** если `cinema-clapboard` не появился в CMD/PowerShell сразу,
> выполни `uv tool update-shell` и перезапусти терминал.

---

### ffmpeg

**Устанавливается автоматически** — пакет `static-ffmpeg` входит в зависимости
и при первом запуске скачивает статический бинарник ffmpeg для твоей платформы.
Ничего дополнительно делать не нужно.

> Если на машине уже установлен системный ffmpeg — он будет использован вместо bundled.

---

## Настройка API-ключей

Конфиг и ключи хранятся в `~/.cinema_clapboard/` (создаётся автоматически при первом запуске).

### OpenRouter API Key — нужен для LLM-парсинга сцены/дубля

```bash
cinema-clapboard --settings
# → выбрать "LLM API Key Manager" → "Set / Edit OpenRouter API Key"
```

Или вручную в файл `~/.cinema_clapboard/.env`:
```dotenv
OPENROUTER_API_KEY=ваш_ключ_openrouter_здесь
```

### Hugging Face Token — нужен для скачивания CLAP и Whisper

В тот же файл `~/.cinema_clapboard/.env`:
```dotenv
HUGGINGFACE_HUB_TOKEN=hf_ваш_токен
```

---

## Быстрый старт

```bash
# Обработать один файл (распознать хлопок → обрезать → транскрибировать → распарсить)
cinema-clapboard audio.wav

# Обработать и переименовать
cinema-clapboard audio.wav --rename

# Обработать папку — откроется TUI с выбором файлов
cinema-clapboard ./recordings/ --rename

# Настройки
cinema-clapboard --settings
```

---

## Use Case 1 — Полный пайплайн

Оркестратор последовательно прогоняет файл через все этапы:

1. **Preprocessor** — стандартизация аудио в памяти.
2. **Clapper** — поиск хлопка (CLAP-модель, ~600 MB, скачивается при первом запуске).
3. **Cutter** — обрезка после хлопка. Если хлопок не найден — транскрибируется весь файл.
4. **Whisper** — распознавание речи.
5. **Scene Parser** — извлечение сцены/дубля через LLM.

```bash
# Базовый запуск
cinema-clapboard claps/тест.wav

# С переименованием
cinema-clapboard claps/тест.wav --rename

# Сохранить результат в JSON рядом с файлом
cinema-clapboard claps/тест.wav --save

# Вывод сырого JSON (удобно для скриптов)
cinema-clapboard claps/тест.wav --json
```

**Пример вывода:**
```
🎬  cinema_clapboard
📁  тест.wav
────────────────────────────────────────────────────────
🎚  Clapper     hits: 1 | best: 2.50s (score 0.421)
✂️   Cut         2.50s → 7.30s
🗣  Whisper     сцена три дубль два  [fr | transcribe | whisper-small]
🧠  LLM         seq=None  shot=3  take=2  [gemini-3.1-flash-lite]
📝  New name    Sequence_None_shot_3_dubl_2.wav
────────────────────────────────────────────────────────
```

---

## Use Case 2 — Пакетная обработка (batch mode)

```bash
# Указать директорию — откроется TUI с выбором файлов
cinema-clapboard ./recordings/

# С переименованием и сохранением JSON для каждого файла
cinema-clapboard ./recordings/ --rename --save
```

При запуске на директории программа:
1. Рекурсивно находит все аудиофайлы (`.wav .mp3 .flac .aac .m4a .ogg .aif .aiff .bwf .rf64`).
2. Показывает TUI-список с чекбоксами (Space — снять/поставить, A — все, Enter — запустить).
3. Спрашивает подтверждение.
4. Обрабатывает файлы по очереди; ошибка на одном файле не останавливает остальные.
5. Выводит итоговую сводку.

> **Важно:** CLAP (~600 MB) и Whisper загружаются ровно **один раз** на всю сессию —
> в отличие от bash-loop, который запускал бы новый процесс на каждый файл.

---

## Use Case 3 — Настройки (`--settings`)

```bash
cinema-clapboard --settings
```

Открывает интерактивное меню:
- **Whisper language** — язык аудиодорожки (`fr`, `en`, `ru`, …).
- **Whisper mode** — `transcribe` (транскрипция) или `translate` (перевод → en).
- **Whisper model** — локальная модель (`whisper-tiny` … `whisper-large-v3`).
- **Rename rendering** — шаблон и суффикс имени файла.
- **LLM API Key Manager** — установить/сменить ключ OpenRouter.
- **Delete downloaded Whisper weights** — удалить скачанные веса.

Настройки сохраняются в `~/.cinema_clapboard/config.yaml`.

---

## Переименование файлов

По умолчанию файл переименовывается по шаблону из конфига:
`Sequence_{sequence}_shot_{shot}_dubl_{take}`.

Часто нужно сохранить информацию из оригинального имени (например, номер канала ZOOM).

### Примеры конфигураций

**ZOOM-рекордер** (извлечение номера канала):
```yaml
renamer:
  naming_template: "Sequence_{sequence}_shot_{shot}_dubl_{take}"
  suffix: ""
  extract_from_source: true
  # Вытягивает "Tr3" из "ZOOM0746_Tr3 [2026-05-28 152751].wav"
  extract_pattern: "_([Tt]r\\d+)"
  extract_format: "_{match}"
```

| Исходный файл | → | Новое имя |
|---|---|---|
| `ZOOM0746_Tr3 [2026-05-28 152751].wav` | → | `Sequence_1_shot_3_dubl_5_Tr3.wav` |

**Число после второго подчёркивания:**
```yaml
renamer:
  extract_from_source: true
  # Вытягивает "12" из "260603_0001_12.wav"
  extract_pattern: "^[^_]*_[^_]*_(.+?)$"
  extract_format: "_TR{match}"
```

| Исходный файл | → | Новое имя |
|---|---|---|
| `260603_0001_12.wav` | → | `Sequence_1_shot_3_dubl_5_TR12.wav` |

### Параметры renamer в config.yaml

| Параметр | Описание |
|---|---|
| `naming_template` | Шаблон с плейсхолдерами `{sequence}`, `{shot}`, `{take}` |
| `suffix` | Фиксированный суффикс после шаблона |
| `extract_from_source` | `true` / `false` — включить извлечение из исходного имени |
| `extract_pattern` | Regex-паттерн для извлечения |
| `extract_format` | Формат: `{match}` = найденное значение |

---

## Для разработчиков

### Клонировать и запустить из исходников

```bash
git clone https://github.com/ВАШ_НИК/Audio_scene_cutter_renamer
cd Audio_scene_cutter_renamer
uv sync
uv pip install -e .

# Запуск
cinema-clapboard audio.wav
# или через uv run:
uv run cinema-clapboard audio.wav
```

### Тесты

Все тесты работают на моках — **скачивание моделей не требуется**.

```bash
# Запустить все тесты
uv run pytest tests/ -v

# По модулям
uv run pytest tests/test_clapper.py -v
uv run pytest tests/test_whisper.py -v
uv run pytest tests/test_preprocessor.py -v
uv run pytest tests/test_cutter.py -v
uv run pytest tests/test_scene_parser.py -v
uv run pytest tests/test_pipeline.py -v
uv run pytest tests/test_renamer.py -v
uv run pytest tests/test_settings.py -v
```

| Модуль | Файл тестов | Что тестируется |
|---|---|---|
| `clapper.py` | `test_clapper.py` | загрузка конфига, ранжирование, analyze_audio с fake-моделью, CLI |
| `whisper.py` | `test_whisper.py` | загрузка конфига, транскрибация через StandardizedAudio и CutterResult |
| `preprocessor.py` | `test_preprocessor.py` | загрузка конфига, standardize_audio, decode_standardized_audio |
| `cutter.py` | `test_cutter.py` | выбор лучшего хита, обрезка через ffmpeg, ошибка на пустых хитах |
| `scene_parser.py` | `test_scene_parser.py` | загрузка конфига, Pydantic-валидация, parse_scene с моками, CLI |
| `pipeline.py` | `test_pipeline.py` | сквозной пайплайн с моками (с хлопком и без), CLI, сохранение результатов |
| `pipeline.py` (renamer) | `test_renamer.py` | переименование файлов, конфликты имён, извлечение из оригинального имени |
| `settings.py` | `test_settings.py` | Settings UI, менеджер ключей, удаление весов |

---

## Структура проекта

```
.
├── cinema_clapboard_app/       # Основной пакет
│   ├── clapper.py              # CLAP-детекция хлопушки
│   ├── whisper.py              # Транскрибация речи (Whisper)
│   ├── preprocessor.py         # Стандартизация аудио → AAC in-memory
│   ├── cutter.py               # Обрезка аудио по метке хлопушки
│   ├── scene_parser.py         # Парсинг сцены/дубля через LLM
│   ├── pipeline.py             # Оркестратор одного файла (lib)
│   ├── batch_runner.py         # CLI-оркестратор (файл / директория)
│   ├── cli.py                  # Точка входа: cinema-clapboard
│   ├── settings.py             # Интерактивный Settings UI (InquirerPy)
│   ├── presentator_cli.py      # Форматирование результата в терминал
│   ├── config_utils.py         # Общие утилиты: конфиг, env, пути
│   ├── default_config.yaml     # Дефолтный конфиг (копируется в ~/.cinema_clapboard/)
│   ├── default_env             # Дефолтный .env (копируется в ~/.cinema_clapboard/)
│   └── prompts/
│       └── scene_parser.md     # Промпт для LLM-парсинга
├── tests/                      # Тесты (все на моках)
├── pyproject.toml              # Зависимости и точка входа (uv/hatch)
└── uv.lock
```

**Конфиг пользователя** (`~/.cinema_clapboard/`):
- `config.yaml` — настройки (создаётся автоматически из `default_config.yaml`)
- `.env` — API-ключи (создаётся автоматически из `default_env`)

---

## Известные нюансы

- Модели CLAP (~600 MB) и Whisper скачиваются при первом запуске.
- ffmpeg (через `static-ffmpeg`) скачивается автоматически при первом запуске если нет системного.
- При batch-режиме модели загружаются один раз на всю сессию.
- Поддерживаемые форматы: `.wav .mp3 .flac .aac .m4a .ogg .aif .aiff .bwf .rf64`.
