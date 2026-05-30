# Audio Scene Cutter & Renamer

Утилита для автоматической обработки аудиофайлов со съёмочной площадки:
распознавание хлопушки, транскрибация речи, парсинг номера сцены/дубля через LLM.

---

## Требования

- **Python** ≥ 3.12
- **uv** — менеджер зависимостей ([установка](https://docs.astral.sh/uv/getting-started/installation/))
- **ffmpeg** — для декодирования AAC/M4A и работы `preprocessor.py`
- *(опционально)* **CUDA** — ускорение моделей CLAP / Whisper на GPU

---

## Установка

```bash
# Клонируем и ставим зависимости
cd Audio_scene_cutter_renamer
uv sync
```

### Hugging Face токен

Модели CLAP и Whisper скачиваются с Hugging Face.
Скопируйте `.env` и вставьте свой токен:

```bash
cp .env .env.local   # или отредактируйте .env напрямую
```

```dotenv
HUGGINGFACE_HUB_TOKEN=hf_ваш_токен
```

---

## Use Case 1 — Распознавание хлопушки на одном файле

Модуль `clapper.py` использует CLAP-модель (text ↔ audio similarity) для поиска хлопушки.
При первом запуске модель скачивается (~600 MB) в `.cache/clap`.

```bash
# базовый запуск — вывод в человекочитаемом формате
uv run python clapper.py claps/тест.wav

# вывод JSON
uv run python clapper.py claps/тест.wav --json

# debug — показывает каждый hit в stderr
uv run python clapper.py claps/тест.wav --debug

# свой конфиг
uv run python clapper.py claps/тест.wav --config config.yaml --debug --json
```

**Пример вывода (human-readable):**
```
🎬 Clapper summary
📁 File: тест.wav
🎚 Threshold: 0.280
🎯 Hits: 1
  1. ⏱ 2.50s | score 0.421 | sharp clap sound
     🥇 sharp clap sound 0.421, loud click 0.310
```

Настройки в `config.yaml`, секция `clapper`:
- `audio_model.threshold` — минимальный score (по умолчанию `0.28`)
- `text_search.text_keys` — текстовые ключи для CLAP-сопоставления
- `clip.post_hit_seconds` — сколько секунд оставить после хлопка

---

## Use Case 2 — Распознавание речи на одном файле

```bash
# транскрибация (по умолчанию язык ru из config.yaml)
uv run python whisper.py claps/тест.wav

# JSON-вывод
uv run python whisper.py claps/тест.wav --json

# другой язык
uv run python whisper.py claps/тест.wav --language en

# перевод на английский (Whisper автоматом переводит на en)
uv run python whisper.py claps/тест.wav --task translate

# перевод + JSON
uv run python whisper.py claps/тест.wav --task translate --json
```

**Пример вывода (human-readable):**
```
🗣 Whisper summary
📁 File: тест.wav
🧠 Model: openai/whisper-small
🌐 Language: ru
🧾 Task: transcribe
🕒 Sample rate: 16000
📝 Text: сцена три дубль два
🧩 Chunks: 1
  1. (0.0, 3.2) |  сцена три дубль два
```

**Флаги:**

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--config` | `config.yaml` | Путь к конфигу |
| `--json` | выкл | Вывод сырого JSON |
| `--language` | `ru` (из конфига) | Язык распознавания |
| `--task` | `transcribe` | `transcribe` или `translate` (перевод → en) |

Настройки в `config.yaml`, секция `whisper`:
- `model_id` — модель Whisper (`openai/whisper-small` по умолчанию)
- `cache_dir` — куда скачивать модель (`.cache/whisper`)
- `device` — `auto` / `cpu` / `cuda`

---

## Тесты

Все тесты работают на моках — **скачивание моделей не требуется**.

```bash
# запуск всех тестов
uv run pytest tests/ -v

# только тесты clapper-детекции
uv run pytest tests/test_clapper.py -v

# только тесты whisper
uv run pytest tests/test_whisper.py -v

# только тесты preprocessor
uv run pytest tests/test_preprocessor.py -v

# только тесты cutter
uv run pytest tests/test_cutter.py -v
```

### Что покрыто тестами

| Модуль | Файл тестов | Что тестируется |
|---|---|---|
| `clapper.py` | `tests/test_clapper.py` | загрузка конфига, ранжирование top-matches, analyze_audio с fake-моделью, CLI |
| `whisper.py` | `tests/test_whisper.py` | загрузка конфига, транскрибация через StandardizedAudio и CutterResult |
| `preprocessor.py` | `tests/test_preprocessor.py` | загрузка конфига, standardize_audio, decode_standardized_audio |
| `cutter.py` | `tests/test_cutter.py` | выбор лучшего хита, обрезка через ffmpeg, ошибка на пустых хитах |

---

## Структура проекта

```
.
├── clapper.py          # CLAP-модель детекции хлопушки
├── whisper.py           # Транскрибация речи (Whisper)
├── preprocessor.py      # Стандартизация аудио → AAC in-memory
├── cutter.py            # Обрезка аудио по метке хлопушки
├── config.yaml          # Единый конфиг для всех модулей
├── .env                 # HF-токен (не коммитится)
├── pyproject.toml       # Зависимости (uv)
├── uv.lock
├── claps/               # Тестовые аудиофайлы
└── tests/
    ├── conftest.py
    ├── test_clapper.py
    ├── test_cutter.py
    ├── test_preprocessor.py
    └── test_whisper.py
```

---

## Известные нюансы

- Модели скачиваются в `.cache/` внутри проекта (см. `config.yaml`).
- Обработка идёт **по одному файлу** за раз. Для пакетной обработки планируется bash-скрипт.
