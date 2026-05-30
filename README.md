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

## Use Case 3 — Извлечение сцены и дубля через LLM (scene_parser.py)

Модуль `scene_parser.py` отправляет транскрибированный текст в LLM через API OpenRouter, извлекает структурированную информацию и валидирует её по Pydantic-схеме `SceneTake`.

Перед запуском убедитесь, что в файле `.env` указан валидный ключ API:
```dotenv
OPENROUTER_API_KEY=ваш_ключ_openrouter_здесь
```

Настройки в `config.yaml`, секция `scene_parser`:
* `model` — модель на OpenRouter (по умолчанию `google/gemini-3.1-flash-lite`).
* `scene_description` и `take_description` — текстовые инструкции, передаваемые в LLM для гибкой настройки формата извлечения сцены и дубля.

Запуск модуля в терминале напрямую:
```bash
# базовый запуск (human-readable вывод)
uv run python scene_parser.py "сцена тридцать два дубль пять"

# JSON-вывод (удобно для скриптов)
uv run python scene_parser.py "сцена тридцать два дубль пять" --json
```

---

## Use Case 4 — Сквозной пайплайн обработки (pipeline.py)

Сводный оркестратор пайплайна `pipeline.py` последовательно прогоняет аудиофайл через все этапы и выводит единый структурированный результат.

**Как это работает:**
1. Приведение аудиофайла к стандарту в памяти (`preprocessor.py`).
2. Поиск хлопка хлопушки в файле (`clapper.py`).
3. Если хлопок найден — обрезка аудио сразу после него (`cutter.py`). Если хлопок не обнаружен — транскрипция выполняется на полной длине аудиофайла.
4. Распознавание речи на основе полученного аудиофрагмента (`whisper.py`).
5. Извлечение структурированной сцены и дубля через LLM (`scene_parser.py`).

Запуск пайплайна:
```bash
# запуск с красивым текстовым выводом всей метаинформации
uv run python pipeline.py claps/тест.wav

# вывод результатов в формате JSON
uv run python pipeline.py claps/тест.wav --json

# запуск с сохранением отчета в файл (создаст claps/тест_result.json)
uv run python pipeline.py claps/тест.wav --save
```

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

# только тесты scene_parser
uv run pytest tests/test_scene_parser.py -v

# только тесты pipeline
uv run pytest tests/test_pipeline.py -v
```

### Что покрыто тестами

| Модуль | Файл тестов | Что тестируется |
|---|---|---|
| `clapper.py` | `tests/test_clapper.py` | загрузка конфига, ранжирование top-matches, analyze_audio с fake-моделью, CLI |
| `whisper.py` | `tests/test_whisper.py` | загрузка конфига, транскрибация через StandardizedAudio и CutterResult |
| `preprocessor.py` | `tests/test_preprocessor.py` | загрузка конфига, standardize_audio, decode_standardized_audio |
| `cutter.py` | `tests/test_cutter.py` | выбор лучшего хита, обрезка через ffmpeg, ошибка на пустых хитах |
| `scene_parser.py` | `tests/test_scene_parser.py` | загрузка конфига, Pydantic-валидация, parse_scene с моками, CLI |
| `pipeline.py` | `tests/test_pipeline.py` | сквозной пайплайн с моками (с хлопками и без), CLI, сохранение результатов |

---

## Структура проекта

```
.
├── clapper.py          # CLAP-модель детекции хлопушки
├── whisper.py           # Транскрибация речи (Whisper)
├── preprocessor.py      # Стандартизация аудио → AAC in-memory
├── cutter.py            # Обрезка аудио по метке хлопушки
├── scene_parser.py      # Извлечение сцены и дубля через LLM
├── pipeline.py          # Сквозной оркестратор пайплайна
├── config.yaml          # Единый конфиг для всех модулей
├── .env                 # Настройки ключей API (не коммитится)
├── pyproject.toml       # Зависимости (uv)
├── uv.lock
├── claps/               # Тестовые аудиофайлы
├── prompts/             # Шаблоны промптов для LLM
│   └── scene_parser.md  # Промпт для извлечения сцены/дубля
└── tests/
    ├── conftest.py
    ├── test_clapper.py
    ├── test_cutter.py
    ├── test_preprocessor.py
    ├── test_whisper.py
    ├── test_scene_parser.py
    └── test_pipeline.py
```

---

## Известные нюансы

- Модели скачиваются в `.cache/` внутри проекта (см. `config.yaml`).
- Обработка идёт **по одному файлу** за раз. Для пакетной обработки планируется bash-скрипт.
