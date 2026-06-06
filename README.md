# cinema-clapboard

CLI-инструмент для звукорежиссёров на съёмочной площадке.
Слышит хлопушку, транскрибирует речь актёра, распознаёт сцену и дубль — и переименовывает файл.

Работает на macOS, Linux, Windows. Модели и ffmpeg скачиваются сами при первом запуске.

---

## Что умеет

- 🎬 Находит хлопок в аудиофайле (CLAP-модель)
- ✂️ Обрезает запись до нужного момента
- 🗣 Транскрибирует речь (Whisper, локально, без облака)
- 🧠 Извлекает номер сцены и дубля через LLM (OpenRouter)
- 📁 Переименовывает файл по шаблону: `Sequence_1_shot_3_dubl_2.wav`
- 🗂 Пакетная обработка папки с TUI-выбором файлов

Поддерживаемые форматы: `.wav .mp3 .flac .aac .m4a .ogg .aif .aiff .bwf .rf64`

---

## Быстрый старт — `uv tool` (рекомендуется)

Самый простой способ: одна команда, ничего клонировать не нужно.

### 1. Установить uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> После установки перезапусти терминал.

### 2. Установить cinema-clapboard

```bash
uv tool install git+https://github.com/ВАШ_НИК/Audio_scene_cutter_renamer
```

`uv` сам скачает Python ≥ 3.12 если его нет, создаст изолированное окружение и зарегистрирует команду `cinema-clapboard` в PATH.

> **Windows:** если команда не появилась — выполни `uv tool update-shell` и перезапусти терминал.

### 3. Добавить API-ключи

При первом запуске создаётся `~/.cinema_clapboard/.env` — открой его в любом редакторе:

```dotenv
OPENROUTER_API_KEY=ваш_ключ      # нужен для LLM-парсинга сцены/дубля
HUGGINGFACE_HUB_TOKEN=hf_токен   # нужен для скачивания CLAP и Whisper
```

Или через интерактивное меню:
```bash
cinema-clapboard --settings
```

### 4. Запустить

```bash
# Один файл
cinema-clapboard audio.wav

# Один файл — обработать и переименовать
cinema-clapboard audio.wav --rename

# Папка — откроется TUI с выбором файлов
cinema-clapboard ./recordings/ --rename
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

> CLAP (~600 MB) и Whisper скачиваются при первом запуске. ffmpeg входит в зависимости и не требует ручной установки.

---

## Альтернатива — запустить из исходников

Для тех, кто хочет потрогать код или внести правки.

```bash
git clone https://github.com/ВАШ_НИК/Audio_scene_cutter_renamer
cd Audio_scene_cutter_renamer
uv run cinema-clapboard audio.wav
```

`uv run` сам создаёт venv и ставит зависимости при первом запуске.
Конфиг и ключи — те же `~/.cinema_clapboard/`, что и при установке через `uv tool`.

---

## Настройки

Конфиг хранится в `~/.cinema_clapboard/` и создаётся автоматически при первом запуске — независимо от способа установки.

```bash
# Интерактивный редактор настроек
cinema-clapboard --settings
```

Что можно настроить:
- **Язык и модель Whisper** — `fr`, `en`, `ru`, … ; `whisper-tiny` … `whisper-large-v3`
- **Режим Whisper** — транскрипция или перевод в английский
- **Шаблон переименования** — например `Sequence_{sequence}_shot_{shot}_dubl_{take}`
- **LLM API Key** — установить или сменить ключ OpenRouter

Или вручную:
```bash
nano ~/.cinema_clapboard/config.yaml
nano ~/.cinema_clapboard/.env
```

---

## Документация

- [`decisions.md`](decisions.md) — архитектурные решения и их обоснование
- `HANDOVER.md` *(в разработке)* — структура проекта, модули, тесты, как контрибьютить
