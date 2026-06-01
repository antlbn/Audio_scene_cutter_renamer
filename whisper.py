"""Local Whisper transcription assembled from the notebook experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import cutter
import preprocessor
from config_utils import load_env_file, merge_config_dict, resolve_path


LOGGER = logging.getLogger("whisper")

DEFAULT_MODEL_ID = "openai/whisper-small"
DEFAULT_CACHE_DIR = ".cache/whisper"
DEFAULT_DEVICE = "auto"
DEFAULT_LANGUAGE = ""
DEFAULT_TASK = "transcribe"
DEFAULT_RETURN_TIMESTAMPS = True


@dataclass(slots=True)
class WhisperConfig:
    """Config for the Whisper transcription step."""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: str = DEFAULT_CACHE_DIR
    device: str = DEFAULT_DEVICE
    language: str = DEFAULT_LANGUAGE
    task: str = DEFAULT_TASK
    return_timestamps: bool = DEFAULT_RETURN_TIMESTAMPS


@dataclass(slots=True)
class WhisperDetection:
    """Minimal Whisper output intended for the LLM stage."""

    text: str
    model_id: str


@dataclass(slots=True)
class WhisperChunk:
    """One timestamped segment returned by Whisper."""

    timestamp: tuple[float | None, float | None] | None
    text: str


@dataclass(slots=True)
class WhisperResult:
    """Normalized transcription result ready for JSON export."""

    file_name: str
    detection: WhisperDetection
    chunks: list[WhisperChunk]
    language: str
    task: str
    sample_rate: int


@dataclass(slots=True)
class WhisperRuntime:
    """Loaded Whisper pipeline bundle."""

    asr_pipeline: Any
    device: torch.device


_RUNTIME_CACHE: dict[tuple[str, str, str], WhisperRuntime] = {}


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def _get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_config(config_path: str | Path | None = None) -> WhisperConfig:
    """Load the whisper section from config.yaml and apply defaults."""

    config_file = Path(config_path) if config_path is not None else _default_config_path()
    raw: dict[str, Any] = {}

    if config_file.exists():
        loaded = yaml.safe_load(config_file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
        raw = loaded.get("whisper", loaded)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid whisper section in {config_file}: expected a mapping")
    else:
        LOGGER.warning("Config file %s not found, using built-in defaults", config_file)

    defaults = {
        "model_id": DEFAULT_MODEL_ID,
        "cache_dir": DEFAULT_CACHE_DIR,
        "device": DEFAULT_DEVICE,
        "language": DEFAULT_LANGUAGE,
        "task": DEFAULT_TASK,
        "return_timestamps": DEFAULT_RETURN_TIMESTAMPS,
    }
    merged = merge_config_dict(defaults, raw)
    return WhisperConfig(
        model_id=str(merged["model_id"]),
        cache_dir=str(merged["cache_dir"]),
        device=str(merged["device"]),
        language=str(merged["language"]),
        task=str(merged["task"]),
        return_timestamps=bool(merged["return_timestamps"]),
    )




def _normalize_timestamp(value: Any) -> tuple[float | None, float | None] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    start, end = value
    return (
        None if start is None else float(start),
        None if end is None else float(end),
    )


def _normalize_chunks(raw_chunks: Any) -> list[WhisperChunk]:
    if not isinstance(raw_chunks, list):
        return []

    chunks: list[WhisperChunk] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        chunks.append(
            WhisperChunk(
                timestamp=_normalize_timestamp(chunk.get("timestamp")),
                text=str(chunk.get("text", "")),
            )
        )
    return chunks


def load_runtime(config: WhisperConfig, *, cache_base_dir: Path | None = None) -> WhisperRuntime:
    """Load and cache the Whisper pipeline."""

    cache_root = cache_base_dir or Path(__file__).resolve().parent
    cache_dir = resolve_path(config.cache_dir, cache_root)
    device = _get_device(config.device)
    cache_key = (config.model_id, str(cache_dir), str(device))

    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    load_env_file(Path(__file__).with_name(".env"))

    print(f"[whisper] loading model: {config.model_id}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(config.model_id, cache_dir=str(cache_dir))
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.model_id,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    pipeline_device = 0 if device.type == "cuda" else -1
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=pipeline_device,
    )
    print("[whisper] model loaded", file=sys.stderr)

    runtime = WhisperRuntime(asr_pipeline=asr_pipeline, device=device)
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _standardize_source(
    source: str | Path | preprocessor.StandardizedAudio | cutter.CutterResult,
) -> preprocessor.StandardizedAudio:
    if isinstance(source, preprocessor.StandardizedAudio):
        return source
    if isinstance(source, cutter.CutterResult):
        return source.standardized_audio
    return preprocessor.standardize_audio(source)


class ConsoleSpinner:
    """A premium console spinner to show progress on stderr."""

    def __init__(self, message: str = "Working"):
        self.message = message
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self.thread = None

    def _spin(self) -> None:
        idx = 0
        while self.running:
            frame = self.frames[idx % len(self.frames)]
            sys.stderr.write(f"\r\033[K[whisper] {frame} {self.message}")
            sys.stderr.flush()
            idx += 1
            time.sleep(0.08)

    def __enter__(self) -> ConsoleSpinner:
        if sys.stderr.isatty():
            self.running = True
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        else:
            print(f"[whisper] {self.message}...", file=sys.stderr, flush=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


def transcribe_audio(
    source: str | Path | preprocessor.StandardizedAudio | cutter.CutterResult,
    config: WhisperConfig | None = None,
    runtime: WhisperRuntime | None = None,
) -> WhisperResult:
    """Transcribe an audio source with Whisper."""

    config = config or load_config()
    runtime = runtime or load_runtime(config)

    standardized_audio = _standardize_source(source)
    waveform, sample_rate = preprocessor.decode_standardized_audio(
        standardized_audio,
        target_sample_rate=16_000,
    )
    audio_array = waveform.detach().cpu().numpy()

    with ConsoleSpinner(f"transcribing audio ({config.language})..."):
        raw_result = runtime.asr_pipeline(
            {"array": audio_array, "sampling_rate": sample_rate},
            return_timestamps=config.return_timestamps,
            generate_kwargs={
                "task": config.task,
                "language": config.language,
            },
        )

    text = str(raw_result.get("text", ""))
    chunks = _normalize_chunks(raw_result.get("chunks"))
    return WhisperResult(
        file_name=standardized_audio.file_name,
        detection=WhisperDetection(text=text, model_id=config.model_id),
        chunks=chunks,
        language=config.language,
        task=config.task,
        sample_rate=sample_rate,
    )


def render_result(result: WhisperResult) -> str:
    """Render a compact human-readable console summary."""

    lines = [
        "🗣 Whisper summary",
        f"📁 File: {result.file_name}",
        f"🧠 Model: {result.detection.model_id}",
        f"🌐 Language: {result.language}",
        f"🧾 Task: {result.task}",
        f"🕒 Sample rate: {result.sample_rate}",
        f"📝 Text: {result.detection.text}",
    ]
    if not result.chunks:
        lines.append("🙈 No timestamp chunks returned")
        return "\n".join(lines)

    lines.append(f"🧩 Chunks: {len(result.chunks)}")
    for index, chunk in enumerate(result.chunks, start=1):
        lines.append(f"  {index}. {chunk.timestamp} | {chunk.text}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Whisper.")
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the human-readable console summary.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override the configured Whisper language.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override the configured Whisper task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        if args.language is not None:
            config.language = args.language
        if args.task is not None:
            config.task = args.task

        result = transcribe_audio(args.audio_file, config=config)
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False))
        else:
            print(render_result(result))
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper
        LOGGER.exception("whisper failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
