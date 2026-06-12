"""CLAP-based clapper detector.

This module loads a CLAP model once per process, scans an audio file with
sliding windows, matches each window against configured text keys, and returns
a JSON-serializable summary of the strongest hits.

Usage:
    uv run python clapper.py path/to/audio.wav --config config.yaml --debug
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from . import preprocessor
from .config_utils import get_user_config_path, get_user_env_path, load_env_file, merge_config_dict, resolve_path

LOGGER = logging.getLogger("clapper")

DEFAULT_MODEL_NAME = "laion/clap-htsat-fused"
DEFAULT_CACHE_DIR = ".cache/clap"
DEFAULT_TARGET_SAMPLE_RATE = 48_000
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_HOP_SECONDS = 0.5
DEFAULT_THRESHOLD = 0.10
DEFAULT_TOP_N = 3
DEFAULT_PRE_HIT_SECONDS = 10.0
DEFAULT_POST_HIT_SECONDS = 2.0
DEFAULT_TEXT_KEYS = [
    "clapper board snap",
    "sharp clap sound",
    "loud click",
    "film slate clap",
    "clap noise",
    "clapperboard closing",
    "slate snap",
    "production clap",
    "sharp film clap",
    "clapper strike",
]


@dataclass(slots=True)
class AudioModelConfig:
    """Model and audio-window settings for the clapper detector."""

    model_name: str = DEFAULT_MODEL_NAME
    cache_dir: str = DEFAULT_CACHE_DIR
    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    hop_seconds: float = DEFAULT_HOP_SECONDS
    threshold: float = DEFAULT_THRESHOLD
    device: str = "auto"


@dataclass(slots=True)
class TextSearchConfig:
    """Text-key configuration for CLAP similarity search."""

    text_keys: list[str] = field(default_factory=lambda: list(DEFAULT_TEXT_KEYS))
    top_n: int = DEFAULT_TOP_N
    show_progress: bool = True


@dataclass(slots=True)
class ClipConfig:
    """Post-processing settings for clapper-driven audio trimming."""

    pre_hit_seconds: float = DEFAULT_PRE_HIT_SECONDS
    post_hit_seconds: float = DEFAULT_POST_HIT_SECONDS


@dataclass(slots=True)
class ClapperConfig:
    """Top-level config for the clapper module."""

    audio_model: AudioModelConfig = field(default_factory=AudioModelConfig)
    text_search: TextSearchConfig = field(default_factory=TextSearchConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)


@dataclass(slots=True)
class ClapperTopMatch:
    """One ranked text-key match for a single time window."""

    text_key: str
    score: float


@dataclass(slots=True)
class ClapperHit:
    """One scored detection."""

    timestamp: float
    score: float
    text_key: str
    top_matches: list[ClapperTopMatch]


@dataclass(slots=True)
class ClapperResult:
    """Final JSON-serializable result returned by the detector."""

    file_name: str
    best_scores: list[ClapperHit]
    num_points: int
    threshold: float


@dataclass(slots=True)
class ClapperRuntime:
    """Loaded CLAP model bundle."""

    model: Any
    processor: Any
    device: torch.device
    cache_dir: Path


_RUNTIME_CACHE: dict[tuple[str, str, str], ClapperRuntime] = {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")




def load_config(config_path: str | Path | None = None) -> ClapperConfig:
    """Load config.yaml and apply defaults for missing values."""

    config_file = Path(config_path) if config_path is not None else _default_config_path()
    raw: dict[str, Any] = {}

    if config_file.exists():
        loaded = yaml.safe_load(config_file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
        raw = loaded.get("clapper", loaded)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid clapper section in {config_file}: expected a mapping")
    else:
        LOGGER.warning("Config file %s not found, using built-in defaults", config_file)

    defaults = {
        "audio_model": {
            "model_name": DEFAULT_MODEL_NAME,
            "cache_dir": DEFAULT_CACHE_DIR,
            "target_sample_rate": DEFAULT_TARGET_SAMPLE_RATE,
            "window_seconds": DEFAULT_WINDOW_SECONDS,
            "hop_seconds": DEFAULT_HOP_SECONDS,
            "threshold": DEFAULT_THRESHOLD,
            "device": "auto",
        },
        "text_search": {
            "text_keys": list(DEFAULT_TEXT_KEYS),
            "top_n": DEFAULT_TOP_N,
            "show_progress": True,
        },
        "clip": {
            "pre_hit_seconds": DEFAULT_PRE_HIT_SECONDS,
            "post_hit_seconds": DEFAULT_POST_HIT_SECONDS,
        },
    }
    merged = merge_config_dict(defaults, raw)

    audio_model_raw = merged["audio_model"]
    text_search_raw = merged["text_search"]
    clip_raw = merged["clip"]

    audio_model = AudioModelConfig(
        model_name=str(audio_model_raw["model_name"]),
        cache_dir=str(audio_model_raw["cache_dir"]),
        target_sample_rate=int(audio_model_raw["target_sample_rate"]),
        window_seconds=float(audio_model_raw["window_seconds"]),
        hop_seconds=float(audio_model_raw["hop_seconds"]),
        threshold=float(audio_model_raw["threshold"]),
        device=str(audio_model_raw["device"]),
    )
    text_keys = [str(item).strip() for item in text_search_raw["text_keys"] if str(item).strip()]
    if not text_keys:
        raise ValueError("text_search.text_keys must contain at least one non-empty key")

    text_search = TextSearchConfig(
        text_keys=text_keys,
        top_n=max(1, int(text_search_raw["top_n"])),
        show_progress=bool(text_search_raw["show_progress"]),
    )
    clip = ClipConfig(
        pre_hit_seconds=max(0.0, float(clip_raw["pre_hit_seconds"])),
        post_hit_seconds=max(0.0, float(clip_raw["post_hit_seconds"])),
    )
    return ClapperConfig(audio_model=audio_model, text_search=text_search, clip=clip)


# ---------------------------------------------------------------------------
# Audio loading and small numeric helpers
# ---------------------------------------------------------------------------


def _load_audio(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio as a mono float32 waveform.

    Torchaudio is the fast path. ffmpeg is a fallback for container formats
    that torchaudio may not decode in this environment.
    """

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.float()
    except Exception as exc:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                f"Failed to load {audio_path} with torchaudio and no ffmpeg fallback is available"
            ) from exc

        ffmpeg_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-f",
            "wav",
            "-",
        ]
        proc = subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        decoded, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(decoded.T.copy())

    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError(f"Audio file is empty or invalid: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0), sample_rate


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    window = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, window, mode="same")


def _zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def _local_maxima(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.array([], dtype=int)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] >= values[2:]
    return np.flatnonzero(left & right) + 1


def _group_peaks(peak_indices: list[int], min_gap_frames: int, scores: np.ndarray) -> list[int]:
    grouped: list[int] = []
    current_group: list[int] = []

    for idx in peak_indices:
        if not current_group or idx - current_group[-1] <= min_gap_frames:
            current_group.append(int(idx))
            continue

        grouped.append(max(current_group, key=lambda item: float(scores[item])))
        current_group = [int(idx)]

    if current_group:
        grouped.append(max(current_group, key=lambda item: float(scores[item])))

    return grouped


def _get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


# ---------------------------------------------------------------------------
# Runtime loading
# ---------------------------------------------------------------------------


def load_runtime(
    config: ClapperConfig, *, cache_base_dir: Path | None = None
) -> ClapperRuntime:
    """Load and cache the CLAP pipeline."""

    cache_root = cache_base_dir or get_user_config_path().parent
    cache_dir = resolve_path(config.audio_model.cache_dir, cache_root)
    device = _get_device(config.audio_model.device)
    cache_key = (config.audio_model.model_name, str(cache_dir), str(device))

    if cache_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[cache_key]

    load_env_file(get_user_env_path())

    print(f"[clapper] loading model: {config.audio_model.model_name}", file=sys.stderr)
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(config.audio_model.model_name, cache_dir=str(cache_dir))
    processor = ClapProcessor.from_pretrained(config.audio_model.model_name, cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    print("[clapper] model loaded", file=sys.stderr)

    runtime = ClapperRuntime(model=model, processor=processor, device=device, cache_dir=cache_dir)
    _RUNTIME_CACHE[cache_key] = runtime
    return runtime


# ---------------------------------------------------------------------------
# Feature extraction and ranking
# ---------------------------------------------------------------------------


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, dim=-1)


def _move_model_inputs_to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: _move_model_inputs_to_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    return inputs


def _build_top_matches(
    scores: torch.Tensor,
    text_keys: list[str],
    top_n: int,
    threshold: float,
) -> list[ClapperTopMatch]:
    top_count = min(top_n, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=top_count)
    top_matches = [
        ClapperTopMatch(text_key=text_keys[int(idx)], score=float(score))
        for score, idx in zip(top_scores, top_indices, strict=True)
    ]
    filtered = [item for item in top_matches if item.score >= threshold]
    return filtered or top_matches


def _build_hit(
    timestamp: float,
    scores: torch.Tensor,
    text_keys: list[str],
    top_n: int,
    threshold: float,
) -> ClapperHit:
    top_matches = _build_top_matches(scores, text_keys, top_n, threshold)
    best_match = top_matches[0]
    return ClapperHit(
        timestamp=round(float(timestamp), 6),
        score=float(best_match.score),
        text_key=best_match.text_key,
        top_matches=top_matches,
    )


def _prepare_audio_windows(
    audio_source: str | Path | preprocessor.StandardizedAudio,
    config: ClapperConfig,
) -> tuple[Path, torch.Tensor, int, int, list[int]]:
    """Load audio, resample to the model target rate, and build the sliding window plan.

    File paths are always loaded directly via torchaudio/ffmpeg to preserve the
    original sample rate and avoid lossy transcoding.  A pre-built
    ``StandardizedAudio`` object is decoded in-memory instead.
    """

    if isinstance(audio_source, preprocessor.StandardizedAudio):
        audio_path = Path(audio_source.file_name)
        waveform, sample_rate = preprocessor.decode_standardized_audio(
            audio_source,
            target_sample_rate=config.audio_model.target_sample_rate,
        )
    else:
        audio_path = Path(audio_source)
        waveform, sample_rate = _load_audio(audio_path)
        target_sr = config.audio_model.target_sample_rate
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr

    window_size = max(1, int(round(config.audio_model.window_seconds * sample_rate)))
    hop_size = max(1, int(round(config.audio_model.hop_seconds * sample_rate)))

    if len(waveform) < window_size:
        window_starts: list[int] = []
    else:
        window_starts = list(range(0, len(waveform) - window_size + 1, hop_size))

    return audio_path, waveform, sample_rate, window_size, window_starts


def _encode_text_features(runtime: ClapperRuntime, text_keys: list[str]) -> torch.Tensor:
    """Build one normalized embedding per text key."""

    text_inputs = runtime.processor(
        text=text_keys,
        return_tensors="pt",
        padding=True,
    )
    text_inputs = _move_model_inputs_to_device(text_inputs, runtime.device)
    with torch.no_grad():
        text_features = runtime.model.get_text_features(**text_inputs)
    return _normalize_features(text_features.to(runtime.device))


def _encode_audio_window(
    runtime: ClapperRuntime,
    chunk: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    """Build one normalized CLAP embedding for a single audio window."""

    audio_inputs = runtime.processor(
        audio=chunk.detach().cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    audio_inputs = _move_model_inputs_to_device(audio_inputs, runtime.device)
    with torch.no_grad():
        audio_features = runtime.model.get_audio_features(**audio_inputs)
    return _normalize_features(audio_features.to(runtime.device))


def _score_audio_window(text_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
    """Compute similarity scores for one audio window against all text keys."""

    scores = (audio_features @ text_features.T).squeeze(0)
    if scores.ndim != 1:
        raise RuntimeError("Expected a 1D score vector per window")
    return scores


def _format_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Render a compact progress bar for terminal output."""

    if total <= 0:
        return "░" * width
    filled = max(0, min(width, round(width * current / total)))
    return "█" * filled + "░" * (width - filled)


def _print_progress(current: int, total: int, *, final: bool = False) -> None:
    """Print a single progress line.

    We keep this on stderr so it never collides with JSON or summary output.
    """

    bar = _format_progress_bar(current, total)
    prefix = "✅" if final else "🔎"
    message = f"[clapper] {prefix} [{bar}] {current}/{total} windows"

    if sys.stderr.isatty():
        end = "\n" if final else "\r"
        print(message, end=end, file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def _print_debug_hit(hit: ClapperHit) -> None:
    """Print one hit in a compact, readable debug format."""

    top_matches = ", ".join(
        f"{match.text_key} {match.score:.3f}" for match in hit.top_matches
    )
    print(
        f"🎯 t={hit.timestamp:.2f}s | score={hit.score:.3f} | key={hit.text_key} | top: {top_matches}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyze_audio(
    audio_source: str | Path | preprocessor.StandardizedAudio,
    config: ClapperConfig,
    runtime: ClapperRuntime,
    *,
    debug: bool = False,
) -> ClapperResult:
    """Run CLAP similarity search over one audio file."""

    audio_path, waveform, sample_rate, window_size, window_starts = _prepare_audio_windows(
        audio_source,
        config,
    )
    total_windows = len(window_starts)

    # Text side: build one embedding per configured key once, then reuse it for
    # every audio window.
    text_features = _encode_text_features(runtime, config.text_search.text_keys)
    hits: list[ClapperHit] = []

    print(f"[clapper] {audio_path.name} clapper recognition in work", file=sys.stderr)
    if total_windows == 0:
        print("[clapper] ⚠️ no windows available for the requested window size", file=sys.stderr)
    else:
        # Audio side: iterate window-by-window, encode the chunk, and score it
        # against all text keys.
        for index, start in enumerate(window_starts, start=1):
            chunk = waveform[start : start + window_size]
            audio_features = _encode_audio_window(runtime, chunk, sample_rate)
            scores = _score_audio_window(text_features, audio_features)

            best_score = float(scores.max().item())
            if best_score >= config.audio_model.threshold:
                hit = _build_hit(
                    timestamp=start / sample_rate,
                    scores=scores,
                    text_keys=config.text_search.text_keys,
                    top_n=config.text_search.top_n,
                    threshold=config.audio_model.threshold,
                )
                hits.append(hit)
                if debug:
                    if sys.stderr.isatty():
                        print("\r\033[K", end="", file=sys.stderr)
                    _print_debug_hit(hit)

            if config.text_search.show_progress:
                _print_progress(index, total_windows, final=False)

        if config.text_search.show_progress:
            _print_progress(total_windows, total_windows, final=True)

    result = ClapperResult(
        file_name=audio_path.name,
        best_scores=hits,
        num_points=len(hits),
        threshold=float(config.audio_model.threshold),
    )
    return result


def detect_clapper(
    audio_source: str | Path | preprocessor.StandardizedAudio,
    *,
    config_path: str | Path | None = None,
    debug: bool = False,
) -> ClapperResult:
    """High-level convenience wrapper used by the CLI."""

    config = load_config(config_path)
    runtime = load_runtime(config)
    return analyze_audio(audio_source, config, runtime, debug=debug)


def render_result(result: ClapperResult) -> str:
    """Render a compact human-readable console summary."""

    lines = [
        "🎬 Clapper summary",
        f"📁 File: {result.file_name}",
        f"🎚 Threshold: {result.threshold:.3f}",
        f"🎯 Hits: {result.num_points}",
    ]
    if not result.best_scores:
        lines.append("🙈 No matches above threshold")
        return "\n".join(lines)

    for index, hit in enumerate(result.best_scores, start=1):
        top_matches = ", ".join(
            f"{match.text_key} {match.score:.3f}" for match in hit.top_matches
        )
        lines.append(
            f"  {index}. ⏱ {hit.timestamp:.2f}s | score {hit.score:.3f} | {hit.text_key}"
        )
        lines.append(f"     🥇 {top_matches}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect clapper events in one audio file.")
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a per-hit debug line with timestamp and text key.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of the human-readable console summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = detect_clapper(
            args.audio_file,
            config_path=args.config,
            debug=args.debug,
        )
        if args.json:
            print(json.dumps(asdict(result), ensure_ascii=False))
        else:
            print(render_result(result))
        return 0
    except Exception as exc:  # pragma: no cover - defensive CLI wrapper
        LOGGER.exception("clapper failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
