"""Detect clap-like transients in an audio file.

The script:
- loads an audio file,
- renders a spectrogram PNG,
- detects clap candidates using high-frequency energy and spectral flux,
- prints the result as JSON on stdout.

Usage:
    uv run python detect_clap.py path/to/audio.wav
"""

from __future__ import annotations

import io
import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio


DEFAULT_HIGH_BAND_HZ = 2000.0
DEFAULT_THRESHOLD_STD = 5.0
DEFAULT_MIN_SEPARATION_S = 1
DEFAULT_MOTION_SMOOTHING_WINDOW = 3
DEFAULT_BAND_ENERGY_WEIGHT = 1.0
DEFAULT_FLUX_WEIGHT = 0.75

# Tuning knobs:
# - DEFAULT_HIGH_BAND_HZ: lower or raise the frequency band we inspect.
# - DEFAULT_THRESHOLD_STD: lower for more detections, higher for fewer false positives.
# - DEFAULT_MIN_SEPARATION_S: increase to avoid repeated hits for one clap.
# - DEFAULT_BAND_ENERGY_WEIGHT / DEFAULT_FLUX_WEIGHT: change the balance between
#   broad high-frequency energy and sudden spectral changes.


@dataclass(slots=True)
class ClapDetectionResult:
    """Structured result returned by :func:`detect_claps`."""

    audio_path: str
    sample_rate: int
    spectrogram_path: str
    clap_times: list[float]
    spectrogram_db: np.ndarray
    frame_hop_seconds: float


def load_audio(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio as a mono float32 tensor.

    We try `torchaudio` first because it is the cheapest path for WAV/FLAC-like
    files. If that fails, we fall back to `ffmpeg`, which can decode many more
    containers such as M4A and AAC.
    """

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.ndim != 2:
            raise ValueError(f"Unexpected audio tensor shape: {tuple(waveform.shape)}")
        waveform = waveform.float()
    except Exception as exc:
        # Some files in this repo are M4A, which torchaudio's default backend
        # may not decode in this environment. `ffmpeg` gives us a robust fallback.
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

    if waveform.shape[1] == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    return waveform, sample_rate


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth a 1D score curve to reduce one-frame spikes."""

    if window_size <= 1 or values.size == 0:
        return values
    window = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, window, mode="same")


def _zscore(values: np.ndarray) -> np.ndarray:
    """Normalize values so different features can be combined on one scale."""

    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def _local_maxima(values: np.ndarray) -> np.ndarray:
    """Return indices of simple local maxima in a 1D array."""

    if values.size < 3:
        return np.array([], dtype=int)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] >= values[2:]
    return np.flatnonzero(left & right) + 1


def _group_peaks(peak_indices: Iterable[int], min_gap_frames: int, scores: np.ndarray) -> list[int]:
    """Merge nearby peaks and keep only the strongest one from each cluster."""

    grouped: list[int] = []
    current_group: list[int] = []

    for idx in peak_indices:
        if not current_group or idx - current_group[-1] <= min_gap_frames:
            current_group.append(int(idx))
            continue

        best = max(current_group, key=lambda i: scores[i])
        grouped.append(best)
        current_group = [int(idx)]

    if current_group:
        best = max(current_group, key=lambda i: scores[i])
        grouped.append(best)

    return grouped


def detect_claps(
    audio_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    high_band_hz: float = DEFAULT_HIGH_BAND_HZ,
    threshold_std: float = DEFAULT_THRESHOLD_STD,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
) -> ClapDetectionResult:
    """Detect clap-like impulses and render a spectrogram image.

    Detection heuristic:
    - compute the STFT magnitude spectrogram,
    - measure energy in the upper frequency band,
    - measure spectral flux between consecutive frames,
    - combine both signals into a single score curve,
    - pick strong local maxima and merge nearby duplicates.
    """

    audio_path = Path(audio_path)
    waveform, sample_rate = load_audio(audio_path)

    if n_fft is None:
        # Use a larger FFT for higher sample rates so the spectrogram stays stable.
        n_fft = 2048 if sample_rate >= 32000 else 1024
    if hop_length is None:
        # A quarter-window hop is a common compromise between time resolution and noise.
        hop_length = max(1, n_fft // 4)

    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    # Magnitude spectrogram is the base representation for both visualization
    # and clap scoring.
    magnitude = stft.abs().cpu().numpy()

    # Convert magnitude to decibels for visualization. We normalize to 0 dB at
    # the loudest bin so the plot is easier to read across different files.
    spectrogram_db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
    spectrogram_db = spectrogram_db - float(spectrogram_db.max())

    # Clap transients usually produce short broadband spikes. We focus on the
    # upper part of the spectrum where those spikes are easy to separate from
    # speech or ambience.
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    high_band_mask = freqs >= min(high_band_hz, float(freqs[-1]))
    if not np.any(high_band_mask):
        high_band_mask = freqs >= freqs[len(freqs) // 2]

    # Two complementary signals:
    # - band_energy: how much energy exists in the high-frequency range
    # - flux: how much the spectrum changed from frame to frame
    high_band_magnitude = magnitude[high_band_mask]
    band_energy = high_band_magnitude.mean(axis=0)

    frame_to_frame_change = np.diff(magnitude, axis=1)
    positive_change_only = np.maximum(0.0, frame_to_frame_change)
    flux = positive_change_only.sum(axis=0)
    flux = np.pad(flux, (1, 0), mode="edge")

    # Bring both features onto comparable scales and combine them.
    normalized_band_energy = _zscore(band_energy)
    normalized_flux = _zscore(flux)
    combined_score = (
        DEFAULT_BAND_ENERGY_WEIGHT * normalized_band_energy
        + DEFAULT_FLUX_WEIGHT * normalized_flux
    )
    smoothed_score = _moving_average(combined_score, DEFAULT_MOTION_SMOOTHING_WINDOW)

    # Local maxima above the threshold are treated as clap candidates.
    score_mean = float(smoothed_score.mean())
    score_std = float(smoothed_score.std())
    threshold = score_mean + threshold_std * score_std
    candidate_peaks = _local_maxima(smoothed_score)
    candidate_peaks = candidate_peaks[smoothed_score[candidate_peaks] >= threshold]

    # Avoid reporting the same clap multiple times when a transient spans
    # several adjacent frames.
    min_gap_frames = max(1, int(math.ceil(min_separation_s * sample_rate / hop_length)))
    selected_frames = _group_peaks(candidate_peaks.tolist(), min_gap_frames, smoothed_score)

    clap_times = [round(frame * hop_length / sample_rate, 6) for frame in selected_frames]

    output_dir = Path(output_dir) if output_dir is not None else audio_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_path = output_dir / f"{audio_path.stem}_spectrogram.png"
    _save_spectrogram_png(
        spectrogram_db,
        sample_rate=sample_rate,
        hop_length=hop_length,
        output_path=spectrogram_path,
        title=audio_path.name,
        clap_times=clap_times,
    )

    return ClapDetectionResult(
        audio_path=str(audio_path),
        sample_rate=sample_rate,
        spectrogram_path=str(spectrogram_path),
        clap_times=clap_times,
        spectrogram_db=spectrogram_db,
        frame_hop_seconds=hop_length / sample_rate,
    )


def _save_spectrogram_png(
    spectrogram_db: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    output_path: Path,
    title: str,
    clap_times: list[float],
) -> None:
    """Save the spectrogram as a PNG and overlay detected clap timestamps."""

    duration_s = spectrogram_db.shape[1] * hop_length / sample_rate
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(
        spectrogram_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, duration_s, 0.0, sample_rate / 2.0],
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="dB")
    ax.set_title(f"Spectrogram: {title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Vertical lines make it easy to visually check if detections match the spikes.
    for clap_time in clap_times:
        ax.axvline(clap_time, color="cyan", linewidth=1.0, alpha=0.75)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI interface."""

    parser = argparse.ArgumentParser(description="Detect clap-like transients in an audio file.")
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts. Defaults to the audio file directory.",
    )
    parser.add_argument(
        "--high-band-hz",
        type=float,
        default=DEFAULT_HIGH_BAND_HZ,
        help="Lower edge of the frequency band used for clap scoring. Tune this if claps are missed or too much speech is detected.",
    )
    parser.add_argument(
        "--threshold-std",
        type=float,
        default=DEFAULT_THRESHOLD_STD,
        help="Score threshold in standard deviations above the mean. Lower it to catch more events, raise it to reduce false positives.",
    )
    parser.add_argument(
        "--min-separation-s",
        type=float,
        default=DEFAULT_MIN_SEPARATION_S,
        help="Minimum separation between clap detections in seconds. Increase it if one clap is being reported several times.",
    )
    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()

    result = detect_claps(
        args.audio_file,
        output_dir=args.output_dir,
        high_band_hz=args.high_band_hz,
        threshold_std=args.threshold_std,
        min_separation_s=args.min_separation_s,
    )

    payload = asdict(result)
    payload["spectrogram_db"] = None
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
