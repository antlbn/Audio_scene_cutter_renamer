from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import clapper
import cutter


def _make_clip_result() -> clapper.ClapperResult:
    return clapper.ClapperResult(
        file_name="input.wav",
        best_scores=[
            clapper.ClapperHit(
                timestamp=0.5,
                score=0.2,
                text_key="first",
                top_matches=[],
            ),
            clapper.ClapperHit(
                timestamp=3.2,
                score=0.9,
                text_key="second",
                top_matches=[],
            ),
        ],
        num_points=2,
        threshold=0.3,
    )


def test_cut_audio_by_clapper_uses_best_hit_and_returns_in_memory_audio(monkeypatch):
    result = _make_clip_result()
    config = clapper.ClapperConfig(
        clip=clapper.ClipConfig(post_hit_seconds=2.0),
    )

    monkeypatch.setattr(
        cutter.clapper,
        "_load_audio",
        lambda audio_path: (torch.zeros(40, dtype=torch.float32), 10),
    )

    recorded: dict[str, list[str]] = {}

    def fake_run(cmd, check, capture_output):
        recorded["cmd"] = cmd
        samples = np.linspace(-1.0, 1.0, 12, dtype=np.float32).astype("<f4")
        return SimpleNamespace(stdout=samples.tobytes())

    monkeypatch.setattr(cutter.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cutter.subprocess, "run", fake_run)

    clipped = cutter.cut_audio_by_clapper(Path("input.wav"), result, config)

    assert clipped.clapper_result is result
    assert clipped.best_hit.timestamp == 3.2
    assert clipped.sample_rate == 16_000
    assert clipped.duration_seconds == pytest.approx(12 / 16_000)
    assert clipped.audio.dtype == torch.float32
    assert clipped.audio.ndim == 1
    assert clipped.clip_start_seconds == pytest.approx(3.2)
    assert clipped.clip_end_seconds == pytest.approx(4.0)

    cmd = recorded["cmd"]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ss") + 1] == "3.200000"
    assert cmd[cmd.index("-t") + 1] == "0.800000"
    assert cmd[cmd.index("-ar") + 1] == "16000"


def test_cut_audio_by_clapper_prints_and_raises_on_empty_hits(capsys):
    result = clapper.ClapperResult(
        file_name="input.wav",
        best_scores=[],
        num_points=0,
        threshold=0.3,
    )

    with pytest.raises(ValueError, match="No timestamps were found"):
        cutter._select_best_hit(result)

    captured = capsys.readouterr()
    assert "no timestamps were found" in captured.err.lower()
