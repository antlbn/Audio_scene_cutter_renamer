from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import clapper
import cutter
import preprocessor


def _make_clip_result() -> clapper.ClapperResult:
    return clapper.ClapperResult(
        file_name="input.aac",
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


def test_cut_audio_by_clapper_uses_best_hit_and_returns_standardized_audio(monkeypatch):
    result = _make_clip_result()
    config = clapper.ClapperConfig(
        clip=clapper.ClipConfig(post_hit_seconds=2.0),
    )

    standardized = preprocessor.StandardizedAudio(
        file_name="input.aac",
        payload=b"source-aac",
        codec="aac",
        sample_rate=16_000,
        channels=1,
        duration_seconds=4.0,
    )

    recorded: dict[str, list[str] | bytes] = {}

    def fake_run(cmd, check, capture_output, input):
        recorded["cmd"] = cmd
        recorded["input"] = input
        return SimpleNamespace(stdout=b"trimmed-aac")

    monkeypatch.setattr(cutter.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cutter.subprocess, "run", fake_run)

    clipped = cutter.cut_audio_by_clapper(standardized, result, config)

    assert clipped.clapper_result is result
    assert clipped.best_hit.timestamp == 3.2
    assert clipped.duration_seconds == pytest.approx(0.8)
    assert clipped.clip_start_seconds == pytest.approx(3.2)
    assert clipped.clip_end_seconds == pytest.approx(4.0)
    assert clipped.standardized_audio.payload == b"trimmed-aac"
    assert clipped.standardized_audio.sample_rate == 16_000
    assert clipped.standardized_audio.duration_seconds == pytest.approx(0.8)

    cmd = recorded["cmd"]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ss") + 1] == "3.200000"
    assert cmd[cmd.index("-t") + 1] == "0.800000"
    assert recorded["input"] == b"source-aac"


def test_cut_audio_by_clapper_prints_and_raises_on_empty_hits(capsys):
    result = clapper.ClapperResult(
        file_name="input.aac",
        best_scores=[],
        num_points=0,
        threshold=0.3,
    )

    with pytest.raises(ValueError, match="No timestamps were found"):
        cutter._select_best_hit(result)

    captured = capsys.readouterr()
    assert "no timestamps were found" in captured.err.lower()
