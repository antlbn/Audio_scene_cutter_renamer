from __future__ import annotations

import json
from pathlib import Path

import torch

from cinema_clapboard_app import clapper
from cinema_clapboard_app import preprocessor



class FakeProcessor:
    def __call__(self, **kwargs):
        return kwargs


class FakeModel:
    def __init__(self, audio_features: list[list[float]], text_features: list[list[float]]):
        self._audio_features = [torch.tensor(item, dtype=torch.float32).unsqueeze(0) for item in audio_features]
        self._text_features = torch.tensor(text_features, dtype=torch.float32)
        self._audio_index = 0

    def get_text_features(self, **kwargs):
        return self._text_features.clone()

    def get_audio_features(self, **kwargs):
        result = self._audio_features[self._audio_index]
        self._audio_index += 1
        return result.clone()


def test_load_config_fills_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
clapper:
  text_search:
    text_keys:
      - custom key
""".strip()
    )

    config = clapper.load_config(config_path)

    assert config.audio_model.model_name == clapper.DEFAULT_MODEL_NAME
    assert config.audio_model.cache_dir == clapper.DEFAULT_CACHE_DIR
    assert config.audio_model.threshold == clapper.DEFAULT_THRESHOLD
    assert config.text_search.text_keys == ["custom key"]
    assert config.text_search.top_n == clapper.DEFAULT_TOP_N
    assert config.clip.post_hit_seconds == clapper.DEFAULT_POST_HIT_SECONDS


def test_select_top_matches_orders_descending():
    scores = torch.tensor([0.2, 0.7, 0.5], dtype=torch.float32)
    matches = clapper._build_top_matches(scores, ["a", "b", "c"], top_n=2, threshold=0.0)

    assert [item.text_key for item in matches] == ["b", "c"]
    assert [round(item.score, 3) for item in matches] == [0.7, 0.5]


def test_select_top_matches_filters_below_threshold():
    scores = torch.tensor([0.9, 0.7, 0.2], dtype=torch.float32)
    matches = clapper._build_top_matches(scores, ["a", "b", "c"], top_n=3, threshold=0.75)

    assert [item.text_key for item in matches] == ["a"]
    assert [round(item.score, 3) for item in matches] == [0.9]


def test_analyze_audio_uses_standardized_audio(monkeypatch):
    config = clapper.ClapperConfig(
        audio_model=clapper.AudioModelConfig(
            threshold=0.85,
            target_sample_rate=4,
            window_seconds=0.5,
            hop_seconds=0.5,
            device="cpu",
        ),
        text_search=clapper.TextSearchConfig(
            text_keys=["first", "second", "third"],
            top_n=2,
            show_progress=False,
        ),
    )

    fake_runtime = clapper.ClapperRuntime(
        model=FakeModel(
            audio_features=[
                [0.9, 0.1, 0.0],
                [0.2, 0.8, 0.0],
                [0.1, 0.2, 0.3],
            ],
            text_features=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        processor=FakeProcessor(),
        device=torch.device("cpu"),
        cache_dir=Path(".cache/clap"),
    )

    standardized = preprocessor.StandardizedAudio(
        file_name="sample.aac",
        payload=b"aac-bytes",
        codec="aac",
        sample_rate=16_000,
        channels=1,
        duration_seconds=1.5,
    )

    monkeypatch.setattr(
        clapper.preprocessor,
        "decode_standardized_audio",
        lambda audio, target_sample_rate: (torch.zeros(6, dtype=torch.float32), target_sample_rate),
    )

    result = clapper.analyze_audio(standardized, config, fake_runtime, debug=True)

    assert result.file_name == "sample.aac"
    assert result.threshold == 0.85
    assert result.num_points == 2
    assert [hit.text_key for hit in result.best_scores] == ["first", "second"]
    assert all(len(hit.top_matches) == 1 for hit in result.best_scores)


def test_main_uses_cli_result_without_real_model(monkeypatch, tmp_path: Path, capsys):
    expected = clapper.ClapperResult(
        file_name="input.aac",
        best_scores=[],
        num_points=0,
        threshold=0.1,
    )

    monkeypatch.setattr(clapper, "detect_clapper", lambda *args, **kwargs: expected)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("clapper: {}\n")

    exit_code = clapper.main(["input.aac", "--config", str(config_path), "--debug", "--json"])
    out = capsys.readouterr().out

    assert exit_code == 0
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["file_name"] == "input.aac"
    assert payload["num_points"] == 0


def test_render_result_is_human_readable():
    result = clapper.ClapperResult(
        file_name="input.aac",
        best_scores=[
            clapper.ClapperHit(
                timestamp=1.25,
                score=0.42,
                text_key="sharp clap sound",
                top_matches=[
                    clapper.ClapperTopMatch(text_key="sharp clap sound", score=0.42),
                    clapper.ClapperTopMatch(text_key="loud click", score=0.31),
                ],
            )
        ],
        num_points=1,
        threshold=0.3,
    )

    rendered = clapper.render_result(result)

    assert "🎬 Clapper summary" in rendered
    assert "📁 File: input.aac" in rendered
    assert "🎯 Hits: 1" in rendered
    assert "⏱ 1.25s" in rendered
    assert "sharp clap sound 0.420" in rendered
