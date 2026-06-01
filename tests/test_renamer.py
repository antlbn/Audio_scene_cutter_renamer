from __future__ import annotations


import pipeline


def test_load_renamer_template_defaults(tmp_path):
    config_path = tmp_path / "non_existing.yaml"
    template = pipeline.load_renamer_template(config_path)
    assert template == "Sequence_{sequence}_shot_{shot}_take_{take}"
    assert pipeline.load_renamer_suffix(config_path) == ""


def test_load_renamer_template_custom(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
renamer:
  naming_template: "seq-{sequence}-shot-{shot}-take-{take}"
  suffix: "_AB"
""")
    template = pipeline.load_renamer_template(config_path)
    assert template == "seq-{sequence}-shot-{shot}-take-{take}"
    assert pipeline.load_renamer_suffix(config_path) == "_AB"


def test_rename_audio_file_happy_path(tmp_path):
    original = tmp_path / "input.wav"
    original.write_text("audio data")
    
    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
    )
    
    assert new_path.name == "Sequence_1_shot_3_take_5.wav"
    assert new_path.exists()
    assert not original.exists()


def test_rename_audio_file_missing_fields_skips(tmp_path):
    original = tmp_path / "input.wav"
    original.write_text("audio data")
    
    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence=None,
        shot=None,
        take=None,
        template=template,
    )
    
    assert new_path.resolve() == original.resolve()
    assert original.exists()


def test_rename_audio_file_clean_double_underscores(tmp_path):
    original = tmp_path / "input.wav"
    original.write_text("audio data")
    
    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot=None,  # missing shot
        take=5,
        template=template,
    )
    
    # "Sequence_1_shot__take_5.wav" -> cleaned to "Sequence_1_shot_take_5.wav"
    assert new_path.name == "Sequence_1_shot_take_5.wav"
    assert new_path.exists()


def test_rename_audio_file_conflict_resolution(tmp_path):
    original = tmp_path / "input.wav"
    original.write_text("audio data")
    
    # Create conflicting files
    conflict1 = tmp_path / "Sequence_1_shot_3_take_5.wav"
    conflict1.write_text("existing data")
    
    conflict2 = tmp_path / "Sequence_1_shot_3_take_5_1.wav"
    conflict2.write_text("more existing data")
    
    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
    )
    
    assert new_path.name == "Sequence_1_shot_3_take_5_2.wav"
    assert new_path.exists()
    assert original.exists() is False


def test_rename_audio_file_appends_suffix_as_is(tmp_path):
    original = tmp_path / "input_AB.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        suffix="_AB",
    )

    assert new_path.name == "Sequence_1_shot_3_take_5_AB.wav"
    assert new_path.exists()
