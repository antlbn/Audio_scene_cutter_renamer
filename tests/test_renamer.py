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


def test_extract_from_source_zoom_format_with_timestamp(tmp_path):
    """Test ZOOM format: ZOOM0746_Tr3 [2026-05-28 152751].wav"""
    original = tmp_path / "ZOOM0746_Tr3 [2026-05-28 152751].wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        suffix="",
        extract_from_source=True,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    assert new_path.name == "Sequence_1_shot_3_take_5_Tr3.wav"
    assert new_path.exists()


def test_extract_from_source_zoom_format_no_timestamp(tmp_path):
    """Test ZOOM format: ZOOM0746_Tr3.wav"""
    original = tmp_path / "ZOOM0746_Tr3.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        extract_from_source=True,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    assert new_path.name == "Sequence_1_shot_3_take_5_Tr3.wav"
    assert new_path.exists()


def test_extract_from_source_after_second_underscore(tmp_path):
    """Test format: 260603_0001_12.wav - extract '12' after second underscore"""
    original = tmp_path / "260603_0001_12.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        extract_from_source=True,
        extract_pattern="^[^_]*_[^_]*_(.+?)$",
        extract_format="_TR{match}",
    )

    assert new_path.name == "Sequence_1_shot_3_take_5_TR12.wav"
    assert new_path.exists()


def test_extract_from_source_disabled(tmp_path):
    """When extract_from_source is False, should not extract"""
    original = tmp_path / "ZOOM0746_Tr3.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        extract_from_source=False,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    assert new_path.name == "Sequence_1_shot_3_take_5.wav"
    assert new_path.exists()


def test_extract_from_source_pattern_no_match(tmp_path):
    """When pattern doesn't match, should just use the base name"""
    original = tmp_path / "no_match_here.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        extract_from_source=True,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    # No extraction, just the base name
    assert new_path.name == "Sequence_1_shot_3_take_5.wav"
    assert new_path.exists()


def test_extract_from_source_with_suffix(tmp_path):
    """Extract from source AND apply suffix"""
    original = tmp_path / "ZOOM0746_Tr3.wav"
    original.write_text("audio data")

    template = "Sequence_{sequence}_shot_{shot}_take_{take}"
    new_path = pipeline.new_name(
        original_path=original,
        sequence="1",
        shot="3",
        take=5,
        template=template,
        suffix="KinoBlia",
        extract_from_source=True,
        extract_pattern="_([Tt]r\\d+)",
        extract_format="_{match}",
    )

    # suffix is added after template, then extraction is appended
    assert new_path.name == "Sequence_1_shot_3_take_5_KinoBlia_Tr3.wav"
    assert new_path.exists()
