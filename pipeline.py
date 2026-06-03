"""Pipeline orchestrator to process an audio file through all stages.

Runs preprocessor, clapper detector, cutter, whisper transcriber, and LLM scene parser,
then aggregates the metadata into a single structured result.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

import clapper
import cutter
import presentator_cli
import preprocessor
import scene_parser
import whisper

LOGGER = logging.getLogger("pipeline")
DEFAULT_RENAMER_TEMPLATE = "Sequence_{sequence}_shot_{shot}_take_{take}"
DEFAULT_RENAMER_SUFFIX = ""
DEFAULT_EXTRACT_FROM_SOURCE = False
DEFAULT_EXTRACT_PATTERN = "_([Tt]r\\d+)"
DEFAULT_EXTRACT_FORMAT = "_{match}"


def load_renamer_config(config_path: str | Path | None = None) -> dict[str, str | bool]:
    config_file = Path(config_path) if config_path is not None else Path("config.yaml")
    default_config = {
        "naming_template": DEFAULT_RENAMER_TEMPLATE,
        "suffix": DEFAULT_RENAMER_SUFFIX,
        "extract_from_source": DEFAULT_EXTRACT_FROM_SOURCE,
        "extract_pattern": DEFAULT_EXTRACT_PATTERN,
        "extract_format": DEFAULT_EXTRACT_FORMAT,
    }
    if not config_file.exists():
        return default_config
    try:
        loaded = yaml.safe_load(config_file.read_text()) or {}
    except (yaml.YAMLError, OSError) as exc:
        LOGGER.warning("Failed to load renamer config %s: %s", config_file, exc)
        return default_config

    if isinstance(loaded, dict):
        renamer_cfg = loaded.get("renamer", {})
        if isinstance(renamer_cfg, dict):
            return {
                "naming_template": str(renamer_cfg.get("naming_template", DEFAULT_RENAMER_TEMPLATE)),
                "suffix": str(renamer_cfg.get("suffix", DEFAULT_RENAMER_SUFFIX)),
                "extract_from_source": bool(renamer_cfg.get("extract_from_source", DEFAULT_EXTRACT_FROM_SOURCE)),
                "extract_pattern": str(renamer_cfg.get("extract_pattern", DEFAULT_EXTRACT_PATTERN)),
                "extract_format": str(renamer_cfg.get("extract_format", DEFAULT_EXTRACT_FORMAT)),
            }
    return default_config


def load_renamer_template(config_path: str | Path | None = None) -> str:
    return load_renamer_config(config_path)["naming_template"]


def load_renamer_suffix(config_path: str | Path | None = None) -> str:
    return load_renamer_config(config_path)["suffix"]


def new_name(
    original_path: Path,
    sequence: str | None,
    shot: str | None,
    take: int | None,
    template: str,
    suffix: str = "",
    extract_from_source: bool = False,
    extract_pattern: str = "",
    extract_format: str = "",
) -> Path:
    """Format the new filename and rename the file on disk, handling naming conflicts.
    
    Args:
        original_path: Path to the original file.
        sequence: Extracted sequence number.
        shot: Extracted shot number.
        take: Extracted take number.
        template: Naming template with {sequence}, {shot}, {take} placeholders.
        suffix: Suffix to append to the filename.
        extract_from_source: Whether to extract part of the original filename.
        extract_pattern: Regex pattern to extract from original filename (e.g., "_([Tt]r\\d+)" for ZOOM).
        extract_format: Format string for the extracted value (e.g., "_{match}").
    
    Returns:
        Path to the renamed file.
    """
    seq_val = sequence if sequence is not None else ""
    shot_val = shot if shot is not None else ""
    take_val = str(take) if take is not None else ""

    if not seq_val and not shot_val and not take_val:
        LOGGER.warning("LLM sequence, shot, and take are all missing. Skipping file rename.")
        return original_path

    new_stem = template.format(sequence=seq_val, shot=shot_val, take=take_val)
    new_stem = re.sub(r'_+', '_', new_stem)
    new_stem = new_stem.strip('_')

    if suffix:
        new_stem = re.sub(r'_+', '_', f"{new_stem}_{suffix}")
        new_stem = new_stem.strip('_')

    # Extract part of the original filename if configured
    if extract_from_source and extract_pattern and extract_format:
        # Remove extension and search for pattern in stem
        original_stem = original_path.stem
        match = re.search(extract_pattern, original_stem)
        if match:
            extracted_value = match.group(1)
            extracted_suffix = extract_format.format(match=extracted_value)
            new_stem = f"{new_stem}{extracted_suffix}"
            LOGGER.info("Extracted '%s' from source filename, appending: %s", extracted_value, extracted_suffix)
        else:
            LOGGER.debug("No match found for extract_pattern '%s' in '%s'", extract_pattern, original_stem)

    if not new_stem:
        LOGGER.warning("Formatted name is empty. Skipping file rename.")
        return original_path

    ext = original_path.suffix
    candidate_name = f"{new_stem}{ext}"
    target_path = original_path.parent / candidate_name

    if target_path.resolve() == original_path.resolve():
        return original_path

    counter = 1
    while target_path.exists():
        candidate_name = f"{new_stem}_{counter}{ext}"
        target_path = original_path.parent / candidate_name
        counter += 1

    LOGGER.info("Renaming %s to %s", original_path.name, target_path.name)
    original_path.rename(target_path)
    return target_path


class PipelineResult(BaseModel):
    """Final aggregated result of the audio scene processing pipeline."""

    input_name: str = Field(description="Name of the original input audio file")
    new_name: str = Field(description="Name of the generated audio file")

    # Preprocessor
    preprocessor_codec: str = Field(description="Audio codec used for standardization")
    preprocessor_sample_rate: int = Field(description="Sample rate of standardized audio")
    preprocessor_duration_seconds: float = Field(description="Duration of standardized audio in seconds")

    # Clapper
    clapper_hits: int = Field(description="Number of detected clapper hits")
    clapper_best_timestamp: float | None = Field(None, description="Timestamp of the best clapper hit")
    clapper_best_score: float | None = Field(None, description="Score of the best clapper hit")
    clapper_best_text_key: str | None = Field(None, description="Matched text key for the best clapper hit")

    # Cutter
    cutter_clip_start: float | None = Field(None, description="Start timestamp of the cut audio clip")
    cutter_clip_end: float | None = Field(None, description="End timestamp of the cut audio clip")

    # Whisper
    whisper_text: str = Field(description="Whisper transcription text")
    whisper_model: str = Field(description="Whisper model ID")
    whisper_language: str = Field(description="Whisper transcription language")
    whisper_task: str = Field(description="Whisper transcription task")

    # LLM Scene Parser
    llm_model: str = Field(description="LLM model used for scene parsing")
    llm_sequence: str | None = Field(None, description="Extracted sequence number")
    llm_shot: str | None = Field(None, description="Extracted shot number")
    llm_take: int | None = Field(None, description="Extracted take number")
    llm_announcement: str = Field(description="Full raw announcement as spoken")


def run_pipeline(
    audio_path: str | Path,
    *,
    config_path: str | Path | None = None,
    whisper_language: str | None = None,
    whisper_task: str | None = None,
    rename: bool = False,
) -> PipelineResult:
    """Execute the full processing pipeline on a single audio file.

    Args:
        audio_path: Path to the input audio file.
        config_path: Path to the config.yaml file.
        whisper_language: Optional override for the Whisper language.
        whisper_task: Optional override for the Whisper task.

    Returns:
        PipelineResult: Struct containing all stages' outputs and metadata.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Source audio file not found: {audio_path}")

    # 1. Load configurations for all steps
    prep_config = preprocessor.load_config(config_path)
    clap_config = clapper.load_config(config_path)
    whisp_config = whisper.load_config(config_path)
    llm_config = scene_parser.load_config(config_path)

    # Apply command line overrides to Whisper if provided
    if whisper_language is not None:
        whisp_config.language = whisper_language
    if whisper_task is not None:
        whisp_config.task = whisper_task

    # 2. Preprocess: standardize audio in-memory
    LOGGER.info("standardizing audio: %s", audio_path.name)
    std_audio = preprocessor.standardize_audio(audio_path, config=prep_config)

    # 3. Clapper: run CLAP detector on the raw source file.
    # File paths are always loaded directly (torchaudio/ffmpeg) to preserve
    # the original sample rate — preprocessing would downsample to 16 kHz and
    # encode AAC, destroying the high-frequency transients CLAP relies on.
    clap_runtime = clapper.load_runtime(clap_config)
    clap_res = clapper.analyze_audio(audio_path, config=clap_config, runtime=clap_runtime)

    # 4. Cutter & Whisper: run based on clapper hits
    best_hit = None
    cutter_clip_start = None
    cutter_clip_end = None
    whisper_source = std_audio

    if clap_res.best_scores:
        # Determine best hit (highest score)
        best_hit = max(clap_res.best_scores, key=lambda hit: (float(hit.score), -float(hit.timestamp)))
        LOGGER.info("best clapper hit at %.2fs (score %.3f)", best_hit.timestamp, best_hit.score)

        # Run cutter to trim audio in memory
        try:
            cut_res = cutter.cut_audio_by_clapper(std_audio, clap_res, config=clap_config)
            whisper_source = cut_res.standardized_audio
            cutter_clip_start = cut_res.clip_start_seconds
            cutter_clip_end = cut_res.clip_end_seconds
            LOGGER.info("audio trimmed to %s s - %s s", cutter_clip_start, cutter_clip_end)
        except Exception as exc:
            LOGGER.warning("cutting failed: %s. transcribing full audio.", exc)
    else:
        LOGGER.warning("no clapper hits detected. transcribing full audio.")

    # 5. Whisper: transcribe the audio source (either cut or full standardized)
    whisp_runtime = whisper.load_runtime(whisp_config)
    whisp_res = whisper.transcribe_audio(whisper_source, config=whisp_config, runtime=whisp_runtime)

    # 6. LLM Scene Parser: parse the transcribed text
    llm_res = scene_parser.parse_scene(whisp_res, config=llm_config)

    final_audio_path = Path(audio_path)
    if rename:
        renamer_config = load_renamer_config(config_path)
        final_audio_path = new_name(
            final_audio_path,
            sequence=llm_res.scene_take.sequence,
            shot=llm_res.scene_take.shot,
            take=llm_res.scene_take.take,
            template=renamer_config["naming_template"],
            suffix=renamer_config["suffix"],
            extract_from_source=renamer_config["extract_from_source"],
            extract_pattern=renamer_config["extract_pattern"],
            extract_format=renamer_config["extract_format"],
        )

    # 7. Aggregate everything into PipelineResult
    return PipelineResult(
        input_name=audio_path.name,
        new_name=final_audio_path.name,
        preprocessor_codec=std_audio.codec,
        preprocessor_sample_rate=std_audio.sample_rate,
        preprocessor_duration_seconds=round(std_audio.duration_seconds, 6),
        clapper_hits=len(clap_res.best_scores),
        clapper_best_timestamp=best_hit.timestamp if best_hit else None,
        clapper_best_score=best_hit.score if best_hit else None,
        clapper_best_text_key=best_hit.text_key if best_hit else None,
        cutter_clip_start=cutter_clip_start,
        cutter_clip_end=cutter_clip_end,
        whisper_text=whisp_res.detection.text,
        whisper_model=whisp_res.detection.model_id,
        whisper_language=whisp_res.language,
        whisper_task=whisp_res.task,
        llm_model=llm_res.model,
        llm_sequence=llm_res.scene_take.sequence,
        llm_shot=llm_res.scene_take.shot,
        llm_take=llm_res.scene_take.take,
        llm_announcement=llm_res.scene_take.announcement,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process an audio file through the full scene-rename pipeline.")
    parser.add_argument("audio_file", type=Path, help="Path to the input audio file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable summary.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to {audio_file}_result.json next to the input file.",
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
    parser.add_argument(
        "--france",
        action="store_true",
        help="Set Whisper transcription language to French.",
    )
    parser.add_argument(
        "--rename",
        "-rename",
        action="store_true",
        help="Rename the source audio file based on sequence/shot/take.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        lang = args.language
        if args.france:
            lang = "french"

        result = run_pipeline(
            args.audio_file,
            config_path=args.config,
            whisper_language=lang,
            whisper_task=args.task,
            rename=args.rename,
        )

        # Present the result using presentator_cli
        presentator_cli.present(
            result.model_dump(),
            use_case="name_extractor",
            json_output=args.json,
        )

        if args.save:
            parent_dir = Path(args.audio_file).parent
            result_json_name = f"{Path(result.new_name).stem}_result.json"
            output_path = parent_dir / result_json_name
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"\n💾 Pipeline result successfully saved to {output_path}", file=sys.stderr)

        return 0
    except Exception as exc:
        LOGGER.exception("pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
