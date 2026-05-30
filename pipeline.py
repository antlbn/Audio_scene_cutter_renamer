"""Pipeline orchestrator to process an audio file through all stages.

Runs preprocessor, clapper detector, cutter, whisper transcriber, and LLM scene parser,
then aggregates the metadata into a single structured result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import clapper
import cutter
import preprocessor
import scene_parser
import whisper

LOGGER = logging.getLogger("pipeline")


class PipelineResult(BaseModel):
    """Final aggregated result of the audio scene processing pipeline."""

    source_file: str = Field(description="Name of the source audio file")

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
    llm_scene: str = Field(description="Extracted scene name/number")
    llm_take: int = Field(description="Extracted take number")
    llm_raw_announcement: str = Field(description="Verification text of the extracted announcement")


def run_pipeline(
    audio_path: str | Path,
    *,
    config_path: str | Path | None = None,
    whisper_language: str | None = None,
    whisper_task: str | None = None,
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
    print(f"[pipeline] standardizing audio: {audio_path.name}", file=sys.stderr)
    std_audio = preprocessor.standardize_audio(audio_path, config=prep_config)

    # 3. Clapper: run CLAP detector
    clap_runtime = clapper.load_runtime(clap_config)
    clap_res = clapper.analyze_audio(std_audio, config=clap_config, runtime=clap_runtime)

    # 4. Cutter & Whisper: run based on clapper hits
    best_hit = None
    cutter_clip_start = None
    cutter_clip_end = None
    whisper_source = std_audio

    if clap_res.best_scores:
        # Determine best hit (highest score)
        best_hit = max(clap_res.best_scores, key=lambda hit: (float(hit.score), -float(hit.timestamp)))
        print(f"[pipeline] best clapper hit at {best_hit.timestamp:.2f}s (score {best_hit.score:.3f})", file=sys.stderr)

        # Run cutter to trim audio in memory
        try:
            cut_res = cutter.cut_audio_by_clapper(std_audio, clap_res, config=clap_config)
            whisper_source = cut_res.standardized_audio
            cutter_clip_start = cut_res.clip_start_seconds
            cutter_clip_end = cut_res.clip_end_seconds
            print(f"[pipeline] audio trimmed to {cutter_clip_start}s - {cutter_clip_end}s", file=sys.stderr)
        except Exception as exc:
            print(f"[pipeline] ⚠️ cutting failed: {exc}. transcribing full audio.", file=sys.stderr)
    else:
        print("[pipeline] ⚠️ no clapper hits detected. transcribing full audio.", file=sys.stderr)

    # 5. Whisper: transcribe the audio source (either cut or full standardized)
    whisp_runtime = whisper.load_runtime(whisp_config)
    whisp_res = whisper.transcribe_audio(whisper_source, config=whisp_config, runtime=whisp_runtime)

    # 6. LLM Scene Parser: parse the transcribed text
    llm_res = scene_parser.parse_scene(whisp_res, config=llm_config)

    # 7. Aggregate everything into PipelineResult
    return PipelineResult(
        source_file=audio_path.name,
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
        llm_scene=llm_res.scene_take.scene,
        llm_take=llm_res.scene_take.take,
        llm_raw_announcement=llm_res.scene_take.raw_announcement,
    )


def render_pipeline_result(result: PipelineResult) -> str:
    """Render a premium human-readable console summary of the pipeline execution."""
    lines = [
        "⚡️ Audio Scene Pipeline Execution Summary",
        f"📁 Source File:      {result.source_file}",
        "========================================",
        "⚙️  Preprocessor Standardized Audio:",
        f"   Codec:            {result.preprocessor_codec}",
        f"   Sample Rate:      {result.preprocessor_sample_rate} Hz",
        f"   Duration:         {result.preprocessor_duration_seconds:.2f} s",
        "",
        "🎬 Clapper Detection:",
        f"   Total Hits:       {result.clapper_hits}",
    ]

    if result.clapper_hits > 0:
        lines.extend([
            f"   Best Hit Time:    {result.clapper_best_timestamp:.2f} s",
            f"   Best Hit Score:   {result.clapper_best_score:.3f}",
            f"   Best Text Key:    '{result.clapper_best_text_key}'",
        ])
    else:
        lines.append("   Best Hit Time:    None (No clap detected)")

    lines.extend([
        "",
        "✂️  Cutter Trim Range:",
    ])
    if result.cutter_clip_start is not None:
        lines.extend([
            f"   Clip Start:       {result.cutter_clip_start:.2f} s",
            f"   Clip End:         {result.cutter_clip_end:.2f} s",
            f"   Clip Duration:    {(result.cutter_clip_end - result.cutter_clip_start):.2f} s",
        ])
    else:
        lines.append("   Clip Start:       None (No trim performed)")

    lines.extend([
        "",
        "🗣  Speech-To-Text (Whisper):",
        f"   Model:            {result.whisper_model}",
        f"   Language:         {result.whisper_language}",
        f"   Task:             {result.whisper_task}",
        f"   Transcription:    \"{result.whisper_text}\"",
        "",
        "🤖 Structured Parser (LLM):",
        f"   Model:            {result.llm_model}",
        f"   🎬 Scene:         {result.llm_scene}",
        f"   🎥 Take:          {result.llm_take}",
        f"   📝 Verification:  \"{result.llm_raw_announcement}\"",
    ])

    return "\n".join(lines)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = run_pipeline(
            args.audio_file,
            config_path=args.config,
            whisper_language=args.language,
            whisper_task=args.task,
        )

        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(render_pipeline_result(result))

        if args.save:
            output_name = f"{args.audio_file.stem}_result.json"
            output_path = args.audio_file.parent / output_name
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"\n💾 Pipeline result successfully saved to {output_path}", file=sys.stderr)

        return 0
    except Exception as exc:
        LOGGER.exception("pipeline execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
