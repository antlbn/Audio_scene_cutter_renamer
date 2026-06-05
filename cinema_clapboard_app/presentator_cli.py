"""Console presenter for structured pipeline results.

Takes the result dict from pipeline and renders it in human-readable format
based on the specified use case. Supports multiple output formats.

Contract:
    name_extractor: requires input_name, new_name, llm_sequence, llm_shot, llm_take, llm_announcement
                    optional clapper_hits, clapper_best_timestamp, whisper_text
    
    whisper_extractor: requires input_name, whisper_text, whisper_language, whisper_model
                       optional clapper_best_timestamp
"""

import json
from typing import Any


def present(
    data: dict[str, Any],
    use_case: str = "name_extractor",
    json_output: bool = False,
) -> None:
    """Present the result data to console based on use case.
    
    Args:
        data: Result dict (typically from PipelineResult.model_dump())
        use_case: One of 'name_extractor' or 'whisper_extractor'
        json_output: If True, output raw JSON instead of formatted text
    """
    if json_output:
        print(json.dumps(data, indent=2))
        return

    if use_case == "name_extractor":
        _present_name_extractor(data)
    elif use_case == "whisper_extractor":
        _present_whisper_extractor(data)
    else:
        # Fallback: name_extractor is default
        _present_name_extractor(data)


def _present_name_extractor(data: dict[str, Any]) -> None:
    """Present results for the name extraction use case."""
    lines = [
        "🎬 Scene Naming Pipeline Result",
        "=" * 50,
        f"📁 Input File:       {data.get('input_name', 'N/A')}",
        f"📦 New File Name:    {data.get('new_name', 'N/A')}",
        "",
        "📋 Extracted Metadata:",
    ]

    # Extracted metadata
    sequence = data.get("llm_sequence")
    shot = data.get("llm_shot")
    take = data.get("llm_take")
    announcement = data.get("llm_announcement", "")

    if sequence or shot or take:
        lines.append(f"   🎬 Sequence:       {sequence or '—'}")
        lines.append(f"   🎞  Shot:          {shot or '—'}")
        lines.append(f"   🎥 Take:          {take or '—'}")
    else:
        lines.append("   (No metadata extracted)")

    if announcement:
        lines.append(f"   📝 Announcement:   \"{announcement}\"")

    # Optional: clapper detection summary
    if data.get("clapper_hits"):
        lines.extend([
            "",
            "🎙️  Clapper Detection:",
            f"   Hits Found:       {data.get('clapper_hits', 0)}",
        ])
        if data.get("clapper_best_timestamp") is not None:
            lines.append(f"   Best Hit Time:    {data.get('clapper_best_timestamp'):.2f} s")

    # Optional: transcription snippet
    if data.get("whisper_text"):
        whisper_text = data.get("whisper_text", "")
        lines.extend([
            "",
            "🗣  Transcription:",
            f"   \"{whisper_text}\"",
        ])

    print("\n".join(lines))


def _present_whisper_extractor(data: dict[str, Any]) -> None:
    """Present results for the Whisper transcription use case."""
    lines = [
        "🗣  Speech Recognition Result",
        "=" * 50,
        f"📁 Input File:       {data.get('input_name', 'N/A')}",
        "",
        "🤖 Transcription:",
        f"   Model:            {data.get('whisper_model', 'N/A')}",
        f"   Language:         {data.get('whisper_language', 'N/A')}",
        f"   Task:             {data.get('whisper_task', 'N/A')}",
        "",
        "📝 Detected Text:",
    ]

    whisper_text = data.get("whisper_text", "")
    lines.append(f"   {whisper_text}")

    # Optional: clapper timestamp if available
    if data.get("clapper_best_timestamp") is not None:
        lines.extend([
            "",
            "🎙️  Detection Context:",
            f"   Clapper Found At: {data.get('clapper_best_timestamp'):.2f} s",
        ])

    print("\n".join(lines))
