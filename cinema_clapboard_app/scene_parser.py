"""LLM Scene Parser to extract scene and take numbers from transcribed text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
import re

from pydantic import BaseModel, Field, field_validator

from .config_utils import load_env_file, merge_config_dict, resolve_path
from .config_utils import get_user_env_path

# Set up logging
LOGGER = logging.getLogger("scene_parser")

project_root = Path(__file__).parent

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_PROMPT_PATH = "prompts/scene_parser.md"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_SEQUENCE_DESCRIPTION = "Sequence number (séquence) (e.g. '1', '10')"
DEFAULT_SHOT_DESCRIPTION = "Shot/plan number (plan) (e.g. '3', '3A')"
DEFAULT_TAKE_DESCRIPTION = "Take/prise number (prise) (e.g. 1, 2, 3)"


class SceneTake(BaseModel):
    """Structured sequence/shot/take information extracted from spoken audio."""

    sequence: str | None = Field(default=None, description="Sequence number (séquence) (e.g. '1', '10')")
    shot: str | None = Field(default=None, description="Shot/plan number (plan) (e.g. '3', '3A')")
    take: int | None = Field(default=None, description="Take/prise number (prise) (e.g. 1, 2, 3)")
    announcement: str = Field(description="The full raw announcement exactly as spoken (unstructured)")

    @field_validator("take", mode="before")
    @classmethod
    def coerce_take_to_int(cls, v: object) -> int | None:
        """Accept int, numeric string, or string containing a number (e.g. '2ème prise')."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        # Extract the first sequence of digits from whatever string the LLM returns
        match = re.search(r"\d+", str(v))
        if match:
            return int(match.group())
        return None


@dataclass(slots=True)
class SceneParserConfig:
    """Config settings for the LLM scene parser."""

    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    api_key_env: str = DEFAULT_API_KEY_ENV
    prompt_path: str = DEFAULT_PROMPT_PATH
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    sequence_description: str = DEFAULT_SEQUENCE_DESCRIPTION
    shot_description: str = DEFAULT_SHOT_DESCRIPTION
    take_description: str = DEFAULT_TAKE_DESCRIPTION


@dataclass(slots=True)
class SceneParseResult:
    """The result of parsing a scene/take announcement via LLM."""

    scene_take: SceneTake
    model: str
    raw_llm_response: str


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")




def load_config(config_path: str | Path | None = None) -> SceneParserConfig:
    """Load the scene_parser section from config.yaml and apply defaults."""
    config_file = Path(config_path) if config_path is not None else _default_config_path()
    raw: dict[str, Any] = {}

    if config_file.exists():
        loaded = yaml.safe_load(config_file.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid config format in {config_file}: expected a mapping")
        raw = loaded.get("scene_parser", loaded)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid scene_parser section in {config_file}: expected a mapping")
    else:
        LOGGER.warning("Config file %s not found, using built-in defaults", config_file)

    # Map old config keys if they exist in config.yaml
    if "scene_description" in raw and "sequence_description" not in raw:
        raw["sequence_description"] = raw["scene_description"]
    if "plan_description" in raw and "shot_description" not in raw:
        raw["shot_description"] = raw["plan_description"]
    if "prise_description" in raw and "take_description" not in raw:
        raw["take_description"] = raw["prise_description"]

    defaults = {
        "model": DEFAULT_MODEL,
        "api_base": DEFAULT_API_BASE,
        "api_key_env": DEFAULT_API_KEY_ENV,
        "prompt_path": DEFAULT_PROMPT_PATH,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "sequence_description": DEFAULT_SEQUENCE_DESCRIPTION,
        "shot_description": DEFAULT_SHOT_DESCRIPTION,
        "take_description": DEFAULT_TAKE_DESCRIPTION,
    }
    merged = merge_config_dict(defaults, raw)
    return SceneParserConfig(
        model=str(merged["model"]),
        api_base=str(merged["api_base"]),
        api_key_env=str(merged["api_key_env"]),
        prompt_path=str(merged["prompt_path"]),
        temperature=float(merged["temperature"]),
        max_tokens=int(merged["max_tokens"]),
        sequence_description=str(merged["sequence_description"]),
        shot_description=str(merged["shot_description"]),
        take_description=str(merged["take_description"]),
    )


def parse_scene(
    whisper_result: Any,
    config: SceneParserConfig | None = None,
) -> SceneParseResult:
    """Parse transcription text via OpenRouter LLM to extract scene and take.

    Args:
        whisper_result: WhisperResult object, dict, or raw transcription string.
        config: SceneParserConfig to override default configuration.

    Returns:
        SceneParseResult: Contains extracted structured data and debug info.
    """
    config = config or load_config()

    # Determine input text from whisper_result
    if isinstance(whisper_result, str):
        text = whisper_result
    elif hasattr(whisper_result, "detection") and hasattr(whisper_result.detection, "text"):
        text = whisper_result.detection.text
    elif isinstance(whisper_result, dict):
        # Allow dictionary input too (e.g. from JSON deserialized Whisper result)
        detection = whisper_result.get("detection", {})
        if isinstance(detection, dict):
            text = detection.get("text", "")
        else:
            text = whisper_result.get("text", "")
    else:
        text = str(whisper_result)

    # Load environment variables
    load_env_file(get_user_env_path())

    # Get API key
    api_key = os.environ.get(config.api_key_env)
    if not api_key or api_key == "your_openrouter_key_here":
        raise ValueError(
            f"API Key not found in environment variable '{config.api_key_env}'. "
            f"Please set it or update your .env file."
        )

    # Read prompt template
    prompt_file = resolve_path(config.prompt_path, project_root)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")

    prompt_template = prompt_file.read_text()

    # Generate JSON schema representation for the LLM
    # We specify sequence, shot and take as requested. We can pass a simplified schema or full JSON schema.
    schema_desc = {
        "type": "object",
        "properties": {
            "sequence": {"type": "string", "description": config.sequence_description},
            "shot": {"type": "string", "description": config.shot_description},
            "take": {"type": "integer", "description": config.take_description},
            "announcement": {"type": "string", "description": "The full raw announcement exactly as spoken"},
        },
        "required": ["sequence", "shot", "take", "announcement"]
    }
    schema_str = json.dumps(schema_desc, indent=2)

    # Format the prompt
    prompt = prompt_template.format(text=text, schema=schema_str)

    print(f"[scene_parser] sending request to OpenRouter ({config.model})...", file=sys.stderr)
    client = OpenAI(base_url=config.api_base, api_key=api_key)

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
    )

    raw_response = response.choices[0].message.content or ""
    print(f"[scene_parser] response received: {raw_response}", file=sys.stderr)

    # Validate JSON response
    try:
        parsed_dict = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {raw_response}") from exc

    # Ensure announcement is present and filled
    if "announcement" not in parsed_dict or not parsed_dict["announcement"]:
        parsed_dict["announcement"] = text

    # Parse and validate with Pydantic
    scene_take = SceneTake.model_validate(parsed_dict)

    return SceneParseResult(
        scene_take=scene_take,
        model=config.model,
        raw_llm_response=raw_response,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract scene and take from transcription using LLM.")
    parser.add_argument("text", type=str, help="Transcription text to parse.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of human-readable summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        result = parse_scene(args.text, config=config)

        if args.json:
            out_data = {
                "sequence": result.scene_take.sequence,
                "shot": result.scene_take.shot,
                "take": result.scene_take.take,
                "announcement": result.scene_take.announcement,
                "llm_model": result.model,
            }
            print(json.dumps(out_data, ensure_ascii=False))
        else:
            print("🎬 LLM Scene Parser summary")
            print(f"  🧠 Model: {result.model}")
            print(f"  🎬 Sequence: {result.scene_take.sequence}")
            print(f"  🎞 Shot: {result.scene_take.shot}")
            print(f"  🎥 Take: {result.scene_take.take}")
            print(f"  📝 Announcement: {result.scene_take.announcement}")
        return 0
    except Exception as exc:
        LOGGER.exception("scene_parser failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
