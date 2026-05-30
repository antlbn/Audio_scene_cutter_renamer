"""LLM Scene Parser to extract scene and take numbers from transcribed text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

# Set up logging
LOGGER = logging.getLogger("scene_parser")

DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_PROMPT_PATH = "prompts/scene_parser.md"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_SCENE_DESCRIPTION = "Full scene description (e.g. 'сцена тридцать два', 'сцена 3B')"
DEFAULT_TAKE_DESCRIPTION = "The take number (e.g. 1, 2, 3)"


class SceneTake(BaseModel):
    """Structured scene and take information extracted from spoken audio."""

    scene: str = Field(description="The scene name or number (e.g. '3', '3A', '10')")
    take: int = Field(description="The take number (e.g. 1, 2, 3)")
    raw_announcement: str = Field(description="The exact text of the announcement")


@dataclass(slots=True)
class SceneParserConfig:
    """Config settings for the LLM scene parser."""

    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    api_key_env: str = DEFAULT_API_KEY_ENV
    prompt_path: str = DEFAULT_PROMPT_PATH
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    scene_description: str = DEFAULT_SCENE_DESCRIPTION
    take_description: str = DEFAULT_TAKE_DESCRIPTION


@dataclass(slots=True)
class SceneParseResult:
    """The result of parsing a scene/take announcement via LLM."""

    scene_take: SceneTake
    model: str
    raw_llm_response: str


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            if key not in os.environ or os.environ[key] in ("your_openrouter_key_here", "YOUR_HUGGING_FACE_TOKEN_HERE", "YOUR_HF_TOKEN_HERE", "YOUR_KEY_HERE", ""):
                os.environ[key] = value


def _merge_config_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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

    defaults = {
        "model": DEFAULT_MODEL,
        "api_base": DEFAULT_API_BASE,
        "api_key_env": DEFAULT_API_KEY_ENV,
        "prompt_path": DEFAULT_PROMPT_PATH,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "scene_description": DEFAULT_SCENE_DESCRIPTION,
        "take_description": DEFAULT_TAKE_DESCRIPTION,
    }
    merged = _merge_config_dict(defaults, raw)
    return SceneParserConfig(
        model=str(merged["model"]),
        api_base=str(merged["api_base"]),
        api_key_env=str(merged["api_key_env"]),
        prompt_path=str(merged["prompt_path"]),
        temperature=float(merged["temperature"]),
        max_tokens=int(merged["max_tokens"]),
        scene_description=str(merged["scene_description"]),
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
    project_root = Path(__file__).resolve().parent
    _load_env_file(project_root / ".env")

    # Get API key
    api_key = os.environ.get(config.api_key_env)
    if not api_key or api_key == "your_openrouter_key_here":
        raise ValueError(
            f"API Key not found in environment variable '{config.api_key_env}'. "
            f"Please set it or update your .env file."
        )

    # Read prompt template
    prompt_file = _resolve_path(config.prompt_path, project_root)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_file}")

    prompt_template = prompt_file.read_text()

    # Generate JSON schema representation for the LLM
    # We specify scene and take as requested. We can pass a simplified schema or full JSON schema.
    schema_desc = {
        "type": "object",
        "properties": {
            "scene": {"type": "string", "description": config.scene_description},
            "take": {"type": "integer", "description": config.take_description}
        },
        "required": ["scene", "take"]
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

    # Ensure raw_announcement is present and filled
    if "raw_announcement" not in parsed_dict or not parsed_dict["raw_announcement"]:
        parsed_dict["raw_announcement"] = text

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
            # Output raw scene_take fields alongside metadata
            out_data = {
                "scene": result.scene_take.scene,
                "take": result.scene_take.take,
                "raw_announcement": result.scene_take.raw_announcement,
                "llm_model": result.model,
            }
            print(json.dumps(out_data, ensure_ascii=False))
        else:
            print("🎬 LLM Scene Parser summary")
            print(f"  🧠 Model: {result.model}")
            print(f"  🎬 Scene: {result.scene_take.scene}")
            print(f"  🎥 Take: {result.scene_take.take}")
            print(f"  📝 Announcement: {result.scene_take.raw_announcement}")
        return 0
    except Exception as exc:
        LOGGER.exception("scene_parser failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
