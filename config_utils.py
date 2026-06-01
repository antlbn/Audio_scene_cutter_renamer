"""Shared configuration and environment helpers used across modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_DEFAULT_ENV_PLACEHOLDERS = {
    "your_openrouter_key_here",
    "YOUR_HUGGING_FACE_TOKEN_HERE",
    "YOUR_HF_TOKEN_HERE",
    "YOUR_KEY_HERE",
    "",
}


def merge_config_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        if key not in os.environ or os.environ[key] in _DEFAULT_ENV_PLACEHOLDERS:
            os.environ[key] = value
