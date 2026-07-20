"""Shared configuration and logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging(level: str = "INFO") -> None:
    """Configure concise application logging."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict[str, Any]:
    """Load and minimally validate the YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "region" not in config or "data" not in config:
        raise ValueError("Configuration must contain 'region' and 'data' mappings")
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
