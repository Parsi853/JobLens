"""Project configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    root: Path
    values: dict[str, Any]

    def path(self, key: str) -> Path:
        """Return an absolute configured project path."""
        value = Path(self.values["paths"][key])
        return value if value.is_absolute() else self.root / value


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load YAML settings, resolving paths relative to the project root."""
    candidate = Path(config_path or os.getenv("JOBLENS_CONFIG", "configs/config.yaml"))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    with candidate.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {candidate}")
    return Settings(PROJECT_ROOT, values)
