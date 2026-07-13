"""Configurable deterministic skill extraction."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class SkillExtractor:
    """Extract canonical skill names using synonym-aware boundary matching."""

    def __init__(self, skills: dict[str, list[str]]) -> None:
        self.skills = skills
        self._patterns = [
            (
                canonical,
                re.compile(
                    r"(?<![\w])(?:" + "|".join(re.escape(term) for term in terms) + r")(?![\w])",
                    re.IGNORECASE,
                ),
            )
            for canonical, aliases in skills.items()
            for terms in [[canonical, *aliases]]
        ]

    @classmethod
    def from_yaml(cls, path: Path) -> SkillExtractor:
        """Build an extractor from a UTF-8 YAML mapping."""
        with path.open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f"Skills config must be a mapping: {path}")
        return cls(values)

    def extract(self, text: str | None) -> list[str]:
        """Return unique canonical names in stable configuration order."""
        content = text or ""
        return [canonical for canonical, pattern in self._patterns if pattern.search(content)]
