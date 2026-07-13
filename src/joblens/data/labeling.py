"""Deterministic weak labels derived from vacancy titles."""

import re

ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "ml_engineering": (
        "ml engineer",
        "machine learning engineer",
        "инженер машинного обучения",
        "разработчик машинного обучения",
        "ml-разработчик",
    ),
    "data_science": ("data scientist", "специалист по данным", "исследователь данных"),
    "data_analysis": ("аналитик данных", "data analyst", "продуктовый аналитик", "bi analyst"),
    "data_engineering": ("data engineer", "инженер данных", "разработчик хранилищ данных"),
    "python_backend": (
        "python developer",
        "python-разработчик",
        "backend python",
        "backend-разработчик python",
    ),
}


def label_role(name: str) -> str:
    """Map a vacancy title to a supported role class."""
    normalized = re.sub(r"\s+", " ", str(name).casefold()).strip()
    for role, patterns in ROLE_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return role
    return "other"


def add_role_target(frame):
    """Return a copy with a target derived only from the title."""
    result = frame.copy()
    result["target"] = result["name"].map(label_role)
    return result
