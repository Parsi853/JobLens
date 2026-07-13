"""Safe local artifact persistence."""

from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model to a local joblib file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Load a trusted local model artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
