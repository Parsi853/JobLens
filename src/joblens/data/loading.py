"""CSV input/output helpers."""

from pathlib import Path

import pandas as pd


def load_vacancies(path: Path) -> pd.DataFrame:
    """Load vacancies from UTF-8 CSV without modifying the source."""
    if not path.exists():
        raise FileNotFoundError(f"Vacancy CSV not found: {path}")
    return pd.read_csv(path)


def save_vacancies(frame: pd.DataFrame, path: Path) -> None:
    """Save prepared vacancies as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
