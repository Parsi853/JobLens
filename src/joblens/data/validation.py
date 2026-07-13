"""Input data validation."""

import pandas as pd

REQUIRED_COLUMNS = {"vacancy_id", "name", "description"}
EXPECTED_COLUMNS = REQUIRED_COLUMNS | {
    "key_skills",
    "area",
    "experience",
    "employment",
    "schedule",
    "salary_from",
    "salary_to",
    "salary_currency",
    "salary_gross",
    "published_at",
    "employer_id",
}


def validate_columns(frame: pd.DataFrame) -> None:
    """Raise an actionable error when required columns are absent."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")


def ensure_optional_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add absent optional columns as null values."""
    result = frame.copy()
    for column in EXPECTED_COLUMNS - set(result.columns):
        result[column] = pd.NA
    return result
