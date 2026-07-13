"""Salary model feature construction."""

import pandas as pd

from joblens.features.skills import SkillExtractor

CATEGORICAL = ["area", "experience", "employment", "schedule"]
NUMERIC = ["description_length", "word_count", "skill_count"]
SALARY_FEATURES = ["description", *CATEGORICAL, *NUMERIC]


def build_salary_features(frame: pd.DataFrame, extractor: SkillExtractor) -> pd.DataFrame:
    """Create text, categorical and simple numeric salary features."""
    result = frame.copy()
    result["description"] = result["description"].fillna("").astype(str)
    for column in CATEGORICAL:
        result[column] = (
            result.get(column, pd.Series(index=result.index, dtype=str)).fillna("").astype(str)
        )
    result["description_length"] = result["description"].str.len()
    result["word_count"] = result["description"].str.split().str.len()
    result["skill_count"] = result["description"].map(lambda value: len(extractor.extract(value)))
    return result[SALARY_FEATURES]
