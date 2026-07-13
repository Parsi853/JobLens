"""Vacancy data cleaning."""

from __future__ import annotations

import html
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from joblens.data.validation import ensure_optional_columns, validate_columns

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, *, lowercase: bool = True) -> str:
    """Strip HTML, unescape entities and normalize whitespace safely."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    text = html.unescape(TAG_RE.sub(" ", str(value)))
    text = SPACE_RE.sub(" ", text).strip()
    return text.lower() if lowercase else text


def parse_key_skills(value: Any) -> list[str]:
    """Parse comma-separated or JSON-list skills into a clean list."""
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return []
    if isinstance(value, list):
        parts = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            parts = decoded if isinstance(decoded, list) else text.split(",")
        except json.JSONDecodeError:
            parts = text.split(",")
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        skill = clean_text(part, lowercase=False)
        key = skill.casefold()
        if skill and key not in seen:
            result.append(skill)
            seen.add(key)
    return result


def prepare_vacancies(frame: pd.DataFrame, salary_min: float, salary_max: float) -> pd.DataFrame:
    """Validate and clean vacancies while preserving the original frame."""
    validate_columns(frame)
    result = ensure_optional_columns(frame)
    result["description"] = result["description"].map(clean_text)
    result["name"] = result["name"].map(lambda value: clean_text(value, lowercase=False))
    result = result[result["description"].str.len() > 0].drop_duplicates("vacancy_id").copy()
    result["key_skills_list"] = result["key_skills"].map(parse_key_skills)
    result["key_skills"] = result["key_skills_list"].map(lambda values: ", ".join(values))
    result["model_text"] = (
        result["description"] + " " + result["key_skills"].fillna("").str.lower()
    ).str.strip()
    for column in ("salary_from", "salary_to"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["salary_mid"] = result[["salary_from", "salary_to"]].mean(axis=1, skipna=True)
    rub = result["salary_currency"].fillna("").str.upper().eq("RUR") | result[
        "salary_currency"
    ].fillna("").str.upper().eq("RUB")
    valid_salary = result["salary_mid"].between(salary_min, salary_max) & rub
    result.loc[~valid_salary, "salary_mid"] = np.nan
    result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
    return result.reset_index(drop=True)
