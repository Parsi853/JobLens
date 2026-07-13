"""Salary regression training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from joblens.features.salary import CATEGORICAL, NUMERIC, build_salary_features
from joblens.features.skills import SkillExtractor


def build_salary_pipeline(config: dict) -> Pipeline:
    """Build heterogeneous salary regression pipeline."""
    tfidf = config["tfidf"]
    features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf["ngram_range"]),
                    min_df=tfidf["min_df"],
                    max_features=tfidf["max_features"],
                    sublinear_tf=True,
                ),
                "description",
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", StandardScaler(with_mean=False), NUMERIC),
        ]
    )
    return Pipeline([("features", features), ("regressor", Ridge(alpha=config["ridge"]["alpha"]))])


def train_salary_models(train: pd.DataFrame, extractor: SkillExtractor, config: dict):
    """Fit Ridge and median baseline on rows with valid salary targets."""
    valid = train.dropna(subset=["salary_mid"])
    minimum = config["training"]["min_salary_rows"]
    if len(valid) < minimum:
        raise ValueError(f"Not enough salary rows: {len(valid)}; at least {minimum} required")
    x, y = build_salary_features(valid, extractor), valid["salary_mid"]
    model = build_salary_pipeline(config).fit(x, y)
    baseline = DummyRegressor(strategy="median").fit(np.zeros((len(y), 1)), y)
    return model, baseline
