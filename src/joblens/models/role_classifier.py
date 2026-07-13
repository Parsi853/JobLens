"""Role classifier training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from joblens.features.text import model_text


def split_frame(
    frame: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Use chronological split when dates are sufficiently complete, otherwise stratify."""
    split = config["split"]
    if frame["published_at"].notna().mean() >= split["min_valid_dates_ratio"]:
        ordered = frame.sort_values("published_at")
        first = int(len(frame) * split["train_size"])
        second = int(len(frame) * (split["train_size"] + split["validation_size"]))
        return (
            ordered.iloc[:first],
            ordered.iloc[first:second],
            ordered.iloc[second:],
            "chronological",
        )
    stratify = frame["target"] if frame["target"].value_counts().min() >= 2 else None
    train, rest = train_test_split(
        frame,
        train_size=split["train_size"],
        random_state=config["random_state"],
        stratify=stratify,
    )
    relative_validation = split["validation_size"] / (split["validation_size"] + split["test_size"])
    rest_stratify = rest["target"] if rest["target"].value_counts().min() >= 2 else None
    validation, test = train_test_split(
        rest,
        train_size=relative_validation,
        random_state=config["random_state"],
        stratify=rest_stratify,
    )
    return train, validation, test, "stratified_random"


def build_role_pipeline(config: dict) -> Pipeline:
    """Build TF-IDF plus balanced logistic regression."""
    tfidf, logistic = config["tfidf"], config["logistic_regression"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf["ngram_range"]),
                    min_df=tfidf["min_df"],
                    max_features=tfidf["max_features"],
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=logistic["C"],
                    max_iter=logistic["max_iter"],
                    class_weight="balanced",
                    random_state=config["random_state"],
                ),
            ),
        ]
    )


def train_role_models(train: pd.DataFrame, config: dict):
    """Fit the main classifier and most-frequent baseline without title leakage."""
    x, y = model_text(train), train["target"]
    if y.nunique() < 2:
        raise ValueError("Role training requires at least two target classes")
    model = build_role_pipeline(config).fit(x, y)
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y), 1)), y)
    return model, baseline
