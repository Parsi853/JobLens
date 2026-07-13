"""Train and evaluate salary regression models."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from joblens.data.loading import load_vacancies
from joblens.features.salary import build_salary_features
from joblens.features.skills import SkillExtractor
from joblens.models.evaluation import regression_metrics, save_regression_plots, write_json
from joblens.models.persistence import save_model
from joblens.models.role_classifier import split_frame
from joblens.models.salary_regressor import train_salary_models
from joblens.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Train salary model, exiting clearly if too few labeled rows exist."""
    settings = load_settings()
    config = settings.values
    frame = load_vacancies(settings.path("processed_data"))
    frame["salary_mid"] = pd.to_numeric(frame["salary_mid"], errors="coerce")
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True)
    extractor = SkillExtractor.from_yaml(settings.path("skills"))
    train, validation, test, method = split_frame(frame, config)
    try:
        model, baseline = train_salary_models(train, extractor, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    evaluation = pd.concat([validation, test]).dropna(subset=["salary_mid"])
    if evaluation.empty:
        raise SystemExit("No salary rows in validation/test split; cannot evaluate")
    truth = evaluation["salary_mid"]
    predictions = model.predict(build_salary_features(evaluation, extractor))
    baseline_predictions = baseline.predict(np.zeros((len(evaluation), 1)))
    metrics = {
        "model": regression_metrics(truth, predictions),
        "baseline": regression_metrics(truth, baseline_predictions),
        "split_method": method,
        "rows": {
            "train_salary": int(train["salary_mid"].notna().sum()),
            "evaluation_salary": len(evaluation),
        },
    }
    output = settings.path("artifacts") / "salary_model"
    save_model(model, output / "pipeline.joblib")
    write_json(metrics, output / "metrics.json")
    save_regression_plots(truth, predictions, settings.path("reports") / "figures")
    LOGGER.info("Salary model saved to %s; MAE %.2f", output, metrics["model"]["mae"])


if __name__ == "__main__":
    main()
