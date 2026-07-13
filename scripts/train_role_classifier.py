"""Train and evaluate role classification models."""

from __future__ import annotations

import logging

import numpy as np

from joblens.data.loading import load_vacancies
from joblens.features.text import model_text
from joblens.models.evaluation import (
    classification_metrics,
    full_classification_report,
    save_confusion,
    write_json,
)
from joblens.models.persistence import save_model
from joblens.models.role_classifier import split_frame, train_role_models
from joblens.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Train classifier, compare baseline and persist real metrics/artifacts."""
    settings = load_settings()
    config = settings.values
    frame = load_vacancies(settings.path("processed_data"))
    minimum = config["training"]["min_role_rows"]
    if len(frame) < minimum:
        raise SystemExit(f"Not enough role rows: {len(frame)}; at least {minimum} required")
    frame["published_at"] = __import__("pandas").to_datetime(
        frame["published_at"], errors="coerce", utc=True
    )
    train, validation, test, method = split_frame(frame, config)
    model, baseline = train_role_models(train, config)
    eval_frame = __import__("pandas").concat([validation, test])
    truth = eval_frame["target"]
    predictions = model.predict(model_text(eval_frame))
    baseline_predictions = baseline.predict(np.zeros((len(eval_frame), 1)))
    metrics = {
        "model": classification_metrics(truth, predictions),
        "baseline": classification_metrics(truth, baseline_predictions),
        "split_method": method,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    output = settings.path("artifacts") / "role_classifier"
    save_model(model, output / "pipeline.joblib")
    write_json(metrics, output / "metrics.json")
    write_json(
        full_classification_report(truth, predictions), output / "classification_report.json"
    )
    write_json({"classes": list(model.classes_), "config": config}, output / "run_metadata.json")
    save_confusion(
        truth,
        predictions,
        list(model.classes_),
        settings.path("reports") / "figures/confusion_matrix.png",
    )
    LOGGER.info("Role classifier saved to %s; macro F1 %.4f", output, metrics["model"]["macro_f1"])


if __name__ == "__main__":
    main()
