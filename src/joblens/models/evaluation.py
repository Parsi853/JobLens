"""Evaluation and reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_json(value: dict[str, Any], path: Path) -> None:
    """Write JSON report with UTF-8 and native scalar conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate classifier headline metrics."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate robust regression metrics, including zero-safe MdAPE."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    nonzero = actual != 0
    mdape = np.median(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return {
        "mae": mean_absolute_error(actual, predicted),
        "median_absolute_error": median_absolute_error(actual, predicted),
        "mdape_percent": mdape,
        "r2": r2_score(actual, predicted) if len(actual) > 1 else float("nan"),
    }


def save_confusion(y_true, y_pred, labels: list[str], path: Path) -> None:
    """Save a confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Role confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_regression_plots(y_true, y_pred, directory: Path) -> None:
    """Save predicted-vs-actual and residual distribution plots."""
    directory.mkdir(parents=True, exist_ok=True)
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.7)
    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(bounds, bounds, "--", color="black")
    ax.set(xlabel="Actual salary", ylabel="Predicted salary", title="Predicted vs actual")
    fig.tight_layout()
    fig.savefig(directory / "predicted_vs_actual.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.hist(predicted - actual, bins=min(20, max(5, len(actual) // 2)))
    ax.set(xlabel="Prediction error", ylabel="Count", title="Error distribution")
    fig.tight_layout()
    fig.savefig(directory / "error_distribution.png")
    plt.close(fig)


def full_classification_report(y_true, y_pred) -> dict[str, Any]:
    """Return precision, recall and F1 per class."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
