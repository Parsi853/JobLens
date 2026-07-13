import pandas as pd

from joblens.models.persistence import load_model, save_model
from joblens.models.role_classifier import build_role_pipeline


def test_training_and_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {
            "text": [
                "python backend api",
                "django python service",
                "sql analytics report",
                "pandas sql dashboard",
                "python fastapi",
                "statistics sql",
            ],
            "target": [
                "python_backend",
                "python_backend",
                "data_analysis",
                "data_analysis",
                "python_backend",
                "data_analysis",
            ],
        }
    )
    config = {
        "random_state": 42,
        "tfidf": {"ngram_range": [1, 2], "min_df": 1, "max_features": 100},
        "logistic_regression": {"C": 1.0, "max_iter": 200},
    }
    model = build_role_pipeline(config).fit(frame["text"], frame["target"])
    path = tmp_path / "pipeline.joblib"
    save_model(model, path)
    restored = load_model(path)
    assert restored.predict(["python api"])[0] == model.predict(["python api"])[0]
