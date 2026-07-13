"""Text feature helpers."""

import pandas as pd


def model_text(frame: pd.DataFrame) -> pd.Series:
    """Return leakage-safe classifier input; vacancy name is intentionally excluded."""
    if "model_text" in frame:
        return frame["model_text"].fillna("").astype(str)
    return frame["description"].fillna("").astype(str)
