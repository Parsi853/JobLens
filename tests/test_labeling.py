import pandas as pd

from joblens.data.labeling import add_role_target, label_role
from joblens.features.text import model_text


def test_target_from_title():
    assert label_role("Senior Data Scientist") == "data_science"
    assert label_role("Python-разработчик") == "python_backend"
    assert label_role("Системный администратор") == "other"


def test_no_title_leakage_in_features():
    frame = pd.DataFrame(
        {
            "name": ["Data Scientist"],
            "description": ["generic description"],
            "model_text": ["generic description sql"],
        }
    )
    labeled = add_role_target(frame)
    assert model_text(labeled).iloc[0] == "generic description sql"
    assert "Data Scientist" not in model_text(labeled).iloc[0]
