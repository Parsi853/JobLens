import pandas as pd

from joblens.data.cleaning import clean_text, prepare_vacancies


def test_clean_html_and_entities():
    assert clean_text("<p>Hello&nbsp; <b>МИР</b></p>") == "hello мир"


def test_empty_text():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_prepare_drops_empty_and_does_not_mutate_source():
    source = pd.DataFrame(
        {"vacancy_id": [1, 2], "name": ["A", "B"], "description": ["<b>Text</b>", None]}
    )
    result = prepare_vacancies(source, 20_000, 1_000_000)
    assert len(result) == 1
    assert source.loc[0, "description"] == "<b>Text</b>"
