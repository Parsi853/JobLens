from joblens.features.skills import SkillExtractor


def extractor():
    return SkillExtractor({"Python": ["python3"], "SQL": ["sql"], "PyTorch": ["torch", "pytorch"]})


def test_extract_expected_skills_in_stable_order():
    assert extractor().extract("PyTorch, SQL and PYTHON3") == ["Python", "SQL", "PyTorch"]


def test_short_skill_word_boundaries():
    assert extractor().extract("nosql database sequential processing") == []
