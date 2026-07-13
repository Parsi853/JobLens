from joblens.features.skills import SkillExtractor
from joblens.matching.matcher import ResumeMatcher


def matcher():
    return ResumeMatcher(
        SkillExtractor({"Python": ["python"], "SQL": ["sql"]}),
        {"text_weight": 0.6, "skill_weight": 0.4, "neutral_skill_coverage": 0.5},
    )


def test_match_score_and_skill_sets():
    result = matcher().match("Python developer", "Python SQL developer")
    assert result.matched_skills == ["Python"]
    assert result.missing_skills == ["SQL"]
    assert result.skill_coverage == 0.5
    assert 0 <= result.score <= 100


def test_empty_vocabulary_is_safe():
    result = matcher().match("", "")
    assert result.text_similarity == 0
    assert result.skill_coverage == 0.5
    assert result.score == 20
