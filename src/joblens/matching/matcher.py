"""Transparent heuristic matching score."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from joblens.features.skills import SkillExtractor
from joblens.schemas import MatchResult


class ResumeMatcher:
    """Combine TF-IDF similarity with required-skill coverage."""

    def __init__(self, extractor: SkillExtractor, config: dict) -> None:
        self.extractor = extractor
        self.config = config

    def match(self, resume_text: str, vacancy_text: str) -> MatchResult:
        """Calculate a bounded 0–100 heuristic score."""
        try:
            vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([resume_text, vacancy_text])
            similarity = float(cosine_similarity(vectors[0], vectors[1])[0, 0])
        except ValueError:
            similarity = 0.0
        vacancy_skills = self.extractor.extract(vacancy_text)
        resume_skills = self.extractor.extract(resume_text)
        resume_set = set(resume_skills)
        matched = [skill for skill in vacancy_skills if skill in resume_set]
        missing = [skill for skill in vacancy_skills if skill not in resume_set]
        # With no explicit requirements, neutral coverage avoids rewarding or penalizing absence.
        coverage = (
            len(matched) / len(vacancy_skills)
            if vacancy_skills
            else self.config["neutral_skill_coverage"]
        )
        score = 100 * (
            self.config["text_weight"] * similarity + self.config["skill_weight"] * coverage
        )
        return MatchResult(
            score=round(max(0.0, min(100.0, score)), 2),
            text_similarity=round(similarity, 4),
            skill_coverage=round(coverage, 4),
            vacancy_skills=vacancy_skills,
            resume_skills=resume_skills,
            matched_skills=matched,
            missing_skills=missing,
        )
