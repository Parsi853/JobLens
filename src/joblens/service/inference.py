"""Long-lived inference facade."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from joblens.data.cleaning import clean_text
from joblens.features.salary import build_salary_features
from joblens.features.skills import SkillExtractor
from joblens.matching.matcher import ResumeMatcher
from joblens.models.persistence import load_model
from joblens.schemas import AnalysisResult, RolePrediction
from joblens.settings import Settings, load_settings

LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a requested trained model is absent."""


class JobLensService:
    """Load trusted artifacts once and expose all inference operations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.extractor = SkillExtractor.from_yaml(self.settings.path("skills"))
        self.matcher = ResumeMatcher(self.extractor, self.settings.values["match"])
        artifact_dir = self.settings.path("artifacts")
        self.role_model = self._optional_load(artifact_dir / "role_classifier" / "pipeline.joblib")
        self.salary_model = self._optional_load(artifact_dir / "salary_model" / "pipeline.joblib")

    @staticmethod
    def _optional_load(path: Path):
        if not path.exists():
            LOGGER.warning("Optional model is unavailable: %s", path)
            return None
        return load_model(path)

    def predict_role(self, text: str) -> RolePrediction:
        """Predict the vacancy direction and calibrated-like class scores."""
        if self.role_model is None:
            raise ModelUnavailableError(
                "Role classifier is not trained; run train_role_classifier.py"
            )
        prepared = clean_text(text)
        predicted = str(self.role_model.predict([prepared])[0])
        probabilities: dict[str, float] = {}
        confidence = None
        if hasattr(self.role_model, "predict_proba"):
            values = self.role_model.predict_proba([prepared])[0]
            classes = self.role_model.classes_
            probabilities = {
                str(label): round(float(value), 6)
                for label, value in zip(classes, values, strict=True)
            }
            confidence = max(probabilities.values())
        return RolePrediction(
            predicted_class=predicted, confidence=confidence, probabilities=probabilities
        )

    def predict_salary(self, description: str, **metadata: str) -> float:
        """Predict salary in RUB if a salary model is available."""
        if self.salary_model is None:
            raise ModelUnavailableError("Salary model is not trained; run train_salary_model.py")
        frame = pd.DataFrame([{**metadata, "description": clean_text(description)}])
        features = build_salary_features(frame, self.extractor)
        return round(float(self.salary_model.predict(features)[0]), 2)

    def extract_skills(self, text: str) -> list[str]:
        """Extract canonical skills."""
        return self.extractor.extract(text)

    def match_resume(self, resume_text: str, vacancy_text: str):
        """Compare a resume and vacancy."""
        return self.matcher.match(resume_text, vacancy_text)

    def analyze_vacancy(
        self, resume_text: str, vacancy_text: str, **metadata: str
    ) -> AnalysisResult:
        """Return all available analysis and explicit partial-result warnings."""
        warnings = ["Match score is heuristic and is not a hiring probability or decision."]
        role = None
        salary = None
        try:
            role = self.predict_role(vacancy_text)
        except ModelUnavailableError as error:
            warnings.append(str(error))
        try:
            salary = self.predict_salary(vacancy_text, **metadata)
        except ModelUnavailableError as error:
            warnings.append(str(error))
        return AnalysisResult(
            role=role,
            match=self.match_resume(resume_text, vacancy_text),
            salary=salary,
            salary_available=salary is not None,
            warnings=warnings,
        )
