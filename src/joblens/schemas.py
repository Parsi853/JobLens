"""Pydantic request and response schemas."""

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """A non-empty text payload."""

    text: str = Field(min_length=1, max_length=100_000)


class SalaryRequest(BaseModel):
    """Features needed for salary inference."""

    description: str = Field(min_length=1, max_length=100_000)
    area: str = ""
    experience: str = ""
    employment: str = ""
    schedule: str = ""


class MatchRequest(BaseModel):
    """Resume and vacancy text pair."""

    resume_text: str = Field(min_length=1, max_length=100_000)
    vacancy_text: str = Field(min_length=1, max_length=100_000)


class AnalyzeRequest(MatchRequest):
    """Full vacancy analysis payload."""

    area: str = ""
    experience: str = ""
    employment: str = ""
    schedule: str = ""


class RolePrediction(BaseModel):
    """Role classifier result."""

    predicted_class: str
    confidence: float | None
    probabilities: dict[str, float]


class MatchResult(BaseModel):
    """Resume matching result."""

    score: float
    text_similarity: float
    skill_coverage: float
    vacancy_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]


class AnalysisResult(BaseModel):
    """Aggregated analysis result."""

    role: RolePrediction | None
    match: MatchResult
    salary: float | None
    salary_available: bool
    warnings: list[str]
