"""JobLens REST API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from joblens.schemas import AnalyzeRequest, MatchRequest, SalaryRequest, TextRequest
from joblens.service.inference import JobLensService, ModelUnavailableError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once for the application process."""
    app.state.service = JobLensService()
    yield


app = FastAPI(title="JobLens ML", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def service(request: Request) -> JobLensService:
    """Access the process-local service."""
    return request.app.state.service


@app.get("/health")
def health(request: Request):
    """Report process health and artifact availability."""
    current = service(request)
    return {
        "status": "ok",
        "role_model": current.role_model is not None,
        "salary_model": current.salary_model is not None,
    }


@app.post("/predict/role")
def predict_role(payload: TextRequest, request: Request):
    """Predict vacancy role."""
    try:
        return service(request).predict_role(payload.text)
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/predict/salary")
def predict_salary(payload: SalaryRequest, request: Request):
    """Predict approximate salary."""
    try:
        return {"salary_rub": service(request).predict_salary(**payload.model_dump())}
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/extract-skills")
def extract_skills(payload: TextRequest, request: Request):
    """Extract configured skills without requiring trained models."""
    return {"skills": service(request).extract_skills(payload.text)}


@app.post("/match")
def match(payload: MatchRequest, request: Request):
    """Calculate resume-vacancy match."""
    return service(request).match_resume(**payload.model_dump())


@app.post("/analyze-vacancy")
def analyze(payload: AnalyzeRequest, request: Request):
    """Run complete or partial vacancy analysis."""
    values = payload.model_dump()
    resume, vacancy = values.pop("resume_text"), values.pop("vacancy_text")
    return service(request).analyze_vacancy(resume, vacancy, **values)
