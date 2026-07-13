from fastapi.testclient import TestClient

from joblens.api.main import app


def test_health_without_models():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_validation():
    with TestClient(app) as client:
        response = client.post("/match", json={"resume_text": "", "vacancy_text": "Python"})
    assert response.status_code == 422


def test_model_free_endpoint():
    with TestClient(app) as client:
        response = client.post("/extract-skills", json={"text": "Python and SQL"})
    assert response.status_code == 200
    assert response.json()["skills"] == ["Python", "SQL"]
