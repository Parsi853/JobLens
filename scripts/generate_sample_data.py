"""Generate a small deterministic dataset for smoke testing only."""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/samples/vacancies_sample.csv"

ROLES = [
    (
        "ML Engineer",
        "Разработка моделей машинного обучения на Python, PyTorch, Docker и MLflow",
        "Python,PyTorch,Docker,MLflow",
    ),
    (
        "Data Scientist",
        "Исследование данных, статистика, Python, pandas, scikit-learn и SQL",
        '["Python", "pandas", "scikit-learn", "SQL"]',
    ),
    (
        "Аналитик данных",
        "Анализ метрик и отчетность: SQL, Python, pandas, статистика",
        "SQL,Python,pandas,статистика",
    ),
    (
        "Data Engineer",
        "Построение ETL на Airflow, Spark, Kafka, PostgreSQL и Python",
        "Airflow,Spark,Kafka,PostgreSQL,Python",
    ),
    (
        "Python developer",
        "Backend-разработка REST API на Python, FastAPI, PostgreSQL, Docker",
        "Python,FastAPI,PostgreSQL,Docker,REST API",
    ),
    (
        "DevOps инженер",
        "Поддержка Linux, Docker, Git и CI/CD инфраструктуры",
        "Linux,Docker,Git,CI/CD",
    ),
]


def main() -> None:
    """Write 60 synthetic rows matching the documented input schema."""
    random.seed(42)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "vacancy_id",
        "name",
        "description",
        "key_skills",
        "area",
        "experience",
        "employment",
        "schedule",
        "salary_from",
        "salary_to",
        "salary_currency",
        "salary_gross",
        "published_at",
        "employer_id",
    ]
    rows = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for index in range(60):
        role_index = index % len(ROLES)
        name, description, skills = ROLES[role_index]
        center = 100_000 + role_index * 25_000 + random.randint(-10_000, 10_000)
        rows.append(
            {
                "vacancy_id": f"sample-{index:03}",
                "name": name,
                "description": (
                    f"<p>{description}</p> Команда создаёт production-сервисы. Вакансия {index}."
                ),
                "key_skills": skills if index % 7 else "",
                "area": ["Москва", "Санкт-Петербург", "Новосибирск"][index % 3],
                "experience": ["Нет опыта", "1-3 года", "3-6 лет"][index % 3],
                "employment": "Полная занятость",
                "schedule": ["Удалённая работа", "Полный день"][index % 2],
                "salary_from": center - 15_000,
                "salary_to": center + 15_000,
                "salary_currency": "RUR",
                "salary_gross": True,
                "published_at": (start + timedelta(days=index)).isoformat(),
                "employer_id": f"employer-{index % 8}",
            }
        )
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} smoke-test rows: {OUTPUT}")


if __name__ == "__main__":
    main()
