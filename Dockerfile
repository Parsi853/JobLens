FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "joblens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

