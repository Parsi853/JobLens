PYTHON ?= python

.PHONY: install sample-data prepare train-role train-salary evaluate api app test lint format
install:
	$(PYTHON) -m pip install -e ".[dev]"
sample-data:
	$(PYTHON) scripts/generate_sample_data.py
prepare:
	$(PYTHON) scripts/prepare_data.py
train-role:
	$(PYTHON) scripts/train_role_classifier.py
train-salary:
	$(PYTHON) scripts/train_salary_model.py
evaluate:
	$(PYTHON) scripts/evaluate_models.py
api:
	$(PYTHON) -m uvicorn joblens.api.main:app --reload
app:
	$(PYTHON) -m streamlit run src/joblens/ui/app.py
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check .
format:
	$(PYTHON) -m ruff format .

