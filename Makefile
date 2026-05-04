.PHONY: install test run lint

install:
	pip install -r requirements.txt && pip install -e ".[dev]"

test:
	OPENAI_API_KEY="" pytest tests/ -v

run:
	uvicorn src.main:app --reload --port 8000

lint:
	ruff check src/ tests/
