.PHONY: install test run lint

install:
	py -m pip install -r requirements.txt && py -m pip install -e ".[dev]"

test:
	OPENAI_API_KEY="" py -m pytest tests/ -v

run:
	py -m uvicorn src.main:app --reload --port 8000

lint:
	py -m ruff check src/ tests/
