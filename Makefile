.PHONY: setup setup-dev test lint format run docker
setup:
	uv venv --python 3.11 --allow-existing
	uv pip install -r requirements.txt -r requirements-dev.txt
setup-dev:
	uv venv --python 3.11 --allow-existing
	uv pip install -r requirements-dev.txt
test:
	.venv/bin/python -m pytest -q
lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .
run:
	.venv/bin/uvicorn serving_app.main:app --host 0.0.0.0 --port 8000
docker:
	docker compose -f serving_app/docker-compose.yml up --build
