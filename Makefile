.PHONY: help install setup train run dev docker-up docker-down docker-logs clean test format lint

help:
	@echo "NBA Betting Agent Pro - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make setup          - Complete setup (install + train)"
	@echo ""
	@echo "Running:"
	@echo "  make run            - Run API server"
	@echo "  make dev            - Run in development mode (auto-reload)"
	@echo "  make scheduler      - Run with scheduler (recommended)"
	@echo ""
	@echo "Training & Updates:"
	@echo "  make train          - Train models from scratch"
	@echo "  make update         - Update data (injuries, games, incremental training)"
	@echo "  make train-inc      - Incremental model update"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start all services"
	@echo "  make docker-down    - Stop all services"
	@echo "  make docker-logs    - View logs"
	@echo "  make docker-restart - Restart services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make test           - Run tests"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make lint           - Run linters"
	@echo "  make clean          - Clean up cache and temp files"
	@echo ""

install:
	pip install -r requirements.txt

setup: install
	mkdir -p data/cache models logs
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "⚠️  Please edit .env and add your ODDS_API_KEY"; \
		exit 1; \
	fi
	python train.py --full
	python daily_update.py

train:
	python train.py --full

train-inc:
	python train.py

update:
	python daily_update.py --retrain-model

run:
	uvicorn main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

scheduler:
	@echo "Starting API and Scheduler..."
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@python daily_update.py --schedule & \
	uvicorn main:app --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "✓ Services started!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart:
	docker-compose restart

test:
	pytest tests/ -v --cov=. --cov-report=html

format:
	black *.py
	isort *.py

lint:
	flake8 *.py --max-line-length=120
	black --check *.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf data/cache/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete

clean-all: clean
	rm -rf data/*.csv data/*.json
	rm -rf models/*.pkl models/*.json
	docker-compose down -v
	docker system prune -f