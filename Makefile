.PHONY: install run-gradio run-api run-grpc test lint format docker-up docker-down clean ingest eval load-test proto

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[all]"
	python -m spacy download en_core_web_sm

install-uv:
	uv sync --all-extras

# ─── Run ──────────────────────────────────────────────────────────────────────
run-gradio:
	python ui/gradio_app.py

run-api:
	uvicorn api.rest.app:app --reload --host 0.0.0.0 --port ${API_PORT:-8000}

run-grpc:
	python api/grpc/server.py

run-worker:
	python api/rest/worker.py

# ─── Data ─────────────────────────────────────────────────────────────────────
ingest:
	python data/ingest.py

build-graph:
	python graph_db/schema/build_graph.py

# ─── Fine-tuning ──────────────────────────────────────────────────────────────
prepare-dataset:
	python finetuning/dataset_prep.py

finetune:
	python finetuning/train.py

eval-ft:
	python finetuning/eval.py

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-retrieval:
	pytest tests/eval/ -v -m eval

load-test:
	locust -f tests/load/locustfile.py --host=http://localhost:8000

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check .
	mypy . --ignore-missing-imports

format:
	ruff format .
	ruff check --fix .

# ─── Proto generation ─────────────────────────────────────────────────────────
proto:
	python -m grpc_tools.protoc \
		-I api/grpc \
		--python_out=api/grpc \
		--grpc_python_out=api/grpc \
		api/grpc/rag_service.proto

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d
	@echo "Waiting for services..."
	@sleep 5
	@docker-compose ps

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-reset:
	docker-compose down -v
	docker-compose up -d

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info
