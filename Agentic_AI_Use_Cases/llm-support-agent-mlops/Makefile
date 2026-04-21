PY     = python
PORT   = 5001   # matches MLFLOW_TRACKING_URI default

# ── Setup ──────────────────────────────────────────────────────────────────
venv:
	uv venv || true
	. .venv/bin/activate && uv pip install -r requirements.txt

# ── Local services ─────────────────────────────────────────────────────────
mlflow:
	mlflow server \
	  --host 127.0.0.1 --port $(PORT) \
	  --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns

qdrant:
	docker run --rm -p 6333:6333 qdrant/qdrant:v1.11.0

# ── Pipeline steps ─────────────────────────────────────────────────────────
ingest:
	$(PY) -m src.apps.ingestor.ingest

eval:
	$(PY) -m src.apps.evaluator.eval

pipeline:
	$(PY) -m src.pipelines.run_pipeline

# ── Inference API ──────────────────────────────────────────────────────────
run-api:
	$(PY) -m src.apps.inference_api.main

# ── Docker ─────────────────────────────────────────────────────────────────
docker-build:
	docker build -f infra/docker/Dockerfile.inference -t inference-api:0.1 .

kind-load:
	kind load docker-image inference-api:0.1 --name llm-mlops

# ── Housekeeping ───────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

.PHONY: venv mlflow qdrant ingest eval pipeline run-api docker-build kind-load clean
