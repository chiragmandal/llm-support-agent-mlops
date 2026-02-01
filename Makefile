PY=python

venv:
	uv venv || true
	. .venv/bin/activate && uv pip install -r requirements.txt

mlflow:
	mlflow server --host 127.0.0.1 --port 5000 \
	  --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns

run-api-local:
	$(PY) -m src.apps.inference_api.main

ingest-local:
	$(PY) -m src.apps.ingestor.ingest

eval-local:
	$(PY) -m src.apps.evaluator.eval

pipeline:
	$(PY) -m src.pipelines.run_pipeline

docker-build-inference:
	docker build -f infra/docker/Dockerfile.inference -t inference-api:0.1 .

kind-load-inference:
	kind load docker-image inference-api:0.1 --name llm-mlops
