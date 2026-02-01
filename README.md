# LLM Support Agent (RAG) — End-to-End MLOps with ZenML + MLflow + Qdrant + Kubernetes

A **free, local-first** end-to-end MLOps project that builds a **RAG-based customer support agent** and tracks quality with a reproducible pipeline.

**Stack**
- **LLM**: Ollama (local, no API keys)
- **Vector DB**: Qdrant
- **API**: FastAPI
- **Pipeline**: ZenML (ingest → eval → register + quality gate)
- **Experiment Tracking**: MLflow (metrics + artifacts)
- **Containers / K8s**: Docker + kind (optional deployment)

---

## Architecture

![LLM Support Agent RAG Architecture](/data/image/LLM_Support_Agent_RAG_MLOps_Architecture.png)

## Features
- RAG ingestion: loads Markdown KB documents, chunks them, embeds chunks, and upserts into Qdrant
- Deterministic evaluation:
    - citation rate
    - groundedness proxy (embedding similarity between answer and retrieved context)
    - refusal correctness (refuse when retrieval is weak)
    - latency p50/p95
    - keyword coverage (basic correctness proxy)
- Quality gate: gate_pass = True/False logged to MLflow
- Idempotent ingestion (stable chunk IDs): upsert uses deterministic hashed IDs to avoid duplicates across runs
- MLflow artifacts: eval_report.json + run_config.json


## Repository Structure

```bash
data/
  sample_kb/          # Knowledge base docs (Markdown)
  eval_set/           # Small eval set (JSONL)

src/
  apps/
    ingestor/         # Ingestion CLI
    evaluator/        # Evaluation CLI
    inference_api/    # FastAPI service
  common/             # RAG core, qdrant client, ollama client, metrics
  pipelines/          # ZenML pipeline definitions

infra/
  k8s/                # Kubernetes manifests (Qdrant + inference)
  docker/             # Dockerfile for inference service
```

## Prerequisites
1. System
    - macOS (Apple Silicon works) / Linux
    - Python 3.11
    - Conda (recommended)
    - Ollama installed and running locally
    - Qdrant running locally OR via port-forward from Kubernetes
2. Install Ollama
    - Install Ollama (macOS): https://ollama.com/
    - Start server and pull a model:
```bash
ollama serve
ollama pull llama3.1
```
## Setup

```bash
conda create -n llm-mlops python=3.11 -y
conda activate llm-mlops
pip install -r requirements.txt
```

Initialize ZenML:

```bash
zenml init
zenml integration install mlflow -y
```

## Start MLFlow (Tracking Server)

MLflow runs locally and stores metadata in mlflow.db and artifacts in ./mlruns.

```bash
mlflow server \
  --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Set the tracking URI:
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Open MLflow UI:
http://127.0.0.1:5001

## Start Qdrant 
### Option A: Run locally (Docker)

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.11.0
```


### Option B: Run in Kubernetes (kind) + port-forward

```bash
kind create cluster --name llm-mlops
kubectl apply -f infra/k8s/namespace.yaml
kubectl -n llm apply -f infra/k8s/qdrant.yaml
kubectl -n llm port-forward svc/qdrant 6333:6333
```

## Run End-to-End Pipeline (ZenML)
```bash
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

python -m src.pipelines.run_pipeline
```

## Expected output:
- ingest_step runs (loads docs → chunks → embeds → upserts)
- eval_step runs and prints metrics
- register_step logs metrics + artifacts to MLflow
- MLflow run shows gate_pass = True/False


## Check Results:

1. MLflow UI
- Open the experiment:
    - llm-support-agent-mlops
- In the latest run, check:
    - Params: gate_pass, config values
    - Metrics: citation rate, groundedness, latency
    - Artifacts: eval_report.json, run_config.json

2. Qdrant status

```bash
curl -s http://localhost:6333/collections | python -m json.tool
```

## Run the Inference API (Local)

```bash
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1

python -m src.apps.inference_api.main

```


Test it:
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"ticket":"I cannot reset my password. It says token expired.","top_k":5}'
```

Response includes:
- answer
- citations
- retrieved contexts
- latency and groundedness proxy