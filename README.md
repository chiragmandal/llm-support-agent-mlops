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

---

## What Changed and Why?

The following improvements were made across four dimensions: **performance**, **accuracy**, **data security**, and **code quality**. Each change includes the rationale so you can evaluate the tradeoff yourself.

### 1. Security

#### `.env.example` added; `.env` added to `.gitignore`
**File:** `.env.example`, `.gitignore`

`.gitignore` previously omitted `.env`. If a developer created a `.env` with service URLs or future API keys, it could be committed accidentally. `.env` is now explicitly ignored. A `.env.example` template documents every tunable variable so contributors know exactly what to configure without guessing from source code.

#### Input length cap on the `/answer` endpoint
**File:** `src/apps/inference_api/main.py`

The `ticket` field had no length limit. An unbounded input allows two classes of abuse:
1. **Prompt injection** — a long preamble can push the system prompt out of the model's effective context window.
2. **Runaway token cost** — an adversarial or misconfigured client can send megabyte-sized payloads that consume the entire context budget of the LLM.

`MAX_TICKET_CHARS` (default `2000`, tunable via env var) is enforced at the Pydantic validator layer before any embedding or LLM call occurs. Requests exceeding the limit receive an HTTP `422` with a clear error message.

#### Raw context text removed from API responses
**File:** `src/apps/inference_api/main.py`

The original `/answer` endpoint returned the full `contexts` list, which included raw KB chunk text. This exposes internal knowledge-base content to every API caller — a data disclosure risk if the KB contains proprietary policy details. The new `AskResponse` model returns only `answer`, `citations` (source + chunk_id), `groundedness_proxy`, and `latency_ms`. Citations give callers enough to audit the answer without receiving raw document text.

#### K8s manifests: resource limits added
**File:** `infra/k8s/inference.yaml`

The original Deployment had no `resources` block. A pod without limits can consume unlimited CPU/memory on the node, starving co-located services. Sensible defaults (`250m`/`512Mi` requests, `1000m`/`1Gi` limits) are now set and documented for tuning.

---

### 2. Performance

#### QdrantClient singleton via `lru_cache`
**File:** `src/common/qdrant_store.py`

The original `get_client()` created a **new `QdrantClient` instance on every call** — which means a new HTTP connection setup on every `/answer` request. `@lru_cache(maxsize=8)` turns it into a per-URL singleton. After the first call, the client and its underlying connection pool are reused across all requests at zero overhead. This is especially impactful when Qdrant is accessed over a network (Docker, k8s).

#### `ensure_collection` removed from the query hot path
**File:** `src/common/rag.py`, `src/apps/ingestor/ingest.py`

`ensure_collection` was called inside `rag_answer()`, meaning **every query issued a `GET /collections` call to Qdrant** just to verify the collection existed. This is a wasted round-trip on every request. Collection setup now happens only inside the ingestor. The inference path trusts the collection exists (as it should after ingest) and skips the check entirely.

#### Ollama retry with exponential back-off
**File:** `src/common/ollama_client.py`

The original client had zero retry logic. A single transient Ollama hiccup (restart, OOM, momentary overload) would immediately return a `500` to the caller. The new `_with_retry()` helper retries up to `OLLAMA_RETRIES` times (default `3`) with exponential back-off starting at `1s`. This is tunable via environment variable for environments where Ollama runs on a separate host with higher variance latency.

#### K8s readiness + liveness probes
**File:** `infra/k8s/inference.yaml`

Without probes, Kubernetes sends traffic to a pod immediately after the container starts — before the embedding model has loaded into memory. The readiness probe delays traffic until `/health` returns `200`. The liveness probe restarts the pod if it becomes permanently unresponsive. Together these eliminate the `503` burst users see during pod startup or after a silent crash.

---

### 3. Accuracy / RAG Quality

#### Sentence-aware chunking replaces character slicing
**File:** `src/apps/ingestor/ingest.py`

The original `chunk_text()` sliced on raw character offsets. This routinely split mid-sentence — and sometimes mid-word for dense text. The embedding model (`all-MiniLM-L6-v2`) was trained on coherent sentence-level text; feeding it a chunk that starts or ends mid-sentence produces a degraded embedding that does not accurately represent the semantic content.

The new `sentence_aware_chunk()` uses a regex on sentence-ending punctuation (`[.!?]` followed by whitespace) to keep sentences whole. When a new sentence would exceed `CHUNK_SIZE_CHARS`, the current chunk is closed and a new one begins with an overlap buffer of whole sentences (up to `CHUNK_OVERLAP_CHARS`). This consistently improves retrieval precision for factual support content where each sentence carries a distinct policy statement.

#### Cosine score threshold filters junk chunks from the prompt
**File:** `src/common/qdrant_store.py`, `src/common/rag.py`, `src/common/config.py`

Previously all `TOP_K` retrieved chunks were unconditionally included in the prompt regardless of how weakly they matched the query. A low-scoring chunk (e.g. score `0.05`) adds noise to the LLM context and can cause the model to hallucinate by stitching together unrelated policy clauses.

`SCORE_THRESHOLD` (default `0.30`, tunable via env var) is passed to Qdrant's native `score_threshold` parameter in the search call. Qdrant filters candidates before returning them, which is more efficient than post-filtering in Python. If no chunk clears the threshold, the prompt receives `(no context retrieved)` and the LLM is expected to gracefully refuse — which the existing refusal-correctness eval metric already tests for.

#### True idempotent re-ingestion via deterministic UUID chunk IDs
**File:** `src/apps/ingestor/ingest.py`

The README stated "idempotent ingestion (stable chunk IDs)" but the implementation used sequential integers starting from `1`. Running ingest twice on the same document tree produced the same integer IDs and overwrote correctly — but changing a document and re-running left orphaned old chunks alongside new ones because the total chunk count shifted. Deleting a KB file left its old chunks permanently in Qdrant.

The fix has two parts:
1. **Deterministic IDs**: chunk IDs are now `sha256(source_filename::chunk_index)` truncated to a UUID. The same document always produces the same IDs, so upsert is a true no-op on unchanged docs.
2. **Collection reset on ingest**: the collection is dropped and recreated at the start of each ingest run. This guarantees stale chunks from removed or renamed KB files cannot persist. The performance cost is negligible since ingest runs offline in the pipeline, not on the serving path.

---

### 4. Code Quality

#### ZenML steps call functions directly instead of `subprocess`
**File:** `src/pipelines/zenml_pipeline.py`, `src/apps/evaluator/eval.py`

The original `ingest_step` and `eval_step` used `subprocess.check_call(["python", "-m", ...])`. This is an anti-pattern in ZenML for three reasons:
1. **Error handling**: a non-zero exit code raises a generic `CalledProcessError` with no stack trace. ZenML's step UI shows the subprocess command, not the actual Python exception.
2. **Performance**: each subprocess forks a full Python interpreter and re-imports all heavy dependencies (sentence-transformers, qdrant-client) from scratch.
3. **Testability**: you cannot unit-test a step that delegates all logic to a subprocess.

The fix: `eval.py` now exports a `run_eval()` function. Both steps import and call their respective functions directly. ZenML captures all logging output and exceptions in the step UI as designed.

#### Structured logging replaces `print()`
**File:** All source files

Every `print()` statement has been replaced with `logging.getLogger(__name__)` calls. `config.py` configures `logging.basicConfig` once at import time so all modules inherit a consistent `%(asctime)s %(levelname)s %(name)s %(message)s` format. This means:
- Log lines are timestamped and include the originating module.
- Log level is controllable via the standard `LOG_LEVEL` env var without code changes.
- ZenML and MLflow both capture structured log output in their respective UIs.

#### Makefile port fixed and targets consolidated
**File:** `Makefile`

The original Makefile started MLflow on port `5000` while the README and `run_pipeline.py` defaulted to `5001`. This silent mismatch caused the pipeline to fail on first run if the developer followed the Makefile. All references now consistently use `5001`. A `qdrant` target and a `clean` target were added to make local development self-contained.

---

## Features

- RAG ingestion: loads Markdown KB documents, chunks them on sentence boundaries, embeds chunks, and upserts into Qdrant with deterministic IDs
- Score-threshold retrieval: only chunks exceeding `SCORE_THRESHOLD` cosine similarity reach the LLM prompt
- Deterministic evaluation:
  - citation rate
  - groundedness proxy (embedding cosine similarity between answer and retrieved context)
  - refusal correctness (refuse when retrieval is weak)
  - latency p50/p95
  - keyword coverage (basic correctness proxy)
- Quality gate: `gate_pass = True/False` logged to MLflow
- True idempotent ingestion: deterministic UUID chunk IDs + collection reset on re-run
- MLflow artifacts: `eval_report.json` + `run_config.json`
- Structured logging throughout with configurable log level

---

## Repository Structure

```
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

.env.example          # Template for all environment variables
```

---

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

---

## Setup

```bash
conda create -n llm-mlops python=3.11 -y
conda activate llm-mlops
pip install -r requirements.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
# edit .env as needed — it is git-ignored and will never be committed
```

Initialize ZenML:

```bash
zenml init
zenml integration install mlflow -y
```

---

## Start MLflow (Tracking Server)

MLflow runs locally and stores metadata in `mlflow.db` and artifacts in `./mlruns`.

```bash
make mlflow
# or manually:
mlflow server \
  --host 127.0.0.1 --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Set the tracking URI:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```

Open MLflow UI: http://127.0.0.1:5001

---

## Start Qdrant

### Option A: Run locally (Docker)

```bash
make qdrant
# or manually:
docker run --rm -p 6333:6333 qdrant/qdrant:v1.11.0
```

### Option B: Run in Kubernetes (kind) + port-forward

```bash
kind create cluster --name llm-mlops
kubectl apply -f infra/k8s/namespace.yaml
kubectl -n llm apply -f infra/k8s/qdrant.yaml
kubectl -n llm port-forward svc/qdrant 6333:6333
```

---

## Run End-to-End Pipeline (ZenML)

```bash
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

python -m src.pipelines.run_pipeline
```

Expected output:
- `ingest_step` runs (loads docs → sentence-aware chunks → embeds → upserts with deterministic IDs)
- `eval_step` runs and prints metrics
- `register_step` logs metrics + artifacts to MLflow
- MLflow run shows `gate_pass = True/False`

---

## Check Results

### MLflow UI

Open the experiment `llm-support-agent-mlops`. In the latest run, check:
- **Params**: `gate_pass`, config values
- **Metrics**: citation rate, groundedness, latency
- **Artifacts**: `eval_report.json`, `run_config.json`

### Qdrant status

```bash
curl -s http://localhost:6333/collections | python -m json.tool
```

---

## Run the Inference API (Local)

```bash
export QDRANT_URL=http://localhost:6333
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1

make run-api
# or: python -m src.apps.inference_api.main
```

Test it:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"ticket":"I cannot reset my password. It says token expired.","top_k":5}'
```

Response includes:
- `answer` — LLM response grounded in KB context
- `citations` — source document + chunk ID for each key claim
- `groundedness_proxy` — cosine similarity between answer and retrieved context (higher is better)
- `latency_ms` — end-to-end wall-clock latency

> **Note:** Raw context chunk text is intentionally excluded from the response to avoid exposing internal KB content to API callers.

---

## Tuning Reference

| Env var | Default | Effect |
|---|---|---|
| `SCORE_THRESHOLD` | `0.30` | Raise to reduce noise in prompts; lower if too many queries return no context |
| `MAX_TICKET_CHARS` | `2000` | Raise for use cases with long structured tickets |
| `TOP_K` | `5` | Increase for broader KB coverage; decrease to reduce prompt length |
| `OLLAMA_RETRIES` | `3` | Increase if Ollama runs on a separate host with variable latency |
| `CHUNK_SIZE_CHARS` | `1100` | Smaller chunks improve precision; larger improve recall |
| `CHUNK_OVERLAP_CHARS` | `180` | Increase if answers frequently span chunk boundaries |

---

## Roadmap / Known Limitations

The following improvements are the natural next steps:

- **Semantic response cache** (Redis + embedding similarity): identical or near-identical tickets should not hit the embedding model + Qdrant + Ollama on every call
- **Streaming responses** (SSE): users currently wait for the full completion before seeing any output
- **API key authentication**: the `/answer` endpoint is currently unauthenticated
- **Per-session rate limiting**: no guard against a single client flooding the inference service
- **LLM-as-judge evaluation**: the current `keyword_coverage` metric is a weak correctness proxy; a secondary LLM call scoring faithfulness and relevance would be more robust
- **Hybrid retrieval** (BM25 + dense): pure vector search misses exact-match queries (e.g. order IDs, error codes); a BM25 first-pass + re-rank would improve precision on structured queries
