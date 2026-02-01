import os
import json
import mlflow
from zenml import pipeline, step

from src.common.config import (
    QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL,
    CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS,
    GATE_CITATION_RATE, GATE_GROUNDEDNESS, GATE_P95_LAT_MS,
)

@step
def ingest_step() -> str:
    import subprocess
    subprocess.check_call(["python", "-m", "src.apps.ingestor.ingest"])
    return "ingest_ok"

@step
def eval_step(ingest_status: str) -> dict:
    # ingest_status is only used to force ordering in the DAG
    import subprocess
    subprocess.check_call(["python", "-m", "src.apps.evaluator.eval"])
    with open("eval_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    return report

@step
def register_step(report: dict) -> bool:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("llm-support-agent-mlops")

    metrics = report["metrics"]

    gate_pass = (
        metrics.get("citation_rate", 0.0) >= GATE_CITATION_RATE
        and metrics.get("groundedness_avg", 0.0) >= GATE_GROUNDEDNESS
        and metrics.get("latency_p95_ms", 10**9) <= GATE_P95_LAT_MS
    )

    run_config = {
        "qdrant_url": os.getenv("QDRANT_URL", QDRANT_URL),
        "qdrant_collection": os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION),
        "embed_model": os.getenv("EMBED_MODEL", EMBED_MODEL),
        "chunk_size_chars": int(os.getenv("CHUNK_SIZE_CHARS", str(CHUNK_SIZE_CHARS))),
        "chunk_overlap_chars": int(os.getenv("CHUNK_OVERLAP_CHARS", str(CHUNK_OVERLAP_CHARS))),
        "ollama_url": os.getenv("OLLAMA_URL", ""),
        "ollama_model": os.getenv("OLLAMA_MODEL", ""),
        "gate_thresholds": {
            "citation_rate_min": GATE_CITATION_RATE,
            "groundedness_avg_min": GATE_GROUNDEDNESS,
            "latency_p95_ms_max": GATE_P95_LAT_MS,
        },
    }

    with mlflow.start_run():
        # params (configs)
        mlflow.log_param("ollama_model", run_config["ollama_model"])
        mlflow.log_param("embed_model", run_config["embed_model"])
        mlflow.log_param("qdrant_collection", run_config["qdrant_collection"])
        mlflow.log_param("chunk_size_chars", run_config["chunk_size_chars"])
        mlflow.log_param("chunk_overlap_chars", run_config["chunk_overlap_chars"])

        # gate + thresholds
        mlflow.log_param("gate_pass", gate_pass)
        mlflow.log_metric("gate_citation_rate_min", float(GATE_CITATION_RATE))
        mlflow.log_metric("gate_groundedness_avg_min", float(GATE_GROUNDEDNESS))
        mlflow.log_metric("gate_latency_p95_ms_max", float(GATE_P95_LAT_MS))

        # metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # artifacts
        with open("run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

        mlflow.log_artifact("run_config.json")
        mlflow.log_artifact("eval_report.json")

    return gate_pass

@pipeline(enable_cache=False)
def llm_support_pipeline():
    ingest_status = ingest_step()
    report = eval_step(ingest_status)   # <- dependency created here
    gate_pass = register_step(report)
    return gate_pass
