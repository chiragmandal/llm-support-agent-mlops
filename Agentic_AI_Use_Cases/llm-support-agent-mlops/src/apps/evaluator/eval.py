"""
Evaluation suite.

Key changes vs original:
- run_eval() is now a callable function (not just __main__) so the ZenML
  pipeline can import and call it directly instead of via subprocess.
- Structured logging replaces bare print() calls.
- Report is written to eval_report.json as before (MLflow artifact).
"""
import json
import logging
import os
from pathlib import Path

from src.common.config import QDRANT_URL, QDRANT_COLLECTION
from src.common.rag import rag_answer, groundedness_proxy
from src.common.eval_metrics import has_refusal, keyword_coverage, percentile

logger = logging.getLogger(__name__)

EVAL_PATH = Path("data/eval_set/eval.jsonl")


def run_eval(
    qdrant_url: str | None = None,
    collection: str | None = None,
) -> dict:
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL", QDRANT_URL)
    collection = collection or os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    rows = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Running eval on %d samples", len(rows))

    latencies: list[int] = []
    grounded: list[float] = []
    cite_ok = 0
    refusal_ok = 0
    kw_scores: list[float] = []
    per_item: list[dict] = []

    for r in rows:
        ticket = r["ticket"]
        must_cite = bool(r.get("must_cite", True))
        expected_keywords = r.get("expected_keywords", [])

        out = rag_answer(ticket=ticket, qdrant_url=qdrant_url, collection=collection, top_k=5)
        g = groundedness_proxy(out["answer"], out["contexts"])
        out["groundedness_proxy"] = g

        has_cite = ("[doc:" in out["answer"]) or bool(out.get("citations"))
        if (not must_cite) or has_cite:
            cite_ok += 1

        weak_retrieval = (
            len(out["contexts"]) == 0
            or max((c["score"] for c in out["contexts"]), default=0.0) < 0.2
        )
        refused = has_refusal(out["answer"])
        if (weak_retrieval and refused) or (not weak_retrieval and not refused):
            refusal_ok += 1

        kws = keyword_coverage(out["answer"], expected_keywords)
        kw_scores.append(kws)
        latencies.append(int(out["latency_ms"]))
        grounded.append(float(g))

        per_item.append(
            {
                "ticket": ticket,
                "latency_ms": out["latency_ms"],
                "groundedness_proxy": round(g, 4),
                "keyword_coverage": round(kws, 4),
                "answer": out["answer"][:600],
            }
        )

    n = len(rows)
    metrics = {
        "n": n,
        "citation_rate": round(cite_ok / n, 4) if n else 0.0,
        "refusal_correctness": round(refusal_ok / n, 4) if n else 0.0,
        "groundedness_avg": round(sum(grounded) / n, 4) if n else 0.0,
        "keyword_coverage_avg": round(sum(kw_scores) / n, 4) if n else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }

    report = {"metrics": metrics, "samples": per_item}
    report_path = Path("eval_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Eval metrics: %s", metrics)
    return report


def main():
    report = run_eval()
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
