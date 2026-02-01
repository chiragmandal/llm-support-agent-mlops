import json
import os
from pathlib import Path
from src.common.config import QDRANT_URL, QDRANT_COLLECTION
from src.common.rag import rag_answer, groundedness_proxy
from src.common.eval_metrics import has_refusal, keyword_coverage, percentile

EVAL_PATH = Path("data/eval_set/eval.jsonl")

def main():
    qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    latencies = []
    grounded = []
    cite_ok = 0
    refusal_ok = 0
    kw_scores = []

    per_item = []

    for r in rows:
        ticket = r["ticket"]
        must_cite = bool(r.get("must_cite", True))
        expected_keywords = r.get("expected_keywords", [])

        out = rag_answer(ticket=ticket, qdrant_url=qdrant_url, collection=collection, top_k=5)
        g = groundedness_proxy(out["answer"], out["contexts"])
        out["groundedness_proxy"] = g

        has_cite = ("[doc:" in out["answer"]) or (len(out.get("citations", [])) > 0)
        if (not must_cite) or has_cite:
            cite_ok += 1

        # If retrieval seems weak, we expect refusal-ish behavior
        weak_retrieval = len(out["contexts"]) == 0 or max([c["score"] for c in out["contexts"]] + [0.0]) < 0.2
        refused = has_refusal(out["answer"])
        if (weak_retrieval and refused) or (not weak_retrieval and not refused):
            refusal_ok += 1

        kws = keyword_coverage(out["answer"], expected_keywords)
        kw_scores.append(kws)

        latencies.append(int(out["latency_ms"]))
        grounded.append(float(g))

        per_item.append({
            "ticket": ticket,
            "latency_ms": out["latency_ms"],
            "groundedness_proxy": g,
            "keyword_coverage": kws,
            "answer": out["answer"][:600],
        })

    n = len(rows)
    metrics = {
        "n": n,
        "citation_rate": cite_ok / n if n else 0.0,
        "refusal_correctness": refusal_ok / n if n else 0.0,
        "groundedness_avg": sum(grounded)/n if n else 0.0,
        "keyword_coverage_avg": sum(kw_scores)/n if n else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
    }

    report = {"metrics": metrics, "samples": per_item}
    Path("eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
