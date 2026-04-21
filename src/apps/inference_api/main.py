"""
FastAPI inference service for the LLM Support Agent.

Key changes vs original:
- Input validation: ticket is capped at MAX_TICKET_CHARS. Unbounded input
  allows prompt injection and runaway token cost.
- Context text is stripped from the API response. Returning raw KB chunks
  to callers exposes internal knowledge-base content unnecessarily. Citations
  (source + chunk_id) are still returned for auditability.
- Groundedness proxy is now rounded to 4 decimal places for clean JSON.
- /health endpoint now also reports the configured collection name so
  infrastructure health checks can detect misconfiguration.
- Structured logging for every request.
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.common.config import QDRANT_URL, QDRANT_COLLECTION, TOP_K, MAX_TICKET_CHARS
from src.common.rag import rag_answer, groundedness_proxy

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Support Agent (RAG)")


class AskRequest(BaseModel):
    ticket: str = Field(..., description="Customer support ticket text")
    top_k: int = Field(default=TOP_K, ge=1, le=20)

    @field_validator("ticket")
    @classmethod
    def ticket_not_empty_or_too_long(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("ticket must not be empty")
        if len(v) > MAX_TICKET_CHARS:
            raise ValueError(
                f"ticket exceeds {MAX_TICKET_CHARS} characters "
                f"(got {len(v)}). Truncate before submitting."
            )
        return v


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    groundedness_proxy: float
    latency_ms: int
    # NOTE: raw context text is intentionally omitted from the response.
    # Returning KB chunk text to callers would expose internal knowledge-base
    # content. Citations (source + chunk_id) are sufficient for auditability.


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION),
        "qdrant_url": os.getenv("QDRANT_URL", QDRANT_URL),
    }


@app.post("/answer", response_model=AskResponse)
def answer(req: AskRequest):
    qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    logger.info(
        "Incoming ticket len=%d top_k=%d collection=%s",
        len(req.ticket), req.top_k, collection,
    )

    try:
        out = rag_answer(
            ticket=req.ticket,
            qdrant_url=qdrant_url,
            collection=collection,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("rag_answer failed: %s", exc)
        raise HTTPException(status_code=503, detail="Inference service temporarily unavailable.")

    gp = round(groundedness_proxy(out["answer"], out["contexts"]), 4)

    logger.info(
        "Response latency_ms=%d groundedness=%.4f citations=%d",
        out["latency_ms"], gp, len(out["citations"]),
    )

    return AskResponse(
        answer=out["answer"],
        citations=out["citations"],
        groundedness_proxy=gp,
        latency_ms=out["latency_ms"],
    )


def main():
    import uvicorn
    uvicorn.run("src.apps.inference_api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
