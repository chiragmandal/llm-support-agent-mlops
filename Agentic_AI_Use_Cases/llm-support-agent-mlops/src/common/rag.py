"""
RAG query pipeline.

Key changes vs original:
- Uses the cached get_client() so no new TCP connection per request.
- ensure_collection() removed from the hot path; ingestor is responsible
  for collection setup.
- Contexts are filtered by SCORE_THRESHOLD inside search() at the Qdrant
  layer. If no context clears the threshold the prompt includes a
  (no context retrieved) marker and the LLM is expected to refuse.
- Structured logging for latency and retrieval quality.
"""
import logging
import time

import numpy as np

from .config import QDRANT_COLLECTION, TOP_K, SCORE_THRESHOLD
from .embedder import embed_texts, cosine
from .ollama_client import generate
from .qdrant_store import get_client, search

logger = logging.getLogger(__name__)

SYSTEM = """You are a customer support assistant.
Rules:
- Use ONLY the provided context. If context is insufficient, say you don't have enough information and ask for what you need.
- Provide a concise helpful answer.
- Always include citations as [doc:<source>#<chunk_id>] for each key claim.
"""


def build_prompt(ticket: str, contexts: list[dict]) -> str:
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[doc:{c['source']}#{c['chunk_id']}]\n{c['text']}\n")
    ctx = "\n".join(ctx_lines) if ctx_lines else "(no context retrieved)"
    return f"{SYSTEM}\n\nContext:\n{ctx}\n\nTicket:\n{ticket}\n\nAnswer:"


def rag_answer(
    ticket: str,
    qdrant_url: str,
    collection: str = QDRANT_COLLECTION,
    top_k: int = TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> dict:
    t0 = time.time()

    # Embed the query
    qvec = embed_texts([ticket])[0]

    # Use cached client — no new TCP connection per request
    client = get_client(qdrant_url)

    # Retrieve only chunks that clear the relevance threshold
    hits = search(client, collection, qvec, top_k=top_k, score_threshold=score_threshold)

    contexts = []
    for h in hits:
        payload = h.payload or {}
        contexts.append(
            {
                "score": float(h.score),
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "chunk_id": payload.get("chunk_id", "0"),
            }
        )

    logger.info(
        "ticket_len=%d retrieved=%d top_score=%.3f",
        len(ticket),
        len(contexts),
        contexts[0]["score"] if contexts else 0.0,
    )

    prompt = build_prompt(ticket, contexts)
    answer = generate(prompt)

    latency_ms = int((time.time() - t0) * 1000)
    citations = [
        f"[doc:{c['source']}#{c['chunk_id']}]"
        for c in contexts[: min(3, len(contexts))]
    ]

    logger.info("latency_ms=%d answer_len=%d", latency_ms, len(answer))

    return {
        "answer": answer,
        "citations": citations,
        "contexts": contexts,
        "latency_ms": latency_ms,
    }


def groundedness_proxy(answer: str, contexts: list[dict]) -> float:
    """Cosine similarity between the answer embedding and mean context embedding."""
    if not contexts:
        return 0.0
    ctx_text = " ".join(c["text"] for c in contexts[:5])
    vecs = embed_texts([answer, ctx_text])
    return cosine(vecs[0], vecs[1])
