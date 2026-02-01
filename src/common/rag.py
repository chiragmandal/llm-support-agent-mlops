import time
import numpy as np
from .embedder import embed_texts, cosine
from .qdrant_store import get_client, ensure_collection, search
from .ollama_client import generate
from .config import QDRANT_COLLECTION, TOP_K

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
) -> dict:
    t0 = time.time()
    qvec = embed_texts([ticket])[0]
    client = get_client(qdrant_url)
    # collection should exist after ingest, but safe-guard:
    ensure_collection(client, collection, vector_size=len(qvec))

    hits = search(client, collection, qvec, top_k=top_k)
    contexts = []
    for h in hits:
        payload = h.payload or {}
        contexts.append({
            "score": float(h.score),
            "text": payload.get("text", ""),
            "source": payload.get("source", "unknown"),
            "chunk_id": payload.get("chunk_id", "0"),
        })

    prompt = build_prompt(ticket, contexts)
    answer = generate(prompt)

    latency_ms = int((time.time() - t0) * 1000)
    citations = [f"[doc:{c['source']}#{c['chunk_id']}]" for c in contexts[: min(3, len(contexts))]]

    return {
        "answer": answer,
        "citations": citations,
        "contexts": contexts,
        "latency_ms": latency_ms,
    }

def groundedness_proxy(answer: str, contexts: list[dict]) -> float:
    if not contexts:
        return 0.0
    # Compare answer embedding to mean context embedding
    ctx_text = " ".join([c["text"] for c in contexts[:5]])
    vecs = embed_texts([answer, ctx_text])
    return cosine(vecs[0], vecs[1])
