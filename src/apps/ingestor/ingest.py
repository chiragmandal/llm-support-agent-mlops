import os
from pathlib import Path
from typing import List, Dict
from src.common.config import (
    QDRANT_URL, QDRANT_COLLECTION, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
)
from src.common.embedder import embed_texts
from src.common.qdrant_store import get_client, ensure_collection, upsert_chunks

DATA_DIR = Path("data/sample_kb")

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def load_docs() -> List[Dict]:
    docs = []
    for fp in DATA_DIR.glob("*.md"):
        docs.append({"source": fp.name, "text": fp.read_text(encoding="utf-8")})
    return docs

def main():
    qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    docs = load_docs()
    if not docs:
        raise RuntimeError(f"No docs found in {DATA_DIR.resolve()}")

    all_chunks = []
    payloads = []

    for d in docs:
        chunks = chunk_text(d["text"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            payloads.append({
                "source": d["source"],
                "chunk_id": str(i),
                "text": ch,
            })

    vecs = embed_texts(all_chunks)
    client = get_client(qdrant_url)
    ensure_collection(client, collection, vector_size=vecs.shape[1])
    next_id = upsert_chunks(client, collection, vecs, payloads, start_id=1)

    print(f"Ingested {len(all_chunks)} chunks into {collection} at {qdrant_url}. Next id: {next_id}")

if __name__ == "__main__":
    main()
