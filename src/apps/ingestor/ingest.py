"""
Knowledge-base ingestor.

Key changes vs original:
- Sentence-aware chunking via regex. The original character-slice approach
  frequently cut through sentence boundaries, which fragments the semantic
  unit the embedding model is trained on and degrades retrieval quality.
  The new chunker accumulates sentences until the chunk size budget is full,
  then carries an overlap buffer of whole sentences into the next chunk.
- Deterministic UUID chunk IDs derived from sha256(source::index). This
  makes re-ingestion a true upsert/no-op: running ingest twice on the same
  document tree produces identical IDs, so Qdrant overwrites rather than
  accumulates duplicate vectors.
- Old collection is deleted and recreated on each ingest run so stale chunks
  from deleted or renamed KB files cannot persist.
- Structured logging throughout.
"""
import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

from src.common.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
)
from src.common.embedder import embed_texts
from src.common.qdrant_store import get_client, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/sample_kb")

# Regex that matches sentence-ending punctuation followed by whitespace or EOS.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_aware_chunk(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into chunks of at most *size* chars on sentence boundaries.

    Sentences are kept whole. When a new sentence would push the current chunk
    over *size*, the chunk is closed and the next one begins with an overlap
    buffer of whole sentences totalling at most *overlap* chars.

    This preserves the semantic unit that embedding models are trained on,
    which materially improves retrieval precision compared to blind character
    slicing.
    """
    sentences = _SENTENCE_SPLIT.split(text.strip())
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If adding this sentence would exceed the budget and we have content,
        # flush the current chunk.
        if current and current_len + sent_len + 1 > size:
            chunks.append(" ".join(current))
            # Build overlap buffer from the tail of the flushed chunk.
            overlap_buf: List[str] = []
            buf_len = 0
            for s in reversed(current):
                if buf_len + len(s) + 1 > overlap:
                    break
                overlap_buf.insert(0, s)
                buf_len += len(s) + 1
            current = overlap_buf
            current_len = buf_len

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Return a stable UUID string for a given (source, chunk_index) pair.

    Using sha256-derived UUIDs means the same document always produces the
    same IDs, so re-running ingest is a pure upsert with no duplicates.
    """
    key = f"{source}::{chunk_index}"
    digest = hashlib.sha256(key.encode()).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def load_docs() -> List[Dict]:
    docs = []
    for fp in DATA_DIR.glob("*.md"):
        docs.append({"source": fp.name, "text": fp.read_text(encoding="utf-8")})
    logger.info("Loaded %d documents from %s", len(docs), DATA_DIR.resolve())
    return docs


def main() -> None:
    qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    docs = load_docs()
    if not docs:
        raise RuntimeError(f"No docs found in {DATA_DIR.resolve()}")

    all_chunks: List[str] = []
    payloads: List[Dict] = []
    ids: List[str] = []

    for d in docs:
        chunks = sentence_aware_chunk(d["text"], CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        logger.info("Document '%s' -> %d chunks", d["source"], len(chunks))
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            payloads.append(
                {
                    "source": d["source"],
                    "chunk_id": str(i),
                    "text": ch,
                }
            )
            ids.append(_make_chunk_id(d["source"], i))

    vecs = embed_texts(all_chunks)
    client = get_client(qdrant_url)

    # Delete and recreate the collection so stale chunks from removed or
    # renamed KB files cannot linger and pollute future retrievals.
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.info("Dropping existing collection '%s' for clean re-ingest", collection)
        client.delete_collection(collection)

    ensure_collection(client, collection, vector_size=vecs.shape[1])
    upsert_chunks(client, collection, vecs, payloads, ids=ids)

    logger.info(
        "Ingest complete: %d chunks into '%s' at %s", len(all_chunks), collection, qdrant_url
    )


if __name__ == "__main__":
    main()
