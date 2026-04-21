"""
Qdrant store helpers.

Key changes vs original:
- get_client() is now @lru_cache so one QdrantClient is reused per URL.
  Creating a new client on every request adds TCP handshake overhead on
  every query. Caching the client drops that overhead to zero after warmup.
- search() now accepts a score_threshold parameter wired to Qdrant's native
  filter. Low-relevance chunks never reach the LLM prompt.
- ensure_collection() is intentionally NOT called from the query path any more.
  It should only run during ingest.
- upsert_chunks() now accepts pre-computed string IDs (UUID) for idempotent
  re-ingestion instead of sequential integers.
"""
import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import numpy as np

from .config import QDRANT_URL, QDRANT_COLLECTION, SCORE_THRESHOLD

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_client(url: str = QDRANT_URL) -> QdrantClient:
    """Return a cached QdrantClient for *url*.

    Using lru_cache means the TCP connection is established once per process
    and reused across all requests, avoiding per-query connection overhead.
    """
    logger.info("Creating QdrantClient for %s", url)
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    """Create *collection* if it does not already exist.

    Call this only from the ingestor, not from the query path.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        logger.info("Creating collection '%s' (vector_size=%d)", collection, vector_size)
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )
    else:
        logger.debug("Collection '%s' already exists", collection)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    vectors: np.ndarray,
    payloads: list[dict],
    ids: list[str],          # now caller-supplied UUID strings for idempotency
) -> None:
    """Upsert *vectors* with *payloads* using caller-supplied *ids*.

    Deterministic IDs mean re-running ingest on the same documents is a true
    no-op (Qdrant upserts overwrite existing points with identical IDs).
    """
    points = [
        qm.PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=payloads[i])
        for i in range(len(payloads))
    ]
    client.upsert(collection_name=collection, points=points)
    logger.info("Upserted %d points into '%s'", len(points), collection)


def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    top_k: int,
    score_threshold: float = SCORE_THRESHOLD,
):
    """Search *collection* and discard hits below *score_threshold*.

    Passing score_threshold to Qdrant's native filter is more efficient than
    post-filtering in Python because Qdrant skips candidate scoring early.
    """
    results = client.search(
        collection_name=collection,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
    )
    logger.debug(
        "Search returned %d hits (threshold=%.2f)", len(results), score_threshold
    )
    return results
