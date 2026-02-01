from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .config import QDRANT_URL, QDRANT_COLLECTION
import numpy as np

def get_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)

def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )

def upsert_chunks(
    client: QdrantClient,
    collection: str,
    vectors: np.ndarray,
    payloads: list[dict],
    start_id: int = 1
) -> int:
    points = []
    for i in range(len(payloads)):
        pid = start_id + i
        points.append(qm.PointStruct(id=pid, vector=vectors[i].tolist(), payload=payloads[i]))
    client.upsert(collection_name=collection, points=points)
    return start_id + len(payloads)

def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    top_k: int
):
    return client.search(
        collection_name=collection,
        query_vector=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    )
