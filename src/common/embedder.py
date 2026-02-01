from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np
from .config import EMBED_MODEL

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)

def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized
    return float((a * b).sum())
