import os

def env(name: str, default: str) -> str:
    return os.getenv(name, default)

QDRANT_URL = env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = env("QDRANT_COLLECTION", "support_kb")

# Colima containers reach host via host.lima.internal
OLLAMA_URL = env("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "mistral")

EMBED_MODEL = env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE_CHARS = int(env("CHUNK_SIZE_CHARS", "1100"))
CHUNK_OVERLAP_CHARS = int(env("CHUNK_OVERLAP_CHARS", "180"))

TOP_K = int(env("TOP_K", "5"))

# Eval gates (tune if your laptop is slower)
GATE_CITATION_RATE = float(env("GATE_CITATION_RATE", "0.90"))
GATE_GROUNDEDNESS = float(env("GATE_GROUNDEDNESS", "0.55"))
GATE_P95_LAT_MS = int(env("GATE_P95_LAT_MS", "20000"))
