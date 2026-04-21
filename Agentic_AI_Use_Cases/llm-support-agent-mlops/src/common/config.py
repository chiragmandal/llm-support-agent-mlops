import os
import logging

# ---------------------------------------------------------------------------
# Logging (configure once at import time so every module inherits it)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

def env(name: str, default: str) -> str:
    return os.getenv(name, default)

# ---------------------------------------------------------------------------
# Infrastructure URLs
# ---------------------------------------------------------------------------
QDRANT_URL        = env("QDRANT_URL",        "http://localhost:6333")
QDRANT_COLLECTION = env("QDRANT_COLLECTION", "support_kb")

OLLAMA_URL   = env("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "mistral")

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL = env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS    = int(env("CHUNK_SIZE_CHARS",    "1100"))
CHUNK_OVERLAP_CHARS = int(env("CHUNK_OVERLAP_CHARS", "180"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K           = int(env("TOP_K",           "5"))
# Chunks scoring below this threshold are discarded before prompt assembly.
# Prevents low-relevance noise from diluting the LLM context window.
SCORE_THRESHOLD = float(env("SCORE_THRESHOLD", "0.30"))

# ---------------------------------------------------------------------------
# Ollama reliability
# ---------------------------------------------------------------------------
OLLAMA_TIMEOUT_S = int(env("OLLAMA_TIMEOUT_S", "120"))
OLLAMA_RETRIES   = int(env("OLLAMA_RETRIES",   "3"))

# ---------------------------------------------------------------------------
# API safety
# ---------------------------------------------------------------------------
# Hard cap on ticket length to prevent prompt injection and runaway token cost.
MAX_TICKET_CHARS = int(env("MAX_TICKET_CHARS", "2000"))

# ---------------------------------------------------------------------------
# Eval quality gates
# ---------------------------------------------------------------------------
GATE_CITATION_RATE  = float(env("GATE_CITATION_RATE",  "0.90"))
GATE_GROUNDEDNESS   = float(env("GATE_GROUNDEDNESS",   "0.55"))
GATE_P95_LAT_MS     = int(env("GATE_P95_LAT_MS",       "20000"))
