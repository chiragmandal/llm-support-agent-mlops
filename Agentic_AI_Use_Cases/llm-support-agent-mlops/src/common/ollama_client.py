"""
Ollama HTTP client with retry + exponential back-off.

Key changes vs original:
- Added _with_retry() so transient Ollama hiccups (503, connection reset)
  do not immediately crash the request. Retries up to OLLAMA_RETRIES times
  with exponential back-off starting at 1 second.
- Timeout is now driven by OLLAMA_TIMEOUT_S from config instead of a hardcoded
  120-second constant, making it tunable per environment.
- Structured logging so every slow or failed generation is traceable.
"""
import logging
import time

import requests

from .config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S, OLLAMA_RETRIES

logger = logging.getLogger(__name__)


def _with_retry(func, retries: int, backoff_base: float = 1.5):
    """Call *func()* up to *retries* times with exponential back-off."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = backoff_base ** attempt
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Ollama call failed after %d attempts: %s", retries, exc
                )
    raise last_exc  # type: ignore[misc]


def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_URL,
    timeout_s: int = OLLAMA_TIMEOUT_S,
    retries: int = OLLAMA_RETRIES,
) -> str:
    """Generate a completion via Ollama, with retry on transient errors."""
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    def _call() -> str:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    logger.debug("Sending prompt to Ollama model='%s' url='%s'", model, url)
    result = _with_retry(_call, retries=retries)
    logger.debug("Ollama response length=%d chars", len(result))
    return result
