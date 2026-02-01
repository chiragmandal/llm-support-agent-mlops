import requests
from .config import OLLAMA_URL, OLLAMA_MODEL

def generate(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()
