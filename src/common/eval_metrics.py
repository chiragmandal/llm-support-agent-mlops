import numpy as np

def has_refusal(text: str) -> bool:
    t = text.lower()
    patterns = [
        "don't have enough information",
        "do not have enough information",
        "insufficient information",
        "i can't determine",
        "i cannot determine",
        "need more details",
    ]
    return any(p in t for p in patterns)

def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    a = answer.lower()
    hits = sum(1 for k in expected_keywords if k.lower() in a)
    return hits / len(expected_keywords)

def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    arr = np.array(sorted(values))
    idx = int(np.ceil((p/100.0) * len(arr))) - 1
    idx = max(0, min(idx, len(arr)-1))
    return int(arr[idx])
