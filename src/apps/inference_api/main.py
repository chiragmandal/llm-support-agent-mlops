import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.common.config import QDRANT_URL, QDRANT_COLLECTION, TOP_K
from src.common.rag import rag_answer, groundedness_proxy

app = FastAPI(title="LLM Support Agent (RAG)")

class AskRequest(BaseModel):
    ticket: str
    top_k: int = TOP_K

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/answer")
def answer(req: AskRequest):
    qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
    collection = os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION)

    out = rag_answer(
        ticket=req.ticket,
        qdrant_url=qdrant_url,
        collection=collection,
        top_k=req.top_k,
    )
    out["groundedness_proxy"] = groundedness_proxy(out["answer"], out["contexts"])
    return out

def main():
    import uvicorn
    uvicorn.run("src.apps.inference_api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
