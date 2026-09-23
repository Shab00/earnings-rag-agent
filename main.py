"""FastAPI server for the earnings call RAG agent.

Run: uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import answer
from retriever import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()  # load once at startup; retrieve() reuses the cached index
    yield


app = FastAPI(title="Earnings Call RAG Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Filters(BaseModel):
    ticker: str | None = None
    year: str | None = None
    quarter: str | None = None
    speaker: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    filters: Filters | None = None
    top_k: int = Field(default=6, ge=1, le=20)


# Plain `def` endpoints run in a threadpool, so blocking OpenAI calls don't stall the server.
@app.post("/ask")
def ask(request: AskRequest):
    filters = request.filters.model_dump(exclude_none=True) if request.filters else None
    try:
        return answer(request.question, filters=filters or None, top_k=request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/companies")
def companies():
    _, metadata = load_index()
    by_ticker = {}
    for chunk in metadata:
        entry = by_ticker.setdefault(chunk["ticker"], {"ticker": chunk["ticker"], "company": chunk["company"], "quarters": set()})
        entry["quarters"].add((chunk["year"], chunk["quarter"]))
    return [
        {**entry, "quarters": [f"{q} {y}" for y, q in sorted(entry["quarters"])]}
        for entry in by_ticker.values()
    ]


@app.get("/health")
def health():
    index, _ = load_index()
    return {"status": "ok", "chunks_loaded": index.ntotal}
