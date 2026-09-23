"""Retrieve transcript chunks from the FAISS index built by ingest.py."""

import json

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from ingest import EMBEDDING_MODEL, INDEX_DIR

load_dotenv()

_index = None
_metadata = None
_client = None


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load the index and metadata from data/faiss_index/ (cached after the first call)."""
    global _index, _metadata
    if _index is None:
        index_path, metadata_path = INDEX_DIR / "index.faiss", INDEX_DIR / "metadata.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No index in {INDEX_DIR}/, run `python ingest.py` first")
        _index = faiss.read_index(str(index_path))
        _metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _index.ntotal != len(_metadata):
            raise ValueError(f"Index has {_index.ntotal} vectors but metadata has {len(_metadata)} chunks")
    return _index, _metadata


def embed_query(query: str) -> np.ndarray:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY
    response = _client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vector = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vector)
    return vector


def matches(chunk: dict, filters: dict) -> bool:
    """ticker/year/quarter match exactly (case-insensitive); speaker matches any
    speaker in the chunk by substring, so "Cook" finds "Tim Cook"."""
    for key, value in filters.items():
        if value is None or value == "":
            continue
        value = str(value).strip().lower()
        if key == "speaker":
            if not any(value in s["name"].lower() for s in chunk["speakers"]):
                return False
        elif key in ("ticker", "year", "quarter"):
            if str(chunk[key]).lower() != value:
                return False
        else:
            raise ValueError(f"Unknown filter {key!r}; use ticker, year, quarter or speaker")
    return True


def retrieve(query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    """Return the top_k chunks most similar to the query, each with a `score` (cosine similarity)."""
    index, metadata = load_index()
    # The index is small, so with filters search everything and filter afterwards;
    # this guarantees top_k results whenever that many chunks match.
    k = index.ntotal if filters else min(top_k, index.ntotal)
    scores, ids = index.search(embed_query(query), k)
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        chunk = metadata[i]
        if filters and not matches(chunk, filters):
            continue
        results.append({**chunk, "score": round(float(score), 4)})
        if len(results) == top_k:
            break
    return results


def format_context(chunks: list[dict]) -> str:
    """Format chunks as '[Apple Q4 2025 — Tim Cook, CEO]\\n{text}' blocks."""
    blocks = [
        f"[{c['company']} {c['quarter']} {c['year']} — {c['speaker']}, {c['speaker_role']}]\n{c['text']}"
        for c in chunks
    ]
    return "\n\n".join(blocks)
