# Earnings Call RAG Agent

A production RAG (Retrieval Augmented Generation) agent 
that answers natural language questions across earnings 
call transcripts from Apple, NVIDIA, Microsoft and Tesla.

Ask questions like:
- "What did Tim Cook say about AI investment across the 
  last 4 quarters?"
- "Compare Jensen Huang's comments on data centre demand 
  between Q3 2026 and Q2 2027"
- "What guidance did Microsoft give on Azure revenue growth?"
- "How did Elon Musk describe Tesla's energy business?"

Every answer cites the exact transcript, quarter, year 
and speaker it came from.

## How it works

**Ingestion pipeline:**
16 earnings call transcripts (4 companies × 4 quarters) 
are fetched from public sources, chunked at speaker 
boundaries into ~500 token segments, embedded using 
OpenAI text-embedding-3-small and stored in a FAISS 
vector index. Each chunk carries metadata: company, 
ticker, quarter, year, speaker name and role.

**Query pipeline:**
A natural language question is embedded with the same 
model and used to retrieve the top-k most semantically 
similar chunks from FAISS. Optional filters narrow 
results by company, quarter, year or speaker. 
GPT-4o-mini receives the retrieved chunks as context 
and generates a cited answer grounded only in the 
transcript excerpts — it never speculates beyond 
what the transcripts contain.

**RAG design decisions:**
- Chunk at speaker boundaries — keeps each executive's 
  remarks together so the retriever finds complete 
  thoughts, not sentence fragments
- Speaker alias normalisation — Tim Cook, Jensen Huang 
  and Amy Hood each appear under one canonical name 
  across all quarters so speaker filtering works 
  correctly
- Filter-then-rank — all 388 vectors are searched, 
  then filtered by metadata, so filtered queries 
  always return the full top_k when enough chunks match
- Source citations on every answer — company, quarter, 
  speaker and a direct link to the original transcript

## Companies and quarters covered

| Company   | Ticker | Quarters                          |
|-----------|--------|-----------------------------------|
| Apple     | AAPL   | Q4 2025, Q1 2026, Q2 2026, Q3 2026 |
| Microsoft | MSFT   | Q1 2026, Q2 2026, Q3 2026, Q4 2026 |
| NVIDIA    | NVDA   | Q3 2026, Q4 2026, Q1 2027, Q2 2027 |
| Tesla     | TSLA   | Q3 2025, Q4 2025, Q2 2026          |

## Stack

- **Embeddings:** OpenAI text-embedding-3-small
- **Vector store:** FAISS (faiss-cpu)
- **LLM:** GPT-4o-mini
- **Backend:** FastAPI
- **Deployment:** Render
- **Data source:** Public earnings call transcripts 
  via Motley Fool

## API

- `POST /ask` — ask a question, optionally filtered 
  by ticker, quarter, year or speaker
- `GET /companies` — list available companies and quarters
- `GET /health` — health check with chunk count

API docs: https://earnings-rag-agent.onrender.com/docs

## How to run locally

```bash
git clone https://github.com/Shab00/earnings-rag-agent
cd earnings-rag-agent
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env

# Add transcripts to data/transcripts/ then build index
python ingest.py

# Start server
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API docs.

## Example query

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What did Jensen Huang say about data centre demand?",
    "filters": {"ticker": "NVDA"},
    "top_k": 4
  }'
```

Response includes the answer plus source citations:
```json
{
  "question": "What did Jensen Huang say about data centre demand?",
  "answer": "Jensen Huang described data centre demand as...",
  "sources": [
    {
      "company": "NVIDIA",
      "ticker": "NVDA",
      "quarter": "Q3",
      "year": "2026",
      "speaker": "Jensen Huang",
      "speaker_role": "CEO",
      "excerpt": "...",
      "source_url": "https://..."
    }
  ],
  "chunks_retrieved": 4
}
```

## What I would add next

- **React frontend** — chat interface on GitHub Pages 
  with company filter dropdown and source cards
- **More companies** — add Amazon, Google, Meta
- **Historical depth** — extend to 8 quarters per company
- **Comparison queries** — "compare Apple and Microsoft 
  on AI investment" using multi-company retrieval
- **Speaker filter UI** — filter by CEO only to cut 
  through analyst Q&A noise
