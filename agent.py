"""RAG agent that answers questions from the earnings call transcripts."""

from dotenv import load_dotenv
from openai import OpenAI

from retriever import format_context, retrieve

load_dotenv()

MODEL = "gpt-4o-mini"
NOT_FOUND = "I could not find information about this in the available transcripts."

SYSTEM_PROMPT = f"""You are a financial analyst assistant with access to earnings call transcripts from Apple, NVIDIA, Microsoft and Tesla covering the last 4 quarters for each company.

Answer questions based ONLY on the transcript excerpts provided. Always cite your sources — name the company, quarter and speaker for each claim you make.

If the answer is not in the provided excerpts say '{NOT_FOUND}'

Do not speculate or use knowledge outside the provided transcripts."""

_client = None


def answer(query: str, filters: dict = None, top_k: int = 6) -> dict:
    global _client
    chunks = retrieve(query, top_k, filters)

    if chunks:
        if _client is None:
            _client = OpenAI()  # reads OPENAI_API_KEY
        response = _client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{format_context(chunks)}\n\nQuestion: {query}"},
            ],
        )
        answer_text = response.choices[0].message.content
    else:
        answer_text = NOT_FOUND  # nothing matched the filters; no point calling the model

    return {
        "question": query,
        "answer": answer_text,
        "sources": [
            {
                "company": c["company"],
                "ticker": c["ticker"],
                "quarter": c["quarter"],
                "year": c["year"],
                "speaker": c["speaker"],
                "speaker_role": c["speaker_role"],
                "excerpt": c["text"][:150],
                "score": c["score"],
                "source_url": c["source_url"],
            }
            for c in chunks
        ],
        "chunks_retrieved": len(chunks),
    }


if __name__ == "__main__":
    import sys

    result = answer(" ".join(sys.argv[1:]) or "What did Tim Cook say about AI investment?")
    print(result["answer"])
    for s in result["sources"]:
        print(f"- {s['company']} {s['quarter']} {s['year']} — {s['speaker']} ({s['speaker_role']}), score {s['score']}")
