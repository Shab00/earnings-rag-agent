"""Build a FAISS vector index from the earnings call transcripts.

Reads data/transcripts/{TICKER}_{Q}_{YEAR}.txt (written by fetch_transcripts.py),
splits each call into ~500-token chunks along speaker turns, embeds them with
OpenAI text-embedding-3-small, and writes:

  data/faiss_index/index.faiss    cosine-similarity index (row i = metadata[i])
  data/faiss_index/metadata.json  list of chunk dicts

Quarter and year are the company's fiscal quarter, as named in the filename.

Usage:
  python ingest.py            # needs OPENAI_API_KEY (env or .env)
  python ingest.py --dry-run  # parse and chunk only, no API calls
"""

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

TRANSCRIPTS_DIR = Path("data/transcripts")
INDEX_DIR = Path("data/faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 500
EMBED_BATCH_SIZE = 100

COMPANIES = {"AAPL": "Apple", "NVDA": "NVIDIA", "MSFT": "Microsoft", "TSLA": "Tesla"}

# Checked in order; the first match wins, so "Chief Accounting Officer" is not read as CFO.
ROLE_PATTERNS = [
    ("Incoming CEO", r"incoming chief executive"),
    ("CEO", r"chief executive officer"),
    ("CFO", r"chief financial officer"),
    ("CAO", r"chief accounting officer"),
    ("Investor Relations", r"investor relations"),
    ("Legal", r"legal|counsel|secretary"),
]

# Executives appear under different names across calls; map them to one name so
# filtering by speaker works across quarters.
SPEAKER_ALIASES = {
    "Timothy Cook": "Tim Cook",
    "Timothy D. Cook": "Tim Cook",
    "Timothy Donald Cook": "Tim Cook",
    "Jen-Hsun Huang": "Jensen Huang",
    "Amy E. Hood": "Amy Hood",
}

TOKENIZER = tiktoken.get_encoding("cl100k_base")  # tokenizer used by text-embedding-3-*
# "Name: text" at the start of a paragraph. Every word must be capitalized so that
# sentences like "Our north star remains the same: ..." are not taken as speakers.
SPEAKER_RE = re.compile(r"^((?:[A-Z][\w.'’-]*)(?: [A-Z][\w.'’-]*){0,3}): (.+)", re.S)
OPERATOR_PHRASES = ("question comes from", "next question", "concludes today", "conference call")


def count_tokens(text):
    return len(TOKENIZER.encode(text))


def role_for_title(title):
    lowered = title.lower()
    for role, pattern in ROLE_PATTERNS:
        if re.search(pattern, lowered):
            return role
    return title.strip()  # e.g. "Vice President of AI"


def parse_file(path):
    """Split a transcript file into header info, participants and speaker turns."""
    ticker, quarter, year = path.stem.split("_")
    header, _, body = path.read_text(encoding="utf-8").partition("\nTranscript:\n")

    source_url = re.search(r"^Source: (\S+)", header, re.M).group(1)
    raw_date = re.search(r"^Date: (.+)", header, re.M).group(1)
    # "Thursday, January 29, 2026 at 5 p.m. ET" -> "January 29, 2026"
    date = re.sub(r"^[A-Z][a-z]+day, ", "", raw_date)
    date = re.split(r",? at |, \d{1,2}:", date)[0].strip()

    # "- Chief Financial Officer — Kevan Parekh" (separator is an em dash or a hyphen)
    participants = {}
    for title, name in re.findall(r"^- (.+?) [—-] (.+)$", header, re.M):
        participants[name.strip()] = role_for_title(title)

    turns = []  # [speaker, [paragraph, ...]]
    for para in (p.strip() for p in body.split("\n\n")):
        if not para:
            continue
        match = SPEAKER_RE.match(para)
        if match:
            turns.append([match.group(1), [match.group(2).strip()]])
        elif turns:
            turns[-1][1].append(para)  # continuation of the current speaker
        else:
            turns.append(["Unknown", [para]])

    return {
        "ticker": ticker,
        "company": COMPANIES.get(ticker, ticker),
        "quarter": quarter,
        "year": year,
        "date": date,
        "source_url": source_url,
        "participants": participants,
        "turns": turns,
    }


def make_speaker_resolver(participants, turns):
    """Map transcript speaker labels to (canonical name, role).

    Labels vary between calls ("Tim Cook" vs "Timothy D. Cook", "Jen-Hsun Huang"
    vs "Jensen Huang"), so executives are matched to the participant list by last
    name. Unlisted speakers are the operator (who reads the question queue) or analysts.
    """
    by_last_name = {name.split()[-1].lower(): (name, role) for name, role in participants.items()}
    operator_like = set()
    for speaker, paras in turns:
        text = " ".join(paras).lower()
        if speaker not in participants and len(speaker.split()) == 1 and any(p in text for p in OPERATOR_PHRASES):
            operator_like.add(speaker)  # e.g. NVIDIA's operator is labelled "Sarah"

    def resolve(label):
        if label == "Operator" or label in operator_like:
            return "Operator", "Operator"
        if label in participants:
            name, role = label, participants[label]
        elif len(label.split()) > 1 and label.split()[-1].lower() in by_last_name:
            name, role = by_last_name[label.split()[-1].lower()]
        else:
            name, role = label, "Analyst"
        return SPEAKER_ALIASES.get(name, name), role

    return resolve


def split_long_turn(speaker, paras):
    """Break one speaker's remarks into pieces of at most ~CHUNK_TOKENS, on paragraph boundaries."""
    pieces, current, size = [], [], 0
    for para in paras:
        tokens = count_tokens(para)
        if current and size + tokens > CHUNK_TOKENS:
            pieces.append((speaker, current))
            current, size = [], 0
        current.append(para)
        size += tokens
    if current:
        pieces.append((speaker, current))
    return pieces


def chunk_transcript(doc):
    """Group consecutive speaker turns into ~CHUNK_TOKENS chunks.

    A turn is never split unless it alone exceeds the limit, so short exchanges
    (operator intro, analyst question, executive answer) stay together. Each chunk's
    `speaker` is whoever contributed the most tokens; `speakers` lists everyone in it.
    """
    resolve = make_speaker_resolver(doc["participants"], doc["turns"])
    pieces = []
    for speaker, paras in doc["turns"]:
        pieces.extend(split_long_turn(speaker, paras))

    groups, current, size = [], [], 0
    for speaker, paras in pieces:
        text = f"{speaker}: " + "\n\n".join(paras)
        tokens = count_tokens(text)
        if current and size + tokens > CHUNK_TOKENS:
            groups.append(current)
            current, size = [], 0
        current.append((speaker, text, tokens))
        size += tokens
    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        tokens_by_speaker = {}
        for speaker, _, tokens in group:
            tokens_by_speaker[speaker] = tokens_by_speaker.get(speaker, 0) + tokens
        main_name, main_role = resolve(max(tokens_by_speaker, key=tokens_by_speaker.get))
        speakers = []
        for speaker in tokens_by_speaker:
            name, role = resolve(speaker)
            if name not in [s["name"] for s in speakers]:
                speakers.append({"name": name, "role": role})
        chunks.append({
            "ticker": doc["ticker"],
            "company": doc["company"],
            "quarter": doc["quarter"],
            "year": doc["year"],
            "date": doc["date"],
            "source_url": doc["source_url"],
            "speaker": main_name,
            "speaker_role": main_role,
            "speakers": speakers,
            "chunk_index": index,
            "text": "\n\n".join(text for _, text, _ in group),
        })
    return chunks


def embedding_input(chunk):
    """Prefix the chunk with its context so a query like 'Apple Q4 2025 margins' can match it."""
    return (
        f"{chunk['company']} ({chunk['ticker']}) fiscal {chunk['quarter']} {chunk['year']} "
        f"earnings call, {chunk['date']}. Speaker: {chunk['speaker']} ({chunk['speaker_role']}).\n\n"
        f"{chunk['text']}"
    )


def embed(client, texts):
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        print(f"[ingest] Embedded {min(start + EMBED_BATCH_SIZE, len(texts))}/{len(texts)} chunks")
    return np.array(vectors, dtype="float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only; skip embedding and saving")
    args = parser.parse_args()

    files = sorted(TRANSCRIPTS_DIR.glob("*_Q?_????.txt"))
    if not files:
        raise SystemExit(f"[ingest] No transcripts found in {TRANSCRIPTS_DIR}/ (run fetch_transcripts.py)")

    all_chunks = []
    for path in files:
        print(f"[ingest] Processing {path.name}...")
        chunks = chunk_transcript(parse_file(path))
        print(f"[ingest] {len(chunks)} chunks created")
        all_chunks.extend(chunks)
        print(f"[ingest] {path.name} done")
    if args.dry_run:
        print(f"[ingest] Total: {len(files)} files, {len(all_chunks)} chunks")
        print("[ingest] Dry run: skipping embedding and index write")
        return

    load_dotenv()
    client = OpenAI()  # reads OPENAI_API_KEY
    vectors = embed(client, [embedding_input(c) for c in all_chunks])
    faiss.normalize_L2(vectors)  # inner product on unit vectors = cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    (INDEX_DIR / "metadata.json").write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] Total: {len(files)} files, {index.ntotal} chunks embedded")
    print(f"[ingest] Index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
