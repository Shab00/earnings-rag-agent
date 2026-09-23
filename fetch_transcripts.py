"""Download earnings call transcripts from The Motley Fool.

Saves each transcript to data/transcripts/{TICKER}_{Q}_{YEAR}.txt, where Q and
YEAR are the company's *fiscal* quarter as named in the transcript title
(e.g. NVIDIA's Aug 2026 call is Q2 fiscal 2027).

Fool's URLs are not fully predictable: dates can be days off from the call and
slugs vary ("earnings-transcript", missing ticker, even the wrong quarter). So
for each target quarter the script tries, in order:
  1. the known URL, then the same slug dated +/-1 day
  2. transcript links listed on the ticker's Fool quote page whose slug mentions the quarter
and only accepts a page whose title matches the expected quarter.
"""

import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.fool.com/earnings/call-transcripts"
TARGETS = {
    "AAPL": [
        ("Q3", 2026, "2026/07/31/apple-aapl-q3-2026-earnings-call-transcript"),
        ("Q2", 2026, "2026/04/30/apple-aapl-q2-2026-earnings-call-transcript"),
        ("Q1", 2026, "2026/01/29/apple-aapl-q1-2026-earnings-call-transcript"),
        ("Q4", 2025, "2025/10/30/apple-aapl-q4-2025-earnings-call-transcript"),
    ],
    "NVDA": [
        ("Q2", 2027, "2026/08/27/nvidia-nvda-q2-2027-earnings-call-transcript"),
        ("Q1", 2027, "2026/05/28/nvidia-nvda-q1-2027-earnings-call-transcript"),
        ("Q4", 2026, "2026/02/26/nvidia-nvda-q4-2026-earnings-call-transcript"),
        ("Q3", 2026, "2025/11/20/nvidia-nvda-q3-2026-earnings-call-transcript"),
    ],
    "MSFT": [
        ("Q4", 2026, "2026/07/29/microsoft-msft-q4-2026-earnings-call-transcript"),
        ("Q3", 2026, "2026/04/30/microsoft-msft-q3-2026-earnings-call-transcript"),
        ("Q2", 2026, "2026/01/29/microsoft-msft-q2-2026-earnings-call-transcript"),
        ("Q1", 2026, "2025/10/29/microsoft-msft-q1-2026-earnings-call-transcript"),
    ],
    "TSLA": [
        ("Q2", 2026, "2026/07/22/tesla-tsla-q2-2026-earnings-call-transcript"),
        ("Q1", 2026, "2026/04/22/tesla-tsla-q1-2026-earnings-call-transcript"),
        ("Q4", 2025, "2026/01/22/tesla-tsla-q4-2025-earnings-call-transcript"),
        ("Q3", 2025, "2025/10/23/tesla-tsla-q3-2025-earnings-call-transcript"),
    ],
}
OUTPUT_DIR = Path("data/transcripts")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5  # be polite to fool.com

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url):
    """Return the page HTML, or None if it doesn't exist."""
    time.sleep(REQUEST_DELAY)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def date_variants(path):
    """The given path, then the same slug dated one day earlier and later."""
    day, slug = path[:10], path[11:]
    d = date.fromisoformat(day.replace("/", "-"))
    return [path] + [f"{(d + timedelta(days=delta)):%Y/%m/%d}/{slug}" for delta in (-1, 1)]


def quote_page_links(ticker):
    html = fetch(f"https://www.fool.com/quote/nasdaq/{ticker.lower()}/") or ""
    return sorted(set(re.findall(r"/earnings/call-transcripts/(\d{4}/\d{2}/\d{2}/[a-z0-9-]+)/", html)), reverse=True)


def title_quarter(soup):
    """(Q, YEAR) from a title like 'Apple Q4 2025 Earnings Call Transcript'."""
    match = re.search(r"\b(Q[1-4])\s+(?:FY\s*)?(\d{4})\b", soup.h1.get_text(" ", strip=True) if soup.h1 else "")
    return (match.group(1), int(match.group(2))) if match else None


def section_after(heading):
    """Text of the <p>/<li> elements between `heading` and the next <h2>."""
    parts = []
    for el in heading.find_next_siblings():
        if el.name == "h2":
            break
        for block in [el] if el.name in ("p", "li") else el.find_all(["p", "li"]):
            text = block.get_text(" ", strip=True)
            if text:
                parts.append(text)
    return parts


def extract_transcript(soup, url):
    """Build a clean transcript: title, call date, participants, then the dialogue only."""
    body = soup.select_one("div.article-body")
    if body is None:
        return None
    headings = {h.get_text(strip=True).lower(): h for h in body.find_all("h2")}
    transcript_heading = next((h for name, h in headings.items() if "transcript" in name), None)
    if transcript_heading is None:
        return None

    dialogue = [p for p in section_after(transcript_heading) if not p.lower().startswith("image source")]
    # Keep only a transcript that actually has "Speaker Name: ..." turns.
    if sum(bool(re.match(r"^[A-Z][\w.,'’ -]{1,60}:\s", p)) for p in dialogue) < 5:
        return None

    lines = [soup.h1.get_text(" ", strip=True), f"Source: {url}"]
    if "date" in headings:
        lines.append("Date: " + " ".join(section_after(headings["date"])))
    participants = []
    if "call participants" in headings:
        participants = [p for p in section_after(headings["call participants"]) if "[email" not in p and "@" not in p]
    if participants:
        lines += ["", "Call participants:"] + [f"- {p}" for p in participants]
    lines += ["", "Transcript:", ""]
    return "\n".join(lines) + "\n\n".join(dialogue) + "\n"


def find_transcript(ticker, quarter, year, known_path, quote_links):
    """Return (url, text) for the target quarter, or (None, None)."""
    tag = f"{quarter.lower()}-{year}"
    candidates = date_variants(known_path) + [p for p in quote_links if tag in p]
    tried = set()
    for path in candidates:
        if path in tried:
            continue
        tried.add(path)
        url = f"{BASE}/{path}/"
        html = fetch(url)
        if html is None:
            print(f"    404  {url}")
            continue
        soup = BeautifulSoup(html, "html.parser")
        found = title_quarter(soup)
        if found != (quarter, year):
            print(f"    skip {url} (page is {found[0]} {found[1]})" if found else f"    skip {url} (no quarter in title)")
            continue
        text = extract_transcript(soup, url)
        if text:
            return url, text
        print(f"    skip {url} (no transcript dialogue found)")
    return None, None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for ticker, targets in TARGETS.items():
        print(f"\n{ticker}: looking up transcript links on fool.com...")
        quote_links = quote_page_links(ticker)
        for quarter, year, known_path in targets:
            out_path = OUTPUT_DIR / f"{ticker}_{quarter}_{year}.txt"
            print(f"  {ticker} {quarter} {year}:")
            url, text = find_transcript(ticker, quarter, year, known_path, quote_links)
            if text is None:
                print(f"    NOT FOUND, no file written")
                missing.append(out_path.name)
                continue
            out_path.write_text(text, encoding="utf-8")
            print(f"    saved {out_path} ({len(text) / 1024:.0f} KB) from {url}")
    print(f"\nDone. Missing: {', '.join(missing) if missing else 'none'}")


if __name__ == "__main__":
    main()
