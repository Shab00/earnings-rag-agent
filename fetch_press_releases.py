"""Download quarterly earnings documents from SEC EDGAR.

For each company, finds the most recent 8-K filings with Item 2.02
("Results of Operations and Financial Condition") and saves the text of
their EX-99 exhibits to data/press_releases/{TICKER}_{quarter}_{year}.txt.

Note: EDGAR earnings 8-Ks contain the earnings *press release* (and, for
some companies, CFO commentary or a shareholder letter), not the verbatim
call transcript. Companies almost never file call transcripts with the SEC.

The quarter/year in the filename is the calendar quarter the results cover
(the quarter before the filing date), not the company's fiscal quarter.
"""

import html
import re
import time
from datetime import date
from pathlib import Path

import requests

COMPANIES = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "TSLA": "0001318605",
}
NUM_QUARTERS = 4
OUTPUT_DIR = Path("data/press_releases")

# SEC requires a descriptive User-Agent with contact info; requests without one get 403.
HEADERS = {"User-Agent": "earnings-rag-agent research contact@example.com"}
REQUEST_DELAY = 0.2  # SEC fair-access limit is 10 requests/second


def get(url):
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def earnings_filings(cik, limit):
    """Return the most recent 8-K earnings filing (Item 2.02) for each of the last `limit` quarters.

    Some companies (e.g. Tesla) also file delivery reports under Item 2.02 earlier in the
    same quarter; keeping only the latest filing per quarter picks the full results release.
    """
    recent = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
    filings = {}
    for form, items, accession, filed in zip(
        recent["form"], recent["items"], recent["accessionNumber"], recent["filingDate"]
    ):
        if form != "8-K" or "2.02" not in items.split(","):
            continue
        filed = date.fromisoformat(filed)
        quarter = reported_quarter(filed)
        if quarter not in filings:  # filings are newest first
            if len(filings) == limit:
                break
            filings[quarter] = {"accession": accession, "filed": filed}
    return filings


def exhibit_urls(cik, accession):
    """Return URLs of the EX-99.x documents in a filing, in index order."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
    index = get(f"{base}/{accession}-index.html").text
    urls = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S | re.I):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
        if len(cells) < 4 or not re.match(r"\s*EX-99", cells[3], re.I):
            continue
        link = re.search(r'href="([^"]+)"', cells[2], re.I)
        if link:
            urls.append("https://www.sec.gov" + link.group(1).replace("/ix?doc=", ""))
    return urls


def html_to_text(raw):
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<br\s*/?>|</(p|div|tr|h\d|li)>", "\n", raw, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw)).replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def reported_quarter(filed):
    """Calendar quarter covered by an earnings release filed on `filed`."""
    q = (filed.month - 1) // 3  # quarter before the filing's quarter, 0 means Q4 of prior year
    return (f"Q{q}", filed.year) if q else ("Q4", filed.year - 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in COMPANIES.items():
        print(f"\n{ticker}: fetching filing history...")
        filings = earnings_filings(cik, NUM_QUARTERS)
        print(f"{ticker}: found {len(filings)} earnings 8-K filings")
        for (quarter, year), filing in filings.items():
            out_path = OUTPUT_DIR / f"{ticker}_{quarter}_{year}.txt"
            urls = exhibit_urls(cik, filing["accession"])
            if not urls:
                print(f"  {out_path.name}: no EX-99 exhibit in {filing['accession']}, skipping")
                continue
            sections = [f"Source: {url}\n\n{html_to_text(get(url).text)}" for url in urls]
            header = f"{ticker} {quarter} {year} (filed {filing['filed']}, accession {filing['accession']})"
            out_path.write_text(header + "\n\n" + "\n\n---\n\n".join(sections) + "\n", encoding="utf-8")
            size_kb = out_path.stat().st_size / 1024
            print(f"  saved {out_path} ({len(urls)} exhibit(s), {size_kb:.0f} KB)")
    print("\nDone.")


if __name__ == "__main__":
    main()
