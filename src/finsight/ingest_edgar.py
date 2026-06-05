"""Download a real SEC filing by ticker and save it as plain text.

Dependency-free (uses the standard library only). SEC requires a descriptive
User-Agent that includes contact info; set your email:

    PowerShell:  $env:SEC_USER_AGENT="Your Name your.email@example.com"

Usage:
    python -m finsight.ingest_edgar AAPL MSFT
    python -m finsight.ingest_edgar AAPL --form 10-Q

Saved files land in data/filings/ and are picked up automatically by the
pipeline (it prefers real filings over the bundled sample once any exist).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.request

_OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "filings"))
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _ua() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        # SEC asks for contact info; please set SEC_USER_AGENT to your email.
        ua = "FinSight research finsight@example.com"
    return ua


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _ua(),
                                               "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            data = gzip.decompress(data)
        return data


def _ticker_to_cik(ticker: str) -> str:
    raw = json.loads(_get(_TICKERS_URL).decode("utf-8"))
    for row in raw.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise SystemExit(f"Ticker not found on EDGAR: {ticker}")


def _latest_filing(cik: str, form: str) -> tuple[str, str]:
    subs = json.loads(_get(_SUBMISSIONS.format(cik=cik)).decode("utf-8"))
    recent = subs["filings"]["recent"]
    for f, acc, doc in zip(recent["form"], recent["accessionNumber"],
                           recent["primaryDocument"]):
        if f == form:
            return acc.replace("-", ""), doc
    raise SystemExit(f"No {form} found for CIK {cik}")


def _html_to_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def fetch(ticker: str, form: str = "10-K") -> str:
    cik = _ticker_to_cik(ticker)
    acc, doc = _latest_filing(cik, form)
    url = _DOC.format(cik=int(cik), acc=acc, doc=doc)
    text = _html_to_text(_get(url))
    os.makedirs(_OUT_DIR, exist_ok=True)
    safe_form = form.replace("-", "").lower()
    out = os.path.join(_OUT_DIR, f"{ticker.lower()}_{safe_form}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"{ticker.upper()} {form} (source: SEC EDGAR)\n\n{text}")
    print(f"saved {ticker} {form}: {len(text):,} chars -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch SEC filings by ticker")
    p.add_argument("tickers", nargs="+", help="e.g. AAPL MSFT")
    p.add_argument("--form", default="10-K", help="10-K (default) or 10-Q")
    args = p.parse_args()
    for t in args.tickers:
        fetch(t, args.form)
        time.sleep(0.5)  # be polite to SEC servers


if __name__ == "__main__":
    main()
