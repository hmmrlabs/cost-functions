#!/usr/bin/env python3
"""Read the 4.02's own prose and check whether it names the filing we joined it to.

`join_probe.py` resolves the repudiated filing positionally: the most recent periodic form
before the 4.02. That produced 91.5 percent joined with a median gap of one quarter, which
is a plausible antecedent and not a verified one. This is the verification, and it is the
step that turns a rate about our rule into a rate about the SEC's record.

# What is checked

An Item 4.02 disclosure names the affected periods in words: "the fiscal year ended
December 31, 2020", "the quarterly period ended June 30, 2021", "the Company's Annual
Report on Form 10-K for the year ended ...". So the check is:

  does the period the joined filing REPORTS ON appear in the 4.02's Item 4.02 text?

`reportDate` from the submissions API is that period, as EDGAR records it. A match means
the positional rule found the filing the company was actually talking about.

# Three outcomes, and the third is the one worth having

  confirmed    the joined filing's period is named in the prose
  contradicted a DIFFERENT period is named and ours is not: the positional rule missed
  unreadable   no period could be parsed out at all

**`unreadable` is reported and never folded into either other bucket.** A 4.02 that
incorporates its periods by reference, or whose text this parser cannot reach, is a fact
about this measurement's instrument and not evidence either way about the join. Counting
it as a failure would understate the rule and counting it as a pass would flatter it.

# Fair access

Declared User-Agent, 0.15s between requests, documents cached under `raw/docs/`.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE / "raw" / "docs"
SUBS = HERE / "raw" / "submissions"
UA = {"User-Agent": "hammer-bench research kartik@multiversal.ventures"}
SLEEP = 0.15

MONTHS = {
    m: i
    for i, m in enumerate(
        "january february march april may june july august september october "
        "november december".split(),
        1,
    )
}
# "December 31, 2020" and "December 31 2020". The comma is optional in filings often
# enough that requiring it loses real matches.
DATE_RE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE
)


def fetch(url: str) -> str | None:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2**attempt)
        except Exception:
            time.sleep(2**attempt)
    return None


def primary_doc(cik: str, adsh: str) -> str | None:
    """The 8-K's own primary document name, off the cached submissions."""
    cache = SUBS / f"CIK{cik}.json"
    if not cache.is_file():
        return None
    for f in json.loads(cache.read_text())["filings"]:
        if f["adsh"] == adsh:
            return f["primaryDocument"] or None
    return None


def document(cik: str, adsh: str) -> str | None:
    """The 8-K text, cached. Tags stripped; entities left alone beyond `&nbsp;`."""
    cache = DOCS / f"{adsh}.txt"
    if cache.is_file():
        return cache.read_text()

    doc = primary_doc(cik, adsh)
    if not doc:
        return None
    nodash = adsh.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{doc}"
    body = fetch(url)
    time.sleep(SLEEP)
    if body is None:
        return None
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", body)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = re.sub(r"\s+", " ", text)
    DOCS.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def item_402_span(text: str) -> str:
    """The Item 4.02 section, or the whole document when its boundaries are not found.

    Falling back to the whole document is the permissive choice and it is the right one
    here: a period named outside the section heading is still the company naming a period,
    and the alternative is discarding a filing whose layout this parser did not anticipate.
    The cost is recorded rather than hidden, since a whole-document span can pick up a
    period from an unrelated item.
    """
    m = re.search(r"item\s*4\.0\s*2", text, re.IGNORECASE)
    if not m:
        return text
    start = m.start()
    nxt = re.search(r"item\s*[0-9]\.[0-9]{2}", text[start + 10 :], re.IGNORECASE)
    return text[start : start + 10 + nxt.start()] if nxt else text[start:]


def dates_in(span: str) -> set[str]:
    out = set()
    for mon, day, year in DATE_RE.findall(span):
        try:
            out.add(date(int(year), MONTHS[mon.lower()], int(day)).isoformat())
        except ValueError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "joins.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("prior_period")]
    import random

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample]

    verdicts = Counter()
    detail = []
    for i, r in enumerate(rows, 1):
        text = document(r["cik"], r["event_adsh"])
        if not text:
            verdicts["unreadable"] += 1
            detail.append({**r, "verdict": "unreadable", "why": "document not fetched"})
            continue
        span = item_402_span(text)
        found = dates_in(span)
        if not found:
            verdicts["unreadable"] += 1
            detail.append({**r, "verdict": "unreadable", "why": "no period parsed"})
        elif r["prior_period"] in found:
            verdicts["confirmed"] += 1
            detail.append({**r, "verdict": "confirmed"})
        else:
            verdicts["contradicted"] += 1
            detail.append(
                {**r, "verdict": "contradicted", "periods_named": sorted(found)[:6]}
            )
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}  {dict(verdicts)}", flush=True)

    n = len(rows)
    readable = verdicts["confirmed"] + verdicts["contradicted"]
    out = {
        "vintage": time.strftime("%Y-%m-%d"),
        "checked": n,
        "seed": args.seed,
        "verdicts": dict(verdicts),
        "readable": readable,
        # The headline: of the ones where a period could be read at all, how often the
        # positional rule named the same filing the company did.
        "confirmed_of_readable": round(verdicts["confirmed"] / readable, 4)
        if readable
        else None,
        "note": "unreadable is neither a pass nor a fail. It is a property of this "
        "parser and of filings that incorporate their periods by reference.",
    }
    (HERE / "verification.json").write_text(json.dumps(out, indent=1) + "\n")
    with (HERE / "verification.jsonl").open("w") as fh:
        for d in detail:
            fh.write(json.dumps(d) + "\n")

    print()
    print(f"checked              {n}")
    for k in ("confirmed", "contradicted", "unreadable"):
        print(f"  {k:<18} {verdicts[k]}")
    if readable:
        print(f"confirmed of readable {out['confirmed_of_readable']:.1%}  (n={readable})")


if __name__ == "__main__":
    main()
