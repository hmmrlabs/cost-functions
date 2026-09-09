#!/usr/bin/env python3
"""Measure whether the disagreement corpus can exist, before anybody builds it.

An Item 4.02 8-K says a company's previously issued financial statements should not be
relied upon. To use that as a label over two views of the repudiated filing, the filing
has to be resolvable from the 4.02 and it has to carry XBRL. Neither is guaranteed and
both are measurable in one request per company, so they are measured first.

`docs/superpowers/specs/2026-09-08-cross-document-disagreement-design.md` in
hammer-worldmodel names this the gate: report the join rate and stop there if it is poor.

# What is being counted, and what it is not

**The join here is a heuristic and this script does not pretend otherwise.** A 4.02 names
the affected periods in its prose; the structured `items` field does not carry them. So
the prior filing is resolved positionally: the most recent periodic filing by the same CIK
strictly before the 4.02's own filing date. That is a plausible antecedent and NOT a
verified one, and a rate reported from it is a rate about that rule.

Two failure modes it cannot see, both stated so the number is read correctly:

- A 4.02 restating a filing several periods back joins to the wrong one.
- A 4.02 whose affected filing predates the company's available submission history
  joins to nothing and is counted as a miss, which is right.

Verifying the join needs the 4.02's own prose, which is the next step and only worth
taking if this rate justifies it.

# Fair access

Declared User-Agent and 0.15s between requests, matching `../sec-8k-402/fetch.py`. One
request per distinct CIK, cached to `raw/submissions/`, so a re-run costs nothing.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw" / "submissions"
HITS = HERE.parent / "sec-8k-402" / "raw" / "efts-hits.jsonl"
UA = {
    "User-Agent": "hammer-bench research kartik@hammer.ai",
    "Accept": "application/json",
}
# The forms a 4.02 can repudiate. An 8-K carries no financial statements to restate.
PERIODIC = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}
SLEEP = 0.15


def get_json(url: str, tries: int = 5):
    """One GET, with the backoff the sibling harvester found EDGAR needs."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt + 1 == tries:
                raise
            time.sleep(2**attempt)
        except Exception:
            if attempt + 1 == tries:
                raise
            time.sleep(2**attempt)
    return None


def submissions(cik: str):
    """Every filing this CIK has, from cache when we already asked.

    `filings.recent` holds roughly the last thousand; `filings.files` names the overflow
    shards. Both are read, because a 4.02 from 2005 at a company that files often would
    otherwise resolve against a window that starts after it.
    """
    cache = RAW / f"CIK{cik}.json"
    if cache.is_file():
        return json.loads(cache.read_text())

    base = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    time.sleep(SLEEP)
    if base is None:
        return None

    rows = []

    def take(block):
        cols = block.get("accessionNumber")
        if not cols:
            return
        n = len(cols)
        for i in range(n):
            rows.append(
                {
                    "adsh": block["accessionNumber"][i],
                    "form": block["form"][i],
                    "filingDate": block["filingDate"][i],
                    "reportDate": block["reportDate"][i],
                    "isXBRL": block["isXBRL"][i],
                    "isInlineXBRL": block["isInlineXBRL"][i],
                    "primaryDocument": block["primaryDocument"][i],
                }
            )

    take(base.get("filings", {}).get("recent", {}))
    for shard in base.get("filings", {}).get("files", []):
        extra = get_json(f"https://data.sec.gov/submissions/{shard['name']}")
        time.sleep(SLEEP)
        if extra:
            take(extra)

    out = {"cik": cik, "name": base.get("name"), "sic": base.get("sic"), "filings": rows}
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def events():
    """The 4.02 events, from the sibling harvest's raw hits.

    Filtered on the STRUCTURED `items` field, never on the phrase that found them, which
    is the same rule `../sec-8k-402/build.py` applies for the same reason.
    """
    seen = set()
    for line in HITS.read_text().splitlines():
        src = json.loads(line).get("_source", {})
        if "4.02" not in (src.get("items") or []):
            continue
        adsh = src.get("adsh")
        ciks = src.get("ciks") or []
        if not adsh or not ciks or adsh in seen:
            continue
        seen.add(adsh)
        yield {
            "adsh": adsh,
            "cik": ciks[0],
            "file_date": src.get("file_date"),
            "sic": (src.get("sics") or [None])[0],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="probe this many events (0 = all). A sample answers the gate question at a "
        "fraction of the request budget, which is the point of asking it first.",
    )
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    evs = list(events())
    if args.sample and args.sample < len(evs):
        import random

        random.Random(args.seed).shuffle(evs)
        evs = evs[: args.sample]

    joined = missed = 0
    with_xbrl = 0
    by_year = Counter()
    joined_by_year = Counter()
    xbrl_by_year = Counter()
    gaps = []
    forms = Counter()
    rows = []

    for i, ev in enumerate(evs, 1):
        year = ev["file_date"][:4]
        by_year[year] += 1
        sub = submissions(ev["cik"])
        prior = None
        if sub:
            cands = [
                f
                for f in sub["filings"]
                if f["form"] in PERIODIC and f["filingDate"] < ev["file_date"]
            ]
            if cands:
                prior = max(cands, key=lambda f: f["filingDate"])

        if prior is None:
            missed += 1
        else:
            joined += 1
            joined_by_year[year] += 1
            gap = (
                __import__("datetime").date.fromisoformat(ev["file_date"])
                - __import__("datetime").date.fromisoformat(prior["filingDate"])
            ).days
            gaps.append(gap)
            forms[prior["form"]] += 1
            if prior["isXBRL"] or prior["isInlineXBRL"]:
                with_xbrl += 1
                xbrl_by_year[year] += 1
            rows.append(
                {
                    "event_adsh": ev["adsh"],
                    "cik": ev["cik"],
                    "event_date": ev["file_date"],
                    "sic": ev["sic"],
                    "prior_adsh": prior["adsh"],
                    "prior_form": prior["form"],
                    "prior_filed": prior["filingDate"],
                    "prior_period": prior["reportDate"],
                    "gap_days": gap,
                    "xbrl": bool(prior["isXBRL"] or prior["isInlineXBRL"]),
                }
            )
        if i % 25 == 0:
            print(f"  {i}/{len(evs)}  joined={joined} xbrl={with_xbrl}", flush=True)

    n = len(evs)
    gaps.sort()
    counts = {
        "vintage": time.strftime("%Y-%m-%d"),
        "probed_events": n,
        "sample": bool(args.sample),
        "seed": args.seed,
        "joined": joined,
        "join_rate": round(joined / n, 4) if n else 0,
        "missed": missed,
        "joined_with_xbrl": with_xbrl,
        "xbrl_rate_of_joined": round(with_xbrl / joined, 4) if joined else 0,
        "gap_days_median": gaps[len(gaps) // 2] if gaps else None,
        "gap_days_p90": gaps[int(len(gaps) * 0.9)] if gaps else None,
        "prior_forms": dict(forms.most_common()),
        "by_year": {
            y: {
                "events": by_year[y],
                "joined": joined_by_year[y],
                "with_xbrl": xbrl_by_year[y],
            }
            for y in sorted(by_year)
        },
        "join_rule": "most recent periodic filing by the same CIK strictly before the "
        "4.02 file_date. Positional and unverified: the 4.02's own prose names the "
        "affected periods and is not read here.",
    }
    (HERE / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    with (HERE / "joins.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print()
    print(f"probed        {n}")
    print(f"joined        {joined}  ({counts['join_rate']:.1%})")
    print(f"with XBRL     {with_xbrl}  ({counts['xbrl_rate_of_joined']:.1%} of joined)")
    print(f"gap days      median {counts['gap_days_median']}, p90 {counts['gap_days_p90']}")
    print(f"prior forms   {dict(forms.most_common(5))}")


if __name__ == "__main__":
    main()
