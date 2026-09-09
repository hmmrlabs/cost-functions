#!/usr/bin/env python3
"""Build the labelled pairs by reading which periods the 4.02 names, not by guessing one.

# Why this replaces the positional join

`join_probe.py` took the most recent periodic filing before the 4.02 and got 91.5 percent
"joined". `verify_join.py` then read the prose and found that only **67.1 percent of the
readable ones named the filing we had picked**. The diagnosis is in the contradicted rows
and it is not a parser problem:

    joined 10-Q period 2012-03-31    prose named 2011-03-31, 2011-06-30,
                                                 2011-09-30, 2011-12-31, 2012-06-30

**A 4.02 usually repudiates a run of periods, not one.** Twenty-three of twenty-eight
contradicted joins were 10-Q joins where the company was restating a sequence. Asking
which single filing a 4.02 refers to is the wrong question, so this asks the right one:
which filings does it name, and the answer is often several.

# The join, and why it cleans itself

Parse every date out of the Item 4.02 span, then keep the ones that equal the `reportDate`
of a periodic filing by that company. A date that is not any filing's period matches
nothing and drops out, which is what happens to the board-meeting and press-release dates
the parser also picks up. **No date list is curated by hand.** The filing record does the
filtering.

A filing named this way is a POSITIVE: the company said its own numbers for that period
should not be relied upon.

# Negatives

Periodic filings by the same company that no 4.02 ever names. Same firms, same forms, same
reporting machinery, and the only difference is the label, which is what stops the arms
learning "which companies restate" instead of "which disclosures precede a restatement".
`docs/.../2026-09-08-cross-document-disagreement-design.md` control C2 exists to check that
they failed to.

# What this does not do

It does not read the prose to decide anything except which periods are named. It fetches no
document bodies for the filings themselves. Those are the feature build and they wait for
C1 and C2.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from verify_join import DOCS, dates_in, document, item_402_span  # same parser, one copy

HERE = Path(__file__).parent
SUBS = HERE / "raw" / "submissions"
PERIODIC = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}


def sic_of(cik: str):
    """The company's SIC from its own submissions record.

    Read here for BOTH classes. It used to be copied off the 4.02 event, which meant only
    positives carried one and every negative carried None. C2 then scored precision 1.000
    because the bucket key leaked the label, which is exactly what a control is for.
    """
    cache = SUBS / f"CIK{cik}.json"
    if not cache.is_file():
        return None
    return json.loads(cache.read_text()).get("sic")


def filings_of(cik: str):
    cache = SUBS / f"CIK{cik}.json"
    if not cache.is_file():
        return []
    return [
        f
        for f in json.loads(cache.read_text())["filings"]
        if f["form"] in PERIODIC and f.get("reportDate")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = every cached event")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    events = [json.loads(l) for l in (HERE / "joins.jsonl").read_text().splitlines()]
    if args.sample and args.sample < len(events):
        import random

        random.Random(args.seed).shuffle(events)
        events = events[: args.sample]

    positives = {}  # (cik, prior_adsh) -> row
    named_per_event = []
    events_with_any = 0
    unreadable = 0
    ciks_seen = set()

    for i, ev in enumerate(events, 1):
        cik = ev["cik"]
        ciks_seen.add(cik)
        text = document(cik, ev["event_adsh"])
        if not text:
            unreadable += 1
            continue
        periods = dates_in(item_402_span(text))
        if not periods:
            unreadable += 1
            continue

        hits = [f for f in filings_of(cik) if f["reportDate"] in periods]
        named_per_event.append(len(hits))
        if hits:
            events_with_any += 1
        for f in hits:
            # A filing named by two different 4.02s is one positive, not two.
            positives[(cik, f["adsh"])] = {
                "cik": cik,
                "adsh": f["adsh"],
                "form": f["form"],
                "period": f["reportDate"],
                "filed": f["filingDate"],
                "xbrl": bool(f["isXBRL"] or f["isInlineXBRL"]),
                "sic": sic_of(cik),
                "label": 1,
                "named_by": ev["event_adsh"],
                "event_date": ev["event_date"],
            }
        if i % 25 == 0:
            print(f"  {i}/{len(events)}  positives={len(positives)}", flush=True)

    # Negatives: everything else those same companies filed.
    pos_keys = set(positives)
    negatives = []
    for cik in sorted(ciks_seen):
        for f in filings_of(cik):
            if (cik, f["adsh"]) in pos_keys:
                continue
            negatives.append(
                {
                    "cik": cik,
                    "adsh": f["adsh"],
                    "form": f["form"],
                    "period": f["reportDate"],
                    "filed": f["filingDate"],
                    "xbrl": bool(f["isXBRL"] or f["isInlineXBRL"]),
                    "sic": sic_of(cik),
                    "label": 0,
                }
            )

    rows = list(positives.values()) + negatives
    with (HERE / "pairs.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    npos, nneg = len(positives), len(negatives)
    pos_x = sum(1 for r in positives.values() if r["xbrl"])
    neg_x = sum(1 for r in negatives if r["xbrl"])
    dist = Counter(named_per_event)
    counts = {
        "vintage": time.strftime("%Y-%m-%d"),
        "events_probed": len(events),
        "events_unreadable": unreadable,
        "events_naming_at_least_one_filing": events_with_any,
        "named_filings_per_event": {
            str(k): dist[k] for k in sorted(dist) if k
        },
        "positives": npos,
        "negatives": nneg,
        "base_rate": round(npos / (npos + nneg), 5) if npos + nneg else None,
        "positives_with_xbrl": pos_x,
        "negatives_with_xbrl": neg_x,
        "xbrl_window": {
            "positives": pos_x,
            "negatives": neg_x,
            "base_rate": round(pos_x / (pos_x + neg_x), 5) if pos_x + neg_x else None,
        },
        "join_rule": "periods named in the filing's own Item 4.02 span, matched against "
        "reportDate of that company's periodic filings. Dates that are not any filing's "
        "period match nothing and drop out.",
    }
    (HERE / "pairs-counts.json").write_text(json.dumps(counts, indent=1) + "\n")

    print()
    print(f"events probed            {len(events)}")
    print(f"  unreadable             {unreadable}")
    print(f"  naming >=1 filing      {events_with_any}")
    print(f"named filings per event  {counts['named_filings_per_event']}")
    print(f"positives                {npos}  ({pos_x} with XBRL)")
    print(f"negatives                {nneg}  ({neg_x} with XBRL)")
    print(f"base rate                {counts['base_rate']}")
    print(f"base rate, XBRL window   {counts['xbrl_window']['base_rate']}")


if __name__ == "__main__":
    main()
