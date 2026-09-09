#!/usr/bin/env python3
"""Make the corpus predictive: declare a horizon, and cut the three leaks measured in it.

`build_pairs.py` labels a filing by whether any Item 4.02 names it. That is a fact, and it
is not yet a prediction task. Asked as one, "does this disclosure get repudiated", three
things in the raw pairs answer it with information nobody would have had at the time.

# Leak 1: positives whose filing came AFTER the 4.02

Measured: **232 of 672 positives have a negative lag.** A 4.02 names a PERIOD, and a
company often refiles that period afterwards as a 10-K/A or 10-Q/A. Those amendments match
on `reportDate` and land in the positives.

They are the CORRECTION, not the thing being predicted. A model handed them learns what a
restated filing looks like, which is trivially separable and useless: it would score well
and mean nothing. Dropped.

# Leak 2: negatives filed after that company's first 4.02

Measured: **3,006 of 8,876 negatives, 33.9 percent.** A filing made after a restatement
comes from a company already under scrutiny, with remediated controls and a different
auditor relationship. Keeping them puts post-treatment rows in the negative class, so an
arm can separate the classes by reading the aftermath rather than the disclosure.

Dropped rather than flagged. A flag would need the arm to use it correctly, and this
corpus is meant to answer whether the disclosure carries signal, not whether the model can
be trusted with a confound.

# Leak 3: right censoring

A filing made shortly before the corpus vintage has not had time to be repudiated, so its
zero is "not yet" rather than "no". Every row must have had the full horizon to be
observed in, which means dropping filings after `vintage - H`.

# The horizon

Measured off the positives' own lag, filing to 4.02:

    p10 21   p25 99   p50 227   p75 391   p90 714   p95 955   max 2171

`--horizon 730` is the default, two years, capturing **90.5 percent** of positives. 365
would capture 70.9 and call the other 29 percent negative, which is not a cleaner task,
just a smaller one with mislabelled rows in it. The number is declared rather than swept,
because sweeping a horizon against a downstream score is how a horizon gets chosen for
flattering a model.
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
# The vintage of the 4.02 event sweep. Nothing after this date was observed at all, so it
# is the clock every censoring decision is made against.
EVENT_SWEEP_VINTAGE = date(2026, 8, 18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=730, help="days, declared not swept")
    args = ap.parse_args()
    H = args.horizon
    cutoff = EVENT_SWEEP_VINTAGE - timedelta(days=H)

    rows = [json.loads(l) for l in (HERE / "pairs.jsonl").read_text().splitlines()]

    # The first 4.02 at each company, which is when that company stops being a clean
    # source of negatives.
    first_event = {}
    for r in rows:
        if r["label"] == 1 and r.get("event_date"):
            c = r["cik"]
            if c not in first_event or r["event_date"] < first_event[c]:
                first_event[c] = r["event_date"]

    kept, drops = [], Counter()
    for r in rows:
        filed = date.fromisoformat(r["filed"])

        if filed > cutoff:
            drops["censored_no_full_horizon"] += 1
            continue

        if r["label"] == 1:
            lag = (date.fromisoformat(r["event_date"]) - filed).days
            if lag < 0:
                drops["positive_filed_after_the_402"] += 1
                continue
            if lag > H:
                # Beyond the horizon this task asks about. Not a negative either: it WAS
                # repudiated, just later than we claim to predict. Dropped, not relabelled.
                drops["positive_beyond_horizon"] += 1
                continue
            r["lag_days"] = lag
        else:
            ev = first_event.get(r["cik"])
            if ev and r["filed"] > ev:
                drops["negative_after_that_companys_first_402"] += 1
                continue

        kept.append(r)

    pos = [r for r in kept if r["label"] == 1]
    neg = [r for r in kept if r["label"] == 0]
    px = [r for r in pos if r["xbrl"]]
    nx = [r for r in neg if r["xbrl"]]

    with (HERE / "pairs-sliced.jsonl").open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    counts = {
        "vintage": time.strftime("%Y-%m-%d"),
        "horizon_days": H,
        "event_sweep_vintage": EVENT_SWEEP_VINTAGE.isoformat(),
        "filed_on_or_before": cutoff.isoformat(),
        "input_rows": len(rows),
        "kept": len(kept),
        "dropped": dict(drops),
        "positives": len(pos),
        "negatives": len(neg),
        "base_rate": round(len(pos) / len(kept), 5) if kept else None,
        "xbrl_window": {
            "positives": len(px),
            "negatives": len(nx),
            "rows": len(px) + len(nx),
            "base_rate": round(len(px) / (len(px) + len(nx)), 5) if px or nx else None,
        },
        "companies": len({r["cik"] for r in kept}),
        "note": "Horizon declared, never swept against a score. Positives beyond it are "
        "dropped rather than relabelled negative: they were repudiated, only later than "
        "this task claims to see.",
    }
    (HERE / "sliced-counts.json").write_text(json.dumps(counts, indent=1) + "\n")

    print(f"horizon                {H} days, filings on or before {cutoff}")
    print(f"input rows             {len(rows)}")
    for k, v in drops.most_common():
        print(f"  dropped {k:<38} {v}")
    print(f"kept                   {len(kept)}")
    print(f"  positives            {len(pos)}")
    print(f"  negatives            {len(neg)}")
    print(f"  base rate            {counts['base_rate']}")
    print(f"XBRL window rows       {counts['xbrl_window']['rows']}"
          f"  (base rate {counts['xbrl_window']['base_rate']})")
    print(f"companies              {counts['companies']}")


if __name__ == "__main__":
    main()
