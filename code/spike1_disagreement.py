#!/usr/bin/env python3
"""Spike 1: is there raw disagreement signal at all, before anything is built?

The hypothesis is that where the words and the numbers disagree about one filing, that
filing is likelier to have been retracted. C3 and C4 already emit calibrated probabilities
over the same rows, so `|p3 - p4|` is a disagreement score that needs no projections, no
Barlow Twins and no new machinery.

**If the top and bottom quartiles of that score carry the same positive rate, the
hypothesis has no raw support and a learned projection will not manufacture it.** That is
the whole point of running this first: it is minutes, and it can stop a week.

Reported as a ratio rather than a p-value, because the question at this stage is whether
there is an effect worth chasing rather than whether a small one is distinguishable from
zero. A ratio near 1.0 is the stop signal.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from arms import (
    BODIES, FACTS, STOP, WORD, numeric_features, predict, text_features, train_logreg,
)

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="views-2021.jsonl")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tags", type=int, default=600)
    ap.add_argument("--vocab", type=int, default=3000)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.views).read_text().splitlines()]
    usable = [r for r in rows if r["chars_narrative"] > 500 and r["n_facts"] > 0]

    facts_by_cik = {}

    def facts(r):
        c = r["cik"]
        if c not in facts_by_cik:
            p = FACTS / f"CIK{c}.json"
            facts_by_cik[c] = json.loads(p.read_text()) if p.is_file() else {}
        return facts_by_cik[c].get(r["adsh"], {})

    def narrative(r):
        p = BODIES / f"{r['adsh']}.json"
        return json.loads(p.read_text())["narrative"] if p.is_file() else ""

    print(f"{'seed':>9} {'n_test':>7} {'base':>6} {'Q1 rate':>8} {'Q4 rate':>8} {'Q4/Q1':>7}")
    ratios = []
    for k in range(args.seeds):
        seed = 20260818 + k
        ciks = sorted({r["cik"] for r in usable})
        random.Random(seed).shuffle(ciks)
        hold = set(ciks[: max(1, len(ciks) // 4)])
        tr = [r for r in usable if r["cik"] not in hold]
        te = [r for r in usable if r["cik"] in hold]

        from collections import Counter

        dft = Counter()
        for r in tr:
            dft.update(set(facts(r)))
        tags = {t: i for i, (t, _) in enumerate(dft.most_common(args.tags))}
        Xtr = [(numeric_features(facts(r), tags, len(tags)), r["label"]) for r in tr]
        w3, b3 = train_logreg(Xtr, len(tags) + 1, seed=seed)

        dfw = Counter()
        for r in tr:
            dfw.update(set(w for w in WORD.findall(narrative(r).lower()) if w not in STOP))
        vocab = {t: i for i, (t, _) in enumerate(dfw.most_common(args.vocab))}
        Xtr4 = [(text_features(narrative(r), vocab, len(vocab)), r["label"]) for r in tr]
        w4, b4 = train_logreg(Xtr4, len(vocab) + 1, seed=seed)

        scored = []
        for r in te:
            p3 = predict(w3, b3, numeric_features(facts(r), tags, len(tags)))
            p4 = predict(w4, b4, text_features(narrative(r), vocab, len(vocab)))
            scored.append((abs(p3 - p4), r["label"]))
        scored.sort(key=lambda t: t[0])

        n = len(scored)
        q = max(1, n // 4)
        q1 = scored[:q]          # the views AGREE most
        q4 = scored[-q:]         # the views DISAGREE most
        r1 = sum(y for _, y in q1) / len(q1)
        r4 = sum(y for _, y in q4) / len(q4)
        base = sum(y for _, y in scored) / n
        ratio = r4 / r1 if r1 else float("inf")
        ratios.append(ratio)
        print(f"{seed:>9} {n:>7} {base:>6.3f} {r1:>8.3f} {r4:>8.3f} {ratio:>7.2f}")

    mean = sum(ratios) / len(ratios)
    print()
    print(f"mean Q4/Q1 ratio {mean:.2f}")
    print(f"splits where disagreement raises the positive rate: "
          f"{sum(1 for x in ratios if x > 1)}/{len(ratios)}")
    print()
    if mean < 1.1:
        print("STOP SIGNAL: disagreement does not raise the positive rate. A learned")
        print("projection will not manufacture a signal that is not in the raw scores.")
    else:
        print("Raw disagreement carries signal. Spike 2 and 3 are worth running.")

    (HERE / "spike1.json").write_text(
        json.dumps({"ratios": [round(x, 4) for x in ratios],
                    "mean_ratio": round(mean, 4),
                    "views": args.views}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
