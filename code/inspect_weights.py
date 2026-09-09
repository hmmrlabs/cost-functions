#!/usr/bin/env python3
"""What are the arms actually reading?

The within-regime design named this check before the arms ran: "every 2021 SPAC restatement
had roughly the same cause, warrant accounting, so the task may be a single accounting
question wearing a corpus. That would show up as a tiny vocabulary carrying all of C4, and
it is checkable by reading the weights rather than guessing."

C4 is logistic regression over words, so the weights ARE the answer. No interpretation
method is needed and none is used: the top positive weights are the terms that push a
filing toward "will be retracted" and the top negative ones push it away.

Two things this can show, and they are different findings:

  a narrow vocabulary   a handful of terms carrying it, all about one accounting topic.
                        Then the arm found the warrant-accounting question, which is a real
                        finding and not the general one.
  a broad vocabulary    signal spread over many unrelated terms. Then it is reading
                        something more like disclosure style.

It also reports how much of the total weight mass the top terms hold, because "the top ten
words look sensible" is a story and "the top ten words carry 40 percent of the mass" is a
measurement.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from arms import (
    BODIES, FACTS, STOP, WORD, cost_of, pick_threshold, predict,
    text_features, numeric_features, train_logreg,
)

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="views-2021.jsonl")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--tags", type=int, default=600)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.views).read_text().splitlines()]
    usable = [r for r in rows if r["chars_narrative"] > 500 and r["n_facts"] > 0]
    ciks = sorted({r["cik"] for r in usable})
    random.Random(args.seed).shuffle(ciks)
    holdout = set(ciks[: max(1, len(ciks) // 4)])
    tr = [r for r in usable if r["cik"] not in holdout]

    def narrative(r):
        p = BODIES / f"{r['adsh']}.json"
        return json.loads(p.read_text())["narrative"] if p.is_file() else ""

    facts_by_cik = {}

    def facts(r):
        c = r["cik"]
        if c not in facts_by_cik:
            p = FACTS / f"CIK{c}.json"
            facts_by_cik[c] = json.loads(p.read_text()) if p.is_file() else {}
        return facts_by_cik[c].get(r["adsh"], {})

    out = {"seed": args.seed, "views": args.views, "train_rows": len(tr)}

    # ---- C4, words ---------------------------------------------------------------------
    df = Counter()
    for r in tr:
        df.update(set(w for w in WORD.findall(narrative(r).lower()) if w not in STOP))
    vocab = {t: i for i, (t, _) in enumerate(df.most_common(args.vocab))}
    inv = {i: t for t, i in vocab.items()}
    log_len = len(vocab)
    X = [(text_features(narrative(r), vocab, log_len), r["label"]) for r in tr]
    w, _b = train_logreg(X, log_len + 1, seed=args.seed)

    words = sorted(
        ((inv[i], w[i]) for i in range(len(vocab))), key=lambda kv: -kv[1]
    )
    mass = sum(abs(v) for _, v in words) or 1.0
    top_pos = words[: args.top]
    top_neg = words[-args.top :][::-1]
    out["C4"] = {
        "vocabulary": len(vocab),
        "top_positive": [[t, round(v, 4)] for t, v in top_pos],
        "top_negative": [[t, round(v, 4)] for t, v in top_neg],
        "mass_in_top_25_each_way": round(
            sum(abs(v) for _, v in top_pos + top_neg) / mass, 4
        ),
        "mass_in_top_100_each_way": round(
            sum(abs(v) for _, v in words[:100] + words[-100:]) / mass, 4
        ),
    }

    # ---- C3, XBRL tags -----------------------------------------------------------------
    dft = Counter()
    for r in tr:
        dft.update(set(facts(r)))
    tags = {t: i for i, (t, _) in enumerate(dft.most_common(args.tags))}
    invt = {i: t for t, i in tags.items()}
    log_n = len(tags)
    Xn = [(numeric_features(facts(r), tags, log_n), r["label"]) for r in tr]
    wn, _bn = train_logreg(Xn, log_n + 1, seed=args.seed)
    tg = sorted(((invt[i], wn[i]) for i in range(len(tags))), key=lambda kv: -kv[1])
    massn = sum(abs(v) for _, v in tg) or 1.0
    out["C3"] = {
        "tags": len(tags),
        "top_positive": [[t, round(v, 4)] for t, v in tg[: args.top]],
        "top_negative": [[t, round(v, 4)] for t, v in tg[-args.top :][::-1]],
        "mass_in_top_25_each_way": round(
            sum(abs(v) for _, v in tg[: args.top] + tg[-args.top :]) / massn, 4
        ),
    }

    (HERE / "weights.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"C4 narrative, {len(vocab)} words, {len(tr)} training rows")
    print(f"  top 25 each way hold {out['C4']['mass_in_top_25_each_way']:.1%} of weight mass")
    print(f"  top 100 each way hold {out['C4']['mass_in_top_100_each_way']:.1%}")
    print("  PUSHES TOWARD retracted:")
    print("   ", ", ".join(t for t, _ in top_pos))
    print("  PUSHES TOWARD clean:")
    print("   ", ", ".join(t for t, _ in top_neg))
    print()
    print(f"C3 numeric, {len(tags)} tags")
    print(f"  top 25 each way hold {out['C3']['mass_in_top_25_each_way']:.1%} of weight mass")
    print("  PUSHES TOWARD retracted:")
    print("   ", ", ".join(t.split(":")[1] for t, _ in tg[:12]))
    print("  PUSHES TOWARD clean:")
    print("   ", ", ".join(t.split(":")[1] for t, _ in tg[-12:][::-1]))


if __name__ == "__main__":
    main()
