#!/usr/bin/env python3
"""C1 and C2: the arms that decide whether the rest is worth running.

The design names these first and says the encoders are not built until they have run. The
reason is in this repository's own history: on the toy corpus a 40-row lookup table tied the
JEPA exactly, 0.656 against 0.656, which retired every architectural claim made up to that
point.

Neither arm reads a document. Both use only what the filing index already gives: SIC, form,
period month, filing year, and how often the company files. If either separates the classes,
the disagreement hypothesis is not what is being measured.

    C1  base rate            predict the majority class. The floor everything must clear.
    C2  40-row lookup        bucket on firm features, predict each bucket's training rate.

# Scored under the asymmetric contract

MODELCARD and E22 both score missed x 10: a missed restatement is expensive and a false
alarm is a review. Reported beside the raw confusion matrix, never instead of it.

# Split

By COMPANY, not by row. Filings from one company are not independent, and a random row
split would put the same firm on both sides and score memorisation as generalisation.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
MISS_COST = 10


def bucket(r) -> tuple:
    """The 40-ish cells C2 predicts in. Deliberately coarse: a lookup table that can
    memorise individual rows is not a control, it is another model."""
    sic = str(r.get("sic") or "0")[:2]
    return (
        sic,
        "K" if r["form"].startswith("10-K") else "Q",
        "amended" if r["form"].endswith("/A") else "orig",
        r["filed"][:3],  # decade
    )


def score(rows, predict) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        p, y = predict(r), r["label"]
        tp += p == 1 and y == 1
        fp += p == 1 and y == 0
        tn += p == 0 and y == 0
        fn += p == 0 and y == 1
    cost = fn * MISS_COST + fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "asymmetric_cost": cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--xbrl-only", action="store_true",
                    help="restrict to the XBRL window, the row set C3 and A1 will use")
    ap.add_argument("--sic", default="",
                    help="restrict to one SIC code. Within a systemic wave the industry "
                         "code can BE the label, and this is how that is checked.")
    ap.add_argument("--regime", default="",
                    help="YYYY or YYYY-YYYY. Inside one regime every row shares the "
                         "calendar, so C1e is degenerate and the era confound is gone by "
                         "construction. See the within-regime design.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / "pairs-sliced.jsonl").read_text().splitlines()]
    if args.xbrl_only:
        rows = [r for r in rows if r["xbrl"]]
    if args.regime:
        lo, _, hi = args.regime.partition("-")
        hi = hi or lo
        rows = [r for r in rows if lo <= r["filed"][:4] <= hi]
    if args.sic:
        rows = [r for r in rows if str(r.get("sic") or "") == args.sic]

    ciks = sorted({r["cik"] for r in rows})
    random.Random(args.seed).shuffle(ciks)
    holdout = set(ciks[: max(1, len(ciks) // 4)])
    train = [r for r in rows if r["cik"] not in holdout]
    test = [r for r in rows if r["cik"] in holdout]

    # C1. The majority class is 0 everywhere here, so it never fires: perfect on the
    # negatives, blind to every positive. Its cost is the number this task has to beat.
    c1 = score(test, lambda r: 0)

    # C0. ALWAYS FIRE, and it is not a joke baseline.
    #
    # Under missed x 10 a false negative costs ten and a false positive costs one, so
    # firing on everything costs the negative count while never firing costs ten times the
    # positive count. Globally, at a 0.10 base rate, never-firing wins easily. Inside the
    # 2021 regime at 0.450 it does not, and an arm that fails to beat "flag everything"
    # has not earned the compute. This baseline was missing from every run before the
    # regime split and it would have been the honest floor for a near-balanced task.
    c0 = score(test, lambda r: 1)

    # C1e. THE YEAR, AND NOTHING ELSE.
    #
    # Measured on the full corpus: the positive rate in the XBRL window runs from 0.037 in
    # 2010 to 0.450 in 2021, a twelvefold swing, because the 2021 SPAC wave restated at a
    # rate nothing else here approaches. An arm that knows only the calendar therefore looks
    # skilled, and removing the era feature from C2 moved its cost from 0.502 of C1 to
    # 0.614, so roughly forty percent of that arm was the date.
    #
    # So era is made FREE TO EVERY ARM rather than hidden from them, and this is the bar.
    # Beating C1 now proves nothing; beating C1e is what says an arm read something the
    # calendar does not already say.
    year_rate = defaultdict(lambda: [0, 0])
    for r in train:
        c = year_rate[r["filed"][:4]]
        c[0] += r["label"]
        c[1] += 1
    train_base = sum(r["label"] for r in train) / len(train) if train else 0
    fires_year = {
        y: (v[0] / v[1]) > train_base for y, v in year_rate.items() if v[1] >= 20
    }
    c1e = score(test, lambda r: 1 if fires_year.get(r["filed"][:4], False) else 0)

    # C2. Each bucket predicts 1 when its TRAINING positive rate clears the base rate.
    # The threshold is the training base rate rather than a swept number, so the arm
    # cannot be tuned into looking good.
    by = defaultdict(lambda: [0, 0])
    for r in train:
        b = by[bucket(r)]
        b[0] += r["label"]
        b[1] += 1
    base = sum(r["label"] for r in train) / len(train) if train else 0
    fire = {k: (v[0] / v[1]) > base for k, v in by.items() if v[1] >= 5}
    c2 = score(test, lambda r: 1 if fire.get(bucket(r), False) else 0)

    out = {
        "seed": args.seed,
        "xbrl_only": args.xbrl_only,
        "rows": len(rows),
        "companies": len(ciks),
        "split": "by company, 25 percent held out",
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positives": sum(r["label"] for r in test),
        "train_base_rate": round(base, 5),
        "buckets_used": len(fire),
        "buckets_firing": sum(fire.values()),
        "C0_always_fire": c0,
        "C1_base_rate": c1,
        "C1e_year_only": c1e,
        "C2_lookup": c2,
        "C2_over_C1e": round(
            c2["asymmetric_cost"] / c1e["asymmetric_cost"], 4
        ) if c1e["asymmetric_cost"] else None,
        "note": "Asymmetric cost is fn*10 + fp, lower is better. C1 never fires, so its cost "
        "is every positive missed. C1e knows only the filing year and is the REAL bar: the "
        "era wave is large enough that beating C1 proves nothing about reading a disclosure.",
    }
    suffix = ("-xbrl" if args.xbrl_only else "") + (f"-{args.regime}" if args.regime else "") + (f"-sic{args.sic}" if args.sic else "")
    (HERE / f"controls{suffix}.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"rows {len(rows)}  companies {len(ciks)}  "
          f"test {len(test)} rows / {out['test_positives']} positives")
    print(f"buckets {len(fire)} used, {out['buckets_firing']} firing")
    for name, s in (("C0 always fire", c0), ("C1 base rate", c1),
                    ("C1e year only", c1e), ("C2 lookup", c2)):
        print(f"  {name:<14} tp={s['tp']:>3} fp={s['fp']:>4} fn={s['fn']:>3}  "
              f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  "
              f"cost={s['asymmetric_cost']}")


if __name__ == "__main__":
    main()
