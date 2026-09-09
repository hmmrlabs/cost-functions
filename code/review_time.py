#!/usr/bin/env python3
"""The cost, denominated in the only currency anybody here actually spends: analyst hours.

`RESULT-costs.md` showed every result in this directory turns on `fn * 10 + fp`, a unitless
constant nobody measured, and that the crossover sits at 5.5. The obvious repair is to stop
inventing the constant and derive it from something a person can be asked about.

    c = (hours a missed restatement eventually costs) / (hours one review costs)

Both numerators and denominators are questions a reviewer can answer about their own shop.
Nobody has to agree on 10.

# The inversion this makes visible, which is the point of the file

Review time sits in the DENOMINATOR. So:

    cheap triage, 10 minutes a filing   ->  c is large   ->  flag everything wins
    real review, 4 hours a filing       ->  c is small   ->  the model wins

**A model is worth building here exactly when review is expensive.** That is also exactly when
you cannot afford to review everything, so the cost view and the capacity view stop disagreeing
the moment the units are real. Under `fn * 10 + fp` they looked like two different arguments.

# Two modes, and the second is the real one

UNCONSTRAINED   spend whatever it takes. Total hours = reviews + misses. This is the sweep
                from costs.py with the ratio derived rather than asserted.

BUDGETED        the team has H hours this quarter, so it can review k = H / review_hours
                filings and no more. **Always-fire is not available in this mode.** It names
                more filings than the team can open and has no opinion about which to drop, so
                what it degrades to is a RANDOM QUEUE of size k. That is the comparison the
                model is actually in, and it is the one no run in this directory has made.

No default hours are shipped for the miss. The published anchors do not settle it: the
restatement announcement effect is about -9.2 percent of market value, roughly half of it
expected litigation, and post-restatement audit fee premia run 21 to 32 percent years three and
four, but none of that converts to a reviewer's hours without an assumption this file declines
to make for you. The grid is printed instead, so a reader can find their own row.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from costs import best_threshold, capacity, confusion, lookup_scores, year_scores

HERE = Path(__file__).parent

# Minutes per filing reviewed. Spans a triage skim to a full workpaper review, because which
# end you are on decides the answer and this file exists to show that.
REVIEW_MINUTES = [5, 15, 30, 60, 120, 240, 480]

# Analyst hours one missed restatement eventually costs: the rework, the auditor's extra
# procedures, the remediation. A day, a work-week, a month.
MISS_HOURS = [8, 40, 160]


def budgeted(train, test, k):
    """Rank, open the top k, and count what that buys. Ties shuffled once with a fixed seed
    before the sort so an arm emitting one score for thousands of rows is not credited with
    the order the rows happened to arrive in."""
    rows = list(test)
    random.Random(20260908).shuffle(rows)
    rows.sort(key=lambda t: -t[0])
    k = max(0, min(k, len(rows)))
    caught = sum(y for _, y in rows[:k])
    pos = sum(y for _, y in rows)
    return caught, pos - caught, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs-sliced.jsonl")
    ap.add_argument("--regime", default="")
    ap.add_argument("--team-hours", type=float, default=0.0,
                    help="analyst hours available for this whole test set. Default 0 prints "
                         "the budget grid instead of one row.")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.pairs).read_text().splitlines()]
    if args.regime:
        lo, _, hi = args.regime.partition("-")
        hi = hi or lo
        rows = [r for r in rows if lo <= r["filed"][:4] <= hi]
    ciks = sorted({r["cik"] for r in rows})
    random.Random(args.seed).shuffle(ciks)
    hold = set(ciks[: max(1, len(ciks) // 4)])
    tr = [r for r in rows if r["cik"] not in hold]
    te = [r for r in rows if r["cik"] in hold]
    train, test = lookup_scores(tr, te)
    ytr, yte = year_scores(tr, te)

    n = len(test)
    pos = sum(y for _, y in test)
    print(f"\n=== {args.regime or 'all years'}: {n} filings on the desk, {pos} of them "
          f"({pos / n:.1%}) later retracted ===")

    # ---------------------------------------------------------------- unconstrained
    print("\n-- UNCONSTRAINED: spend what it takes. Total analyst hours, lower is better.")
    print("   c is derived, not asserted: hours a miss costs over hours a review costs.\n")
    print(f"{'review':>7} {'miss':>6} {'c':>7} | {'flag all':>9} {'flag none':>9} "
          f"{'year only':>9} {'lookup':>9} | {'winner':>10}")
    grid = []
    for mins in REVIEW_MINUTES:
        rh = mins / 60
        for mh in MISS_HOURS:
            c = mh / rh
            # every strategy costs (filings opened) * rh + (positives missed) * mh
            all_h = n * rh
            none_h = pos * mh
            thr_y = best_threshold(ytr, c)
            tp, fp, tn, fn = confusion(yte, thr_y)
            year_h = (tp + fp) * rh + fn * mh
            thr = best_threshold(train, c)
            tp, fp, tn, fn = confusion(test, thr)
            look_h = (tp + fp) * rh + fn * mh
            opts = {"flag all": all_h, "flag none": none_h,
                    "year only": year_h, "lookup": look_h}
            win = min(opts, key=opts.get)
            grid.append({"review_minutes": mins, "miss_hours": mh, "implied_c": round(c, 2),
                         **{k: round(v) for k, v in opts.items()}, "winner": win})
            print(f"{mins:>5}m {mh:>5}h {c:>7.1f} | {all_h:>9,.0f} {none_h:>9,.0f} "
                  f"{year_h:>9,.0f} {look_h:>9,.0f} | {win:>10}")

    # ---------------------------------------------------------------- budgeted
    print("\n-- BUDGETED: the team has H hours. Flag-all is NOT in the choice set, because it")
    print("   names more filings than the team can open and cannot say which to drop. What it")
    print("   degrades to is a random queue of the same size, which is the row below it.\n")
    budgets = ([args.team_hours] if args.team_hours
               else [n * m / 60 * f for m in (60,) for f in (0.01, 0.05, 0.10, 0.25)])
    print(f"{'hours':>8} {'review':>7} {'k':>7} {'k/n':>6} | {'model':>7} {'random':>7} "
          f"{'extra':>6} | {'hours saved vs random':>22}")
    budget_rows = []
    for H in budgets:
        for mins in (30, 60, 240):
            rh = mins / 60
            k = int(H / rh)
            if k <= 0 or k > n:
                continue
            caught, missed, k = budgeted(train, test, k)
            rand = pos * k / n
            for mh in (40,):
                saved = (rand - caught) * -1 * mh  # positive when the model catches more
                budget_rows.append({
                    "team_hours": round(H), "review_minutes": mins, "k": k,
                    "model_caught": caught, "random_caught": round(rand, 1),
                    "extra_caught": round(caught - rand, 1),
                    "miss_hours": mh, "hours_saved": round(saved),
                })
                print(f"{H:>8,.0f} {mins:>6}m {k:>7} {k / n:>6.1%} | {caught:>7} "
                      f"{rand:>7.1f} {caught - rand:>6.1f} | {saved:>18,.0f} h")

    print("\n-- the queue itself, independent of any hours assumption")
    for d in capacity(test):
        print(f"   top {d['k_frac']:>5.0%}  {d['k']:>6} filings  catches {d['caught']:>5} "
              f"of {d['of_positives']}  precision {d['precision_at_k']:.3f}  "
              f"lift {d['lift_over_random']:.2f}x")

    out = f"review-time{('-' + args.regime) if args.regime else ''}.json"
    (HERE / out).write_text(json.dumps({
        "regime": args.regime or "all", "test_rows": n, "test_positives": pos,
        "unconstrained": grid, "budgeted": budget_rows,
        "capacity": capacity(test),
    }, indent=1) + "\n")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
