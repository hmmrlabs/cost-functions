#!/usr/bin/env python3
"""Where the crossover comes from, in closed form, and why these arms sit below it.

RESULT-costs.md measured the crossover at 5.5 to 5.75 by sweeping. It is not an accident of
this corpus and it does not need a sweep. Write the cost per row with base rate pi:

    cost / n  =  c * pi * (1 - TPR)  +  (1 - pi) * FPR

Flag-everything is the point (FPR, TPR) = (1, 1), so its cost is (1 - pi). An arm beats it iff

    c * pi * (1 - TPR)  <  (1 - pi) * (1 - FPR)

and therefore the largest ratio at which the arm still wins is

    c*  =  (1 - pi) / pi   x   max over the ROC of   (1 - FPR) / (1 - TPR)

**Two factors, and only the second is about the model.** The first is the odds of a negative.
The second is base-rate free: how much specificity the arm can buy per unit of recall it gives
up, measured from the top-right corner of ROC space rather than the top-left.

That corner is the geometry nobody looks at. Standard practice reads a ROC curve from (0,0)
and asks how fast it climbs. Flag-everything lives at (1,1), so what decides whether it can be
beaten under an asymmetric cost is the steepest line from (1,1) down to the curve, and a model
can have excellent AUC while being flat exactly there.

Reported here per arm: the achieved factor, the c* it implies, and the ROC in the region that
decides it, which is TPR near 1.
"""

import argparse
import json
import random
from pathlib import Path

from costs import lookup_scores

HERE = Path(__file__).parent


def roc(pairs):
    """(FPR, TPR, threshold) at every distinct score, high to low."""
    rows = sorted(pairs, key=lambda t: -t[0])
    P = sum(y for _, y in rows)
    N = len(rows) - P
    out = [(0.0, 0.0, float("inf"))]
    tp = fp = 0
    i = 0
    while i < len(rows):
        s = rows[i][0]
        while i < len(rows) and rows[i][0] == s:
            tp += rows[i][1]
            fp += 1 - rows[i][1]
            i += 1
        out.append((fp / N if N else 0.0, tp / P if P else 0.0, s))
    return out, P, N


def crossover(pairs):
    pts, P, N = roc(pairs)
    pi = P / (P + N)
    odds = (1 - pi) / pi
    best = 0.0
    best_pt = None
    for fpr, tpr, thr in pts:
        if tpr >= 1.0:
            continue  # (1-TPR) = 0 is flag-everything itself, not a rival to it
        f = (1 - fpr) / (1 - tpr)
        if f > best:
            best, best_pt = f, (fpr, tpr, thr)
    return {"base_rate": pi, "odds_of_negative": odds, "refusal_ratio": best,
            "c_star": odds * best, "at": best_pt, "roc": pts}


def near_top(pts, marks=(0.90, 0.95, 0.98, 0.99)):
    """FPR at each high recall. This is the region that decides c*, and it is the region a
    top-left reading of the curve never looks at."""
    out = {}
    for m in marks:
        cand = [p for p in pts if p[1] >= m]
        out[m] = min(cand, key=lambda p: p[0])[0] if cand else None
    return out


def report(name, pairs):
    d = crossover(pairs)
    pts = d["roc"]
    nt = near_top(pts)
    print(f"\n{name}")
    print(f"  base rate {d['base_rate']:.3f}   odds of a negative {d['odds_of_negative']:.2f}")
    print(f"  best (1-FPR)/(1-TPR) = {d['refusal_ratio']:.2f}"
          f"  at FPR {d['at'][0]:.3f} TPR {d['at'][1]:.3f}")
    print(f"  => c* = {d['c_star']:.2f}   (above this, flag-everything wins)")
    print("  FPR needed at high recall, which is where c* is decided:")
    for m, f in nt.items():
        print(f"     TPR {m:.2f} -> FPR {f:.3f}" if f is not None else
              f"     TPR {m:.2f} -> unreachable")
    need = {}
    for c in (6.46, 10):
        need[c] = c / d["odds_of_negative"]
    print("  to beat flag-everything you would need (1-FPR)/(1-TPR) of:")
    for c, v in need.items():
        gap = v / d["refusal_ratio"]
        print(f"     at c={c:<5} {v:.2f}   ({gap:.2f}x what this arm achieves)")
    return {"arm": name, "base_rate": round(d["base_rate"], 4),
            "odds_of_negative": round(d["odds_of_negative"], 4),
            "refusal_ratio": round(d["refusal_ratio"], 4),
            "c_star": round(d["c_star"], 3),
            "at_fpr": round(d["at"][0], 4), "at_tpr": round(d["at"][1], 4),
            "fpr_at_recall": {str(k): (round(v, 4) if v is not None else None)
                              for k, v in nt.items()},
            "needed_refusal_ratio": {str(c): round(v, 3) for c, v in need.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    out = []

    rows = [json.loads(l) for l in (HERE / "pairs-sliced.jsonl").read_text().splitlines()]
    for regime, label in [("", "lookup / all years"), ("2021", "lookup / 2021")]:
        rs = rows if not regime else [r for r in rows if r["filed"][:4] == regime]
        ciks = sorted({r["cik"] for r in rs})
        random.Random(args.seed).shuffle(ciks)
        hold = set(ciks[: max(1, len(ciks) // 4)])
        tr = [r for r in rs if r["cik"] not in hold]
        te = [r for r in rs if r["cik"] in hold]
        _, test = lookup_scores(tr, te)
        out.append(report(label, test))

    import costs
    for arm in ("numeric", "words", "embed"):
        class A:
            pass
        a = A()
        a.arm, a.views, a.embed_dir, a.seed = arm, "views-2021.jsonl", "nomic-embed-text-mean", args.seed
        _, test, _, _, _ = costs.learned_scores(a)
        out.append(report(f"{arm} / 2021", test))

    (HERE / "crossover.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n-> crossover.json")


if __name__ == "__main__":
    main()
