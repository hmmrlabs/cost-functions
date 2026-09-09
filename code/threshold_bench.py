#!/usr/bin/env python3
"""Eight ways to place the line, benchmarked on the same scores.

`RESULT-calibrate.md` established two things. Selecting a decision threshold on in-sample
scores costs up to 5.5x the floor, and out-of-fold selection recovers a median 95 percent of
that. Out-of-fold is one strategy, not the strategy, and it was the only one tried.

This benchmarks it against seven others, chosen to differ in MECHANISM rather than in tuning,
because a family of near-identical variants would tell us nothing about why any of them works.

    S0  in_sample        threshold on the fitted model's own training scores. The defect.
    S1  oof              5-fold out-of-fold inside train. Costs K fits, no data.
    S2  oof_repeated     3 repeats of 5-fold, out-of-fold scores averaged. Same idea, less
                         variance in the scores the threshold is read off.
    S3  oof_boot_median  bootstrap the out-of-fold pairs 40 times, take the MEDIAN threshold.
                         Attacks variance in the threshold itself rather than in the scores.
    S4  validation       held-out quarter. The gold standard, and the expensive one.
    S5  analytic         no search at all. Under `fn * c + fp` the Bayes rule for a
                         calibrated probability is to fire when `p > 1 / (1 + c)`, because
                         firing costs `(1 - p)` and not firing costs `p * c`. Applied to raw
                         model output, which for a forest is a vote fraction and not a
                         probability. **Predicted to work for the calibrated model and fail
                         for the forest**, which is the whole point of including it.
    S6  analytic_iso     the same rule on scores calibrated out of fold. This is where
                         calibration should finally matter: it was irrelevant to an
                         empirical sweep because a monotone map cannot change which rows a
                         threshold selects, but the analytic rule reads the score's VALUE, so
                         the map is exactly what it needs.
    S7  rate_match       take the firing RATE the out-of-fold sweep chose, then set the
                         threshold at that quantile of the DEPLOYMENT scores. Uses the test
                         features and no test labels, which is what a deployment actually
                         has. Attacks the measured mechanism, firing drift, directly.
    S8  oracle           best threshold on test. Not achievable, reported as the bound.

Every strategy is evaluated on the same untouched test rows at every ratio.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from costs import cost_at                                  # noqa: E402
from split3 import learned_features, sweep_once            # noqa: E402
from strong import load_bank, pairs, to_matrix             # noqa: E402

RATIOS = [3, 5, 10, 20]
STRATEGIES = ["S0_in_sample", "S1_oof", "S2_oof_repeated", "S3_oof_boot_median",
              "S4_validation", "S5_analytic", "S6_analytic_iso", "S7_rate_match"]


def oof_scores(clf, X, y, folds, seed, repeats=1):
    acc = None
    for r in range(repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        p = cross_val_predict(clone(clf), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        acc = p if acc is None else acc + p
    return acc / repeats


def boot_median_threshold(prs, c, B, seed):
    rng = random.Random(seed)
    n = len(prs)
    ts = sorted(sweep_once([prs[rng.randrange(n)] for _ in range(n)], c)[0] for _ in range(B))
    return ts[len(ts) // 2]


def quantile(values, q):
    """The score below which fraction `q` of the pool sits. Plain and explicit rather than
    numpy's interpolation, so the threshold is always a score that actually occurred."""
    v = sorted(values)
    if not v:
        return 0.0
    i = min(len(v) - 1, max(0, int(round((1 - q) * len(v)))))
    return v[i]


def evaluate(clf, Xtr, ytr, Xva, yva, Xte, yte, folds, seed, boot):
    clf.fit(Xtr, ytr)
    p_in = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    p_oof = oof_scores(clf, Xtr, ytr, folds, seed, repeats=1)
    p_oof3 = oof_scores(clf, Xtr, ytr, folds, seed, repeats=3)

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_oof.astype(float), ytr.astype(float))
    q_oof = ir.predict(p_oof.astype(float))
    q_te = ir.predict(p_te.astype(float))

    in_p, oof_p, oof3_p = pairs(p_in, ytr), pairs(p_oof, ytr), pairs(p_oof3, ytr)
    va_p, te_p = pairs(p_va, yva), pairs(p_te, yte)
    qoof_p, qte_p = pairs(q_oof, ytr), pairs(q_te, yte)

    pos = int(sum(yte))
    neg = len(yte) - pos
    out = {}
    for c in RATIOS:
        floor = min(neg, pos * c)
        t_oof = sweep_once(oof_p, c)[0]
        rate = sum(1 for s, _ in oof_p if s >= t_oof) / len(oof_p)
        picks = {
            "S0_in_sample":       (sweep_once(in_p, c)[0], te_p),
            "S1_oof":             (t_oof, te_p),
            "S2_oof_repeated":    (sweep_once(oof3_p, c)[0], te_p),
            "S3_oof_boot_median": (boot_median_threshold(oof_p, c, boot, seed + c), te_p),
            "S4_validation":      (sweep_once(va_p, c)[0], te_p),
            "S5_analytic":        (1.0 / (1.0 + c), te_p),
            "S6_analytic_iso":    (1.0 / (1.0 + c), qte_p),
            "S7_rate_match":      (quantile([s for s, _ in te_p], rate), te_p),
        }
        row = {}
        for name, (t, target) in picks.items():
            k = cost_at(target, t, c)
            fires = sum(1 for s, _ in target if s >= t) / len(target)
            row[name] = {"cost": int(k), "over_floor": round(k / floor, 4) if floor else None,
                         "threshold": round(float(t), 6), "fires_on_test": round(fires, 4)}
        orc = sweep_once(te_p, c)[1]
        row["S8_oracle"] = {"cost": int(orc),
                            "over_floor": round(orc / floor, 4) if floor else None,
                            "threshold": None, "fires_on_test": None}
        row["_floor"] = int(floor)
        row["_oof_rate"] = round(rate, 4)
        out[c] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank", choices=["bank", "sec2021"])
    ap.add_argument("--arm", default="numeric", choices=["numeric", "words", "embed"])
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    if args.dataset == "bank":
        (tr, va, te), dim = load_bank(args.train_cap, args.seed)
        tag = f"bank{'-n' + str(args.train_cap) if args.train_cap else ''}"
    else:
        (tr, va, te), dim = learned_features(args.arm, args.seed)
        tag = f"sec-{args.arm}"

    Xtr, ytr = to_matrix(tr, dim, True)
    Xva, yva = to_matrix(va, dim, True)
    Xte, yte = to_matrix(te, dim, True)

    clfs = {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                n_jobs=1, random_state=args.seed),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                    early_stopping=False,
                                                    random_state=args.seed),
        "logreg_l2": LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed),
    }
    res = {}
    print(f"{tag} seed {args.seed}: train {Xtr.shape} test {Xte.shape}  "
          f"cost over the best floor, lower is better")
    for nm, clf in clfs.items():
        res[nm] = evaluate(clf, Xtr, ytr, Xva, yva, Xte, yte,
                           args.folds, args.seed, args.boot)
        print(f"\n  {nm}")
        print(f"  {'strategy':<20}" + "".join(f"{'c=' + str(c):>9}" for c in RATIOS))
        for s in STRATEGIES + ["S8_oracle"]:
            print(f"  {s:<20}" + "".join(
                f"{res[nm][c][s]['over_floor']:>9.3f}" for c in RATIOS))

    (HERE / f"thresholds-{tag}-{args.seed}.json").write_text(json.dumps(
        {"tag": tag, "seed": args.seed, "folds": args.folds, "models": res},
        indent=1) + "\n")
    print(f"\n-> thresholds-{tag}-{args.seed}.json")


if __name__ == "__main__":
    main()
