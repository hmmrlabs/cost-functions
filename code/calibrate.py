#!/usr/bin/env python3
"""Can the threshold be fixed without spending a validation split?

`RESULT-strong.md` measured the artefact and named its mechanism: a threshold is chosen so
that some fraction of TRAIN fires, and on new rows it fires on a different fraction. It also
named the cheap fix it had not tested. This tests it, and tests something more useful.

**"Use a validation set" is a weak recommendation.** Every practitioner would say they already
do, and holding out a quarter of the data to place one scalar is expensive on a small corpus.
The question worth answering is whether the threshold can be placed correctly using the
TRAINING SET ALONE, and the obvious candidate is out-of-fold prediction: K-fold inside train,
each row scored by a model that did not see it, threshold picked on those scores. It costs K
extra fits and no data.

Six policies, all evaluated on the same untouched test rows:

    A  in-sample                 threshold on the fitted model's own training scores.
                                 The practice this directory used throughout.
    B  in-sample + isotonic      calibrate on the training scores, then threshold. **This is
                                 the placebo.** Calibrating a model against the labels it
                                 already memorised cannot recover a distribution it never
                                 got wrong on those rows, so if this fixes anything the
                                 mechanism claim is wrong.
    C  out-of-fold               5-fold inside train, threshold on out-of-fold scores. No
                                 held-out data spent.
    D  out-of-fold + isotonic    as C, with the calibrator also fitted out of fold.
    E  held-out validation       the gold standard, and what RESULT-strong.md used.
    F  oracle                    best threshold on test. Not achievable, reported as the bound.

If C lands on E, the recommendation changes from "hold out data" to "stop scoring your
threshold in sample", which costs nothing anyone is unwilling to spend.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from costs import cost_at                                  # noqa: E402
from split3 import sweep_once                              # noqa: E402
from strong import load_bank, pairs, to_matrix             # noqa: E402
from split3 import learned_features                        # noqa: E402

RATIOS = [3, 5, 10, 20]


def isotonic(scores, labels):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return lambda s: ir.predict(np.asarray(s, dtype=float))


def run_cell(name, clf, Xtr, ytr, Xva, yva, Xte, yte, folds, seed):
    clf.fit(Xtr, ytr)
    p_in = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    p_oof = cross_val_predict(clone(clf), Xtr, ytr, cv=cv, method="predict_proba",
                              n_jobs=1)[:, 1]

    cal_in = isotonic(p_in, ytr)
    cal_oof = isotonic(p_oof, ytr)

    # Each policy is (scores the threshold is chosen on, the map applied to test before
    # comparing). A calibrator changes the scale, so the test scores must move with it or
    # the threshold is being read against a different ruler.
    policies = {
        "A_in_sample":       (pairs(p_in, ytr),            pairs(p_te, yte)),
        "B_in_sample_iso":   (pairs(cal_in(p_in), ytr),    pairs(cal_in(p_te), yte)),
        "C_out_of_fold":     (pairs(p_oof, ytr),           pairs(p_te, yte)),
        "D_oof_iso":         (pairs(cal_oof(p_oof), ytr),  pairs(cal_oof(p_te), yte)),
        "E_validation":      (pairs(p_va, yva),            pairs(p_te, yte)),
    }

    base = pairs(p_te, yte)
    pos = sum(y for _, y in base)
    neg = len(base) - pos
    out = {}
    print(f"\n  {name}")
    print(f"  {'policy':<18}" + "".join(f"{'c=' + str(c):>12}" for c in RATIOS)
          + f"{'  fires tr/te @20':>20}")
    for pol, (sel, tst) in policies.items():
        row, drift = [], None
        for c in RATIOS:
            floor = min(neg, pos * c)
            t = sweep_once(sel, c)[0]
            cost = cost_at(tst, t, c)
            row.append((c, cost, floor))
            if c == 20:
                f_sel = sum(1 for s, _ in sel if s >= t) / len(sel)
                f_te = sum(1 for s, _ in tst if s >= t) / len(tst)
                drift = (round(f_sel, 3), round(f_te, 3))
        out[pol] = {"per_ratio": [{"ratio": c, "cost": int(k), "floor": int(f),
                                   "over_floor": round(k / f, 4) if f else None}
                                  for c, k, f in row],
                    "fires_at_20": drift}
        print(f"  {pol:<18}" + "".join(f"{k / f:>12.3f}" for _, k, f in row)
              + f"{str(drift):>20}")
    orc = []
    for c in RATIOS:
        floor = min(neg, pos * c)
        orc.append({"ratio": c, "cost": int(sweep_once(base, c)[1]), "floor": int(floor),
                    "over_floor": round(sweep_once(base, c)[1] / floor, 4) if floor else None})
    out["F_oracle"] = {"per_ratio": orc, "fires_at_20": None}
    print(f"  {'F_oracle':<18}" + "".join(f"{o['over_floor']:>12.3f}" for o in orc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank", choices=["bank", "sec2021"])
    ap.add_argument("--arm", default="numeric", choices=["numeric", "words", "embed"])
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    if args.dataset == "bank":
        (tr, va, te), dim = load_bank(args.train_cap, args.seed)
        tag = f"bank{'-n' + str(args.train_cap) if args.train_cap else ''}"
    else:
        (tr, va, te), dim = learned_features(args.arm, args.seed)
        tag = f"sec2021-{args.arm}"

    Xtr, ytr = to_matrix(tr, dim, True)
    Xva, yva = to_matrix(va, dim, True)
    Xte, yte = to_matrix(te, dim, True)
    print(f"{tag} seed {args.seed}: train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
    print("cost over the best floor, lower is better")

    clfs = {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                n_jobs=1, random_state=args.seed),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                    early_stopping=False,
                                                    random_state=args.seed),
        "logreg_l2": LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed),
    }
    res = {n: run_cell(n, c, Xtr, ytr, Xva, yva, Xte, yte, args.folds, args.seed)
           for n, c in clfs.items()}

    (HERE / f"calibrate-{tag}-{args.seed}.json").write_text(json.dumps(
        {"tag": tag, "seed": args.seed, "folds": args.folds, "models": res}, indent=1) + "\n")
    print(f"\n-> calibrate-{tag}-{args.seed}.json")


if __name__ == "__main__":
    main()
