#!/usr/bin/env python3
"""Does the threshold artefact survive a model that is not a hand-written logistic regression?

The first objection to RESULT-threshold.md is that its models are weak: a bag-of-words logistic
regression trained by hand, 40 epochs, no regularisation search. A reviewer is entitled to
suspect the whole effect is an artefact of a bad learner and that a real one would not display
it.

**The prediction runs the other way, and this file is written to be wrong if that is wrong.**

The mechanism is a mismatch between the score distribution a model produces on the rows it was
fitted on and the one it produces on new rows. A threshold chosen on the first is displaced
when applied to the second, and the deeper into a tail the cost ratio drives that threshold,
the more the displacement costs. A STRONGER learner separates the training classes MORE, so it
should show a LARGER mismatch and therefore a larger artefact, not a smaller one.

So: gradient boosting and a random forest, which reach near-perfect training scores, against
regularised logistic regression and against this repository's own arm, on identical rows,
identical features and identical splits. Reported beside the train-minus-test AUC gap, because
if the prediction is right the artefact should track that gap and not the model's quality.

Runs under a virtualenv with scikit-learn; everything else in this directory is stdlib only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from split3 import learned_features, sweep_once          # noqa: E402
from costs import cost_at                                 # noqa: E402

RATIOS = [1, 2, 3, 5, 8, 10, 15, 20]


def to_matrix(rows, dim, dense_ok):
    """Feature dicts to a matrix. Sparse for the 3,001-term word arm, dense for everything
    else, because the tree learners here do not take sparse input."""
    data, ri, ci = [], [], []
    for i, (x, _) in enumerate(rows):
        for j, v in x.items():
            if j < dim:
                ri.append(i)
                ci.append(j)
                data.append(v)
    m = csr_matrix((data, (ri, ci)), shape=(len(rows), dim))
    y = np.array([y for _, y in rows])
    return (m.toarray() if dense_ok else m), y


def models(seed, n_features):
    return {
        # deliberately generous: enough capacity to fit the training set hard
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=False, random_state=seed),
        "random_forest": RandomForestClassifier(
            # n_jobs=1 deliberately: with threads, predict_proba's summation order varies
            # and the AUC moved in the third decimal between identical runs. A result that
            # cannot be reproduced to the precision it is quoted at is not a result.
            n_estimators=300, min_samples_leaf=1, n_jobs=1, random_state=seed),
        # the well-specified comparison: L2 with a real penalty
        "logreg_l2": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=seed),
    }


def pairs(scores, labels):
    """numpy scalars out, plain Python in. The stdlib scorers in costs.py and split3.py are
    shared with the pure-Python path and their results are serialised to JSON."""
    return [(float(s), int(y)) for s, y in zip(scores, labels)]


def evaluate(name, strain, sval, stest, gap):
    pos = int(sum(y for _, y in stest))
    neg = len(stest) - pos
    out = []
    print(f"\n  {name:<14} train-test AUC gap {gap['train']:.3f} - {gap['test']:.3f} "
          f"= {gap['train'] - gap['test']:+.3f}")
    print(f"  {'c':>4} {'floor':>8} {'train-pick':>11} {'val-pick':>9} {'oracle':>8} | "
          f"{'train/val':>10} {'val/floor':>10} | {'fires tr':>9} {'fires te':>9} "
          f"{'drift':>7}")
    for c in RATIOS:
        floor = min(neg, pos * c)
        t_tr = sweep_once(strain, c)[0]
        t_va = sweep_once(sval, c)[0]
        c_tr = cost_at(stest, t_tr, c)
        c_va = cost_at(stest, t_va, c)
        c_or = sweep_once(stest, c)[1]
        # The displaced quantity itself. A threshold is chosen so that some fraction of
        # TRAIN fires; applied to new rows it fires on a different fraction, and that drift
        # is what the cost pays for. AUC gap measures a mismatch in RANKING; this measures
        # the mismatch in the score DISTRIBUTION at the operating point, which is the thing
        # a threshold is exposed to and the reason an uncalibrated forest is worst.
        f_tr = sum(1 for s, _ in strain if s >= t_tr) / len(strain)
        f_te = sum(1 for s, _ in stest if s >= t_tr) / len(stest)
        out.append({"ratio": c, "floor": floor, "train_picked": c_tr, "val_picked": c_va,
                    "oracle": c_or,
                    "train_over_val": round(c_tr / c_va, 4) if c_va else None,
                    "val_over_floor": round(c_va / floor, 4) if floor else None,
                    "fires_on_train": round(f_tr, 4), "fires_on_test": round(f_te, 4),
                    "fire_drift": round(f_te - f_tr, 4)})
        print(f"  {c:>4} {floor:>8} {c_tr:>11} {c_va:>9} {c_or:>8} | "
              f"{c_tr / max(c_va, 1):>10.3f} {c_va / floor:>10.3f} | "
              f"{f_tr:>9.3f} {f_te:>9.3f} {f_te - f_tr:>+7.3f}")
    return out


def load_bank(train_cap, seed):
    import random as _r
    from replicate import featurise, read_arff
    names, types, rows = read_arff(HERE / "raw" / "public" / "bank-marketing.arff")
    data, dim = featurise(names, types, rows, names[-1], "2")
    _r.Random(seed).shuffle(data)
    n = len(data)
    a, b = n // 2, (3 * n) // 4
    tr, va, te = data[:a], data[a:b], data[b:]
    if train_cap:
        tr = tr[:train_cap]
    return (tr, va, te), dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sec2021", choices=["sec2021", "bank"])
    ap.add_argument("--arm", default="numeric", choices=["numeric", "words", "embed"])
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()

    if args.dataset == "bank":
        (tr, va, te), dim = load_bank(args.train_cap, args.seed)
        tag = f"bank{'-n' + str(args.train_cap) if args.train_cap else ''}"
    else:
        (tr, va, te), dim = learned_features(args.arm, args.seed)
        tag = f"sec2021-{args.arm}"

    dense = dim <= 1200
    Xtr, ytr = to_matrix(tr, dim, dense)
    Xva, yva = to_matrix(va, dim, dense)
    Xte, yte = to_matrix(te, dim, dense)
    print(f"{tag}: train {Xtr.shape} val {Xva.shape} test {Xte.shape}  "
          f"p/n {dim / len(tr):.3f}  base rate {ytr.mean():.4f}")

    results = {}
    for name, clf in models(args.seed, dim).items():
        if not dense and name in ("hist_gbdt",):
            Xa, Xb, Xc = Xtr.toarray(), Xva.toarray(), Xte.toarray()
        else:
            Xa, Xb, Xc = Xtr, Xva, Xte
        try:
            clf.fit(Xa, ytr)
        except Exception as e:                       # noqa: BLE001
            print(f"\n  {name}: skipped, {type(e).__name__}: {e}")
            continue
        pa = clf.predict_proba(Xa)[:, 1]
        pb = clf.predict_proba(Xb)[:, 1]
        pc = clf.predict_proba(Xc)[:, 1]
        gap = {"train": roc_auc_score(ytr, pa), "val": roc_auc_score(yva, pb),
               "test": roc_auc_score(yte, pc)}
        results[name] = {
            "auc": {k: round(v, 4) for k, v in gap.items()},
            "grid": evaluate(name, pairs(pa, ytr), pairs(pb, yva), pairs(pc, yte), gap),
        }

    (HERE / f"strong-{tag}.json").write_text(json.dumps({
        "tag": tag, "dim": dim, "train_rows": len(tr), "p_over_n": round(dim / len(tr), 4),
        "seed": args.seed, "models": results,
    }, indent=1) + "\n")
    print(f"\n-> strong-{tag}.json")


if __name__ == "__main__":
    main()
