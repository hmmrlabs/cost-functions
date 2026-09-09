#!/usr/bin/env python3
"""Dump the real score distributions behind the threshold artefact, for drawing.

RESULT-strong.md states the mechanism as two numbers: a random forest on bank-marketing picks
a threshold that fires on 0.117 of its training rows and 0.050 of test. Those two numbers are
a summary of two distributions, and the distributions are what actually explain it, so this
writes them out rather than the summary.

Nothing is simulated. Same model, same seed, same split as the result file.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from strong import load_bank, to_matrix, pairs
from split3 import sweep_once
from costs import cost_at

SEED, C, BINS = 20260818, 20, 200

(tr, va, te), dim = load_bank(0, SEED)
Xtr, ytr = to_matrix(tr, dim, True)
Xva, yva = to_matrix(va, dim, True)
Xte, yte = to_matrix(te, dim, True)

clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, n_jobs=1, random_state=SEED)
clf.fit(Xtr, ytr)
p_tr, p_va, p_te = (clf.predict_proba(X)[:, 1] for X in (Xtr, Xva, Xte))

thr_in = sweep_once(pairs(p_tr, ytr), C)[0]
thr_val = sweep_once(pairs(p_va, yva), C)[0]
edges = np.linspace(0, 1, BINS + 1)

def hist(p, y):
    return {"pos": np.histogram(p[y == 1], bins=edges)[0].tolist(),
            "neg": np.histogram(p[y == 0], bins=edges)[0].tolist()}

# The firing-rate curve: for each candidate threshold, the fraction of rows at or above it.
# This is the quantity a threshold actually controls, and plotting it puts the artefact on one
# pair of axes: the curve you tune on sits above the curve you deploy on, at every cut.
grid = np.linspace(0, 1, 201)
fire_tr = [(float((p_tr >= g).mean())) for g in grid]
fire_te = [(float((p_te >= g).mean())) for g in grid]

out = {
    "fire_grid": grid.tolist(), "fire_train": fire_tr, "fire_test": fire_te,
    "dataset": "bank-marketing", "model": "random_forest", "seed": SEED, "cost_ratio": C,
    "bins": edges.tolist(),
    "train": hist(p_tr, ytr), "test": hist(p_te, yte),
    "threshold_in_sample": float(thr_in), "threshold_validation": float(thr_val),
    "fires_train_at_in_sample": float((p_tr >= thr_in).mean()),
    "fires_test_at_in_sample": float((p_te >= thr_in).mean()),
    "fires_test_at_validation": float((p_te >= thr_val).mean()),
    "cost_in_sample": int(cost_at(pairs(p_te, yte), thr_in, C)),
    "cost_validation": int(cost_at(pairs(p_te, yte), thr_val, C)),
    "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
    "base_rate": float(ytr.mean()),
}
(HERE / "score-shift.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"threshold in-sample {thr_in:.4f}  fires train {out['fires_train_at_in_sample']:.3f} "
      f"-> test {out['fires_test_at_in_sample']:.3f}")
print(f"cost {out['cost_in_sample']} vs {out['cost_validation']} "
      f"= {out['cost_in_sample']/out['cost_validation']:.2f}x")
