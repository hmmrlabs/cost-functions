#!/usr/bin/env python3
"""The forward-in-time split, and the company split run beside it on identical rows.

Every threshold result in this directory splits by `cik`. A reviewer of a financial
prediction paper will not accept that: in deployment you hold the past and score the
future, so a split that lets a 2019 row train a model that scores a 2015 row is answering
an easier question than the one anybody has. `RESULT-calibrate.md`, `RESULT-strong.md` and
`RESULT-thresholds.md` all close by naming a temporal split as still open. This closes it.

# The scheme

Rolling origin, one origin per deployment date, with a gap block:

    TRAIN   filings filed in years [Y - 4, Y]      fit the model
    VAL     filings filed in year   Y + 1          the held-out block, S4's selection rows
    TEST    filings filed in year   Y + 2          report

Five origins: Y = 2006, 2010, 2014, 2018, 2020. Test years 2008, 2012, 2016, 2020 and
2022, which between them cover the crisis, the quiet middle, the 2020 shock and the SPAC
wave of 2021 to 2022. One lucky year cannot carry the result.

Two choices in that scheme are worth defending.

**Why a fixed five-year window and not an expanding one.** An expanding window would run
this corpus's training set from 22,500 rows at Y = 2006 to 105,000 at Y = 2020.
`RESULT-thresholds.md` established that the ranking of threshold strategies is a SIZE rule:
out-of-fold is free and exactly as good as a validation split at 22,605 training rows and
is worth a tenth of the floor less at 2,167. Letting the training set quadruple across
origins would confound the origin with that size rule, and the origin is the thing being
varied. A five-year window holds every origin between 15,091 and 34,106 rows.

**Why the test block is one year and not "Y + 2 onward".** Testing on everything after Y + 2
gives the 2006 origin a sixteen-year deployment horizon and the 2020 origin a two-year one,
so an origin-to-origin difference could be horizon length rather than anything about the
origin. One year per origin makes the five numbers comparable.

# Features

Tabular only: SIC division, form type, amended flag, filing year, filing month, reporting
lag, and two filing-frequency features. The narrative and XBRL views in `views-2021.jsonl`
exist for 2021 alone, so `learned_features` from `split3.py` cannot produce a train block
that precedes its test block at all. Choosing those views would mean not running this
experiment. The cost of the choice is that this file cannot say whether the finding holds
for a text model, and it does not claim to.

Both frequency features are computed from filings strictly BEFORE the row's own `filed`
date, so they carry no future information under either split axis.

`filed_year` is included because the brief asks for it and because leaving it out would
hide something. Under a temporal split every test row's year lies outside the training
range, so a tree cannot split on it usefully and a linear model extrapolates off the end of
its fitted range. That is a real property of deploying forward and not a bug in the setup,
but it does degrade the MODEL as well as the threshold, so every cell here also reports the
oracle threshold and the regret against it. Regret isolates the line from the learner, which
is the only column that can tell a threshold result from a model-quality result.

# Rows

Restricted to filings filed 2004 onward. 8-K Item 4.02 is the labelling instrument and it
took effect on 23 August 2004, so a row's label is only as complete as the part of its
730-day horizon that Item 4.02 was in force for. The 46,433 rows filed 1994 to 2002 carry
three positives between them and the 8,231 filed in 2003 carry a base rate of 0.020 against
2005's 0.185, not because those filings were sound but because there was no instrument to
repudiate them with. Training a temporal origin on a block whose base rate is set by the
SEC's rulemaking calendar would measure the rulemaking calendar.

**2004 is itself partly truncated**: a filing made in January 2004 has seven of its
twenty-four horizon months outside the rule. It is kept because dropping it would leave the
2006 origin with a single training year, and the 2006 origin is the only one whose test
block reaches the crisis. The cost of keeping it is that the earliest origin's training base
rate is biased downward by an artefact of the calendar, and `RESULT-temporal.md` says so.

The 2004 floor also clips the earliest window: the 2006 origin trains on 2004 to 2006 rather
than 2002 to 2006, so its window is three years and not five. Every block records the years
it actually holds rather than the years the scheme asked for.

# The control

The question is not "how does a temporal split score" but "does the in-sample threshold
artefact change size when the split becomes temporal". A number without its control cannot
answer that, and the company-split numbers already in this directory are on different
features and a different row pool. So each origin is ALSO run as a company split over the
identical pooled rows at the identical block sizes, changing the split axis and nothing
else. The paired difference is the answer.

Strategies are the ones this directory already benchmarks, names unchanged so the columns
line up with `thresholds-summary.json`:

    S0_in_sample   threshold on the fitted model's own training scores. The defect.
    S1_oof         5-fold out-of-fold inside train. Random folds, not time-ordered ones,
                   because changing the strategy and the split at once would leave the
                   comparison unreadable.
    S4_validation  the held-out block, which under the temporal axis is a future year.
    S8_oracle      best threshold on test. Not achievable, reported as the bound.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from costs import cost_at                     # noqa: E402
from split3 import sweep_once                 # noqa: E402
from strong import pairs                      # noqa: E402

RATIOS = [3, 5, 10, 20]
ORIGINS = [2006, 2010, 2014, 2018, 2020]
WINDOW = 5
FIRST_YEAR = 2004
STRATEGIES = ["S0_in_sample", "S1_oof", "S4_validation"]


# ---------------------------------------------------------------- rows and features


def load_rows(path):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows = [r for r in rows if r["filed"][:4] >= str(FIRST_YEAR)]
    rows.sort(key=lambda r: (r["filed"], r["adsh"]))
    return rows


def _days(s):
    y, m, d = int(s[:4]), int(s[5:7]), int(s[8:10])
    return date(y, m, d).toordinal()


def history_features(rows):
    """Per-row filing history, using only filings strictly earlier than this one.

    A count of a company's filings taken over the whole corpus would tell a 2008 row how
    many times its filer appears in 2019, which is precisely the leak a temporal split
    exists to forbid. Rows are already sorted by `filed`, so one pass suffices; ties on the
    same date are resolved by not counting the tied rows, which keeps the feature a
    function of strictly prior information.

    Keyed on `(cik, adsh)` and not on `adsh` alone. 227 accession numbers in this corpus
    belong to joint filings made by two co-registrants, so the same document appears once
    per filer under one number. Keying on the number alone would hand one of the two
    companies the other's filing history.
    """
    seen = defaultdict(list)
    out = {}
    for r in rows:
        d = _days(r["filed"])
        prior = [t for t in seen[r["cik"]] if t < d]
        gap = (d - max(prior)) if prior else -1
        out[(r["cik"], r["adsh"])] = (len(prior), gap)
        seen[r["cik"]].append(d)
    return out


def vocabularies(train_rows):
    """Built on TRAIN only. A SIC division or a form type first seen in the test year must
    land in an `other` slot, because a deployment cannot widen its own feature space."""
    sics = sorted({str(r.get("sic") or "0")[:2] for r in train_rows})
    forms = sorted({r["form"] for r in train_rows})
    return {s: i for i, s in enumerate(sics)}, {f: i for i, f in enumerate(forms)}


def matrix(rows, sic_ix, form_ix, hist):
    ns, nf = len(sic_ix), len(form_ix)
    dim = ns + 1 + nf + 1 + 6
    X = np.zeros((len(rows), dim), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        s = str(r.get("sic") or "0")[:2]
        X[i, sic_ix.get(s, ns)] = 1.0
        X[i, ns + 1 + form_ix.get(r["form"], nf)] = 1.0
        b = ns + 1 + nf + 1
        n_prior, gap = hist[(r["cik"], r["adsh"])]
        lag = _days(r["filed"]) - _days(r["period"]) if r.get("period") else -1
        X[i, b + 0] = 1.0 if r["form"].endswith("/A") else 0.0
        # Filing year as years since 2004 rather than as 2018. A constant shift of one
        # feature moves only the (unpenalised) intercept, so the fitted logistic regression
        # is the same model, and a tree splits on a shifted feature at shifted thresholds
        # and partitions the rows identically. What it does change is the conditioning:
        # against log-scale neighbours a column near 2000 leaves lbfgs unconverged at 2000
        # iterations, and an unconverged cell is not a measurement.
        X[i, b + 1] = int(r["filed"][:4]) - FIRST_YEAR
        X[i, b + 2] = int(r["filed"][5:7])
        X[i, b + 3] = math.log1p(n_prior)
        X[i, b + 4] = math.log1p(gap) if gap >= 0 else -1.0
        X[i, b + 5] = math.log1p(lag) if 0 <= lag <= 3650 else -1.0
        y[i] = r["label"]
    return X, y


# ---------------------------------------------------------------- blocks


def temporal_blocks(rows, origin, window):
    lo, hi = origin - window + 1, origin
    tr = [r for r in rows if lo <= int(r["filed"][:4]) <= hi]
    va = [r for r in rows if int(r["filed"][:4]) == origin + 1]
    te = [r for r in rows if int(r["filed"][:4]) == origin + 2]
    return tr, va, te


def company_blocks(pool, n_tr, n_va, seed):
    """The same rows, the same block sizes, split by company instead of by date.

    Companies are shuffled and assigned greedily until each block reaches its target count,
    so the sizes match the temporal blocks to within one company's worth of filings. Sizes
    have to match: `RESULT-thresholds.md` showed the strategy ranking moves with training
    size, so a control that also changed n would not be a control.
    """
    import random as _r
    by_cik = defaultdict(list)
    for r in pool:
        by_cik[r["cik"]].append(r)
    ciks = sorted(by_cik)
    _r.Random(seed).shuffle(ciks)
    tr, va, te = [], [], []
    for c in ciks:
        if len(tr) < n_tr:
            tr.extend(by_cik[c])
        elif len(va) < n_va:
            va.extend(by_cik[c])
        else:
            te.extend(by_cik[c])
    return tr, va, te


# ---------------------------------------------------------------- evaluation


def oof_scores(clf, X, y, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(clone(clf), X, y, cv=cv,
                             method="predict_proba", n_jobs=1)[:, 1]


def evaluate(clf, Xtr, ytr, Xva, yva, Xte, yte, folds, seed):
    clf.fit(Xtr, ytr)
    p_in = clf.predict_proba(Xtr)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    p_oof = oof_scores(clf, Xtr, ytr, folds, seed)

    in_p = pairs(p_in, ytr)
    oof_p = pairs(p_oof, ytr)
    va_p = pairs(p_va, yva)
    te_p = pairs(p_te, yte)

    pos = int(yte.sum())
    neg = len(yte) - pos
    auc = {"train": round(float(roc_auc_score(ytr, p_in)), 4),
           "val": round(float(roc_auc_score(yva, p_va)), 4) if len(set(yva)) > 1 else None,
           "test": round(float(roc_auc_score(yte, p_te)), 4) if len(set(yte)) > 1 else None}

    out = {"auc": auc, "test_pos": pos, "test_n": len(yte), "ratios": {}}
    for c in RATIOS:
        floor = min(neg, pos * c)
        # (threshold, the rows the threshold was READ OFF, for the drift measurement)
        picks = {
            "S0_in_sample":  (sweep_once(in_p, c)[0], in_p),
            "S1_oof":        (sweep_once(oof_p, c)[0], oof_p),
            "S4_validation": (sweep_once(va_p, c)[0], va_p),
        }
        orc_t, orc = sweep_once(te_p, c)
        row = {"_floor": int(floor)}
        for name, (t, sel) in picks.items():
            k = cost_at(te_p, t, c)
            f_sel = sum(1 for s, _ in sel if s >= t) / len(sel)
            f_te = sum(1 for s, _ in te_p if s >= t) / len(te_p)
            row[name] = {
                "cost": int(k),
                "over_floor": round(k / floor, 4) if floor else None,
                "regret": round((k - orc) / floor, 4) if floor else None,
                "threshold": round(float(t), 6),
                "fires_on_selection": round(f_sel, 4),
                "fires_on_test": round(f_te, 4),
                "fire_drift": round(f_te - f_sel, 4),
            }
        row["S8_oracle"] = {"cost": int(orc),
                            "over_floor": round(orc / floor, 4) if floor else None,
                            "regret": 0.0,
                            "threshold": round(float(orc_t), 6),
                            "fires_on_selection": None,
                            "fires_on_test": round(
                                sum(1 for s, _ in te_p if s >= orc_t) / len(te_p), 4),
                            "fire_drift": None}
        out["ratios"][c] = row
    return out


def models(seed):
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                n_jobs=1, random_state=seed),
        "hist_gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                    early_stopping=False,
                                                    random_state=seed),
        "logreg_l2": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    }


def run_axis(axis, tr, va, te, hist, folds, seed):
    sic_ix, form_ix = vocabularies(tr)
    Xtr, ytr = matrix(tr, sic_ix, form_ix, hist)
    Xva, yva = matrix(va, sic_ix, form_ix, hist)
    Xte, yte = matrix(te, sic_ix, form_ix, hist)
    print(f"  {axis:<8} train {Xtr.shape} val {len(va)} test {len(te)}  "
          f"base rate tr {ytr.mean():.4f} te {yte.mean():.4f}")
    res = {}
    for nm, clf in models(seed).items():
        res[nm] = evaluate(clf, Xtr, ytr, Xva, yva, Xte, yte, folds, seed)
        r = res[nm]["ratios"]
        print(f"    {nm:<14}" + "".join(f"{'c=' + str(c):>8}" for c in RATIOS))
        for s in STRATEGIES + ["S8_oracle"]:
            print(f"      {s:<12}" + "".join(
                f"{r[c][s]['over_floor']:>8.3f}" for c in RATIOS))
        print(f"      {'drift S0':<12}" + "".join(
            f"{r[c]['S0_in_sample']['fire_drift']:>+8.3f}" for c in RATIOS))
    return {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "base_rate_train": round(float(ytr.mean()), 4),
            "base_rate_test": round(float(yte.mean()), 4),
            "models": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default=",".join(str(o) for o in ORIGINS))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="20260818,20260819")
    ap.add_argument("--out", default="temporal.json")
    args = ap.parse_args()

    rows = load_rows(HERE / "pairs-sliced.jsonl")
    hist = history_features(rows)
    print(f"{len(rows)} filings filed {rows[0]['filed']} to {rows[-1]['filed']}, "
          f"{len({r['cik'] for r in rows})} companies")

    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"seeds": seeds, "window": args.window, "folds": args.folds,
           "ratios": RATIOS, "first_year": FIRST_YEAR, "runs": {}}
    # The temporal blocks do not depend on the seed at all: a calendar year is a calendar
    # year. What does depend on it is the fold assignment inside train and, on the control
    # axis, which companies land in which block. The headline here is a NULL, and a null
    # from one draw of the control is a null about one draw of the control, so both are run.
    for seed in seeds:
        out["runs"][seed] = {}
        for origin in [int(o) for o in args.origins.split(",")]:
            tr, va, te = temporal_blocks(rows, origin, args.window)
            if not tr or not va or not te:
                print(f"origin {origin}: a block is empty, skipped")
                continue
            print(f"\nseed {seed} origin {origin}: train {origin - args.window + 1}-"
                  f"{origin}  val {origin + 1}  test {origin + 2}")
            # the years the train block actually holds, not the ones the scheme asked for:
            # the 2004 floor clips the earliest window to three years
            rec = {"train_years": [int(tr[0]["filed"][:4]), int(tr[-1]["filed"][:4])],
                   "val_year": origin + 1, "test_year": origin + 2, "axes": {}}
            rec["axes"]["temporal"] = run_axis("temporal", tr, va, te, hist,
                                               args.folds, seed)
            ctr, cva, cte = company_blocks(tr + va + te, len(tr), len(va), seed)
            rec["axes"]["company"] = run_axis("company", ctr, cva, cte, hist,
                                              args.folds, seed)
            out["runs"][seed][origin] = rec
            (HERE / args.out).write_text(json.dumps(out, indent=1) + "\n")

    (HERE / args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
