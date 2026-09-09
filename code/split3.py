#!/usr/bin/env python3
"""Three-way split, so the threshold is chosen on data the model never saw and never scored.

`crossover.py` decomposed the loss into discrimination and threshold, using an ORACLE
threshold picked on test. That is an upper bound and a reviewer's first objection: one free
parameter fitted to the evaluation set. This file removes the objection.

    TRAIN 50%   fit the arm
    VAL   25%   pick the threshold, on rows the arm never saw
    TEST  25%   report

Split by COMPANY throughout. Three threshold policies are compared on the same test rows:

    train-picked   the policy every run in this directory used. The model was fitted on
                   these rows, so its scores here are optimistic and the threshold inherits
                   that.
    val-picked     the correct policy, and the one a deployment could actually follow.
    oracle         best threshold on test. Not achievable. Reported as the bound, so the
                   gap between val-picked and oracle is the part of the loss that no amount
                   of threshold care can recover.

# The mechanism, measured rather than asserted

The claim is that under a high cost ratio the optimal operating point migrates toward a
corner of ROC space where it is set by a handful of rows, so its estimate does not transfer.
That predicts the VARIANCE of the picked threshold grows with c. Bootstrapping the validation
set measures exactly that, and it is the difference between a story and a result.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from costs import cost_at, lookup_scores

HERE = Path(__file__).parent
RATIOS = [1, 2, 3, 5, 8, 10, 15, 20]


def sweep_once(pairs, c):
    """Best threshold and its cost in one descending pass.

    The generic sweep in costs.py re-scores every row at every candidate threshold, which is
    fine for a handful of calls and hopeless for forty bootstraps at eight ratios. Sorting
    once and accumulating gives the identical answer in O(n log n): walk the scores from high
    to low, and at each distinct value the confusion matrix for "fire at or above this" is
    already in hand.
    """
    rows = sorted(pairs, key=lambda t: -t[0])
    P = sum(y for _, y in rows)
    # fire on nothing: every positive missed, no false alarm
    best, best_thr = P * c, rows[0][0] + 1.0 if rows else 1.0
    tp = fp = 0
    i, n = 0, len(rows)
    while i < n:
        s = rows[i][0]
        while i < n and rows[i][0] == s:
            tp += rows[i][1]
            fp += 1 - rows[i][1]
            i += 1
        cost = (P - tp) * c + fp
        if cost < best:
            best, best_thr = cost, s
    return best_thr, best


def best_threshold(pairs, c):
    return sweep_once(pairs, c)[0]


def oracle_cost(pairs, c):
    return sweep_once(pairs, c)[1]


def bootstrap_thresholds(val, c, B, seed):
    """Resample the validation set with replacement and re-pick. The spread of what comes
    back is the estimation variance the deployed threshold is exposed to."""
    rng = random.Random(seed)
    n = len(val)
    out = []
    for b in range(B):
        samp = [val[rng.randrange(n)] for _ in range(n)]
        out.append(best_threshold(samp, c))
    out.sort()
    return out


def three_way(rows, key, fit, seed, boot=40):
    ciks = sorted({key(r) for r in rows})
    random.Random(seed).shuffle(ciks)
    n = len(ciks)
    a, b = n // 2, (3 * n) // 4
    g_tr, g_va, g_te = set(ciks[:a]), set(ciks[a:b]), set(ciks[b:])
    tr = [r for r in rows if key(r) in g_tr]
    va = [r for r in rows if key(r) in g_va]
    te = [r for r in rows if key(r) in g_te]
    return fit(tr, va, te)


def report(name, train, val, test, boot, seed):
    pos = sum(y for _, y in test)
    neg = len(test) - pos
    print(f"\n{name}")
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}  "
          f"base rate {pos / len(test):.3f}")
    print(f"{'c':>4} {'floor':>8} {'train-pick':>11} {'val-pick':>9} {'oracle':>8} | "
          f"{'val/floor':>10} {'penalty':>8} | {'thr IQR on val':>15} {'cost IQR':>10}")
    rows_out = []
    for c in RATIOS:
        floor = min(neg, pos * c)
        t_tr = best_threshold(train, c)
        t_va = best_threshold(val, c)
        c_tr = cost_at(test, t_tr, c)
        c_va = cost_at(test, t_va, c)
        c_or = oracle_cost(test, c)
        bs = bootstrap_thresholds(val, c, boot, seed + c)
        lo, hi = bs[len(bs) // 4], bs[(3 * len(bs)) // 4]
        costs_bs = sorted(cost_at(test, t, c) for t in bs)
        clo, chi = costs_bs[len(bs) // 4], costs_bs[(3 * len(bs)) // 4]
        pen = (c_va - c_or) / floor if floor else None
        rows_out.append({
            "ratio": c, "floor": floor, "train_picked": c_tr, "val_picked": c_va,
            "oracle": c_or, "val_over_floor": round(c_va / floor, 4) if floor else None,
            "penalty_over_floor": round(pen, 4) if pen is not None else None,
            "thr_val": round(t_va, 4), "thr_iqr": [round(lo, 4), round(hi, 4)],
            "thr_iqr_width": round(hi - lo, 4),
            "cost_iqr": [clo, chi], "cost_iqr_width_over_floor":
                round((chi - clo) / floor, 4) if floor else None,
            "beats_floor": c_va < floor,
        })
        print(f"{c:>4} {floor:>8} {c_tr:>11} {c_va:>9} {c_or:>8} | "
              f"{c_va / floor:>10.3f} {pen:>8.3f} | "
              f"{lo:>6.3f}-{hi:<8.3f} {(chi - clo) / floor:>10.3f}")
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all",
                    choices=["all", "lookup", "lookup2021", "numeric", "words", "embed"])
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--boot", type=int, default=40)
    args = ap.parse_args()

    out = {}
    pairs = [json.loads(l) for l in (HERE / "pairs-sliced.jsonl").read_text().splitlines()]

    def lookup_fit(tr, va, te):
        # bucket rates are fitted on TRAIN only; val and test are both scored by them
        trp, vap = lookup_scores(tr, va)
        _, tep = lookup_scores(tr, te)
        return trp, vap, tep

    if args.arm in ("all", "lookup"):
        a, b, c = three_way(pairs, lambda r: r["cik"], lookup_fit, args.seed)
        out["lookup/all"] = report("lookup / all years", a, b, c, args.boot, args.seed)
    if args.arm in ("all", "lookup2021"):
        rs = [r for r in pairs if r["filed"][:4] == "2021"]
        a, b, c = three_way(rs, lambda r: r["cik"], lookup_fit, args.seed)
        out["lookup/2021"] = report("lookup / 2021", a, b, c, args.boot, args.seed)

    if args.arm in ("all", "numeric", "words", "embed"):
        arms = ["numeric", "words", "embed"] if args.arm == "all" else [args.arm]
        for arm in arms:
            a, b, c = learned_three_way(arm, args.seed)
            out[f"{arm}/2021"] = report(f"{arm} / 2021", a, b, c, args.boot, args.seed)

    (HERE / "split3.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n-> split3.json")


def learned_three_way(arm, seed):
    """Scores from this repository's own logistic regression, for the pure-stdlib path."""
    from arms import predict, train_logreg
    (tr, va, te), dim = learned_features(arm, seed)
    w, b = train_logreg(tr, dim, seed=seed)
    sc = lambda rs: [(predict(w, b, x), y) for x, y in rs]
    return sc(tr), sc(va), sc(te)


def learned_features(arm, seed):
    """The three splits as (feature dict, label) lists, plus the dimension.

    Split out from learned_three_way so a different learner can be handed the same rows and
    the same features. strong.py uses this to ask whether the threshold artefact is an
    artefact of the weak model, and that question is only answerable if nothing else moves.
    """
    from arms import (
        BODIES, FACTS, STOP, WORD, numeric_features, text_features,
    )

    rows = [json.loads(l) for l in (HERE / "views-2021.jsonl").read_text().splitlines()]
    usable = [r for r in rows if r["chars_narrative"] > 500 and r["n_facts"] > 0]
    ciks = sorted({r["cik"] for r in usable})
    random.Random(seed).shuffle(ciks)
    n = len(ciks)
    g_tr, g_va = set(ciks[: n // 2]), set(ciks[n // 2: (3 * n) // 4])
    g_te = set(ciks[(3 * n) // 4:])
    tr = [r for r in usable if r["cik"] in g_tr]
    va = [r for r in usable if r["cik"] in g_va]
    te = [r for r in usable if r["cik"] in g_te]

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

    if arm == "numeric":
        df = Counter()
        for r in tr:
            df.update(set(facts(r)))
        tags = {t: i for i, (t, _) in enumerate(df.most_common(600))}
        feat = lambda r: numeric_features(facts(r), tags, len(tags))
        dim = len(tags) + 1
    elif arm == "words":
        df = Counter()
        for r in tr:
            df.update(set(w for w in WORD.findall(narrative(r).lower()) if w not in STOP))
        vocab = {t: i for i, (t, _) in enumerate(df.most_common(3000))}
        feat = lambda r: text_features(narrative(r), vocab, len(vocab))
        dim = len(vocab) + 1
    else:
        vd = HERE / "raw" / "vecs" / "nomic-embed-text-mean"
        cache = {}

        def vec(r):
            if r["adsh"] not in cache:
                p = vd / f"{r['adsh']}.json"
                cache[r["adsh"]] = json.loads(p.read_text())["vec"] if p.is_file() else None
            return cache[r["adsh"]]

        tr = [r for r in tr if vec(r)]
        va = [r for r in va if vec(r)]
        te = [r for r in te if vec(r)]
        dim = len(vec(tr[0]))
        feat = lambda r: {i: v for i, v in enumerate(vec(r))}

    build = lambda rs: [(feat(r), r["label"]) for r in rs]
    return (build(tr), build(va), build(te)), dim


if __name__ == "__main__":
    main()
