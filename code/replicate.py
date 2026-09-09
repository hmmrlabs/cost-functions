#!/usr/bin/env python3
"""Does the threshold-selection artefact appear outside this corpus?

`split3.py` found that the crossover recorded in RESULT-costs.md was a property of the
THRESHOLD POLICY rather than of the models: selecting the operating point on the same rows
the model was fitted on inflates test cost by up to 2.5x at a high cost ratio, and switching
to a held-out validation split flips the conclusion on the same data.

That is either a fact about SEC filings or a fact about cost-sensitive evaluation. This file
decides which, on two public datasets chosen to sit either side of our own base rate.

    bank-marketing   45,211 rows, base rate ~0.117, close to this corpus pooled at 0.104.
                     The floor flips from never-fire to always-fire at c ~ 7.5, so the
                     contested operating point is near the TOP-RIGHT corner of ROC space,
                     which is the regime our finding came from.

    creditcard       284,807 rows, base rate ~0.0017. Always-fire never wins below c = 578,
                     so the contested point sits near the BOTTOM-LEFT corner instead.

**Those two make the claim falsifiable in a useful way.** If the inflation is about corners
of ROC space, where the threshold is set by few rows, it should appear in both. If it is
about high base rates or about filings, it should appear only in the first.

Both are fetched from OpenML, ARFF, no authentication. Split 50/25/25 by row; these have no
grouping structure to respect, unlike filings from one company.
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

from arms import predict, sigmoid, train_logreg
from split3 import sweep_once

HERE = Path(__file__).parent
PUBLIC = HERE / "raw" / "public"
RATIOS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]


def read_arff(path, limit_rows=0):
    """Minimal ARFF reader: attribute names and types from the header, CSV rows after
    @data. Enough for these two files and deliberately not a general parser."""
    names, types = [], []
    rows = []
    in_data = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if not in_data:
                low = s.lower()
                if low.startswith("@attribute"):
                    m = re.match(r"@attribute\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.*)",
                                 s, re.I)
                    nm = m.group(1) or m.group(2) or m.group(3)
                    rest = m.group(4).strip()
                    names.append(nm)
                    types.append("nominal" if rest.startswith("{") else "numeric")
                elif low.startswith("@data"):
                    in_data = True
                continue
            rows.append(next(csv_split(s)))
            if limit_rows and len(rows) >= limit_rows:
                break
    return names, types, rows


def csv_split(line):
    """One ARFF data line into fields, honouring single quotes."""
    out, cur, q = [], [], False
    for ch in line:
        if ch == "'":
            q = not q
        elif ch == "," and not q:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    yield out


def featurise(names, types, rows, target, positive):
    """Numeric columns z-scored, nominal columns one-hot. The scaler is fitted on ALL rows
    rather than on train alone, which is a small leak and is stated: it moves feature scale,
    not the label, and the question here is about threshold selection rather than about
    squeezing the last point of AUC."""
    ti = names.index(target)
    cols = [i for i in range(len(names)) if i != ti]

    stats, levels = {}, {}
    for i in cols:
        if types[i] == "numeric":
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[i]))
                except (ValueError, IndexError):
                    pass
            m = sum(vals) / len(vals) if vals else 0.0
            v = sum((x - m) ** 2 for x in vals) / len(vals) if vals else 1.0
            stats[i] = (m, math.sqrt(v) or 1.0)
        else:
            levels[i] = sorted({r[i] for r in rows if i < len(r)})

    index, nxt = {}, 0
    for i in cols:
        if types[i] == "numeric":
            index[(i, None)] = nxt
            nxt += 1
        else:
            for lv in levels[i]:
                index[(i, lv)] = nxt
                nxt += 1
    dim = nxt + 1  # last slot is the bias-ish constant arms.train_logreg expects to exist

    out = []
    for r in rows:
        x = {}
        for i in cols:
            if types[i] == "numeric":
                try:
                    m, sd = stats[i]
                    x[index[(i, None)]] = (float(r[i]) - m) / sd
                except (ValueError, IndexError):
                    pass
            else:
                j = index.get((i, r[i] if i < len(r) else None))
                if j is not None:
                    x[j] = 1.0
        y = 1 if (r[ti] if ti < len(r) else "") == positive else 0
        out.append((x, y))
    return out, dim


def run(name, data, dim, seed, boot=40, epochs=40, train_cap=0, noise=0):
    random.Random(seed).shuffle(data)
    n = len(data)
    a, b = n // 2, (3 * n) // 4
    tr, va, te = data[:a], data[a:b], data[b:]
    if train_cap:
        tr = tr[:train_cap]
    if noise:
        # pure noise columns, to raise p/n without touching the signal. The point of the
        # knob is that a threshold picked on training scores is displaced by OVERFITTING,
        # and this is the cheapest way to add overfitting to a dataset that has none.
        rng = random.Random(seed + 7)
        base = dim
        for split in (tr, va, te):
            for x, _ in split:
                for j in range(noise):
                    x[base + j] = rng.gauss(0, 1)
        dim = base + noise

    w, bi = train_logreg([(x, y) for x, y in tr], dim, epochs=epochs, seed=seed)
    sc = lambda rs: [(predict(w, bi, x), y) for x, y in rs]
    strain, sval, stest = sc(tr), sc(va), sc(te)

    pos = sum(y for _, y in stest)
    neg = len(stest) - pos
    print(f"\n{name}")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}  "
          f"base rate {pos / len(stest):.5f}")
    print(f"{'c':>4} {'floor':>9} {'floor is':>9} {'train-pick':>11} {'val-pick':>9} "
          f"{'oracle':>8} | {'train/val':>10} {'val/floor':>10} {'penalty':>8}")
    out = []
    for c in RATIOS:
        always, never = neg, pos * c
        floor = min(always, never)
        t_tr, _ = sweep_once(strain, c)
        t_va, _ = sweep_once(sval, c)
        from costs import cost_at
        c_tr = cost_at(stest, t_tr, c)
        c_va = cost_at(stest, t_va, c)
        c_or = sweep_once(stest, c)[1]
        out.append({
            "ratio": c, "floor": floor,
            "floor_is": "always" if always <= never else "never",
            "train_picked": c_tr, "val_picked": c_va, "oracle": c_or,
            "train_over_val": round(c_tr / c_va, 4) if c_va else None,
            "val_over_floor": round(c_va / floor, 4) if floor else None,
            "penalty_over_floor": round((c_va - c_or) / floor, 4) if floor else None,
            "thr_train": round(t_tr, 6), "thr_val": round(t_va, 6),
        })
        print(f"{c:>4} {floor:>9} {'always' if always <= never else 'never':>9} "
              f"{c_tr:>11} {c_va:>9} {c_or:>8} | {c_tr / max(c_va, 1):>10.3f} "
              f"{c_va / floor:>10.3f} {(c_va - c_or) / floor:>8.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank", choices=["bank", "creditcard"])
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap for runtime. Reported, because a cap changes the base rate "
                         "unless it is applied after stratification and this one is not.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-cap", type=int, default=0,
                    help="shrink the training set without touching val or test. The "
                         "threshold artefact is predicted to appear as p/n rises.")
    ap.add_argument("--noise", type=int, default=0,
                    help="pure noise features added to every split, to raise p/n directly")
    args = ap.parse_args()

    if args.dataset == "bank":
        names, types, rows = read_arff(PUBLIC / "bank-marketing.arff", args.max_rows)
        target, positive = names[-1], "2"
        # OpenML's bank-marketing encodes the class as 1 and 2; 2 is "subscribed".
        if not any(r[-1] == positive for r in rows[:5000]):
            positive = sorted({r[-1] for r in rows})[-1]
    else:
        names, types, rows = read_arff(PUBLIC / "creditcard.arff", args.max_rows)
        target = names[-1]
        positive = "1"
        if not any(r[-1] == positive for r in rows[:20000]):
            positive = sorted({r[-1] for r in rows})[-1]

    data, dim = featurise(names, types, rows, target, positive)
    pos = sum(y for _, y in data)
    print(f"{args.dataset}: {len(data)} rows, {dim - 1} features, "
          f"{pos} positives, base rate {pos / len(data):.5f}, class token {positive!r}")
    tag = args.dataset + (f"-n{args.train_cap}" if args.train_cap else "") + \
        (f"-p{args.noise}" if args.noise else "")
    res = run(f"{args.dataset} train_cap={args.train_cap or 'none'} noise={args.noise}",
              data, dim, args.seed, epochs=args.epochs,
              train_cap=args.train_cap, noise=args.noise)
    (HERE / f"replicate-{tag}.json").write_text(json.dumps({
        "dataset": args.dataset, "rows": len(data), "features": dim - 1,
        "positives": pos, "base_rate": round(pos / len(data), 6),
        "max_rows": args.max_rows, "epochs": args.epochs, "seed": args.seed,
        "train_cap": args.train_cap, "noise_features": args.noise,
        "grid": res,
    }, indent=1) + "\n")
    print(f"\n-> replicate-{tag}.json")


if __name__ == "__main__":
    main()
