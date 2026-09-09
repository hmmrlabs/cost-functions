#!/usr/bin/env python3
"""What the cost function was hiding, and four replacements for it.

Every null in RESULT-null.md and RESULT-regimes.md is a null against `fn * 10 + fp`. That
number came from MODELCARD.md, which took it from E22, which chose it for a different
corpus. Nobody measured it against a filing reviewer's actual budget, and it decides every
result here: at 10, always-fire is the floor in all seven regimes, so an arm with real
discrimination is scored against a baseline that reviews 100 percent of filings and is
never charged for the capacity to do so.

**That is the defect. Not the 10.** A linear `fn*c + fp` with no capacity term makes
"review everything" purchasable at any scale, and once it is purchasable it wins whenever
the base rate clears 1/(c+1). No ranking model can beat a strategy that takes every
positive by taking every row. So the corpus has been answering "is the base rate high"
dressed up as "does the document predict".

Four cost functions, each attacking a different assumption.

  SWEEP        keep the form, vary c from 1 to 30. Does a window exist where an arm beats
               both floors, and does any defensible c fall inside it? Threshold re-picked
               on TRAIN at each c, because a threshold tuned for 10 is not the arm's best
               play at 3.

  CAPACITY     drop the form. A reviewer has k slots, so the decision is a QUEUE and not a
               label. recall@k, precision@k, and lift over a random queue. Always-fire is
               not in the choice set: it cannot name which k to open. This is the metric
               the operational problem actually has, and the one under which a ranking
               model is allowed to be useful.

  PROPER       drop the decision. Average precision, ROC AUC, Brier against the base-rate
               constant. Threshold-free, so a degenerate predictor is pinned at the base
               rate and cannot win by refusing to discriminate.

  INSTANCE     keep the form, vary the cost PER ROW. The literature puts the restatement
               announcement effect near -9 percent of market value, so a miss is
               proportional to the firm, not constant. A missed restatement at a blank-check
               shell and one at an operating company are the same row under `fn*10`. Here
               the miss is weighted by the filing's own reported assets.

The first is the user's one-line change. The second is the argument that the one-line change
is still the wrong shape.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from controls import bucket

HERE = Path(__file__).parent
FACTS = HERE / "raw" / "facts"

RATIOS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
KFRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


# ---------------------------------------------------------------- scoring


def confusion(pairs, thr):
    tp = fp = tn = fn = 0
    for s, y in pairs:
        p = 1 if s >= thr else 0
        tp += p == 1 and y == 1
        fp += p == 1 and y == 0
        tn += p == 0 and y == 0
        fn += p == 0 and y == 1
    return tp, fp, tn, fn


def cost_at(pairs, thr, c):
    tp, fp, tn, fn = confusion(pairs, thr)
    return fn * c + fp


def best_threshold(pairs, c):
    """Minimise cost at ratio c ON TRAIN. Swept over the observed score values rather than
    a fixed 1/100 grid: a lookup arm emits a few dozen distinct bucket rates and a fixed
    grid can miss every one of them."""
    cand = sorted({s for s, _ in pairs} | {0.0, 1.01})
    best, best_thr = None, 0.5
    for t in cand:
        c_t = cost_at(pairs, t, c)
        if best is None or c_t < best:
            best, best_thr = c_t, t
    return best_thr


def sweep(train, test, rivals=None):
    """The arm against every floor, at every ratio.

    `floor` is the best of always-fire, never-fire, and any rival baseline passed in, AT
    THAT RATIO. Which floor wins is itself a function of c, so comparing against a fixed
    one would confound the two effects.

    **Rivals are cost-tuned exactly as the arm is.** The year-only baseline is the real bar
    on the pooled corpus, since the era wave is large enough that beating never-fire says
    nothing about having read a filing, and tuning the arm's threshold while leaving the
    bar at its untuned operating point would manufacture the win this whole run exists to
    test for.
    """
    pos = sum(y for _, y in test)
    neg = len(test) - pos
    rivals = rivals or {}
    out = []
    for c in RATIOS:
        thr = best_threshold(train, c)
        tp, fp, tn, fn = confusion(test, thr)
        arm = fn * c + fp
        floors = {"always": neg, "never": pos * c}
        for name, (rtr, rte) in rivals.items():
            floors[name] = cost_at(rte, best_threshold(rtr, c), c)
        best_name = min(floors, key=floors.get)
        floor = floors[best_name]
        out.append({
            "ratio": c, "threshold": round(thr, 4),
            "arm_cost": arm, "floors": floors,
            "floor": floor, "floor_is": best_name,
            "arm_over_floor": round(arm / floor, 4) if floor else None,
            "beats_floor": arm < floor,
            "tp": tp, "fp": fp, "fn": fn,
        })
    return out


def capacity(test):
    """The queue. Sort by score, open the top k, and ask how many of the positives that
    catches. Ties are broken by shuffling once with a fixed seed BEFORE the sort, so a
    lookup arm emitting one rate for thousands of rows is not silently credited with the
    order they happened to arrive in."""
    rows = list(test)
    random.Random(20260908).shuffle(rows)
    rows.sort(key=lambda t: -t[0])
    n = len(rows)
    pos = sum(y for _, y in rows)
    out = []
    for f in KFRACS:
        k = max(1, int(round(f * n)))
        caught = sum(y for _, y in rows[:k])
        rec = caught / pos if pos else 0.0
        prec = caught / k
        out.append({
            "k_frac": f, "k": k, "caught": caught, "of_positives": pos,
            "recall_at_k": round(rec, 4), "precision_at_k": round(prec, 4),
            # a random queue catches k/n of the positives, so lift is recall / k_frac
            "lift_over_random": round(rec / (k / n), 4) if k else None,
        })
    return out


def proper(test):
    """Threshold-free. Average precision is the area under precision-recall, ROC AUC is the
    probability a random positive outranks a random negative, and Brier is squared error
    against the outcome. The base-rate constant predictor is reported beside Brier because
    it is the thing to beat: a model that has learned nothing scores exactly the base rate's
    variance, and one that is miscalibrated scores worse than knowing nothing."""
    rows = sorted(test, key=lambda t: -t[0])
    n = len(rows)
    pos = sum(y for _, y in rows)
    neg = n - pos

    ap = 0.0
    tp = 0
    for i, (_, y) in enumerate(rows, 1):
        if y:
            tp += 1
            ap += tp / i
    ap = ap / pos if pos else 0.0

    # ROC AUC by rank sum, with ties averaged.
    asc = sorted(test, key=lambda t: t[0])
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and asc[j + 1][0] == asc[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rsum = sum(r for r, (_, y) in zip(ranks, asc) if y)
    auc = (rsum - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else 0.0

    base = pos / n if n else 0.0
    brier = sum((s - y) ** 2 for s, y in test) / n if n else 0.0
    return {
        "average_precision": round(ap, 4),
        "ap_over_base_rate": round(ap / base, 4) if base else None,
        "roc_auc": round(auc, 4),
        "brier": round(brier, 5),
        "brier_of_base_rate_constant": round(base * (1 - base), 5),
        "base_rate": round(base, 4),
    }


def instance(train_w, test_w, c):
    """`fn * c + fp` with the miss weighted per row, and the threshold picked ON TRAIN.

    Weights are normalised to mean 1 within each set, so the total stays comparable with the
    flat run and any movement is the REDISTRIBUTION of cost across rows rather than a change
    of units. Tuning on train matters more here than anywhere else in this file: the weight
    is a second thing a test-set sweep could fit, so a best-case-over-test number would be
    an optimistic bound wearing a result's clothes.
    """
    def norm(pairs_w):
        rows = [(s, y, w) for (s, y), w in pairs_w]
        m = sum(w for _, _, w in rows) / len(rows)
        return [(s, y, w / m) for s, y, w in rows]

    tr, te = norm(train_w), norm(test_w)

    def cost(rows, thr):
        tot = 0.0
        for s, y, w in rows:
            p = 1 if s >= thr else 0
            if y and not p:
                tot += c * w
            elif p and not y:
                tot += 1.0
        return tot

    cand = sorted({s for s, _, _ in tr} | {0.0, 1.01})
    thr = min(cand, key=lambda t: cost(tr, t))
    arm = cost(te, thr)
    always = sum(1.0 for _, y, _ in te if not y)
    never = sum(c * w for _, y, w in te if y)
    floor = min(always, never)
    return {
        "ratio": c, "threshold": round(thr, 4),
        "arm_cost": round(arm, 1),
        "always_fire": round(always, 1),
        "never_fire": round(never, 1),
        "floor": round(floor, 1),
        "floor_is": "always" if always <= never else "never",
        "arm_over_floor": round(arm / floor, 4) if floor else None,
        "beats_floor": arm < floor,
    }


# ---------------------------------------------------------------- arms


def lookup_scores(tr, te):
    """C2 as a SCORE rather than a label: each bucket's training positive rate, smoothed
    toward the global rate so a bucket seen twice does not emit 0.0 or 1.0."""
    agg = defaultdict(lambda: [0, 0])
    for r in tr:
        cell = agg[bucket(r)]
        cell[0] += r["label"]
        cell[1] += 1
    base = sum(r["label"] for r in tr) / len(tr)
    prior = 10.0

    def rate(r):
        pos, n = agg.get(bucket(r), (0, 0))
        return (pos + prior * base) / (n + prior)

    return [(rate(r), r["label"]) for r in tr], [(rate(r), r["label"]) for r in te]


def year_scores(tr, te):
    """C1e as a score: the filing year's own training positive rate, and nothing else.
    Smoothed like the lookup so a thin year does not emit an extreme."""
    agg = defaultdict(lambda: [0, 0])
    for r in tr:
        cell = agg[r["filed"][:4]]
        cell[0] += r["label"]
        cell[1] += 1
    base = sum(r["label"] for r in tr) / len(tr)
    prior = 10.0

    def rate(r):
        pos, n = agg.get(r["filed"][:4], (0, 0))
        return (pos + prior * base) / (n + prior)

    return [(rate(r), r["label"]) for r in tr], [(rate(r), r["label"]) for r in te]


def assets_of(facts):
    """Firm size from the filing's own XBRL, largest of the plausible size tags. Rows with
    no size tag get the median rather than zero, because absence of a tag is a disclosure
    fact and not a firm worth nothing."""
    best = 0.0
    for tag, v in facts.items():
        if tag.endswith(":USD") and tag.split(":")[1] in (
            "Assets", "AssetsCurrent", "LiabilitiesAndStockholdersEquity", "Revenues"
        ):
            try:
                best = max(best, abs(float(v)))
            except (TypeError, ValueError):
                pass
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="lookup", choices=["lookup", "numeric", "words", "embed"])
    ap.add_argument("--pairs", default="pairs-sliced.jsonl")
    ap.add_argument("--views", default="views-2021.jsonl")
    ap.add_argument("--embed-dir", default="nomic-embed-text-mean")
    ap.add_argument("--regime", default="", help="YYYY-YYYY, inclusive on filing year")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm == "lookup":
        rows = [json.loads(l) for l in (HERE / args.pairs).read_text().splitlines()]
        if args.regime:
            lo, hi = (int(x) for x in args.regime.split("-"))
            rows = [r for r in rows if lo <= int(r["filed"][:4]) <= hi]
        ciks = sorted({r["cik"] for r in rows})
        random.Random(args.seed).shuffle(ciks)
        hold = set(ciks[: max(1, len(ciks) // 4)])
        tr = [r for r in rows if r["cik"] not in hold]
        te = [r for r in rows if r["cik"] in hold]
        train, test = lookup_scores(tr, te)
        rivals = {"year": year_scores(tr, te)}
        tr_rows, te_rows = tr, te
    else:
        train, test, tr_rows, te_rows, rivals = learned_scores(args)

    label = f"{args.arm}{'/' + args.regime if args.regime else ''}"
    print(f"\n=== {label}   train {len(train)}  test {len(test)} ===")

    sw = sweep(train, test, rivals)
    names = list(sw[0]["floors"])
    print("\n-- sweep: does a ratio exist where the arm beats every floor?")
    print(f"{'c':>4} {'thr':>6} {'arm':>9} " + " ".join(f"{n:>9}" for n in names)
          + f" {'floor':>9} {'is':>7} {'arm/floor':>10} {'beats':>6}")
    for d in sw:
        print(f"{d['ratio']:>4} {d['threshold']:>6.3f} {d['arm_cost']:>9} "
              + " ".join(f"{d['floors'][n]:>9}" for n in names)
              + f" {d['floor']:>9} {d['floor_is']:>7} "
              f"{d['arm_over_floor']:>10.3f} {'YES' if d['beats_floor'] else '.':>6}")
    win = [d["ratio"] for d in sw if d["beats_floor"]]
    print(f"\nbeats every floor at c in {win}" if win
          else "\nbeats every floor at NO ratio from 1 to 30")

    cap = capacity(test)
    print("\n-- capacity: a reviewer with k slots, and always-fire is not in the choice set")
    print(f"{'k_frac':>7} {'k':>7} {'caught':>7} {'recall@k':>9} {'prec@k':>8} {'lift':>7}")
    for d in cap:
        print(f"{d['k_frac']:>7.2f} {d['k']:>7} {d['caught']:>7} "
              f"{d['recall_at_k']:>9.3f} {d['precision_at_k']:>8.3f} "
              f"{d['lift_over_random']:>7.2f}")

    pr = proper(test)
    print("\n-- proper: threshold-free")
    for k, v in pr.items():
        print(f"   {k:<28} {v}")

    inst = None
    facts_by_cik = {}

    def facts(r):
        c = r["cik"]
        if c not in facts_by_cik:
            p = FACTS / f"CIK{c}.json"
            facts_by_cik[c] = json.loads(p.read_text()) if p.is_file() else {}
        return facts_by_cik[c].get(r["adsh"], {})

    sizes_te = [assets_of(facts(r)) for r in te_rows]
    have = [s for s in sizes_te if s > 0]
    if len(have) >= 0.5 * len(sizes_te) and tr_rows:
        med = sorted(have)[len(have) // 2]
        # log size, because a linear weight would make one megacap the whole test set
        wt = lambda s: math.log10(1 + (s if s > 0 else med))
        train_w = list(zip(train, [wt(assets_of(facts(r))) for r in tr_rows]))
        test_w = list(zip(test, [wt(s) for s in sizes_te]))
        inst = [instance(train_w, test_w, c) for c in RATIOS]
        print(f"\n-- instance: miss weighted by log10 reported assets "
              f"({len(have)}/{len(sizes_te)} test rows carry a size tag, rest get the median)")
        print(f"{'c':>4} {'thr':>6} {'arm':>9} {'always':>9} {'never':>9} "
              f"{'floor':>9} {'arm/floor':>10} {'beats':>6}")
        for d in inst:
            print(f"{d['ratio']:>4} {d['threshold']:>6.3f} {d['arm_cost']:>9.1f} "
                  f"{d['always_fire']:>9.1f} {d['never_fire']:>9.1f} {d['floor']:>9.1f} "
                  f"{d['arm_over_floor']:>10.3f} {'YES' if d['beats_floor'] else '.':>6}")
    else:
        print(f"\n-- instance: skipped, only {len(have)}/{len(sizes_te)} test rows carry a "
              f"size tag, so most weights would be the median and the run would compare "
              f"the flat cost with itself")

    out = args.out or f"costs-{args.arm}{('-' + args.regime) if args.regime else ''}.json"
    (HERE / out).write_text(json.dumps({
        "arm": args.arm, "regime": args.regime or "all",
        "train_rows": len(train), "test_rows": len(test),
        "sweep": sw, "capacity": cap, "proper": pr, "instance": inst,
    }, indent=1) + "\n")
    print(f"\n-> {out}")


def learned_scores(args):
    """C3, C4 and the embedding arm, trained the way arms.py trains them so the numbers here
    are comparable with the ones already recorded rather than a second implementation."""
    from arms import (
        BODIES, STOP, WORD, numeric_features, predict, text_features, train_logreg,
    )

    rows = [json.loads(l) for l in (HERE / args.views).read_text().splitlines()]
    usable = [r for r in rows if r["chars_narrative"] > 500 and r["n_facts"] > 0]
    ciks = sorted({r["cik"] for r in usable})
    random.Random(args.seed).shuffle(ciks)
    hold = set(ciks[: max(1, len(ciks) // 4)])
    tr = [r for r in usable if r["cik"] not in hold]
    te = [r for r in usable if r["cik"] in hold]

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

    if args.arm == "numeric":
        df = Counter()
        for r in tr:
            df.update(set(facts(r)))
        tags = {t: i for i, (t, _) in enumerate(df.most_common(600))}
        feat = lambda r: numeric_features(facts(r), tags, len(tags))
        dim = len(tags) + 1
    elif args.arm == "words":
        df = Counter()
        for r in tr:
            df.update(set(w for w in WORD.findall(narrative(r).lower()) if w not in STOP))
        vocab = {t: i for i, (t, _) in enumerate(df.most_common(3000))}
        feat = lambda r: text_features(narrative(r), vocab, len(vocab))
        dim = len(vocab) + 1
    else:
        vd = HERE / "raw" / "vecs" / args.embed_dir
        cache = {}

        def vec(r):
            if r["adsh"] not in cache:
                p = vd / f"{r['adsh']}.json"
                cache[r["adsh"]] = json.loads(p.read_text())["vec"] if p.is_file() else None
            return cache[r["adsh"]]

        tr = [r for r in tr if vec(r)]
        te = [r for r in te if vec(r)]
        dim = len(vec(tr[0]))
        feat = lambda r: {i: v for i, v in enumerate(vec(r))}

    w, b = train_logreg([(feat(r), r["label"]) for r in tr], dim, seed=args.seed)
    train = [(predict(w, b, feat(r)), r["label"]) for r in tr]
    test = [(predict(w, b, feat(r)), r["label"]) for r in te]
    # Inside one regime every row shares a calendar, so a year baseline is degenerate and
    # equals never-fire. Offered only when the views span more than one year.
    rivals = {}
    if len({r["filed"][:4] for r in tr}) > 1:
        rivals["year"] = year_scores(tr, te)
    return train, test, tr, te, rivals


if __name__ == "__main__":
    main()
