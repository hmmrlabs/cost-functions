#!/usr/bin/env python3
"""C3 and C4: each view alone, before anything reads the two together.

The design says these decide the experiment. A cross-modal claim means the PAIR carries
something neither side has, so if either single view matches the disagreement arm there is
no cross-modal finding and the write-up says so.

    C3   the numeric view alone: which XBRL tags a filing reports, and how many
    C4   the narrative view alone: the words in its MD&A and risk factors

Both must beat the bar `controls.py` set on the same rows: **C1 cost 460, C2 cost 459**
under missed x 10 on the XBRL window.

# No numpy, and that is deliberate

`hammer-bench` carries one dependency, `jsonschema`. It is a harvest repository and adding
a numerical stack to it so a baseline can run would be the wrong trade. Logistic regression
with L2 is thirty lines, the corpus is 1,988 rows, and a dependency-free arm replays
anywhere, which is what this repository asks of every number it keeps.

# What is fitted on what

Vocabulary, tag set, feature scaling and the decision threshold are all fitted on TRAIN
only. The threshold in particular: it is chosen to minimise the asymmetric cost on the
training rows and then applied unchanged to test. Choosing it on test is how a weak arm
reports a strong number.

The split is BY COMPANY, matching `controls.py`, because filings from one firm are not
independent and a row split scores memorisation as generalisation.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
BODIES = HERE / "raw" / "bodies"
VECS = HERE / "raw" / "vecs"
FACTS = HERE / "raw" / "facts"
MISS_COST = 10
WORD = re.compile(r"[a-z]{3,}")

# Words that separate a 10-K from a 10-Q, not a restatement from a clean filing. Left in
# would let C4 learn the form, which `form` already encodes and C2 already tested.
STOP = {
    "the", "and", "for", "that", "with", "was", "were", "are", "our", "its", "has", "have",
    "had", "not", "this", "which", "from", "such", "any", "all", "may", "will", "would",
    "been", "than", "other", "under", "these", "their", "there", "also", "into", "more",
    "who", "you", "including", "december", "march", "june", "september", "january",
    "february", "april", "july", "august", "october", "november", "year", "years", "quarter",
    "month", "months", "period", "periods", "ended", "ending", "company", "companies",
    # LEAKAGE, found by inspect_weights.py. These are the top positive weights in C4 and
    # they are the answer rather than a predictor: a filing that discusses a restatement,
    # a material weakness or a re-issuance is frequently an amendment OF a retracted
    # filing, or one already disclosing the problem. Predicting "will be retracted" from
    # the word "restated" is reading the label.
    "restatement", "restatements", "restated", "restate", "restating", "weakness",
    "weaknesses", "misstatement", "misstatements", "reaudit", "reissued", "revision",
    "revisions", "nonreliance", "unreliable",
}


def sigmoid(z: float) -> float:
    if z < -30:
        return 1e-13
    if z > 30:
        return 1 - 1e-13
    return 1 / (1 + math.exp(-z))


def train_logreg(rows, dim, epochs=60, lr=0.25, l2=1e-4, seed=20260818):
    """Plain L2 logistic regression by SGD. Sparse features as {index: value}."""
    w = [0.0] * dim
    b = 0.0
    order = list(range(len(rows)))
    rnd = random.Random(seed)
    for ep in range(epochs):
        rnd.shuffle(order)
        step = lr / (1 + ep * 0.15)
        for i in order:
            x, y = rows[i]
            z = b + sum(w[j] * v for j, v in x.items())
            e = sigmoid(z) - y
            b -= step * e
            for j, v in x.items():
                w[j] -= step * (e * v + l2 * w[j])
    return w, b


def predict(w, b, x) -> float:
    return sigmoid(b + sum(w[j] * v for j, v in x.items()))


def cost_of(rows, scores, thr) -> tuple[int, dict]:
    tp = fp = tn = fn = 0
    for (_, y), s in zip(rows, scores):
        p = 1 if s >= thr else 0
        tp += p == 1 and y == 1
        fp += p == 1 and y == 0
        tn += p == 0 and y == 0
        fn += p == 0 and y == 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return fn * MISS_COST + fp, {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
    }


def pick_threshold(rows, scores) -> float:
    """The threshold minimising asymmetric cost ON TRAIN. Never chosen on test."""
    best, best_thr = None, 0.5
    for t in [i / 100 for i in range(1, 100)]:
        c, _ = cost_of(rows, scores, t)
        if best is None or c < best:
            best, best_thr = c, t
    return best_thr


def numeric_features(adsh_facts, tags, log_n_idx):
    """C3. Which tags this filing reported, plus how many facts in total.

    Presence rather than value: XBRL values span revenue in billions and share counts in
    units, and a linear model over raw magnitudes would be fitting scale. Which tags a
    filer chose to report is the shape of its disclosure, which is the thing being asked
    about.
    """
    x = {}
    for t in adsh_facts:
        j = tags.get(t)
        if j is not None:
            x[j] = 1.0
    x[log_n_idx] = math.log1p(len(adsh_facts)) / 10.0
    return x


def text_features(text, vocab, log_len_idx):
    """C4. Log term frequency over a training vocabulary, length as its own feature."""
    counts = Counter(w for w in WORD.findall(text.lower()) if w not in STOP)
    total = sum(counts.values()) or 1
    x = {}
    for w, c in counts.items():
        j = vocab.get(w)
        if j is not None:
            x[j] = math.log1p(c) / math.log1p(total)
    x[log_len_idx] = math.log1p(total) / 15.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--tags", type=int, default=600, help="top XBRL tags kept, by train DF")
    ap.add_argument("--vocab", type=int, default=3000, help="top words kept, by train DF")
    ap.add_argument("--views", default="views.jsonl",
                    help="which views file, e.g. views-2021.jsonl for one regime")
    ap.add_argument("--embed", default="",
                    help="use cached embeddings for C4 instead of the word list, e.g. "
                         "nomic-embed-text-mean. inspect_weights showed the word arm was "
                         "reading vocabulary presence with no composition; this is whether "
                         "reading the sentence is worth more.")
    ap.add_argument("--with-year", action="store_true",
                    help="give the arms the filing year, so the question is what the "
                         "document adds ON TOP of the calendar")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.views).read_text().splitlines()]
    # Only rows whose two views actually landed. Reported, so the row count is not silently
    # different from the one the controls ran on.
    usable = [r for r in rows if r["chars_narrative"] > 500 and r["n_facts"] > 0]

    ciks = sorted({r["cik"] for r in usable})
    random.Random(args.seed).shuffle(ciks)
    holdout = set(ciks[: max(1, len(ciks) // 4)])
    tr = [r for r in usable if r["cik"] not in holdout]
    te = [r for r in usable if r["cik"] in holdout]

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

    # C1e on THESE rows and THIS split. The bar, because the era wave is large enough
    # that beating "never fires" says nothing about having read a document.
    yr = defaultdict(lambda: [0, 0])
    for r in tr:
        c = yr[r["filed"][:4]]
        c[0] += r["label"]
        c[1] += 1
    base = sum(r["label"] for r in tr) / len(tr) if tr else 0
    fires = {y: (v[0] / v[1]) > base for y, v in yr.items() if v[1] >= 20}
    c1e_rows = [({}, r["label"]) for r in te]
    c1e_scores = [1.0 if fires.get(r["filed"][:4], False) else 0.0 for r in te]
    c1e_cost, c1e_conf = cost_of(c1e_rows, c1e_scores, 0.5)
    c1_cost = sum(r["label"] for r in te) * MISS_COST
    # C0, always fire. Inside a near-balanced regime this is the floor an arm has to beat,
    # and it dominates everything else: see controls.py for why it was missing until the
    # regime split made the classes comparable.
    c0_cost, c0_conf = cost_of([({}, r["label"]) for r in te], [1.0] * len(te), 0.5)

    # Year as a feature, offered to the arms so the question becomes what the DOCUMENT
    # adds on top of the calendar rather than whether it can rediscover it.
    years = sorted({r["filed"][:4] for r in tr})
    year_idx = {y: i for i, y in enumerate(years)}

    out = {
        "seed": args.seed,
        "with_year": args.with_year,
        "C0_always_fire": {"asymmetric_cost": c0_cost, **c0_conf},
        "C1_cost": c1_cost,
        "C1e_year_only": {"asymmetric_cost": c1e_cost, **c1e_conf},
        "rows_in_views": len(rows),
        "rows_usable": len(usable),
        "companies": len(ciks),
        "train_rows": len(tr),
        "test_rows": len(te),
        "test_positives": sum(r["label"] for r in te),
        "bar": {"C1_cost": 460, "C2_cost": 459, "source": "controls-xbrl.json"},
    }

    # ---- C3, numeric view -------------------------------------------------------------
    df = Counter()
    for r in tr:
        df.update(set(facts(r)))
    tags = {t: i for i, (t, _) in enumerate(df.most_common(args.tags))}
    log_n = len(tags)
    def with_year(x, r, off):
        if args.with_year:
            j = year_idx.get(r["filed"][:4])
            if j is not None:
                x[off + j] = 1.0
        return x

    n3 = log_n + 1 + (len(year_idx) if args.with_year else 0)
    Xtr = [(with_year(numeric_features(facts(r), tags, log_n), r, log_n + 1), r["label"]) for r in tr]
    Xte = [(with_year(numeric_features(facts(r), tags, log_n), r, log_n + 1), r["label"]) for r in te]
    w, b = train_logreg(Xtr, n3, seed=args.seed)
    thr = pick_threshold(Xtr, [predict(w, b, x) for x, _ in Xtr])
    cost, conf = cost_of(Xte, [predict(w, b, x) for x, _ in Xte], thr)
    out["C3_numeric"] = {"features": len(tags) + 1, "threshold": thr,
                         "asymmetric_cost": cost, **conf}

    # ---- C4, narrative view -----------------------------------------------------------
    if args.embed:
        vdir = VECS / args.embed

        def vec(r):
            p = vdir / f"{r['adsh']}.json"
            if not p.is_file():
                return None
            v = json.loads(p.read_text())["vec"]
            # L2 normalise so the regression is not fitting document length through the
            # vector magnitude, which is the same trap the word arm avoided by dividing
            # term counts by the document total.
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return {i: x / n for i, x in enumerate(v)}

        etr = [(vec(r), r["label"]) for r in tr]
        ete = [(vec(r), r["label"]) for r in te]
        etr = [(x, y) for x, y in etr if x]
        ete = [(x, y) for x, y in ete if x]
        dim = (max(max(x) for x, _ in etr) + 1) if etr else 0
        w, b = train_logreg(etr, dim, seed=args.seed)
        thr = pick_threshold(etr, [predict(w, b, x) for x, _ in etr])
        cost, conf = cost_of(ete, [predict(w, b, x) for x, _ in ete], thr)
        out["C4_narrative"] = {"features": dim, "threshold": thr, "embed": args.embed,
                               "rows_scored": len(ete), "asymmetric_cost": cost, **conf}
        (HERE / "arms.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"usable {len(usable)} of {len(rows)} rows, {len(ciks)} companies")
        print(f"test {len(te)} rows / {out['test_positives']} positives")
        print(f"C0 (always fire) {c0_cost}  <- the floor")
        s4 = out["C4_narrative"]
        print(f"  C4_embed({args.embed})  tp={s4['tp']} fp={s4['fp']} fn={s4['fn']}  "
              f"P={s4['precision']:.3f} R={s4['recall']:.3f} F1={s4['f1']:.3f}  "
              f"cost={s4['asymmetric_cost']}")
        return
    df = Counter()
    for r in tr:
        df.update(set(w2 for w2 in WORD.findall(narrative(r).lower()) if w2 not in STOP))
    vocab = {t: i for i, (t, _) in enumerate(df.most_common(args.vocab))}
    log_len = len(vocab)
    n4 = log_len + 1 + (len(year_idx) if args.with_year else 0)
    Xtr = [(with_year(text_features(narrative(r), vocab, log_len), r, log_len + 1), r["label"]) for r in tr]
    Xte = [(with_year(text_features(narrative(r), vocab, log_len), r, log_len + 1), r["label"]) for r in te]
    w, b = train_logreg(Xtr, n4, seed=args.seed)
    thr = pick_threshold(Xtr, [predict(w, b, x) for x, _ in Xtr])
    cost, conf = cost_of(Xte, [predict(w, b, x) for x, _ in Xte], thr)
    out["C4_narrative"] = {"features": len(vocab) + 1, "threshold": thr,
                           "asymmetric_cost": cost, **conf}

    (HERE / "arms.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"usable {len(usable)} of {len(rows)} rows, {len(ciks)} companies")
    print(f"test {len(te)} rows / {out['test_positives']} positives")
    print(f"C0 (always fire) {c0_cost}  <- the floor    C1 (never fires) {c1_cost}   "
          f"C1e (year only) {c1e_cost}")
    for name in ("C3_numeric", "C4_narrative"):
        s = out[name]
        print(f"  {name:<13} tp={s['tp']:>3} fp={s['fp']:>4} fn={s['fn']:>3}  "
              f"P={s['precision']:.3f} R={s['recall']:.3f}  "
              f"thr={s['threshold']:.2f}  cost={s['asymmetric_cost']}")


if __name__ == "__main__":
    main()
