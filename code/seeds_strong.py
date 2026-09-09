#!/usr/bin/env python3
"""Run strong.py across seeds and report the spread rather than a point.

The sec2021 test folds are about 1,100 rows, so a single seed's `val / floor` moves by more
than the differences being discussed. Anything quoted from one seed at this size is a number
that will not reproduce, so this driver runs several and prints the range.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY_ = "/tmp/mlenv/bin/python"
SEEDS = [20260818, 20260819, 20260820]
CELLS = [
    ("bank", ["--dataset", "bank"], "strong-bank.json"),
    ("bank-n300", ["--dataset", "bank", "--train-cap", "300"], "strong-bank-n300.json"),
    ("sec-numeric", ["--dataset", "sec2021", "--arm", "numeric"],
     "strong-sec2021-numeric.json"),
    ("sec-words", ["--dataset", "sec2021", "--arm", "words"], "strong-sec2021-words.json"),
    ("sec-embed", ["--dataset", "sec2021", "--arm", "embed"], "strong-sec2021-embed.json"),
]


def rng(xs):
    return f"{min(xs):.2f}-{max(xs):.2f}"


def main():
    acc = {}
    for name, args, jf in CELLS:
        for seed in SEEDS:
            r = subprocess.run([PY_, "strong.py", *args, "--seed", str(seed)],
                               cwd=HERE, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{name} seed {seed} FAILED\n{r.stderr[-500:]}")
                continue
            d = json.loads((HERE / jf).read_text())
            for m, res in d["models"].items():
                g = {x["ratio"]: x for x in res["grid"]}
                acc.setdefault((name, m), []).append({
                    "seed": seed,
                    "p_over_n": d["p_over_n"],
                    "auc_gap": res["auc"]["train"] - res["auc"]["test"],
                    "auc_train": res["auc"]["train"],
                    "infl20": g[20]["train_over_val"],
                    "infl10": g[10]["train_over_val"],
                    "drift20": g[20]["fire_drift"],
                    "vf10": g[10]["val_over_floor"],
                    "vf20": g[20]["val_over_floor"],
                })
            print(f"  ran {name} {seed}", flush=True)

    print(f"\n{'cell':<13}{'model':<15}{'p/n':>6}{'AUCtr':>7}{'gap':>15}"
          f"{'infl@10':>13}{'infl@20':>13}{'val/floor@10':>15}")
    rows = []
    for (name, m), rs in acc.items():
        rows.append((sum(r["infl20"] for r in rs) / len(rs), name, m, rs))
    for _, name, m, rs in sorted(rows):
        print(f"{name:<13}{m:<15}{rs[0]['p_over_n']:>6.3f}"
              f"{max(r['auc_train'] for r in rs):>7.3f}"
              f"{rng([r['auc_gap'] for r in rs]):>15}"
              f"{rng([r['infl10'] for r in rs]):>13}"
              f"{rng([r['infl20'] for r in rs]):>13}"
              f"{rng([r['vf10'] for r in rs]):>15}")
    (HERE / "strong-seeds.json").write_text(json.dumps(
        {f"{k[0]}/{k[1]}": v for k, v in acc.items()}, indent=1) + "\n")
    print("\n-> strong-seeds.json")


if __name__ == "__main__":
    main()
