#!/usr/bin/env python3
"""Aggregate the threshold benchmark: mean, rank, worst case, and the calibration split.

Mean cost is the obvious summary and the wrong one to lead with. A threshold strategy is a
risk control, so what matters is what it does on its bad day: a policy that is best on average
and catastrophic once is worse than one that is second everywhere. Worst case and mean rank
are therefore reported beside the mean.

Regret against the oracle is reported too, because `over_floor` mixes together how good the
model is with how well the line was placed, and only the second is on trial here.
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RATIOS = [3, 5, 10, 20]
STRATS = ["S0_in_sample", "S1_oof", "S2_oof_repeated", "S3_oof_boot_median",
          "S4_validation", "S5_analytic", "S6_analytic_iso", "S7_rate_match"]


def main() -> None:
    cells = []
    for f in sorted(glob.glob(str(HERE / "thresholds-*.json"))):
        if "summary" in f:      # this script's own output, written next to its inputs
            continue
        d = json.loads(Path(f).read_text())
        for model, per_c in d["models"].items():
            for c in RATIOS:
                row = per_c[str(c)] if str(c) in per_c else per_c[c]
                cells.append({"tag": d["tag"], "seed": d["seed"], "model": model, "ratio": c,
                              "vals": {s: row[s]["over_floor"] for s in STRATS},
                              "oracle": row["S8_oracle"]["over_floor"],
                              "fires": {s: row[s]["fires_on_test"] for s in STRATS}})
    if not cells:
        print("no thresholds-*.json yet")
        return
    tags = sorted({c["tag"] for c in cells})
    seeds = sorted({c["seed"] for c in cells})
    print(f"{len(cells)} cells: {len(tags)} datasets x {len(seeds)} seeds x 3 models "
          f"x {len(RATIOS)} ratios")
    print(f"datasets: {', '.join(tags)}\n")

    def summarise(sel, title):
        if not sel:
            return
        print(title)
        print(f"  {'strategy':<20}{'mean':>8}{'median':>8}{'worst':>8}{'mean rank':>11}"
              f"{'regret':>9}{'beats oof':>11}")
        rank_acc = defaultdict(list)
        for c in sel:
            order = sorted(STRATS, key=lambda s: c["vals"][s])
            for i, s in enumerate(order):
                rank_acc[s].append(i + 1)
        rows = []
        for s in STRATS:
            v = [c["vals"][s] for c in sel]
            reg = [c["vals"][s] - c["oracle"] for c in sel]
            beats = sum(1 for c in sel if c["vals"][s] <= c["vals"]["S1_oof"])
            rows.append((sum(v) / len(v), s, sorted(v)[len(v) // 2], max(v),
                         sum(rank_acc[s]) / len(rank_acc[s]),
                         sum(reg) / len(reg), beats / len(sel)))
        for mean, s, med, worst, rank, reg, beats in sorted(rows):
            print(f"  {s:<20}{mean:>8.3f}{med:>8.3f}{worst:>8.3f}{rank:>11.2f}"
                  f"{reg:>9.3f}{beats:>10.0%}")
        print()

    summarise(cells, "ALL CELLS")
    for c_ in RATIOS:
        summarise([x for x in cells if x["ratio"] == c_], f"c = {c_}")
    for m in sorted({c["model"] for c in cells}):
        summarise([x for x in cells if x["model"] == m], f"model = {m}")

    print("THE CALIBRATION SPLIT: the analytic rule reads a score's VALUE, so it should work")
    print("where the model emits probabilities and fail where it emits vote fractions.")
    print(f"  {'model':<16}{'S5 analytic':>13}{'S6 analytic+iso':>18}{'S1 oof':>10}")
    for m in sorted({c["model"] for c in cells}):
        sel = [x for x in cells if x["model"] == m]
        f = lambda s: sum(x["vals"][s] for x in sel) / len(sel)
        print(f"  {m:<16}{f('S5_analytic'):>13.3f}{f('S6_analytic_iso'):>18.3f}"
              f"{f('S1_oof'):>10.3f}")

    (HERE / "thresholds-summary.json").write_text(json.dumps(cells, indent=1) + "\n")
    print("\n-> thresholds-summary.json")


if __name__ == "__main__":
    main()
