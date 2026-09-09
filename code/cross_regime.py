#!/usr/bin/env python3
"""Does a world get easier or harder to predict as its error rate rises?

The regimes in this corpus differ enormously in how often filings were retracted, from
0.037 in 2010 to 0.450 in 2021. That is a natural experiment nobody had to construct, and
it asks a question the single-regime runs cannot:

**When mistakes in an industry go up, does prediction get easier or harder?**

Two answers are defensible before looking:

  EASIER   more positives, more balanced classes, more signal per split. The usual
           statistical argument.
  HARDER   a wave has a systemic cause. In 2021 it was warrant reclassification, which hit
           SPACs indiscriminately, so what separates a retracted filing from a clean one
           narrows to almost nothing. In a quiet period a restatement is a firm-specific
           failure and firm characteristics should carry it.

These predict opposite things about the same number, which is what makes it worth running.

# What is measured

Per regime: the two floors and the lookup arm.

    C0   always fire.  cost = negatives                (wins when the base rate is high)
    C1   never fire.   cost = positives x 10           (wins when it is low)
    C2   lookup on SIC, form, amendment

The comparable quantity is **C2 against the better of the two floors**, because which floor
wins is itself a function of the base rate and comparing against a fixed one would confound
the two effects. A ratio below 1.0 means the lookup found something the floor did not.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Pre-2011 has no XBRL, so the cross-regime comparison runs on the FULL corpus. Restricting
# to the XBRL window would drop the two earliest regimes and leave the comparison unable to
# see the range it exists to measure.
REGIMES = [
    ("2004-2007", "SOX era"),
    ("2008-2010", "financial crisis"),
    ("2011-2015", "quiet, early"),
    ("2016-2019", "quiet, late"),
    ("2020-2020", "covid onset"),
    ("2021-2021", "SPAC wave"),
    ("2022-2024", "post-wave"),
]


def run(regime: str) -> dict | None:
    out = subprocess.run(
        [sys.executable, "controls.py", "--regime", regime],
        cwd=HERE, capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"  {regime}: failed\n{out.stderr[-400:]}")
        return None
    p = HERE / f"controls-{regime}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> None:
    print(f"{'regime':<12} {'label':<17} {'rows':>7} {'base':>6} "
          f"{'C0':>7} {'C1':>7} {'floor':>7} {'C2':>7} {'C2/floor':>9}")
    table = []
    for regime, label in REGIMES:
        d = run(regime)
        if not d:
            continue
        c0 = d["C0_always_fire"]["asymmetric_cost"]
        c1 = d["C1_base_rate"]["asymmetric_cost"]
        c2 = d["C2_lookup"]["asymmetric_cost"]
        floor = min(c0, c1)
        pos = d["test_positives"]
        n = d["test_rows"]
        base = pos / n if n else 0
        ratio = c2 / floor if floor else None
        which = "C0" if c0 < c1 else "C1"
        table.append({
            "regime": regime, "label": label, "rows": d["rows"], "base_rate": round(base, 4),
            "C0": c0, "C1": c1, "floor": floor, "floor_is": which, "C2": c2,
            "C2_over_floor": round(ratio, 4) if ratio else None,
        })
        print(f"{regime:<12} {label:<17} {d['rows']:>7} {base:>6.3f} "
              f"{c0:>7} {c1:>7} {floor:>7} {c2:>7} {ratio:>9.3f}")

    (HERE / "cross-regime.json").write_text(json.dumps(table, indent=1) + "\n")

    print()
    ok = [t for t in table if t["C2_over_floor"] is not None]
    lo = [t for t in ok if t["base_rate"] < 0.15]
    hi = [t for t in ok if t["base_rate"] >= 0.15]
    if lo and hi:
        ml = sum(t["C2_over_floor"] for t in lo) / len(lo)
        mh = sum(t["C2_over_floor"] for t in hi) / len(hi)
        print(f"low-error regimes  (base < 0.15, n={len(lo)}):  C2/floor mean {ml:.3f}")
        print(f"high-error regimes (base >= 0.15, n={len(hi)}): C2/floor mean {mh:.3f}")
        print()
        if mh > ml:
            print("HARDER when mistakes are common: the lookup's edge over the floor shrinks")
            print("as the error rate rises, which is what a systemic cause looks like. When")
            print("one accounting rule hits every filer, firm characteristics stop separating")
            print("anybody from anybody.")
        else:
            print("EASIER when mistakes are common: the lookup's edge grows with the error")
            print("rate, so the extra positives buy more than the systemic cause costs.")


if __name__ == "__main__":
    main()
