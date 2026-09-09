# The cost function was the finding

Measured 2026-09-08. `costs.py`, on `pairs-sliced.jsonl` and `views-2021.jsonl`, company split,
seed 20260818. Four cost functions over the same scores.

Every null in `RESULT-null.md` and `RESULT-regimes.md` is a null against `fn * 10 + fp`. That
constant came from `MODELCARD.md`, which took it from E22, which chose it for a different
corpus. It was never measured against a filing reviewer's budget, and it decides every result
in this directory.

## 1. The sweep: the crossover is 5.5, and 10 is on the far side of it

Threshold re-picked on TRAIN at each ratio, because a threshold tuned for 10 is not the arm's
best play at 3. Floors are always-fire, never-fire, and the year-only baseline, **each of them
cost-tuned the same way**: tuning the arm while leaving the bar at its untuned operating point
would manufacture the win this run exists to test for.

2021 regime, arm cost over the best floor:

```
   c    numeric    words    embed
   1      0.414    0.307    0.345
   2      0.527    0.423    0.464
   3      0.652    0.576    0.661
   4      0.796    0.729    0.802
   5      0.903    0.882    0.916
   6      1.036    1.035    0.986
   8      1.304    1.160    1.141
  10      1.572    1.407    1.273
```

On a quarter-point grid the crossover is **5.5 for numeric, 5.75 for words, 5.5 for embed**.
Three different feature spaces agree to within a quarter of a point.

So the answer is: the models were never bad. **At 3:1 the narrative arm costs 58 percent of the
best floor.** At 10:1 it costs 141 percent of it. Same model, same rows, same split. The entire
published null is one constant.

## 2. What the constant should be, as far as anyone has measured it

The published anchors do not settle it, and they bracket the crossover rather than clearing it.

- The restatement announcement effect averages about **-9.2 percent** over a two-day window,
  roughly half of it expected litigation, with operation and integrity issues at 10 to 15
  percent ([GAO-03-138](https://www.gao.gov/products/gao-03-138),
  [Herly 2023](https://onlinelibrary.wiley.com/doi/10.1111/jbfa.12615)).
- Cost-sensitive fraud detection work picks its ratio the same way this repository did, by
  assertion. One recent study uses **6.46:1**
  ([review](https://www.researchgate.net/publication/393569888_Cost-Sensitive_Learning_in_Financial_Fraud_Detection_Models)).

**6.46 is a quarter-point above the crossover.** The literature's own number and this corpus's
tipping point are the same number to within the precision either is stated at. That is the
result, and it is worse than either "10 was wrong" or "10 was fine": the answer to whether a
model is worth building here is decided entirely by a parameter nobody has measured, sitting on
a knife edge, and every finding recorded before today was a report about that parameter.

## 3. The capacity view, which is the one the problem actually has

`fn * c + fp` charges nothing for the capacity to review. Always-fire reviews 100 percent of
filings and pays only the per-review price, so it is purchasable at any scale, and once it is
purchasable it wins whenever the base rate clears `1/(c+1)`. No ranking model can beat a
strategy that takes every positive by taking every row. **That is the defect. Not the 10.**

Give the reviewer k slots instead. Always-fire is then not in the choice set: it cannot say
which k to open. `precision@k` is the right metric when k is known at model-selection time
([Consequentialist Critique, arXiv 2504.04528](https://arxiv.org/pdf/2504.04528)).

```
arm       base    prec@1%  prec@5%  prec@10%  recall@10%  lift@1%   max lift
lookup   0.104      0.662    0.379     0.310       0.297     6.34       9.57
numeric  0.468      0.818    0.889     0.898       0.192     1.75       2.14
words    0.468      1.000    0.982     0.982       0.210     2.14       2.14
embed    0.468      1.000    0.963     0.963       0.206     2.14       2.14
```

`max lift` is `1 / base rate`, the ceiling a perfect queue reaches.

**The narrative arm hits the ceiling.** Every one of the top 11 filings it ranks is a filing that
was later retracted, and so is every one of the top 22. A reviewer with capacity for one percent
of the quarter's filings spends all of it on genuine restatements and none on false alarms. Under
`fn * 10 + fp` that arm is 41 percent worse than useless.

The pooled lookup is the weakest classifier here and the strongest queue: **lift 6.34** at one
percent, precision 0.662 against a 0.104 base rate. Lift has room to be large only where the base
rate is small, which is the opposite of the regime where always-fire wins, so the two views of
this corpus disagree about which regime is the interesting one.

## 4. Proper scores, for the question that has no threshold

```
arm       AP     AP/base    AUC    Brier   Brier of base-rate constant
lookup   0.291     2.79    0.721   0.0855            0.0936
numeric  0.813     1.74    0.858   0.1724            0.2490
words    0.909     1.94    0.902   0.1175            0.2490
embed    0.886     1.89    0.880   0.1258            0.2490
```

Every arm beats the constant predictor on Brier, so all of them are calibrated rather than
merely ranked, and the earlier reporting of cost alone hid it. AUC 0.90 for the word arm is not
a model that failed.

## 5. Instance-dependent cost, which is a null

If a miss costs about 9 percent of firm value, it is not constant per row: a missed restatement
at a blank-check shell and one at an operating company are the same row under `fn * 10`. Weighted
by the filing's own reported assets, log scale, normalised to mean 1, threshold still picked on
train:

```
words, arm over floor      c=3     c=10
flat                      0.576    1.407
asset weighted            0.551    1.343
```

**Null.** The weighting moves nothing, so the arm's misses are not concentrated at either end of
the size distribution, and there is no size-aware version of this that rescues it at 10. Worth
running because it was cheap and it closes a hypothesis: 1,063 of 1,080 test rows carry a size
tag, so this is not a coverage failure dressed as a null.

## 6. Two corrections to the record

**The recorded C2 was never cost-tuned.** `controls.py` fires a bucket when its training rate
clears the base rate, deliberately, so the arm "cannot be tuned into looking good". Under a 10:1
cost that operating point is far too conservative, and the recorded numbers are a report about
the threshold rather than the arm. Cost-tuned, 2021: **C2 costs 563 against always-fire's 605**,
where the recorded figure was 2,092. `RESULT-regimes.md` says "nothing built here has beaten C0
reliably", and part of that was the threshold.

**Inside SIC 6770 the finding survives and gets sharper.** Cost-tuned at every ratio from 1 to
30, the lookup's optimal threshold is **0.000**: the best thing the model can do with the cost
function is become always-fire. Capacity lift 1.08 to 1.22 against a ceiling of 1.22, AUC 0.634.
The within-industry null was not a threshold artefact.

## 7. The same thing in analyst hours, which is the only currency spent here

`review_time.py`. Stop asserting the ratio and derive it:

```
c  =  hours a missed restatement eventually costs  /  hours one review costs
```

Both are questions a reviewer can answer about their own shop. Nobody has to agree on 10.

**Review time sits in the denominator, and that produces an inversion nothing above made
visible.** 2021 regime, 1,127 filings on the desk, 522 later retracted, total analyst hours:

```
 review   miss       c    flag all  flag none  year only    lookup    winner
     5m     8h    96.0          94      4,176         94        94    flag all
    15m    40h   160.0         282     20,880        282       282    flag all
    60m     8h     8.0       1,127      4,176      1,127     1,095    lookup
   120m    40h    20.0       2,254     20,880      2,254     2,170    lookup
   240m     8h     2.0       4,508      4,176      4,508     3,712    lookup
   480m     8h     1.0       9,016      4,176      4,176     5,160    flag none
```

At a five-minute triage the whole quarter costs 94 hours and **nobody needs a model**: read
everything, it is two analyst-weeks. At four hours a filing it costs 4,508 hours, which is two
analyst-years for one quarter's filings, and the ranking is what makes the work fit.

**A model is worth building here exactly when review is expensive.** That is also exactly when
you cannot afford to review everything, so the cost view and the capacity view stop being two
arguments the moment the units are real. Under `fn * 10 + fp` they looked like two.

The pooled grid also lands the crossover in the right place from the other direction: at eight
hours a review and forty hours a miss, `c = 5`, the lookup wins on total hours, 144,496 against
the year baseline's 148,080. Section 1 put the crossover at 5.5 from the unitless side.

### The budget, where flag-everything is not on the menu

Give the team H hours. It can open `k = H / review_hours` filings and no more. **Flag-all cannot
say which k to drop, so what it degrades to is a random queue of the same size.** That is the
comparison the model is actually in, and no run in this directory had made it.

Pooled corpus, 38,231 filings, 3,994 of them later retracted, thirty-minute triage:

```
  hours       k     k/n    model catches    random catches    extra
    382     764    2.0%              403              79.8    +323
  1,912   3,823   10.0%            1,187             399.4    +788
  3,823   7,646   20.0%            1,758             798.8    +959
```

**382 analyst hours, spent on the ranked queue instead of an arbitrary one, is 403 restatements
found instead of 80.** Same hours, same team, same filings. If a missed restatement really does
cost forty hours downstream, that one decision is worth about 12,900 hours.

Nothing in that table required picking the miss cost. The extra catch is the model's, whatever a
miss turns out to be worth; the hours figure is the only line that needs the assumption.

### What is deliberately not assumed

No default miss cost ships in this file. The published anchors bracket without settling: the
announcement effect is about -9.2 percent of market value, roughly half of it expected
litigation, and post-restatement audit fee premia run
[21 to 32 percent in years three and four](https://www.academia.edu/93160203/Audit_Fees_after_Remediation_of_Internal_Control_Weaknesses).
None of that converts into a reviewer's hours without an assumption, and PCAOB engagement hours
are only now becoming reportable under
[Form AP metrics](https://pcaobus.org/resources/staff-publications/audit-focus/audit-focus--form-ap),
so the aggregate does not yet exist to look up. The grid is printed instead and the reader finds
their own row.

## What this changes

The 10:1 contract should not be inherited again without a measurement behind it, and no result in
this directory that reads as "the model is not worth it" should be quoted without the ratio
attached. The default for further work here is `precision@k` at a stated capacity, with the cost
sweep reported beside it, because that is the form of the decision a reviewer actually faces and
it does not hand the win to a strategy nobody can staff.

And where a ratio is unavoidable, derive it from hours rather than assert it. `10` is not a fact
about restatements. `four hours a review and a week of rework per miss` is a claim somebody can
be wrong about out loud, which is the only kind worth putting in a model card.
