# The crossover measured a threshold policy, not the models

> **Corrected 2026-09-09 by `RESULT-strong.md`.** This file attributes the artefact to `p/n`.
> That was a proxy: shrinking the training set was the only knob available to one hand-written
> learner. With scikit-learn models on the same data, `p/n` no longer separates anything, and a
> random forest at `p/n = 0.002` on `bank-marketing` shows the LARGEST inflation in the grid,
> 4.9x, in the exact cell section 2 below reports as artefact-free. The variable that separates
> is train-set memorisation: every cell reaching a training AUC of 1.000 shows the effect and no
> cell at 0.91 does. Sections 1, 3b and 4 stand; section 2's control result holds only for the
> learner it was run with, and section 3's framing is superseded.

Measured 2026-09-08. `split3.py` and `replicate.py`. Three seeds on this corpus, two public
datasets as controls, and one knob that turns the effect on and off.

`RESULT-costs.md` reported a crossover at `c = 5.5` and read it as a property of the models.
It is not. It is a property of **where the decision threshold was chosen**, and every run in
this directory chose it on the same rows the model was fitted on.

## 1. Choose the threshold on held-out data and the loss mostly disappears

Split by company, `TRAIN 50 / VAL 25 / TEST 25`. The arm is fitted on train, the threshold is
picked on val, and both are reported on the same test rows. Cost over the best floor, three
seeds:

```
                       c = 5              c = 10             c = 20
arm        policy   seeds mean      seeds mean         seeds mean
words      train    0.80 0.69 0.91  1.36 1.23 1.72     2.42 2.24 3.32   2.66
words      val      0.76 0.69 0.91  0.96 0.95 1.01     0.97 1.00 0.95   0.97
numeric    train    0.76 0.78 1.06  1.18 1.37 1.99     2.12 2.37 3.85   2.78
numeric    val      0.77 0.75 0.92  1.07 1.00 1.02     1.00 1.00 1.16   1.05
embed      train    0.75 0.78 0.92  1.09 1.18 1.23     0.96 1.18 1.26   1.13
embed      val      0.72 0.70 0.95  0.90 1.00 1.34     0.97 1.01 1.08   1.02
```

At `c = 20` the train-picked threshold costs about **2.7 times the floor**. The val-picked
threshold costs about **1.0**. Same model, same rows, same split: only the data the operating
point was chosen on changed.

**The corrected reading, and it is narrower than the one a single seed suggested.** With a
held-out threshold these arms do not beat flagging everything at high ratios. They **tie** it.
The value of the validation split at `c = 20` is not that it makes the model win; it is that
it stops the model from losing.

That is what a correctly tuned model should do here. At a high enough ratio the floor is the
right answer, and a threshold picked on held-out data discovers that and collapses onto it. A
threshold picked on training scores does not: the model separates the classes far more on the
rows it was fitted on than on new ones, so the chosen point sits in the wrong place, and the
further into the tail the cost ratio pushes it, the more that displacement costs.

## 2. It does not reproduce on public data with a well-specified model

Two datasets from OpenML, no authentication, split 50/25/25 by row, same hand-written logistic
regression. `train / val` is the inflation from choosing the threshold on training scores:

```
                          rows     features   base rate    train/val at c=10   at c=20
bank-marketing          45,211           51      0.117               1.012      0.980
creditcard             284,807           30      0.002               1.086      1.046
```

**No artefact.** Both sit within a few percent of 1.0 across the whole grid. On `creditcard`
the model also beats the floor by a factor of four at every ratio, because at a base rate of
0.002 the contested operating point is nowhere near a corner.

So the effect is not a general fact about asymmetric cost, and reporting it as one would have
been wrong. Two controls, both negative.

## 3. One knob turns it on, and it lands on our own number

Hold `bank-marketing` fixed and vary only how badly the model overfits. Training rows shrink;
val and test do not move.

```
train rows   features   p/n      train/val at c=5   at c=10   at c=20
    22,605         51   0.002               1.002     1.012     0.980
     3,000         51   0.017               1.033     1.010     1.025
     1,000         51   0.051               1.049     0.998     1.172
       300         51   0.170               1.044     1.079     1.548
     2,000      3,051   1.526               1.013     1.244     2.440
```

Monotone in `p/n`, and growing with `c` within every row.

**The last line is the match.** Our narrative arm trains on 2,167 rows with 3,001 features,
`p/n = 1.4`, and inflates by 2.66 at `c = 20`. A public dataset with no relationship to SEC
filings, forced to the same `p/n`, inflates by 2.44. The effect is about the ratio of
parameters to rows and about the cost ratio, and about nothing else.

At `p/n = 1.5` the val-picked cost equals the floor exactly from `c = 10` upward: **the
validation split correctly recognises that the model is useless and says flag everything,
while the training split says something confident and wrong at 2.4 times the price.**

## 3b. It bites at whichever corner the floor occupies, which the theory predicted

Force `creditcard` to the same `p/n = 1.5` and the artefact appears there too, but running the
other way along the grid:

```
                       c=1     c=5    c=10    c=20    c=50
train / val          36.10    7.80    4.26    2.49    1.43
val / floor           1.00    1.00    1.00    1.00    1.00
```

**Inflation falls with `c` here, where it rose everywhere else.** That is the prediction, not
a surprise. At a base rate of 0.0018 never-fire is the floor at every ratio on this grid, so
the contested operating point sits near the **bottom-left** corner of ROC space rather than the
top-right, and what pushes it into that tail is a LOW cost ratio. On this corpus and on
`bank-marketing` the base rate is high enough that the floor flips to always-fire, the corner
is top-right, and a high ratio is what pushes it there.

One statement covers all three: **the threshold fails when the cost ratio pushes the optimal
operating point deep into a tail, where it is set by a handful of rows, and which tail depends
on the base rate.** The exact functional form is not established here and should not be
claimed; the direction is, on three datasets, and it reverses exactly where the theory says it
should.

The val column is also worth reading: `1.000` at every ratio. The validation split says
"never fire" every time, which is correct, while the training split spends up to 36 times the
floor being confident.

## 4. What this corrects, and what survives

**Corrected.** The crossover at 5.5 in `RESULT-costs.md` is the crossover *under a
train-selected threshold*. Under a val-selected one there is no clean crossover: the arms win
below about 5 and converge to the floor above it. The reported 1.41 and 1.57 losses at
`c = 10` were mostly the policy, not the models. Every cost figure in `RESULT-null.md`,
`RESULT-regimes.md` and `RESULT-costs.md` carries this defect, because `controls.py` and
`arms.py` both select on train.

**Survives, and one of them is strengthened.**

- **The capacity argument, now with a mechanism.** `precision@k` was recommended because a
  review budget is the decision a reviewer actually faces. It is also **the operating point
  with no threshold to estimate.** A budget is fixed in advance and cannot be displaced by
  overfitting, which is exactly the failure this file measures. That was an argument from
  realism yesterday and it is an argument from variance today.
- **The hours inversion.** A model is worth building exactly when review is expensive, which
  is exactly when you cannot review everything. Independent of thresholds.
- **The instance-dependent null.** Weighting the miss by firm size moves nothing.
- **The queue results.** `precision@k` and lift never involved a threshold.

**Needs re-checking under the val policy and has not been.** The SIC 6770 result, where the
cost-optimal threshold was `0.000`, was measured on 262 test rows with a train-selected
threshold. The direction is probably right and the number is not trustworthy.

## 5. The claim, stated so somebody can attack it

> Selecting a decision threshold on the rows a model was fitted on inflates its test cost
> under asymmetric misclassification cost. The inflation is jointly a function of `p/n` and of
> the cost ratio: absent at `p/n = 0.002`, 1.5x at `p/n = 0.17`, and 2.4x at `p/n = 1.5`, all
> at `c = 20`, on one public dataset with the model's specification as the only thing varied.
> The direction reverses with the base rate: where never-fire is the floor, the inflation is
> largest at LOW ratios instead, 36x at `c = 1` on `creditcard` at the same `p/n`. Reported
> failures of cost-sensitive models in the small-sample high-dimensional regime may therefore
> be measuring the threshold policy rather than the model.

Three seeds, one corpus, two public controls and one manipulation. What it needs next is more
seeds, a second manipulated dataset, and a strong model rather than a logistic regression.
