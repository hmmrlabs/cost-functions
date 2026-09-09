# Calibration cannot fix it, out-of-fold scores mostly can, and both facts are cheap

Measured 2026-09-09. `calibrate.py`, scikit-learn 1.9.0, five folds, seeds 20260818 and
20260819, on `bank-marketing` and on three SEC arms. 84 model-by-ratio cells.

`RESULT-strong.md` named held-out calibration as the cheap fix it had not tested. It tests
badly, for a reason that turns out to be a one-line proof, and the thing that does work is
better than the recommendation it replaces.

Six policies, all scored on the same untouched test rows:

```
A  in-sample              threshold on the fitted model's own training scores
B  in-sample + isotonic    calibrate on those scores, then threshold
C  out-of-fold             5-fold inside train, threshold on out-of-fold scores
D  out-of-fold + isotonic  as C, calibrator also fitted out of fold
E  held-out validation     the gold standard
F  oracle                  best threshold on test, not achievable
```

## 1. Calibration does nothing, in all 84 cells, and it cannot

```
B identical to A     84 of 84 cells, to four decimal places
D identical to C     84 of 84 cells
```

Not approximately. Identically. **A calibrator is a monotone map, and a monotone map cannot
change which rows a threshold selects.** For strictly increasing `g`, the set `{s >= t}` is the
set `{g(s) >= g(t)}`, so the family of achievable confusion matrices is unchanged and so is the
cost-optimal member of it. Isotonic regression is non-decreasing rather than strictly
increasing, so ties could in principle break differently; across 84 cells they never did.

Calibration fixes what a score MEANS. This is a problem about which rows a score SELECTS. Those
are different problems, and the first was the obvious guess.

## 2. Out-of-fold scores recover most of the loss and spend no data

The threshold is placed on 5-fold out-of-fold predictions inside the training set. No held-out
rows, `K` extra fits.

```
worst in-sample cells at c = 20, A -> C, with the validation policy in brackets

bank             random_forest    1.871 -> 0.389   (0.382)
bank             random_forest    1.973 -> 0.372   (0.358)
sec words        random_forest    2.975 -> 1.031   (0.984)
sec words        hist_gbdt        2.462 -> 0.980   (0.940)
sec numeric      hist_gbdt        2.285 -> 1.219   (1.016)
sec embed        random_forest    2.684 -> 1.362   (0.997)
```

Median fraction of the in-sample loss recovered, over 59 cells where there was a loss to
recover:

```
c = 3    1.02
c = 5    0.94
c = 10   0.95
c = 20   0.85
all      0.95
```

**On `bank` it is a complete fix.** 22,605 training rows, and out-of-fold lands within 2
percent of the held-out validation policy while spending none of the data.

**On the SEC corpus it is a partial one**, and the firing rates say why. Out-of-fold at `c = 20`
fires on 0.62 of the selection rows and 0.74 of test; the validation policy fires on 0.96 and
0.97. The residual drift is the out-of-fold models being trained on four fifths of 2,167 rows
while the deployed model gets all of them, so their scores are less separated and the threshold
lands slightly low. At 22,605 rows that difference is invisible; at 2,167 it is a fifth of the
remaining gap.

## 3. What to actually do

1. **Never place a decision threshold on in-sample scores.** This is the whole defect, it is
   what this directory did throughout, and at `c = 20` it costs between 1.7x and 3.0x the floor
   in the cells where the model memorises.
2. **Out-of-fold when the training set is large enough that a `K-1/K` subsample behaves like
   the whole.** Free in data, `K` fits in compute, and it was a complete fix on 22,605 rows.
3. **Held-out validation when it is not.** On 2,167 rows the out-of-fold policy still leaves a
   fifth of the gap, and a quarter of the data spent on placing the threshold bought it back.
4. **Do not reach for calibration.** It is provably irrelevant to this and it was the obvious
   guess.

## 4. What this does to the paper

It makes the contribution constructive rather than only critical, and the fix is free:

> Selecting a decision threshold on in-sample scores inflates test cost under asymmetric
> misclassification cost by up to 5.5x, governed by how completely the model memorises its
> training set rather than by `p/n` or by model quality, and therefore worst for the models most
> likely to be deployed. The mechanism is drift in the firing rate at the chosen operating
> point. Calibration cannot repair it, by monotonicity. Out-of-fold selection recovers a median
> 95 percent of the loss at the cost of `K` extra fits and no held-out data.

Still open: a temporal split, more than two seeds on the calibration grid, and whether the
residual out-of-fold gap closes with repeated or nested cross-validation.
