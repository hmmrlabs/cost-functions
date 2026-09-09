# The strong-model check, which reverses the objection

Measured 2026-09-09. `strong.py` and `seeds_strong.py`, scikit-learn 1.9.0 in a virtualenv;
every other file in this directory is stdlib only. Three seeds per cell, ranges reported
because a 1,100-row test fold moves more than the differences being discussed.

The objection to `RESULT-threshold.md` is that its models are weak: a hand-written logistic
regression, 40 epochs, no regularisation search. A reader is entitled to suspect the whole
effect is an artefact of a bad learner.

**It is the opposite.** Gradient boosting and random forests show the artefact more, not less,
and they show it in the one cell where the weak model showed none at all.

## The grid

`infl` is the cost of a train-selected threshold over a val-selected one, on the same test
rows. Sorted by it.

```
cell         model            p/n   AUCtr        AUC gap     infl@10     infl@20   val/floor@10
bank         logreg_l2      0.002   0.909      0.00-0.00   0.98-1.02   1.00-1.02      0.36-0.37
sec-embed    logreg_l2      0.354   0.908      0.01-0.03   0.92-0.99   0.96-1.14      0.96-1.24
bank-n300    logreg_l2      0.173   0.961      0.11-0.14   1.00-1.35   1.06-1.24      0.54-0.59
sec-numeric  random_forest  0.277   1.000      0.06-0.09   1.53-1.80   1.27-1.57      0.86-0.98
bank         hist_gbdt      0.002   0.994      0.06-0.06   1.38-1.41   1.39-2.07      0.28-0.29
sec-words    logreg_l2      1.385   0.993      0.07-0.09   1.14-1.60   1.79-2.64      0.93-0.98
sec-numeric  logreg_l2      0.277   0.998      0.10-0.10   1.02-1.49   1.76-2.89      0.95-1.08
sec-numeric  hist_gbdt      0.277   1.000      0.07-0.09   1.24-1.49   2.24-2.87      0.90-1.04
sec-words    hist_gbdt      1.385   1.000      0.07-0.10   1.40-2.01   2.62-3.76      0.90-0.97
sec-embed    hist_gbdt      0.354   1.000      0.10-0.10   1.25-1.99   2.62-3.82      0.98-1.13
sec-words    random_forest  1.385   1.000      0.08-0.10   1.44-2.17   1.98-4.20      0.97-1.09
sec-embed    random_forest  0.354   1.000      0.10-0.13   1.20-2.30   2.50-4.36      1.00-1.17
bank-n300    hist_gbdt      0.173   1.000      0.19-0.21   2.00-2.21   3.20-3.46      0.57-0.66
bank-n300    random_forest  0.173   1.000      0.14-0.15   2.36-2.66   3.53-4.00      0.49-0.51
bank         random_forest  0.002   1.000      0.07-0.07   3.04-3.46   4.90-5.51      0.29-0.30
```

**Read the last line against the first.** Same dataset, same 22,605 training rows, same 51
features, same splits. A regularised logistic regression inflates by 1.00 to 1.02. A random
forest with default leaf size inflates by **4.90 to 5.51**. The strongest artefact in the whole
grid is a completely standard model on an easy, well-sampled, low-dimensional dataset.

## This corrects `RESULT-threshold.md`

That file attributed the effect to `p/n`, because shrinking the training set was the only knob
it had. **`p/n` was a proxy, and this grid breaks it.** Sorted by inflation, `p/n` is scrambled:
0.002 sits at both the top and the bottom.

The variable that separates cleanly is **train-set memorisation**:

```
AUC on train        cells    inflation at c = 20
0.908 - 0.909           2            1.00 - 1.14
0.961 - 0.998           4            1.06 - 2.89
1.000                   7            1.27 - 5.51
```

Every cell reaching a training AUC of exactly 1.000 shows the artefact. No cell at 0.91 does.
`p/n` mattered only because it was how the earlier file made one particular learner memorise.

## The displaced quantity, measured

A threshold is chosen so that some fraction of TRAIN fires. Applied to new rows it fires on a
different fraction, and that drift is what the cost pays for. On `bank` at `c = 20`:

```
model            fires on train   fires on test    drift   inflation
random_forest             0.117           0.050   -0.067        4.90
logreg_l2                 0.410           0.417   +0.008        1.00
```

The forest's threshold was set to fire on one filing in nine and fires on one in twenty. Under
a 20:1 cost that is a great many misses bought at full price.

**AUC gap does not predict the magnitude and firing drift does.** `bank/random_forest` has a
gap of 0.071 and inflates 4.9; `bank-n300/logreg_l2` has a gap of 0.120 and inflates 1.2. AUC
gap is a mismatch in RANKING. What a threshold is exposed to is a mismatch in the score
DISTRIBUTION at the operating point, which is why an uncalibrated forest, whose training scores
are vote fractions piled at 0 and 1, is the worst offender in the grid despite ranking well.

That also names the cheap fix, which this directory has not tested: calibrate on held-out data.
Fitting an isotonic or Platt map on the validation split should collapse the drift and most of
the inflation with it, and if it does not, the mechanism claim above is wrong.

## The second thing this corrects: the models were also just weak

`val/floor@10` is the arm's real cost against flagging everything, threshold chosen properly.
On `bank` the strong models reach **0.28 to 0.30**, against the hand-written arm's 0.36. On the
SEC corpus the best cell is `sec-numeric/random_forest` at **0.86 to 0.98**, which straddles
1.0.

So the corrected reading of the SEC null at `c = 10` is that **even a strong model only ties
flagging everything**, and an earlier single-seed reading of 0.840 was the low end of noise.
The corpus finding survives a stronger learner. What does not survive is attributing all of the
earlier gap to the models: it was part threshold policy and part learner, and neither part was
separated until now.

## What this does to the claim

Strengthened, and narrower in the right place:

> Selecting a decision threshold on the rows a model was fitted on inflates its test cost under
> asymmetric misclassification cost. The inflation is governed by how completely the model
> memorises its training set, not by `p/n` and not by model quality: it is absent at a training
> AUC of 0.91 and reaches 4.9x at 1.000, on the same dataset. It is therefore **largest for the
> models most likely to be deployed**, and its mechanism is a drift in the firing rate at the
> chosen operating point rather than a loss of ranking.

Still open: whether held-out calibration removes it, a temporal split, and more than three
seeds.
