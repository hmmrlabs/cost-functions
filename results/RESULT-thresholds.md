# Eight ways to place the line, and only one thing about the choice matters

Measured 2026-09-09. `threshold_bench.py`, `agg_thresholds.py`. **180 cells**: 5 datasets x 3
seeds x 3 models x 4 cost ratios. scikit-learn 1.9.0.

`RESULT-calibrate.md` compared in-sample against out-of-fold and a held-out split. Out-of-fold
was one strategy and it was the only one tried. Seven alternatives now, chosen to differ in
mechanism rather than in tuning.

## The strategies

```
S0  in_sample        threshold on the fitted model's own training scores
S1  oof              5-fold out-of-fold inside train
S2  oof_repeated     3 repeats of 5-fold, out-of-fold scores averaged
S3  oof_boot_median  bootstrap the out-of-fold pairs 40x, take the median threshold
S4  validation       held-out quarter
S5  analytic         no search: fire when p > 1 / (1 + c), the Bayes rule for this cost
S6  analytic_iso     the same rule on scores calibrated out of fold
S7  rate_match       take the firing RATE the out-of-fold sweep chose, set the threshold at
                     that quantile of the DEPLOYMENT scores. Test features, no test labels.
```

## Overall

```
strategy                mean  median   worst  mean rank   regret   >floor
S4_validation          0.695   0.670   1.242       3.09    0.033      13%
S6_analytic_iso        0.765   0.689   1.968       4.61    0.103      23%
S1_oof                 0.766   0.702   1.966       3.49    0.105      23%
S3_oof_boot_median     0.768   0.682   1.966       4.08    0.106      23%
S2_oof_repeated        0.778   0.697   2.200       4.03    0.117      23%
S7_rate_match          0.809   0.696   2.290       4.83    0.147      25%
S5_analytic            0.851   0.696   3.551       5.14    0.189      23%
S0_in_sample           1.156   0.910   4.581       6.72    0.494      39%
```

`regret` is the mean gap to the oracle threshold, which is the only column that isolates the
line from the model. `>floor` is how often the strategy did worse than flagging everything,
which is the operational way to be wrong.

## 1. The size of the training set decides whether a validation split is worth buying

```
dataset       train n   in_sample     oof   validation   S1 - S4
bank            22,605      0.672   0.399        0.399     0.001
bank-n300          300      1.201   0.699        0.656     0.043
sec-embed        2,167      1.331   0.947        0.853     0.094
sec-numeric      2,167      1.178   0.874        0.772     0.102
sec-words        2,167      1.398   0.912        0.795     0.117
```

**On 22,605 training rows out-of-fold is free and exactly as good**, 0.399 against 0.399, worst
case 0.567 against 0.566. Spending a quarter of the data to place one scalar buys nothing.

**On 2,167 rows it is not**, and the gap is a tenth of the floor. The mechanism is the one
`RESULT-calibrate.md` named: the out-of-fold models are fitted on four fifths of an already
small set, so their scores are less separated than the deployed model's and the threshold lands
slightly low. That difference is invisible at 22,605 rows and material at 2,167.

The tail says it louder than the mean:

```
worst case         in_sample    oof   validation
bank                   1.973  0.567        0.566
sec-numeric            3.057  1.931        1.076
sec-words              4.224  1.966        1.123
```

**So the rule is a size rule, not a preference.** Out-of-fold when a `K-1/K` subsample behaves
like the whole training set; buy the validation split when it does not.

## 2. Two obvious refinements to out-of-fold buy nothing at all

S2 averages out-of-fold scores over three repeats, attacking variance in the scores. S3
bootstraps and takes the median threshold, attacking variance in the threshold itself. Both are
the natural next thing to try and neither moves:

```
                bank   bank-n300   sec-numeric   sec-words
S1 oof         0.399       0.699         0.874       0.912
S2 repeated    0.398       0.706         0.896       0.936
S3 boot median 0.399       0.701         0.879       0.918
```

Worst case is 1.966 for both S1 and S3, and 2.200 for S2, so the repeats are marginally worse in
the tail than doing nothing. **Threshold variance is not the lever.** Recorded so nobody spends
an afternoon on it: the residual gap between out-of-fold and a validation split is a BIAS, the
out-of-fold models being weaker, and averaging does not touch a bias.

## 3. Calibration plus the textbook rule reproduces the sweep, in all 180 cells

```
                bank   bank-n300   sec-embed   sec-numeric   sec-words
S1 oof         0.399       0.699       0.947         0.874       0.912
S6 analytic    0.399       0.693       0.945         0.873       0.912
```

Overall 0.765 against 0.766, worst case 1.968 against 1.966. Calibrate out of fold, then apply
`p > 1 / (1 + c)` with no search of any kind, and you land where the empirical sweep lands.

`RESULT-calibrate.md` showed calibration is provably irrelevant to an empirical sweep, because a
monotone map cannot change which rows a threshold selects. **Here is the other half: the
analytic rule reads the score's VALUE rather than its rank, so the same map is exactly what it
needs.** One tool, useless in one place and load-bearing in the other, and the difference is
whether the procedure reads ranks or values.

## 4. The analytic rule alone fails on saturation, not on miscalibration

```
model             S5 analytic   S6 analytic + iso   S1 oof
hist_gbdt               1.067               0.776    0.778
logreg_l2               0.794               0.781    0.784
random_forest           0.693               0.736    0.737
```

Gradient boosting is the only strategy-model pair in the grid whose mean is **worse than
flagging everything**. The firing rates say why:

```
                 c = 3   c = 20     what the sweep wanted at c = 20
hist_gbdt        0.289    0.340                               0.637
logreg_l2        0.357    0.667                               0.684
random_forest    0.372    0.767                               0.650
```

**Boosting's firing rate cannot respond to the cost ratio.** Its scores pile at 0 and 1, so
dragging the threshold from 0.25 down to 0.048 crosses almost no rows, and the rule is stuck at
0.34 when the optimum is 0.64. Logistic regression tracks the sweep to within 0.02 at every
ratio, which is the prediction that motivated including S5.

The forest looks best on S5 and should not be trusted for it: it OVER-fires, 0.767 against a
wanted 0.650, and over-firing is cheap when a miss costs twenty. That is the sign of its error
being forgiving, not the rule working.

## 5. Rate matching is a null

S7 was the strategy aimed squarely at the measured mechanism: transfer the firing rate rather
than the threshold, re-deriving the cut from the deployment pool's own scores using its features
and none of its labels. It is worse than plain out-of-fold on four datasets of five, beats it in
43 percent of cells, and has the second worst tail at 2.290.

Attacking the mechanism directly did not beat simply measuring on clean rows.

## 6. At the ratios this corpus actually faces, none of it rescues the model

```
c = 20            mean   worst   regret
S4_validation    0.831   1.123    0.035
S1_oof           1.046   1.966    0.251
S0_in_sample     2.062   4.581    1.266
```

Only the validation policy beats flagging everything on average at `c = 20`, and `COST-RATIO.md`
puts the defensible ratio for a preparer's seat in the tens to hundreds. **Placing the line
correctly stops a model from losing badly; it does not make it win.** Both halves of that
sentence are results.

## What to do

1. **Never place a threshold on in-sample scores.** Worst case 4.581, regret 0.494, and worse
   than doing nothing in 39 percent of cells. It is the only strategy here that is dominated on
   every column.
2. **Out-of-fold when the training set is large.** Free, and identical to a validation split at
   22,605 rows.
3. **A held-out split when it is small.** At 2,167 rows it is worth a tenth of the floor on the
   mean and nearly half of it in the tail.
4. **Skip the refinements.** Repeated folds and bootstrap-median thresholds move nothing,
   because the residual is bias and not variance.
5. **If you want a threshold without a search, calibrate out of fold and use `1/(1+c)`.** It
   reproduces the sweep. Do not use it on raw boosting output.

Still open: a temporal split, and whether the out-of-fold bias closes with nested or
leave-one-out schemes on small corpora.
