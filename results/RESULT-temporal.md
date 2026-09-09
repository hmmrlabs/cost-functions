# A forward split taxes every threshold, and the in-sample one no more than the rest

Measured 2026-09-09. `temporal.py`, `temporal.json`. **240 cells**: 5 rolling origins x 2 split
axes x 3 models x 4 cost ratios x 2 seeds. scikit-learn 1.9.0.

`RESULT-strong.md`, `RESULT-calibrate.md` and `RESULT-thresholds.md` each close by naming a
temporal split as still open. Every threshold result in this directory splits by `cik`, and a
reviewer of a financial prediction paper will not take a company split for an answer, because a
deployment only ever holds the past.

## The scheme

Rolling origin with a gap block, five origins, tabular features over the full corpus:

```
TRAIN   filings filed in years [Y-4, Y]     fit
VAL     filings filed in year   Y+1         S4 reads its threshold here
TEST    filings filed in year   Y+2         report

Y      train years   n_train   val   n_val   test   n_test   base rate tr -> te
2006   * 2004-2006    22,500  2007   5,205   2008    7,245      0.152 -> 0.106
2010     2006-2010    34,106  2011   6,725   2012    5,777      0.137 -> 0.140
2014     2010-2014    29,012  2015   3,885   2016    3,342      0.135 -> 0.098
2018     2014-2018    17,680  2019   2,738   2020    2,923      0.111 -> 0.166
2020     2016-2020    15,091  2021   4,539   2022    2,779      0.113 -> 0.213

* clipped by the 2004 floor: three training years rather than five
```

**Each origin is run twice: once split forward in time, once split by company over the identical
pooled rows at the identical block sizes.** That control is the whole experiment. A temporal
number on its own cannot say whether anything changed, and the company-split numbers already in
this directory are on different features and a different row pool.

Rows are restricted to filings filed 2004 onward, because 8-K Item 4.02 took effect in August
2004 and before then there was no instrument to repudiate a filing with. That floor clips the
earliest window: **the 2006 origin trains on three years, not five.**

Features are tabular: SIC division, form type, amended flag, filing year, filing month,
reporting lag, and two filing-frequency features computed from strictly prior filings only. The
narrative and XBRL views exist for 2021 alone, so a text arm cannot have a training block that
precedes its test block at all.

## The answer, plainly: **it stays the same**

The artefact is the cost of an in-sample threshold over a validation-picked one on the same test
rows, `S0 / S4`:

```
                      c = 3   c = 5   c = 10   c = 20     all
temporal              1.076   1.031    1.180    1.663   1.238
company               1.026   0.997    1.163    1.625   1.203
paired difference    +0.050  +0.034   +0.017   +0.038  +0.035
descriptive t          1.63    1.71     0.48     0.44    1.39
```

Larger under the temporal split in **69 of 120 paired cells**, which is close to a coin flip. At
the ratio `COST-RATIO.md` says this corpus actually faces, `c = 20`, the artefact is **1.66x
forward in time against 1.63x by company**, and the paired standard error on that difference is
0.088.

The point estimate is small and positive rather than exactly zero, and it is not stable in which
origins carry it. Dropping one origin at a time:

```
dropped   temporal   company     diff      se      t   bigger in
none         1.238     1.203   +0.035   0.025   1.39    69 / 120
2006         1.262     1.204   +0.058   0.025   2.30    58 / 96
2010         1.235     1.211   +0.024   0.031   0.77    53 / 96
2014         1.263     1.196   +0.067   0.028   2.35    59 / 96
2018         1.228     1.219   +0.009   0.026   0.36    55 / 96
2020         1.200     1.183   +0.017   0.030   0.57    51 / 96
```

**The direction is weakly positive and never reverses; the magnitude never exceeds 0.067 on a
baseline of 1.20, which is under six percent.** A prior that distribution shift multiplies the
memorisation effect predicts something on the scale of the effect itself, and the effect itself
is 24 to 66 percent above the floor. Six percent, at a `t` between 0.4 and 2.4 depending on which
origin you leave out, is not that. **Call it the same size and do not report a growth factor.**

Per origin at `c = 20`, seed-averaged, temporal against company:

```
Y     test    random_forest      hist_gbdt      logreg_l2
2006  2008     1.40 / 2.22     1.21 / 1.61    1.00 / 1.04
2010  2012     2.35 / 2.12     1.71 / 1.41    1.11 / 1.02
2014  2016     1.79 / 2.46     1.49 / 1.66    1.02 / 0.99
2018  2020     2.89 / 1.97     2.20 / 1.41    0.99 / 1.04
2020  2022     2.94 / 2.58     1.96 / 1.78    0.89 / 1.07
```

Temporal is larger in **8 of these 15 pairs**: at all three models in 2010, at both tree models
in 2018 and 2020, at one model in 2014, and at none in 2006. Which origin a reviewer happens to
draw decides the sign of the comparison, which is what a null looks like when it is read one
origin at a time.

## What a temporal split does instead: a flat tax on threshold transfer

Mean cost over the best floor, 120 cells per axis:

```
strategy          temporal   company   difference   descriptive t
S0_in_sample         1.220     1.121       +0.099            2.74
S1_oof               1.003     0.934       +0.068            3.84
S4_validation        0.969     0.923       +0.046            3.40
S8_oracle            0.901     0.890       +0.011            1.16
```

**The oracle does not move.** The best threshold obtainable on the test block costs the same
whether the block was reached by calendar or by company, so the model's achievable cost is not
what a forward split takes away. Every strategy that has to CARRY a threshold across the split
pays, and the oracle, which does not carry one, pays nothing.

Regret against the oracle, which is the column that isolates the line from the learner:

```
strategy          temporal   company   ratio
S0_in_sample         0.320     0.231    1.38
S1_oof               0.102     0.044    2.30
S4_validation        0.069     0.033    2.08
```

**The careful strategies lose proportionally more than the sloppy one.** Out-of-fold selection
and a held-out validation block roughly double their regret going forward in time; the in-sample
threshold, already bad, worsens by 38 percent. That is the same null read from the other side: a
temporal split is not selectively hard on in-sample selection, it is slightly harder on
everything and hardest in relative terms on the strategies that were working.

## The measured mechanism triples and buys almost nothing

`RESULT-strong.md` named firing-rate drift as the mechanism: a threshold is chosen so that some
fraction of the selection rows fires, and applied to test rows it fires on a different fraction.
Mean absolute drift at the chosen threshold:

```
strategy          temporal   company   ratio
S0_in_sample         0.090     0.027    3.32
S1_oof               0.091     0.026    3.49
S4_validation        0.042     0.014    3.08
```

Drift triples on every strategy. Cost over floor moves by 0.046 to 0.099. **Within the temporal
axis, the correlation between absolute drift and in-sample cost is +0.09, against +0.20 on the
company axis.** So the quantity that explained the magnitude under a company split is three
times larger under a temporal one and explains less.

Part of the extra drift is base rate rather than score distribution. Mean absolute shift in base
rate from train block to test block is **0.048 forward in time against 0.013 by company**, and
that shift correlates with in-sample drift at `r = +0.24` pooled over both axes. A base-rate
shift moves the firing rate and moves the floor with it, so `cost / floor` absorbs much of it.
That accounts for some of the gap and plainly not most of it. **The remainder is not decomposed
here and this file does not claim to have decomposed it.**

## Two earlier findings that survive a forward split, and two that shift

**1. The memorisation rule holds, unchanged, on both axes.** `RESULT-strong.md` found the
artefact governed by how completely a model fits its training set. `S0 / S4` at `c = 20`, by
training AUC band:

```
train AUC        temporal                 company
< 0.80           1.00  (0.89 - 1.11)      1.02  (0.93 - 1.04)
0.80 - 0.99      1.71  (1.21 - 2.20)      1.49  (0.91 - 2.06)
>= 0.99          2.27  (1.39 - 3.08)      2.27  (1.92 - 2.75)
```

On these features the bands are the models: the logistic regression reaches a training AUC of
0.64 to 0.84 and shows no artefact on either axis, gradient boosting reaches 0.88 to 0.98, and
the random forest reaches 0.995 to 1.000 and shows the artefact on both axes. Going forward in
time changes neither end of that table, and at the top of it the two axes agree to the second
decimal.

**2. Never place a threshold on in-sample scores, still.** S0 is the worst strategy at every
ratio on both axes, and its worst temporal cell is 3.395 (origin 2018, random forest, `c = 20`)
against 1.200 for the validation policy on the same rows.

**3. Out-of-fold selection recovers less.** Median fraction of the in-sample loss recovered by
5-fold out-of-fold, over cells where there was a loss to recover:

```
company    0.914   (n = 68)
temporal   0.807   (n = 66)
```

`RESULT-calibrate.md` quoted a median 0.95 on its own grid. Out-of-fold remains the cheap fix
and it is a slightly less complete one going forward.

**4. The size rule from `RESULT-thresholds.md` acquires an exception.** That file established
that out-of-fold is free and exactly as good as a validation split at 22,605 training rows.
Every origin here sits between 15,091 and 34,106 rows, so on the size rule alone the validation
split should buy nothing:

```
S1_oof minus S4_validation      mean      se   descriptive t
temporal                     +0.0331  0.0110            3.02
company                      +0.0114  0.0054            2.10
difference of differences    +0.0218  0.0118            1.84
```

The company axis reproduces the size rule: at these sizes the gap is 0.011 of the floor, near
enough to nothing. The temporal axis does not: the gap is 0.033. **Under distribution shift the
choice of selection rows is not only about sample size, it is about which rows.** The
out-of-fold rows are drawn from the training years and the validation block is drawn from the
year nearest deployment, and that recency is worth something a `K-1/K` subsample cannot supply.
At `t = 1.84` on 120 correlated cells this is the weakest claim in the file.

## What would falsify this

- **A corpus with a larger shift.** The mean base-rate shift across this corpus's two-year gap is
  0.048. If the artefact separates on the axes at a shift of 0.15 or 0.30, the null here is a
  statement about a mild shift and not about temporal splits.
- **A longer gap.** Test on `Y+5` rather than `Y+2`. If the paired difference grows with the gap,
  the mechanism is shift after all and this scheme was too short to see it.
- **A text model.** These features give a test AUC of 0.56 to 0.85 and an oracle cost of 0.901 of
  the floor. A model that actually separates the classes might place its threshold deeper into a
  tail, where transfer is more fragile.
- **Time-ordered folds inside train.** S1 here uses random 5-fold, deliberately, so that only one
  thing changed. If blocked or forward-chaining folds close the 0.033 gap to S4, the exception in
  section 4 is about fold construction and not about recency.

## What is NOT established

- **That any of these models is useful.** The oracle threshold averages 0.901 of the floor and the
  best achievable strategy is worse than the better of flagging every filing and flagging none in
  35 of 120 temporal cells. This file is about where to put a line on a weak ranking. It is not
  evidence that the ranking works.
- **The small-training-set regime.** All five origins fall between 15,091 and 34,106 training
  rows, by design, so that origin is not confounded with size. The 2,167-row regime where
  `RESULT-thresholds.md` found a validation split worth a tenth of the floor is not probed here at
  all, and nothing in this file speaks to it.
- **Text or XBRL views under a temporal split.** Not possible on this corpus: those views exist
  for 2021 only.
- **That the 2006 origin's training base rate is a real base rate.** Item 4.02 took effect in
  August 2004, so a filing made in January 2004 spent seven of its twenty-four horizon months
  outside the rule and could not be repudiated during them. The 2004 rows are kept because
  dropping them would leave the 2006 origin one training year, but its training base rate of
  0.152 is biased downward by the calendar and its window is three years rather than five. It is
  also the origin where the temporal artefact comes out smallest, so it pulls the paired
  difference toward zero: drop it and the difference goes from +0.035 to +0.058. The
  leave-one-out table above is there so a reader can price that rather than take the pooled
  number on faith.
- **Independence of the 120 cells.** Three models and four ratios share each test block, and the
  temporal blocks carry no randomness at all: a calendar year is a calendar year, and the seed
  redraws only the folds and the control split. The effective sample size on the temporal side is
  **five origins**, not 120 cells. Every `t` quoted above is descriptive of the cell spread and is
  not a test of anything. The result rests on the difference being negligible in size, on its
  direction being absent in 51 of 120 cells and in 7 of 15 origin-by-model pairs at `c = 20`, and
  on the pooled gap being the same small positive number under both seeds (+0.042 and +0.028),
  not on a p-value.
- **Why the flat tax exists.** Threshold transfer degrades by roughly 0.05 to 0.10 of the floor
  for every strategy that carries a threshold across a forward split. Base-rate shift is part of
  it at `r = +0.24`. What the rest is remains open.

## What to do

Nothing in the recommendations changes, and that is the finding.

1. **Never place a threshold on in-sample scores**, forward in time exactly as by company.
2. **Out-of-fold when the training set is large**, with the caveat that it recovers 0.81 rather
   than 0.91 of the in-sample loss under a forward split.
3. **Prefer the most recent held-out block when there is drift**, even at a training size where
   the size rule says out-of-fold should be free. Worth 0.022 of the floor here, measured weakly.
4. **Do not expect a temporal split to reveal a bigger artefact.** It does not. Report the company
   split and the temporal split as the same finding, which is what they are.

Still open: a longer forecast gap, time-ordered folds inside train, a corpus with a larger shift,
and whether the flat tax on threshold transfer is base rate or score distribution.
