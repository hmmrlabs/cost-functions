# Regimes, and the thing that beats every model

> **Superseded in part, 2026-09-08. See `RESULT-costs.md`.** Every null below is scored
> against `fn * 10 + fp`. The crossover where these arms stop beating the floor is c = 5.5 to
> 5.75, so at any ratio below that the conclusions reverse. Two figures here are also a
> threshold artefact: C2 was never cost-tuned, and cost-tuned it beats always-fire in 2021,
> 563 against 605, where this file records 2,092.

Measured 2026-09-08 on `pairs-sliced.jsonl`, 156,732 filings, 5,085 companies, scored under
missed x 10, split by company.

## The operational finding, which needs no model at all

Under a cost where a missed restatement is worth ten false alarms, **always fire** beats
**never fire** as soon as the base rate clears `1/11 = 0.0909`. Every regime in this corpus
clears it.

```
regime            years        rows    base    C0 always   C1 never    C2 lookup
SOX era           2004-2007   27,705   0.147      5,919      10,190       8,117
financial crisis  2008-2010   22,729   0.129      5,045       7,440       5,969
quiet, early      2011-2015   25,656   0.136      5,415       8,500       6,841
quiet, late       2016-2019   12,168   0.098      2,714       2,960       2,735
covid onset       2020         2,923   0.188        618       1,430         961
SPAC wave         2021         4,539   0.463        605       5,220       2,092
post-wave         2022-2024    6,348   0.229      1,251       3,720       2,131
```

**C0 is the floor in all seven, and nothing built here has beaten it reliably.** So:

> Once more than one filing in eleven is going to be retracted, and a miss costs ten times a
> review, stop predicting and review everything.

That statement does not depend on a model, a corpus vintage or an embedding.

## A ratio that was an artefact, corrected

The first cross-regime run reported `C2/floor` rising monotonically with the base rate across
all seven regimes, 1.008 to 3.458, and concluded prediction gets harder as errors rise.

**That was measuring the floor, not the arm.** C0's cost IS the negative count, so it falls
mechanically as the base rate rises, and any fixed-quality arm looks worse against it.

Base-rate-free measures say something different:

```
                          precision lift over base rate
low-error regimes  (<0.15)            1.18
high-error regimes (>=0.15)           1.33
```

Discrimination is roughly flat across a 4.7x swing in error rate. **The world does not get much
easier or harder to predict; it gets easier or harder to beat a trivial baseline**, which is a
different claim.

## Where the systemic effect actually lives

2021 had the highest precision lift of any regime, 1.57, which read as evidence AGAINST a
systemic cause washing out discrimination. It was the opposite, one level up.

```
2021, by industry              pos      n    rate
SIC 6770  blank checks         983  1,133   0.868
everything else                        ~     0.328
```

SIC 6770 is 25 percent of 2021 rows and **48 percent of its positives**. One industry code
separates 0.868 from 0.328, and C2 already has that code.

So the model was identifying **which industry had the problem**, not which firms within it. Run
inside SIC 6770 alone, at a base rate of 0.821:

```
C0  always fire    cost    47    F1 0.901
C2  lookup         cost   232    F1 0.882
```

**The lookup is 4.9x worse than flagging everything, and worse on F1.** Within the industry the
systemic cause struck, firm characteristics separate nobody from anybody. That is the systemic
signature, and the earlier "neither, much" conclusion was looking at the wrong level.

## Spikes 2 and 4: embeddings

`nomic-embed-text` over Ollama, 4,268 of 4,336 narratives in the 2021 regime, chunked into
400-word windows and pooled.

```
                mean ratio to C0    beats C0
mean pooling          1.047           3/7
max pooling           1.034           4/7
```

**Spike 4 is a null.** Max beats mean on 4 of 7 splits and the means differ by 0.013. The
hypothesis was that mean pooling washes out the one anomalous paragraph where a disclosure
problem shows; if it does, that is not where the signal is.

**Spike 2 is the closest anything has come.** The embedding arm reaches recall 0.92 to 0.98
against the word arm's 0.86, and misses are what cost ten. It pays in precision, which is why its
F1 (0.65 to 0.76) is WORSE than the word list's (0.78 to 0.83) while its cost is BETTER.

Words and embeddings are optimising different things, and the cost-versus-F1 split that runs
through this whole corpus appears again inside a single arm.

## What has been ruled out

- **Cross-modal disagreement.** `spike1_disagreement.py`: the quartile where the two views
  disagree most carries a positive rate 0.93x the quartile where they agree, mean over 5 splits,
  2 of 5 above 1.0. No raw signal to project into. The Barlow-Twins-style arm was not built.
- **The narrative view carrying anything past the calendar** in the pooled corpus, see
  `RESULT-null.md`.
- **Firm characteristics inside a systemic wave**, above.

## What has not

Whether any arm beats C0 under a cost where a miss is worth two or three false alarms rather than
ten. Every null here is a null against 10:1, and that ratio was inherited from `MODELCARD.md`
rather than measured from anybody's actual review budget. It is the single assumption most likely
to be wrong, and it is one line to change.
