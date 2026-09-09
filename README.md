# What Is a False Negative Worth?

Grounding asymmetric cost in labor hours, and the threshold bug it uncovers.

Code, results and paper source for a study of what happens when the cost ratio in
`FN * c + FP` is asserted rather than measured. Everything here is reproducible from a
public corpus and two public control datasets.

## The short version

1. **The ratio decides the result.** Three unrelated feature spaces agree that our models beat
   the trivial baseline below `c = 5.6` and lose above it. Our own constant, `c = 10`, traced
   to a model-selection key for a different task in a different repository.
2. **You can derive the ratio from labor hours**, and we do: a restatement's measured audit fee
   premium of $191,000 works out to 424 to 1,273 professional hours, putting `c` between 24 and
   300 depending on review length and detection rate. One cell in twenty reaches 10.
3. **But `c` belongs to a seat, not to a task.** A preparer, an investor and a regulator have
   three different numerators and one of them is not a ratio. A task specification that does
   not name a seat does not determine a cost ratio.
4. **The threshold was a bigger error than the ratio.** Selecting the decision threshold on
   in-sample scores inflates test cost by up to **5.5x**. It is governed by how completely the
   model memorises its training set, so it is worst for the models most likely to be deployed.
   Calibration provably cannot fix it. Out-of-fold selection recovers a median 95 percent.

## Layout

```
paper/     LaTeX source and the compiled PDF
code/      the experiments, stdlib-only except where sklearn is named
data/      corpus construction: the 8-K Item 4.02 join, verification, time slicing
results/   one markdown file per experiment, written as the result landed
```

`results/` is meant to be read in the order the work happened, because several files correct
earlier ones and the corrections are annotated in place rather than edited away:

```
RESULT-null.md         the original null            (superseded in part)
RESULT-regimes.md      regimes, and the SPAC wave   (superseded in part)
RESULT-costs.md        the cost sweep, capacity, hours
COST-RATIO.md          where c = 10 came from, and what cleanup actually costs
RESULT-threshold.md    the crossover measured a policy  (framing superseded)
RESULT-strong.md       strong models make it worse, p/n was the wrong variable
RESULT-calibrate.md    calibration cannot fix it, out-of-fold nearly does
RESULT-thresholds.md   eight threshold strategies, 180 cells
```

## Data

The corpus is 156,732 SEC filings from 5,085 companies, 2004 to 2024, labelled from 8-K Item
4.02: the item a registrant files to state that previously issued financials should no longer
be relied upon. The 8-K names the filings it retracts, so the labels are mechanical rather than
annotated. Source documents are public EDGAR filings.

`data/` holds the construction scripts. The built corpus is published separately as a dataset
so that the results can be checked without re-crawling EDGAR.

Two public control datasets, `bank-marketing` and `creditcard`, are fetched from OpenML by
`code/replicate.py`.

## Reproducing

Most of `code/` is standard-library Python with no dependencies. The strong-model work needs
scikit-learn:

```bash
python3 -m venv .venv && ./.venv/bin/pip install scikit-learn
./.venv/bin/python code/threshold_bench.py --dataset bank --seed 20260818
./.venv/bin/python code/strong.py --dataset sec2021 --arm numeric
python3 code/agg_thresholds.py
```

Seeds are fixed and printed. Where a number could not be reproduced to the precision it was
quoted at, the cause is named in the relevant result file; the random forest is run with
`n_jobs=1` for exactly this reason.

## Status

Draft. Not yet submitted. Open items are listed in the paper's limitations section: a temporal
split, more seeds on the smaller cells, and whether the out-of-fold bias closes with larger `K`.

## License

Code under MIT. Paper text CC BY 4.0. Source filings are US government works.
