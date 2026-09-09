# What `c` we used, where it came from, and what the cleanup actually costs

Written 2026-09-09. Traces the constant that decides every result in this directory, then
tries to measure it.

## 1. The constant, and its actual provenance

```
controls.py:34    MISS_COST = 10
arms.py:44        MISS_COST = 10
controls.py:18    "MODELCARD and E22 both score missed x 10"
```

Follow that citation. `MODELCARD.md` is not in this repository; it is in
`repos/hammer-worldmodel`, and the line is a table row:

```
| selection | val flip-pair accuracy, best epoch 4 | val asymmetric key (missed x10 + false), best epoch 6 |
```

**It is a model-selection key.** The metric used to pick which training epoch to keep, for an
overreach assessor, trained on 178 synthesised rows, in a different repository, on a different
corpus, answering a different question. It was never a claim about restatements, about
auditors, or about anybody's review budget.

That is worse than "unmeasured". An unmeasured constant is a guess about the right domain. This
one was carried across a repository, a corpus and a decision problem, and then every result
here was reported against it as though it described the world.

## 2. What a restatement actually costs, as far as it is published

**The auditor's side is measured.** An interim restatement carries a mean audit fee premium of
**$191,000**, the gap between a restating client's $1.583M mean fee and the $1.392M it would
otherwise have paid, a 13.7 percent premium
([Interim restatements and the audit engagement](https://www.sciencedirect.com/science/article/pii/S0278425425000699)).
Post-remediation the premium persists at
[21 to 32 percent in years three and four](https://www.academia.edu/93160203/Audit_Fees_after_Remediation_of_Internal_Control_Weaknesses).

Converted at the published blended range for US audit work, roughly $150 to $450 an hour:

```
  $150/hr   1,273 audit hours
  $250/hr     764
  $350/hr     546
  $450/hr     424
```

So **roughly 400 to 1,300 hours of professional time**, and that is the auditor alone.

**The elapsed clock.** An 8-K is due within four business days of the non-reliance
determination. The restatement itself typically runs
[around ninety days](https://roseryan.com/finance-accounting-solutions/strategic-projects/restatements/),
and six to nine months when the periods are many or the errors are structural.

**The company's own side is not published.** Internal controller and legal hours, and the
consultants brought in for the project, do not appear in any standardised disclosure. Several
searches returned market-reaction studies and fee studies and nothing on internal effort. So
the 400 to 1,300 figure is a **lower bound on the professional hours a restatement consumes**,
not an estimate of the whole.

## 3. Turning that into `c`, and the discount that decides it

`c` is hours a miss costs over hours a review costs. A flagged filing that is clean costs one
review. A missed one costs the cleanup, but only in proportion to how often a review would
actually have caught the error. Call that detection rate `d`.

```
c = d * H_miss / H_review          H_miss = 760 audit hours

 detection    0.25h review   0.5h    1h     4h     8h
      0.10             304    152    76     19     10
      0.25             760    380   190     48     24
      0.50            1520    760   380     95     48
      1.00            3040   1520   760    190     95
```

**One cell in twenty reaches 10, and it is the corner: a review that finds the error one time
in ten, costing a full working day.** Every other parameterisation is between two and three
hundred times the constant this directory used.

The crossover measured in `RESULT-costs.md` is 5.5. So the honest reading is that **the real
ratio is far above the crossover, flagging everything is correct on cost, and the models were
never going to win that argument.**

## 4. The qualification that matters more than the arithmetic

The 760 hours belongs to a seat this corpus does not occupy.

This task is **post-filing**: given a document already filed, predict whether it will later be
retracted. Flagging it **does not prevent the restatement.** The error is in the numbers, the
company still has to fix them, and the 400 to 1,300 hours is spent either way. What early
detection buys depends entirely on who is asking:

- **A preparer or auditor before filing.** Catching it here does avoid the restatement, and the
  760 hours is the right numerator. But this corpus cannot be that seat, because its label only
  exists after the fact.
- **An investor.** A miss is holding a position through a
  [-9.2 percent announcement effect](https://www.gao.gov/products/gao-03-138). The cost scales
  with position size, so a unitless ratio against "one analyst hour" is not even the right shape
  for the question.
- **A regulator or a reviewer triaging filings.** The remediation happens regardless. What is
  bought is *earlier*, and the value of earlier is a delay measured in months, not a pile of
  hours avoided.

**So the corpus cannot fix `c`, because `c` is a property of who is asking and this task names
nobody.** That is not a gap to be filled by better research. It is a statement about what the
task is.

## 5. What follows

1. **Stop quoting `c = 10`.** It has no provenance in this domain. Where a cost figure is needed
   from this corpus, report the sweep, or name the seat and derive the ratio from that seat's
   own hours.
2. **At any defensible ratio for the preparer's seat, flag everything.** The measured hours put
   `c` in the tens to hundreds, the crossover is 5.5, and above it a model has nothing to add on
   cost alone. That is a real finding and not a failure.
3. **Which is exactly why the recommendation is `precision@k` at a stated capacity.** A ranked
   queue under a review budget **does not need `c` at all.** It needs a number every review
   team already knows, which is how many filings they can open. Section 4 says the ratio is
   unknowable without naming a seat; the capacity framing routes around that instead of
   pretending to resolve it.

The strongest argument for the capacity view was never that flag-everything is unstaffable. It
is that the alternative requires a constant nobody can supply.
