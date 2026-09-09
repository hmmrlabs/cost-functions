# sec-8k-402-prior

The filing an Item 4.02 repudiates, resolved from the 4.02 itself, so that a restatement
event can be used as a label over two views of the document it condemns.

Built 2026-09-08. The design it answers is
`hammer-worldmodel/docs/superpowers/specs/2026-09-08-cross-document-disagreement-design.md`,
which names the join rate as the first number and says to stop if it is poor.

**Read this file in order, because the first rule in it was refuted by the second.** The
positional join below was the gate measurement; verifying it against the filings' own prose cut
it from 91.5 to 67.1 percent and replaced it. Both are kept: a rule that was wrong and the
measurement that showed it is more useful than the surviving rule on its own.

## Source

Not fetched again. The events come from `../sec-8k-402/raw/efts-hits.jsonl`, filtered on the
**structured `items` field** containing `4.02`, never on the phrase that found them, which is the
same rule the sibling harvest applies for the same reason.

For each event, one request to `https://data.sec.gov/submissions/CIK<cik>.json`, plus its
overflow shards under `filings.files` where a company files often enough to have them. Declared
User-Agent, 0.15s between requests, cached to `raw/submissions/` so a re-run costs nothing.

## The join rule, and what it is not

**Positional and unverified.** The prior filing is the most recent periodic form (`10-K`, `10-Q`,
`20-F`, `40-F`, and their amendments) by the same CIK strictly before the 4.02's own filing date.

A 4.02 names the affected periods in its prose. The structured item field does not carry them. So
this is a plausible antecedent and not a confirmed one, and the rate below is a rate about this
rule rather than about the SEC's record. Two things it cannot see, stated so the number is read
correctly:

- a 4.02 restating a filing several periods back joins to the wrong one
- a 4.02 whose affected filing predates the available submission history joins to nothing, and is
  counted as a miss, which is right

Verifying the join means reading the 4.02's own prose. That is the next step and it was worth
deferring until the rate justified it.

## Result, on a 200-event sample, seed 20260818

```
probed        200
joined        183   (91.5%)
with XBRL      83   (45.4% of joined)
gap days      median 91, p90 203
prior forms   10-Q 143, 10-K 23, 10-K/A 10, 10-Q/A 7
```

**The median gap is 91 days**, which is one quarter, and the modal antecedent is a `10-Q`. That is
what a plausible positional join should look like and it is the strongest evidence here that the
rule is not resolving to noise.

**XBRL is the binding constraint, not the join.** 45.4 percent of joined filings carry XBRL, and
every miss is pre-2011: the phase-in ran from large accelerated filers in June 2009 to all filers
by June 2011, and this corpus starts in 2004 with its heaviest years in 2005 and 2006.

That figure was predicted before it was measured. 5,062 of the 9,202 events are pre-2011, which
puts the XBRL-bearing share at **45.0 percent** from the year distribution alone. The probe
returned **45.4 percent** by a route that never looked at the year. Two independent estimates
agreeing is the reason to trust it.

## What the corpus would be

Projected across all 9,202 events:

| | |
|---|---|
| joined to a prior filing | ~8,420 |
| joined **and** XBRL-bearing | **~3,819** |

So the numeric-view arm loses more than half the corpus, and the narrative-only arms keep nearly
all of it. **The arms are therefore not run on the same rows unless the whole experiment is
restricted to the XBRL window**, and they are not comparable across different row sets. Restrict
first, and report the restriction, or the comparison is between an arm with 8,420 rows and an arm
with 3,819 and the difference is the sample.


## The positional join was wrong, and reading the prose replaced it

`verify_join.py` fetched each 4.02's own document, cut the Item 4.02 span, parsed the periods it
names and compared them with the period the positional rule had picked. On 100 joined events,
seed 20260818:

```
confirmed             57
contradicted          28
unreadable            15
confirmed of readable  67.1%   (n=85)
```

**So the positional rule was right about two thirds of the time, not 91.5 percent.** The 91.5 was
a rate about resolving *something*; this is the rate about resolving the *right* something, and
the gap between them is why the verification step existed.

`unreadable` is reported separately and folded into neither bucket. A 4.02 that incorporates its
periods by reference, or whose layout this parser cannot cut, is a fact about the instrument and
not evidence either way.

**The diagnosis is in the contradicted rows and it is not a parser fault:**

```
joined 10-Q period 2012-03-31   prose named 2011-03-31, 2011-06-30,
                                            2011-09-30, 2011-12-31, 2012-06-30
```

Twenty-three of the twenty-eight contradicted joins were 10-Q joins at companies restating a run
of periods. **A 4.02 usually repudiates several filings, not one**, so "which filing does this
4.02 refer to" was the wrong question.

## What the corpus actually is

`build_pairs.py` asks the right question instead: which periods does the 4.02 name, matched
against `reportDate` on that company's periodic filings. Dates that are not any filing's period
match nothing and drop out, so the board-meeting and press-release dates the parser also picks up
are filtered by the filing record rather than by a hand-kept list.

Over the same 183 events:

```
events probed              183
  unreadable                31
  naming >=1 filing        136
filings named per event    mean 4.9, median 4, max 26
positives                  672   (325 with XBRL)
negatives                8,876   (4,084 with XBRL)
base rate                 7.04%
base rate, XBRL window    7.37%
```

**4.9 positives per event against 1 by construction under the positional rule**, and every one of
them verified against the company's own words rather than assumed from filing order.

Negatives are periodic filings by the same companies that no 4.02 names: same firms, same forms,
same reporting machinery, label the only difference. That is what stops an arm learning which
companies restate instead of which disclosures precede a restatement, and control C2 in the
design exists to check that it failed to.

The base rate barely moves between the full set and the XBRL window, 7.04 against 7.37 percent,
so restricting to the XBRL window costs rows without distorting the task.

## Files, after the rebuild

| file | what it is |
|---|---|
| `join_probe.py` | the first, positional gate measurement. Kept: it is what `verify_join.py` refuted |
| `verify_join.py` | reads the 4.02 prose and scores the positional join. `counts` in `verification.json` |
| `build_pairs.py` | the corpus build that replaced it, prose-named periods matched on `reportDate` |
| `pairs.jsonl` | one row per filing: cik, adsh, form, period, filed, xbrl, label |
| `pairs-counts.json` | the numbers above |

## Files, as first written

| file | what it is |
|---|---|
| `join_probe.py` | the measurement. `--sample N` probes a subset; 0 does all |
| `counts.json` | the numbers above, machine-readable |
| `joins.jsonl` | one row per joined event: event accession, CIK, SIC, prior accession, form, period, gap, XBRL flag |
| `raw/submissions/` | cached EDGAR submissions per CIK, gitignored |

## Licence

SEC EDGAR is a US federal work and public domain. Fair-access terms observed: declared
User-Agent, well under 10 requests per second.

## Not done

- **Only 183 events are built.** That is the sample `join_probe.py` cached; the full 9,202 needs
  the same two passes and roughly an hour of fair-access requests.
- **No document bodies for the FILINGS.** Only the 4.02s were fetched, to read their periods. The
  MD&A and risk-factor text and the XBRL facts are the feature build, and the design gates them
  behind controls C1 and C2 running first, so that a base rate and a lookup table get their shot
  before anything is encoded.
- **Negatives are not time-sliced.** Every non-named periodic filing by a restating company counts
  as a negative, including ones filed after the 4.02. For a predictive framing that is leakage,
  and the horizon has to be declared before C3 runs.
