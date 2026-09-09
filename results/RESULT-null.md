# Null: reading the filing adds nothing over knowing the year

> **Superseded in part, 2026-09-08. See `RESULT-costs.md`.** Every null below is scored
> against `fn * 10 + fp`. The crossover where these arms stop beating the floor is c = 5.5 to
> 5.75, so at any ratio below that the conclusions reverse. Two figures here are also a
> threshold artefact: C2 was never cost-tuned, and cost-tuned it beats always-fire in 2021,
> 563 against 605, where this file records 2,092.

Measured 2026-09-08 on `views.jsonl`, 1,967 filings with both views, 103 companies, split by
company, scored under missed x 10.

## The result

```
                       mean cost vs C1e     beats the calendar
C3  numeric + year          0.998                3/5 splits
C4  narrative + year        0.985                3/5 splits
```

A coin flip is 2.5 of 5. **Neither arm is distinguishable from a baseline that knows only the
filing year**, and two of five splits are worse than it. Without the year feature at all, C4
returned a cost of 432 against C1e's 432, reproducing the calendar exactly and finding nothing
else.

## The number that made this look like a finding

Earlier the same day, against the wrong baseline:

```
                vs C1 (never fires)      vs C1e (year only)
C3  numeric          0.878                     0.998
C4  narrative        0.896                     0.985
```

The first column was reported as "both beat the base rate on every split, sign test p=0.0078, so
the effect is real". Every word of that was true and the baseline was wrong. **Both arms were
rediscovering the era wave out of the documents and it was being scored as skill.**

## Why the calendar is so strong

Positive rate by filing year, XBRL window:

```
2010  0.037        2020  0.167
2013  0.122        2021  0.450     <- SPAC restatement wave
2016  0.092        2023  0.272
2019  0.086        2024  0.275
```

A twelvefold swing. Text is the most era-contaminated view available: a 2021 filing talks about
SPACs and COVID, and a model that learns that vocabulary looks like it is reading risk.

The controls at full scale say the same thing:

```
full corpus                 XBRL window
C1  never fires   1.000     C1  never fires   1.000
C1e year only     0.535     C1e year only     0.577
C2  + SIC/form    0.670     C2  + SIC/form    0.502
```

On the full corpus **the calendar alone beats the full lookup table.** Adding industry and form
to the year makes it worse.

## What this settles, and what it does not

**It does not refute the disagreement hypothesis.** Disagreement between two views is not a
function of either view alone, so an interaction can in principle carry what the main effects do
not. A1 is untested and this result does not touch it directly.

**It does make A1 an expensive bet.** Testing it on the full window needs roughly 45,000 document
fetches, about two hours of fair-access requests, to look for an interaction whose two main
effects are both null. That is how a null becomes an expensive null.

**It does say the current framing is wrong.** The task as posed asks a document to beat a
calendar that already knows the answer, and the calendar wins.

## Three passes, three different conclusions

Each was reported at the time and each overturned the last:

```
1,988 rows           C2 ties C1              "firm identity carries nothing"
45,005 rows          C2 halves C1            "firm identity carries a lot"
+ calendar baseline  C1e beats C2            "most of it was the date"
+ arms vs C1e        C3, C4 tie C1e          "the documents carry nothing extra"
```

The 46-positive sample could not distinguish any of these. That is the cost of measuring a rare
event on a small slice, and it is the reason the full corpus was worth building before any
encoder was.
