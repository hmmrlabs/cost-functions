#!/usr/bin/env python3
"""The two views of each filing: the narrative it wrote and the numbers it reported.

C1 and C2 have run. `controls.json`: C2 ties C1 on the XBRL window, 459 against 460 under
missed x 10, so firm identity at that granularity carries nothing and the arms that read the
document get their turn. The bar they have to clear is 459.

    view N   the filing's own prose, MD&A and risk factors where they can be cut
    view X   its XBRL facts, from EDGAR's companyfacts, restricted to that accession

# Why companyfacts rather than the filing's own XBRL attachments

One request per COMPANY returns every fact that company has ever reported, each tagged with
the accession it came from. Filings per company here run to dozens, so this is one request
instead of dozens, and it is the same bytes. The cost is that the response is large and is
cached whole.

# The narrative cut, and what it costs

`item_402_span`'s sibling problem. A 10-K's MD&A is Item 7 and its risk factors are Item 1A;
a 10-Q numbers them differently and an amendment may carry neither. Where the headings can
be found the section is cut, and **where they cannot the whole document is kept with
`narrative_cut: "whole"` recorded on the row**, because discarding a filing whose layout this
parser did not anticipate would silently select the corpus down to filings that format
conventionally.

That flag is not decoration. An arm that scores well only on `narrative_cut: "section"` rows
has found a formatting artefact, and the flag is what makes that checkable.

# Fair access

Declared User-Agent, 0.15s between requests, everything cached under `raw/`.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
BODIES = HERE / "raw" / "bodies"
FACTS = HERE / "raw" / "facts"
SUBS = HERE / "raw" / "submissions"
UA = {"User-Agent": "hammer-bench research kartik@hammer.ai"}
SLEEP = 0.15

# Item 7 is MD&A in a 10-K, Item 2 in a 10-Q. Item 1A is risk factors. Both are matched
# loosely because filers punctuate them every way there is.
MDNA = re.compile(r"item\s*[72][^0-9a-z]{0,4}(management|quantitative)", re.IGNORECASE)
RISK = re.compile(r"item\s*1a[^0-9a-z]{0,4}risk\s*factors", re.IGNORECASE)
NEXT_ITEM = re.compile(r"item\s*\d{1,2}[ab]?[^0-9a-z]{0,4}[a-z]", re.IGNORECASE)


def fetch(url: str, tries: int = 4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2**attempt)
        except Exception:
            time.sleep(2**attempt)
    return None


def strip(html: bytes) -> str:
    t = html.decode("utf-8", "replace")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t)


def primary_doc(cik: str, adsh: str):
    cache = SUBS / f"CIK{cik}.json"
    if not cache.is_file():
        return None
    for f in json.loads(cache.read_text())["filings"]:
        if f["adsh"] == adsh:
            return f["primaryDocument"] or None
    return None


def cut(text: str) -> tuple[str, str]:
    """The narrative sections, or the whole document with the reason recorded.

    **Every occurrence is tried and the LONGEST span wins.** Taking the first match cuts the
    table of contents instead of the section: a 10-K lists "Item 7. Management's Discussion"
    in its TOC, the next Item heading is the very next line, and the span is a few characters
    of index. Measured on the first 120 rows, that produced 117 "section" cuts of which only
    9 carried more than 500 characters. The body of a section is far longer than its index
    entry, so length is the discriminator and it needs no page-layout heuristics.
    """
    spans = []
    for pat in (MDNA, RISK):
        best = ""
        for m in pat.finditer(text):
            start = m.start()
            nxt = NEXT_ITEM.search(text[start + 20 :])
            span = text[start : start + 20 + nxt.start()] if nxt else text[start:]
            if len(span) > len(best):
                best = span
        if best:
            spans.append(best)
    joined = " ".join(spans)
    # A cut that found headings but almost no text found the index, not the section.
    if len(joined) < 500:
        return text, "whole"
    return joined, "section"


def body(cik: str, adsh: str):
    cache = BODIES / f"{adsh}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    doc = primary_doc(cik, adsh)
    if not doc:
        return None
    raw = fetch(
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-','')}/{doc}"
    )
    time.sleep(SLEEP)
    if raw is None:
        return None
    text = strip(raw)
    narrative, how = cut(text)
    out = {
        "adsh": adsh,
        "chars_total": len(text),
        "chars_narrative": len(narrative),
        "narrative_cut": how,
        "narrative": narrative[:200_000],
    }
    BODIES.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def facts_for(cik: str):
    """Every XBRL fact this company has reported, keyed by the accession that carried it."""
    cache = FACTS / f"CIK{cik}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    raw = fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    time.sleep(SLEEP)
    by_adsh: dict[str, dict] = {}
    if raw:
        doc = json.loads(raw)
        for taxonomy, tags in (doc.get("facts") or {}).items():
            for tag, meta in tags.items():
                for unit, obs in (meta.get("units") or {}).items():
                    for o in obs:
                        a = o.get("accn")
                        if not a:
                            continue
                        by_adsh.setdefault(a, {})[f"{taxonomy}:{tag}:{unit}"] = o.get("val")
    FACTS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(by_adsh))
    return by_adsh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="pairs-sliced.jsonl")
    ap.add_argument("--xbrl-only", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--regime", default="", help="YYYY or YYYY-YYYY, matching controls.py")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.rows).read_text().splitlines()]
    if args.xbrl_only:
        rows = [r for r in rows if r["xbrl"]]
    if args.regime:
        lo, _, hi = args.regime.partition("-")
        hi = hi or lo
        rows = [r for r in rows if lo <= r["filed"][:4] <= hi]
    if args.limit:
        rows = rows[: args.limit]

    facts_cache: dict[str, dict] = {}
    stats = Counter()
    out_rows = []

    for i, r in enumerate(rows, 1):
        cik, adsh = r["cik"], r["adsh"]
        b = body(cik, adsh)
        if cik not in facts_cache:
            facts_cache[cik] = facts_for(cik)
        f = facts_cache[cik].get(adsh, {})

        stats["rows"] += 1
        stats["with_narrative"] += bool(b and b["chars_narrative"] > 500)
        stats["with_facts"] += bool(f)
        stats["both"] += bool(b and b["chars_narrative"] > 500 and f)
        if b:
            stats[f"cut_{b['narrative_cut']}"] += 1

        out_rows.append(
            {
                **{k: r[k] for k in ("cik", "adsh", "form", "period", "filed", "sic", "label")},
                "chars_narrative": b["chars_narrative"] if b else 0,
                "narrative_cut": b["narrative_cut"] if b else None,
                "n_facts": len(f),
            }
        )
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  narrative={stats['with_narrative']} "
                  f"facts={stats['with_facts']} both={stats['both']}", flush=True)

    name = f"views-{args.regime}.jsonl" if args.regime else "views.jsonl"
    with (HERE / name).open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    n = stats["rows"] or 1
    counts = {
        "vintage": time.strftime("%Y-%m-%d"),
        "rows": stats["rows"],
        "with_narrative": stats["with_narrative"],
        "with_facts": stats["with_facts"],
        "with_both": stats["both"],
        "both_rate": round(stats["both"] / n, 4),
        "narrative_cut": {
            "section": stats["cut_section"],
            "whole": stats["cut_whole"],
        },
        "positives_with_both": sum(
            1 for r, o in zip(rows, out_rows)
            if r["label"] == 1 and o["chars_narrative"] > 500 and o["n_facts"]
        ),
        "note": "narrative_cut whole means the MD&A and risk-factor headings were not "
        "found and the entire document was kept. An arm scoring well only on `section` "
        "rows has found a formatting artefact.",
    }
    (HERE / (f"views-counts-{args.regime}.json" if args.regime else "views-counts.json")).write_text(json.dumps(counts, indent=1) + "\n")

    print()
    print(f"rows            {stats['rows']}")
    print(f"  narrative     {stats['with_narrative']}")
    print(f"  XBRL facts    {stats['with_facts']}")
    print(f"  BOTH          {stats['both']}  ({counts['both_rate']:.1%})")
    print(f"  cut section   {stats['cut_section']}, whole {stats['cut_whole']}")
    print(f"positives with both views  {counts['positives_with_both']}")


if __name__ == "__main__":
    main()
