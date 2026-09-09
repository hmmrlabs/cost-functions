#!/usr/bin/env python3
"""Embed the narrative with a local model instead of counting words.

`inspect_weights.py` showed what the bag-of-words arm was reading: SPAC vocabulary
(`sponsor`, `rata`, `deposited`, `lock`) plus a handful of words that were the answer rather
than a predictor (`restatement`, `restated`, `weakness`). Removing the leak words barely
moved it, F1 0.835 to 0.832, because the top 25 terms carried only 6.3 percent of the weight
mass. The arm was reading broadly and shallowly: presence of vocabulary, no composition.

An embedding reads the sentence. Whether that is worth anything here is the measurement.

# Local, and why

`nomic-embed-text` over Ollama: 4,307 documents, no rate limit, no fair-access budget spent,
nothing leaving the machine. Vertex `gemini-embedding-001` is what the coverage world's own
semantic index uses and it would cost money and wall-clock for a run whose whole purpose is
to find out whether embeddings help at all. If they do, re-running on the hosted model is a
one-line change and worth the spend then.

# Chunking, stated because it decides what is measured

Narratives run to 200,000 characters and the model's context does not. Each document is cut
into word windows, each window embedded, and the document vector is their MEAN.

Mean pooling is a choice with a cost: it washes out a single anomalous paragraph, which is
plausibly exactly where a disclosure problem shows. `--pool max` keeps the strongest
component per dimension instead and is the alternative worth running when mean disappoints.
Reported on every row so the comparison is not silently between two different things.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
BODIES = HERE / "raw" / "bodies"
VECS = HERE / "raw" / "vecs"
OLLAMA = "http://localhost:11434/api/embed"


def embed(model: str, texts: list[str], tries: int = 3):
    body = json.dumps({"model": model, "input": texts}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                OLLAMA, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r).get("embeddings") or []
        except Exception:
            if attempt + 1 == tries:
                raise
            time.sleep(2**attempt)
    return []


def windows(text: str, words_per: int, max_windows: int) -> list[str]:
    w = text.split()
    if not w:
        return []
    out = [" ".join(w[i : i + words_per]) for i in range(0, len(w), words_per)]
    return out[:max_windows]


def pool(vecs: list[list[float]], how: str) -> list[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    if how == "max":
        return [max(v[i] for v in vecs) for i in range(dim)]
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="views-2021.jsonl")
    ap.add_argument("--model", default="nomic-embed-text:latest")
    ap.add_argument("--pool", choices=["mean", "max"], default="mean")
    ap.add_argument("--words-per-window", type=int, default=400)
    ap.add_argument("--max-windows", type=int, default=8,
                    help="caps a long 10-K at roughly 3,200 words. The MD&A opening is "
                         "where management explains itself and is what this reads.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (HERE / args.views).read_text().splitlines()]
    rows = [r for r in rows if r["chars_narrative"] > 500]
    if args.limit:
        rows = rows[: args.limit]

    tag = f"{args.model.split(':')[0]}-{args.pool}"
    outdir = VECS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cache = outdir / f"{r['adsh']}.json"
        if cache.is_file():
            skipped += 1
            continue
        body = BODIES / f"{r['adsh']}.json"
        if not body.is_file():
            continue
        text = json.loads(body.read_text())["narrative"]
        chunks = windows(text, args.words_per_window, args.max_windows)
        if not chunks:
            continue
        vecs = embed(args.model, chunks)
        if not vecs:
            continue
        cache.write_text(json.dumps({"adsh": r["adsh"], "n_windows": len(vecs),
                                     "vec": pool(vecs, args.pool)}))
        done += 1
        if done % 100 == 0:
            rate = done / max(1e-9, time.time() - t0)
            left = (len(rows) - i) / max(rate, 1e-9) / 60
            print(f"  {i}/{len(rows)}  embedded={done} cached={skipped} "
                  f"{rate:.1f}/s  ~{left:.0f}m left", flush=True)

    counts = {
        "vintage": time.strftime("%Y-%m-%d"),
        "model": args.model,
        "pool": args.pool,
        "words_per_window": args.words_per_window,
        "max_windows": args.max_windows,
        "rows": len(rows),
        "embedded_this_run": done,
        "already_cached": skipped,
        "dir": str(outdir.relative_to(HERE)),
    }
    (HERE / f"embed-counts-{tag}.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(f"\nrows {len(rows)}  embedded {done}  cached {skipped}  -> {outdir.name}")


if __name__ == "__main__":
    main()
