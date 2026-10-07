#!/usr/bin/env python3
"""Decode speed over HTTP, the same client for prod's Python server and the Zig server: m2bench's workloads (code and
prose, 384 tokens, thinking off, a distinct prompt a stream as ``--c-distinct``), T0 and T0.7 (top-p 0.95, seed 1234),
1 and 4 concurrent streams. A stream's decode tok/s is m2bench's: completion tokens after the first, over the first to
the last streamed token. A cell of N streams reports each stream's and their mean, and the aggregate over the span all
N are live (m2bench ``steady``). Standard library only (runs on the Spark hosts as is).

    python3 bench_http.py --url http://127.0.0.1:8000 --out prod.json [--streams 1,4] [--temps 0,0.7] [--reps 2]
    python3 bench_http.py --compare prod.json zig.json      # the table: Zig vs prod a cell
    python3 bench_http.py --url ... --out long.json --long 32768,131072,524288   # long context, one stream a length:
        an N-token document (about N tokens: the reply's usage.prompt_tokens is the exact count) and a question; the
        prompt's tok/s (prompt tokens / time to the first token) and the decode tok/s of a 128-token answer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.request

CODE = """Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked
list, with docstrings and type hints, then three unit tests with pytest."""
PROSE = """Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what
replaced them. Plain prose, no lists or headings."""
CODE_MORE = (
    """Write a Python class `Trie` with `insert(word)`, `search(word)` and `starts_with(prefix)`, with docstrings and
type hints, then three unit tests with pytest.""",
    """Write a Python class `TokenBucket` rate limiter with `allow(now: float) -> bool`, refill rate and burst size,
with docstrings and type hints, then three unit tests with pytest.""",
    """Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs in
O(n log n), with a docstring and type hints, then three unit tests with pytest.""")
PROSE_MORE = (
    """Write a 400-word essay on why canals were dug where they were, how the people who worked them lived, and what
replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why early observatories were built where they were, how their astronomers lived, and
what replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why windmills were built where they were, how their millers lived, and what replaced
them. Plain prose, no lists or headings.""")
PROMPTS = {"code": (CODE, *CODE_MORE), "prose": (PROSE, *PROSE_MORE)}


def one(url: str, model: str, msg: str, temp: float, max_tokens: int, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0:
        body.update(top_p=0.95, seed=1234)
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    times, usage, t0, text = [], None, time.perf_counter(), []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    times.append(time.perf_counter())
                    text.append(delta.get("content") or delta.get("reasoning_content") or "")
    n = (usage or {}).get("completion_tokens") or len(times)
    out.update(t0=t0, first=times[0] if times else None, last=times[-1] if times else None, tokens=n,
               prompt_tokens=(usage or {}).get("prompt_tokens"),
               sha=hashlib.sha256("".join(text).encode()).hexdigest()[:16],
               decode_tok_s=round((n - 1) / (times[-1] - times[0]), 2) if len(times) > 1 and times[-1] > times[0] else None,
               ttft_s=round(times[0] - t0, 3) if times else None)


def cell(url: str, model: str, work: str, temp: float, streams: int, max_tokens: int) -> dict:
    res = [dict() for _ in range(streams)]
    th = [threading.Thread(target=one, args=(url, model, PROMPTS[work][i % 4], temp, max_tokens, res[i]))
          for i in range(streams)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    rates = [r["decode_tok_s"] for r in res if r.get("decode_tok_s")]
    lo = max(r["first"] for r in res if r.get("first"))
    hi = min(r["last"] for r in res if r.get("last"))
    agg = None
    if streams > 1 and hi > lo:     # tokens inside the all-live span, apportioned at each stream's own rate
        agg = round(sum(r["decode_tok_s"] * (hi - lo) for r in res if r.get("decode_tok_s")) / (hi - lo), 2)
    return {"work": work, "temp": temp, "streams": streams, "per_stream": rates,
            "mean_tok_s": round(sum(rates) / len(rates), 2) if rates else None, "aggregate_tok_s": agg,
            "ttft_s": [r.get("ttft_s") for r in res], "tokens": [r.get("tokens") for r in res],
            "sha": [r.get("sha") for r in res]}


def document(n: int) -> str:
    """About n tokens of distinct text (numbered paragraphs of the prose prompts, ~0.75 words a token) and a question."""
    paras, words, i = [], 0, 0
    base = [p.replace("Write a 400-word essay on ", "").replace("\n", " ") for p in PROMPTS["prose"] + PROMPTS["code"]]
    while words < 0.75 * n:
        t = f"Section {i}: note {i * 7919 % 100003} on {base[i % len(base)]}"
        paras.append(t)
        words += len(t.split())
        i += 1
    return "\n\n".join(paras) + f"\n\nWhich note number does section {i // 2} carry? Then summarize the document in 100 words."


def long_cell(url: str, model: str, n: int, max_tokens: int) -> dict:
    r: dict = {}
    one(url, model, document(n), 0.0, max_tokens, r)
    pt = r.get("prompt_tokens")
    return {"work": f"long-{n}", "temp": 0.0, "streams": 1, "prompt_tokens": pt, "ttft_s": r.get("ttft_s"),
            "prompt_tok_s": round(pt / r["ttft_s"], 1) if pt and r.get("ttft_s") else None,
            "mean_tok_s": r.get("decode_tok_s"), "aggregate_tok_s": None, "per_stream": [r.get("decode_tok_s")],
            "tokens": [r.get("tokens")]}


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))

    def key(c):
        return (c["work"], c["temp"], c["streams"])

    best = lambda cs: {k: max(v) for k, v in _group(cs, key).items()}       # noqa: E731 (best of the reps)
    pa, pb = best(a["cells"]), best(b["cells"])
    print(f"| workload | T | streams | {a['label']} tok/s | {b['label']} tok/s | {b['label']} vs {a['label']} |")
    print("|---|---|---|---:|---:|---:|")
    for k in sorted(pa):
        if k in pb and pa[k] and pb[k]:
            print(f"| {k[0]} | {k[1]} | {k[2]} | {pa[k]:.1f} | {pb[k]:.1f} | {100 * (pb[k] / pa[k] - 1):+.1f} % |")
    # greedy (T0) replies: the same text on both sides, prompt by prompt (Zig's tokens == prod's)
    eq = tot = 0
    for ca in a["cells"]:
        for cb in b["cells"]:
            if key(ca) == key(cb) and ca["temp"] == 0 and ca.get("sha") and cb.get("sha") and ca.get("rep") == cb.get("rep"):
                for x, y in zip(ca["sha"], cb["sha"]):
                    tot += 1
                    eq += x == y
    if tot:
        print(f"\nT0 replies equal ({b['label']} vs {a['label']}): {eq}/{tot}")
    la = {c["work"]: c for c in a["cells"] if c["work"].startswith("long-")}
    lb = {c["work"]: c for c in b["cells"] if c["work"].startswith("long-")}
    if la and lb:
        print(f"\n| long context | prompt tokens | {a['label']} prompt tok/s | {b['label']} prompt tok/s | "
              f"{a['label']} TTFT s | {b['label']} TTFT s |")
        print("|---|---:|---:|---:|---:|---:|")
        for w in sorted(la, key=lambda x: int(x[5:])):
            if w in lb:
                print(f"| {w} | {la[w]['prompt_tokens']} | {la[w]['prompt_tok_s']} | {lb[w]['prompt_tok_s']} | "
                      f"{la[w]['ttft_s']} | {lb[w]['ttft_s']} |")


def _group(cs, key):
    out = {}
    for c in cs:
        v = c["mean_tok_s"] if c["streams"] == 1 else (c["aggregate_tok_s"] or c["mean_tok_s"])
        out.setdefault(key(c), []).append(v or 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--label", default="")
    ap.add_argument("--streams", default="1,4")
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--work", default="code,prose")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--long", default="", help="long-context lengths (tokens, comma separated): one stream each, T0")
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return 0
    cell(a.url, a.model, "code", 0.0, 1, 32)                                 # warm-up, not recorded
    cells = []
    if a.long:
        for n in [int(x) for x in a.long.split(",")]:
            c = long_cell(a.url, a.model, n, 128)
            cells.append(c)
            print(json.dumps(c), flush=True)
        if a.out:
            json.dump({"label": a.label or a.url, "cells": cells}, open(a.out, "w"), indent=1)
        return 0
    for rep in range(a.reps):
        for work in a.work.split(","):
            for temp in [float(t) for t in a.temps.split(",")]:
                for s in [int(x) for x in a.streams.split(",")]:
                    c = cell(a.url, a.model, work, temp, s, a.max_tokens)
                    c["rep"] = rep
                    cells.append(c)
                    print(json.dumps(c), flush=True)
    if a.out:
        json.dump({"label": a.label or a.url, "cells": cells}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
