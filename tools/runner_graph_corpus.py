#!/usr/bin/env python3
"""Project the original golden corpus onto routable preloaded-graph binding packets."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from runner_graph_plans import Unsupported, binary_request, compile_expression, native_packet


def generate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.goldens / "cases.jsonl.gz", "rt") as stream:
        originals = [json.loads(line) for line in stream]
    rows, sample, decisions = [], [], []
    modes, kinds = Counter(), Counter()
    for i, row in enumerate(originals):
        expression = row.get("input", {}).get("expression")
        try:
            if expression is not None:
                try:
                    packet = binary_request(expression)
                    mode = "named"
                except Unsupported:
                    if not args.allow_generic:
                        raise Unsupported("compound syntax retains the original circuit route")
                    packet = native_packet(compile_expression(expression))
                    mode = "bindings"
            elif row["id"] == "loop-squares":
                packet = {"version": 1, "plan": "loop-squares-10-62", "values": []}
                mode = "named"
            else:
                raise Unsupported("unsupported program input")
            case = {"id": row["id"], "packet": packet, "expected_output": row["final_output"],
                    "route": "graph", "mode": mode, "operation": row["operation"],
                    "reference_steps": row["generated_steps"], "outcome": row["outcome"]}
            rows.append(case)
            modes[mode] += 1
            kinds[row["operation"]] += 1
            decision = {"id": row["id"], "route": "graph", "mode": mode, "outcome": row["outcome"]}
        except (Unsupported, RecursionError, SyntaxError) as error:
            case = {"id": row["id"], "route": "circuit", "reason": str(error),
                    "expression": expression, "expected_output": row["final_output"]}
            decision = {"id": row["id"], "route": "circuit", "reason": str(error)}
        decisions.append(decision)
        if i < 20:
            sample.append(case)
    for name, cases in (("routable.jsonl", rows), ("samples-graph.jsonl", [r for r in sample if r["route"] == "graph"])):
        with (args.output / name).open("w") as stream:
            for case in cases:
                stream.write(json.dumps(case, separators=(",", ":")) + "\n")
    (args.output / "samples.json").write_text(json.dumps(sample, sort_keys=True, indent=2) + "\n")
    (args.output / "routing.json").write_text(json.dumps(decisions, sort_keys=True, indent=2) + "\n")
    limits = [r["id"] for r in rows if r["outcome"] == "machine_limit"]
    manifest = {"version": 1, "source_cases": len(originals), "routable": len(rows), "modes": dict(modes),
                "operations": dict(kinds), "unroutable": len(originals) - len(rows), "machine_limits": limits,
                "sample_graph": sum(r["route"] == "graph" for r in sample),
                "sample_circuit": sum(r["route"] == "circuit" for r in sample),
                "source_cases_sha256": hashlib.sha256((args.goldens / "cases.jsonl.gz").read_bytes()).hexdigest(),
                "selection": "syntax and bounded plan grammar only; never selected by reference outcome"}
    manifest["allow_generic"] = args.allow_generic
    (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-generic", action="store_true")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
