#!/usr/bin/env python3
"""Build reusable arithmetic weights and binding plans offline, then prepare native bank files."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runner_circuit import read_circuit
from runner_export import write_circuit
from runner_graph_ir import BUCKETS, build_template
from runner_graph_plans import plan_catalog


OPERATIONS = ("add", "sub", "mul", "divstep", "divfinal")


def bridge(doc, baseline):
    doc = dict(doc, format="tensorfold.runner.p2")
    doc["experimental"] = {"kind": "graph_template", "contract": doc["program"], "entry": "State.init then GO"}
    doc["program"] = copy.deepcopy(baseline["program"])
    doc["program"].update(kind="elf", expression=None)
    return doc


def export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    full = read_circuit(args.baseline)
    baseline = {"source": full["source"], "program": full["program"]}
    del full
    gc.collect()
    descriptors, receipts = [], []
    selections = [(op, width) for op in OPERATIONS for width in args.widths]
    selections += [("emit", 62)]
    for op, width in selections:
        start = time.monotonic()
        doc, metadata = build_template(args.qwenputer, op, width)
        stem = f"{op}-{width}"
        source, packed = args.output / f"{stem}.tfimr", args.output / f"{stem}.tfgp"
        info = write_circuit(bridge(doc, baseline), source)
        stats = doc["stats"]
        del doc
        gc.collect()
        subprocess.run([str(args.packer.resolve()), str(source), str(packed)], check=True)
        descriptors.append({"name": op, "width": width, "path": packed.name,
                            "inputs": metadata["input_arity"],
                            "outputs": max(slot["index"] for slot in metadata["output_slots"]) + 1})
        receipt = {"name": stem, "source": info, "packed_bytes": packed.stat().st_size,
                   "packed_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
                   "build_seconds": time.monotonic() - start, "slots": metadata}
        receipts.append(receipt)
        print(json.dumps({"template": stem, "bytes": receipt["packed_bytes"], "stats": stats}), flush=True)
        if not args.keep_ir:
            source.unlink()
            source.with_name(source.name + ".sha256").unlink()
    plans = plan_catalog()
    allowed = {(op, width) for op, width in selections}
    plans = [plan for plan in plans if all(call["op"] == "bind" or (call["op"], call["width"]) in allowed
                                         for call in plan["ops"])]
    manifest = {"version": 1, "templates": descriptors, "plans": plans}
    (args.output / "bank.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    (args.output / "Qwenputer-MIT.txt").write_text(baseline["source"]["license_text"])
    (args.output / "provenance.json").write_text(json.dumps({"source": baseline["source"],
        "format": "tensorfold.runner.graph-bank-provenance", "version": 1,
        "compiler": "offline Qwenputer P2 IR -> T2 prepared graph pack", "request_graph_compilation": False,
        "templates": receipts, "plans": len(plans), "widths": args.widths}, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"pass": True, "templates": len(descriptors), "plans": len(plans),
                      "pack_bytes": sum(r["packed_bytes"] for r in receipts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", default=",".join(map(str, BUCKETS)))
    parser.add_argument("--keep-ir", action="store_true")
    args = parser.parse_args()
    args.widths = [int(value) for value in args.widths.split(",")]
    if not args.widths or len(set(args.widths)) != len(args.widths) or any(w not in BUCKETS for w in args.widths):
        parser.error("widths must be distinct supported digit buckets")
    export(args)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
