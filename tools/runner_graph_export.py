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
import shutil

from runner_circuit import read_circuit
from runner_export import write_circuit
from runner_graph_ir import BUCKETS, build_template
from runner_graph_plans import plan_catalog
from runner_graph_algorithms import OPERATIONS as INTEGER_KERNELS, catalog
from runner_graph_whole import OPERATIONS as WHOLE_KERNELS, plans as whole_plans


OPERATIONS = ("add", "sub", "mul", "select", "divstep", "divfinal", *INTEGER_KERNELS, *WHOLE_KERNELS)


def bridge(doc, baseline):
    doc = dict(doc, format="tensorfold.runner.p2")
    doc["experimental"] = {"kind": "graph_template", "contract": doc["program"], "entry": "State.init then GO"}
    doc["program"] = copy.deepcopy(baseline["program"])
    doc["program"].update(kind="elf", expression=None)
    return doc


def export(args):
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sources = ("runner_graph_ir.py", "runner_graph_algorithms.py", "runner_graph_div.py",
               "runner_graph_byte.py", "runner_graph_export.py", "runner_graph_plans.py", "runner_graph_whole.py", "runner_graph_format.py")
    identity = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    identity["widths"] = args.widths
    legacy = json.loads(Path(__file__).with_name("runner-graph-legacy.json").read_text())
    build_file = args.output / "build.json"
    if build_file.exists() and json.loads(build_file.read_text()) != identity:
        raise ValueError("resume compiler identity or widths changed")
    build_file.write_text(json.dumps(identity, sort_keys=True) + "\n")
    progress_file = args.output / "progress.json"
    progress = json.loads(progress_file.read_text()) if args.resume and progress_file.exists() else []
    full = read_circuit(args.baseline)
    baseline = {"source": full["source"], "program": full["program"]}
    del full
    gc.collect()
    descriptors, receipts = [], []
    selections = [(op, width) for op in OPERATIONS for width in args.widths if op != "meta" or width >= 8]
    if any(width < 8 for width in args.widths) and 8 not in args.widths:
        selections.extend((op, 8) for op in ("meta", "add", "half", "divstep"))
    if ("meta", 128) not in selections:
        selections.append(("meta", 128))
    selections += [("byte_check", 128), ("emit_byte", 4), ("format_f64", 62)]
    selections += [("raw_meta", width) for width in (8, 16, 32, 62, 128)]
    selections += [(op, width) for op in ("emit", "format") for width in (62, 128)]
    selections += [(op, width) for op in ("emit_first", "emit_last") for width in (62, 128)
                   if width == 62 or width in args.widths]
    for op, width in selections:
        start = time.monotonic()
        stem = f"{op}-{width}"
        prior = None
        reuse_root = args.output
        prior = next((r for r in progress if r["name"] == stem), None)
        if prior is None and args.reuse_bank:
            reuse_root = args.reuse_bank
            provenance = json.loads((args.reuse_bank / "provenance.json").read_text())
            prior = next((r for r in provenance["templates"] if r["name"] == stem), None)
            if prior is not None and provenance.get("compiler_sources") != identity:
                compatible = all(identity[name] == digest for name, digest in legacy["compatible_sources"].items())
                if not compatible or legacy["templates"].get(stem) != prior["packed_sha256"]:
                    raise ValueError(f"unqualified compiler reuse: {stem}")
        if prior is not None:
            packed = reuse_root / f"{stem}.tfgp"
            if hashlib.sha256(packed.read_bytes()).hexdigest() != prior["packed_sha256"]:
                raise ValueError(f"reuse checksum mismatch: {stem}")
            if reuse_root != args.output:
                shutil.copyfile(packed, args.output / packed.name)
            metadata = prior["slots"]
            descriptors.append({"name": op, "width": width, "path": packed.name,
                                "inputs": metadata["input_arity"],
                                "outputs": max(slot["index"] for slot in metadata["output_slots"]) + 1})
            receipts.append(dict(prior, reused=True))
            progress_file.write_text(json.dumps(receipts, separators=(",", ":")) + "\n")
            print(json.dumps({"template": stem, "reused": True}), flush=True)
            continue
        doc, metadata = build_template(args.qwenputer, op, width)
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
        progress_file.write_text(json.dumps(receipts, separators=(",", ":")) + "\n")
        print(json.dumps({"template": stem, "bytes": receipt["packed_bytes"], "stats": stats}), flush=True)
        if not args.keep_ir:
            source.unlink()
            source.with_name(source.name + ".sha256").unlink()
    plans = plan_catalog()
    allowed = {(op, width) for op, width in selections}
    plans = [plan for plan in plans if all(call["op"] == "bind" or (call["op"], call["width"]) in allowed
                                         for call in plan["ops"])]
    plans += catalog(args.widths)
    plans += whole_plans(args.widths)
    manifest = {"version": 1, "templates": descriptors, "plans": plans}
    (args.output / "bank.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    (args.output / "Qwenputer-MIT.txt").write_text(baseline["source"]["license_text"])
    (args.output / "provenance.json").write_text(json.dumps({"source": baseline["source"],
        "format": "tensorfold.runner.graph-bank-provenance", "version": 1,
        "compiler": "offline Qwenputer P2 IR -> T2 prepared graph pack", "request_graph_compilation": False,
        "compiler_sources": identity,
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
    parser.add_argument("--reuse-bank", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.widths = [int(value) for value in args.widths.split(",")]
    if not args.widths or len(set(args.widths)) != len(args.widths) or any(w not in BUCKETS for w in args.widths):
        parser.error("widths must be distinct supported digit buckets")
    export(args)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
