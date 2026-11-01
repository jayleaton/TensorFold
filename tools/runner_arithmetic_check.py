#!/usr/bin/env python3
"""Replay every byte and HALT of an experimental compiled arithmetic circuit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

from runner_arithmetic import build, read, write
from runner_export_check import predict, restore
from runner_trace import load_upstream


def check(doc, root, loop_elf=None):
    (calc, ref, spec, elf), _ = load_upstream(root.resolve())
    import numpy as np
    from chipformer.q35 import ir

    program = doc["program"]
    if program["loop_bound"] is None:
        machine = calc.machine(program["expression"])
        expected = calc.expected(program["expression"]).encode()
    else:
        if loop_elf is None or program["loop_bound"] != 10:
            raise ValueError("loop verification requires the original bound-10 ELF")
        machine = ref.Machine.from_program(elf.program(loop_elf))
        expected = b"285\n"
    reference = machine.run(10**7)
    if not machine.halted or machine.fault is not None or machine.out != expected:
        raise ValueError("upstream reference fault/output mismatch")
    if machine.exit_code != int(expected == b"error\n"):
        raise ValueError("upstream exit mismatch")
    trace = [spec.tok("GO")] + [spec.tok("BYTE", b) for b in machine.out] + [spec.tok("HALT")]
    graph = restore(doc, ir, np)
    start = time.monotonic()
    matrix = ir.evaluate(graph, trace[:-1], check=True)
    seconds = time.monotonic() - start
    got = predict(doc, matrix)
    if got != trace[1:]:
        i = next(i for i, (a, b) in enumerate(zip(got, trace[1:])) if a != b)
        raise ValueError(f"next-token mismatch at {i}: {got[i]} != {trace[i + 1]}")
    # Causal next-token equality for every prefix proves free-run equivalence by induction.
    raw = struct.pack(f"<{len(trace)}I", *trace)
    old_raw = struct.pack(f"<{len(reference)}I", *reference)
    return {"pass": True, "stdout": machine.out.decode(), "generated_steps": len(trace) - 1,
            "baseline_generated_steps": len(reference) - 1, "baseline_instructions": machine.steps,
            "baseline_trace_sha256": hashlib.sha256(old_raw).hexdigest(),
            "step_reduction": (len(reference) - 1) / (len(trace) - 1),
            "trace_sha256": hashlib.sha256(raw).hexdigest(), "stats": doc["stats"],
            "tokens_including_GO": trace,
            "reference_ir_seconds": seconds, "numeric_checks": "upstream BF16/SwiGLU/DeltaNet exactness",
            "token_contract": "BYTE stream and HALT exact; internal RV32 tokens intentionally removed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", type=Path, required=True)
    parser.add_argument("--circuit", type=Path)
    parser.add_argument("--expr")
    parser.add_argument("--loop-bound", type=int)
    parser.add_argument("--loop-elf", type=Path)
    parser.add_argument("--export", type=Path, help="write only after successful in-memory replay")
    args = parser.parse_args()
    if args.circuit:
        if args.expr is not None or args.loop_bound is not None or args.export:
            parser.error("--circuit cannot be combined with build arguments")
        doc = read(args.circuit)
    else:
        if (args.expr is None) == (args.loop_bound is None):
            parser.error("provide --circuit or exactly one of --expr/--loop-bound")
        doc = build(args.qwenputer, args.expr, args.loop_bound)
    result = check(doc, args.qwenputer, args.loop_elf)
    if args.export:
        result["artifact"] = write(doc, args.export)
        result["serialized_replay"] = check(read(args.export), args.qwenputer, args.loop_elf)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
