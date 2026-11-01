#!/usr/bin/env python3
"""Qualify and export step-reduction experiments against pinned Qwenputer reference outputs."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

from runner_arithmetic import build, read, write
from runner_arithmetic_check import check
from runner_circuit import read_circuit
from runner_export import write_circuit
from runner_profile import Symbols, profile
from runner_trace import MULTIPLY, load_upstream


def native_bridge(doc, baseline, path):
    """Wrap a raw graph for State.init-only benchmarking; this is not a Machine.boot program."""
    bridged = dict(doc, format="tensorfold.runner.p2")
    bridged["program"] = copy.deepcopy(baseline["program"])
    bridged["program"].update(kind="elf", expression=None)
    bridged["experimental"] = {"contract": doc["program"], "entry": "State.init; feed GO",
                               "warning": "Benchmark bridge only. Do not use Machine.boot or ELF replacement."}
    return write_circuit(bridged, path)


def run(args):
    (calc, ref, spec, elf), _ = load_upstream(args.qwenputer.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = read_circuit(args.baseline)
    rows = []
    jobs = [("multiply", MULTIPLY, None), ("power", "2**100", None), ("loop", None, 10)]
    for name, expression, bound in jobs:
        start = time.monotonic()
        cpu_start = time.process_time()
        doc = build(args.qwenputer, expression, bound)
        build_seconds = time.monotonic() - start
        build_cpu_seconds = time.process_time() - cpu_start
        before = check(doc, args.qwenputer, args.loop_elf)
        asset = args.output / f"{name}.tfac"
        artifact = write(doc, asset)
        restored = check(read(asset), args.qwenputer, args.loop_elf)
        if {k: v for k, v in before.items() if k != "reference_ir_seconds"} != {
                k: v for k, v in restored.items() if k != "reference_ir_seconds"}:
            raise ValueError("serialized replay differs")
        bridge = native_bridge(doc, baseline, args.output / f"{name}.bench.tfpc")
        (args.output / f"{name}.stdout").write_text(restored["stdout"])
        tokens = restored["tokens_including_GO"]
        (args.output / f"{name}.tokens.leu32").write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
        rows.append(dict(name=name, build_seconds=build_seconds, build_cpu_seconds=build_cpu_seconds,
                         artifact=artifact, bridge=bridge, **restored))
    samples, rejected = [], []
    for i, (operation, expression) in enumerate(calc.cases(args.cases, 0)):
        stem = args.output / f"sample-{i:04d}"
        stem.with_suffix(".expr").write_text(expression)
        try:
            doc = build(args.qwenputer, expression)
        except (ValueError, SyntaxError) as error:
            machine = calc.machine(expression)
            tokens = machine.run(10**7)
            oracle = calc.expected(expression).encode()
            limited = machine.out == b"error (too complex: split it)\n"
            if (not machine.halted or machine.fault is not None or (machine.out != oracle and not limited)
                    or machine.exit_code != int(limited or oracle == b"error\n")):
                raise ValueError(f"unsupported sample {i} reference mismatch")
            raw = struct.pack(f"<{len(tokens)}I", *tokens)
            stem.with_suffix(".stdout").write_bytes(machine.out)
            rejected.append({"index": i, "operation": operation, "expression": expression, "reason": str(error),
                             "outcome": "machine_limit" if limited else "ok",
                             "stdout": machine.out.decode(), "generated_steps": len(tokens) - 1,
                             "trace_sha256": hashlib.sha256(raw).hexdigest()})
            continue
        asset = args.output / f"sample-{i:04d}.tfac"
        result = check(doc, args.qwenputer)
        artifact = write(doc, asset)
        restored = check(read(asset), args.qwenputer)
        stem.with_suffix(".stdout").write_text(restored["stdout"])
        native_bridge(doc, baseline, args.output / f"sample-{i:04d}.bench.tfpc")
        samples.append(dict(index=i, operation=operation, expression=expression, artifact=artifact, **restored))
    boundary = []
    for expression in ("0*0", "0**0", "1**999", "9*9", "999+1", "999*999", "9" * 62 + "+1"):
        doc = build(args.qwenputer, expression)
        boundary.append(dict(expression=expression, **check(doc, args.qwenputer)))
    controls = []
    corrupt = args.output / "corrupt-control.tfac"
    data = bytearray((args.output / "multiply.tfac").read_bytes())
    data[-1] ^= 1
    corrupt.write_bytes(data)
    try:
        read(corrupt)
    except ValueError:
        controls.append({"corrupted_payload": "rejected by checksum"})
    else:
        raise ValueError("corrupt payload was accepted")
    finally:
        corrupt.unlink()
    for expression in ("-1", "1//2", "gcd(2,3)", "0xff", "2**1000", "1.5", "1<<2"):
        try:
            build(args.qwenputer, expression)
        except (ValueError, SyntaxError):
            controls.append({"expression": expression, "rejected": True})
        else:
            raise ValueError(f"unsupported input accepted: {expression}")
    damaged = read(args.output / "multiply.tfac")
    first = next(n for n in damaged["nodes"] if n["kind"] == "head" and n["name"].startswith("operand"))
    # Flip an actual operand bit; an unused matrix entry alone need not affect output.
    sparse = first["value_heads"][0]["init"][0][1]
    at = next((i for i, (col, _) in enumerate(sparse) if col == 2), None)
    if at is None:
        sparse.append([2, 128.0])
    else:
        sparse.pop(at)
    try:
        check(damaged, args.qwenputer)
    except (ValueError, AssertionError):
        controls.append({"mutated_operand_snapshot": "rejected by next-token replay"})
    else:
        raise ValueError("mutated operand was not detected")
    damaged["program"]["expression"] = MULTIPLY.replace("7890*", "7891*")
    controls.append({"same_weights_new_operand": check(damaged, args.qwenputer)})
    firmware = []
    for name, path in (("multiply", args.multiply_elf), ("power", args.power_elf), ("loop-o2", args.loop_o2_elf)):
        machine = ref.Machine.from_program(elf.program(path))
        result = profile(machine, Symbols(path), ref, spec, "elf")
        expected = next(row["stdout"] for row in rows if row["name"] == ("loop" if name == "loop-o2" else name))
        if result["output"] != expected or result["exit_code"] != 0:
            raise ValueError(f"wide-limb firmware failed: {name}")
        firmware.append(dict(name=name, **result))
    report = {"circuits": rows, "sample": samples, "unsupported_sample": rejected,
              "boundary": boundary, "negative_controls": controls, "firmware": firmware}
    (args.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"pass": True, "sample_fast": len(samples), "sample_unsupported": len(rejected),
                      "circuits": [{k: row[k] for k in ("name", "generated_steps", "baseline_generated_steps")}
                                   for row in rows],
                      "firmware": [{"name": row["name"], "steps": row["generated_steps"]} for row in firmware]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qwenputer", "baseline", "loop-elf", "multiply-elf", "power-elf", "loop-o2-elf", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
