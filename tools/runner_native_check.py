#!/usr/bin/env python3
"""Match native CPU traces and timings to serialized experimental and baseline circuits."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess


def run(binary, circuit, stdout, expected, expression=None, repeats=5):
    result = subprocess.run([str(binary.resolve()), str(circuit), str(stdout), expression or "-", str(repeats)],
                            text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in result.stderr.splitlines() if line]
    if len(rows) != repeats:
        raise ValueError("native receipt count mismatch")
    for row in rows:
        if not row["pass"] or row["steps"] != expected["generated_steps"]:
            raise ValueError("native step/output mismatch")
        if row["trace_sha256"] != expected["trace_sha256"]:
            raise ValueError("native full token trace differs from reference")
    return {"runs": rows, "load_ns": rows[0]["load_ns"], "load_cpu_ns": rows[0]["load_cpu_ns"],
            "median_cpu_ns": statistics.median(row["cpu_ns"] for row in rows),
            "median_ns": statistics.median(row["run_ns"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--baseline-multiply", type=Path, required=True)
    parser.add_argument("--baseline-loop", type=Path, required=True)
    parser.add_argument("--baseline-calc", type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory
    report = json.loads((folder / "report.json").read_text())
    main_rows = []
    for row in report["circuits"]:
        name = row["name"]
        optimized = run(args.benchmark, folder / f"{name}.bench.tfpc", folder / f"{name}.stdout", row)
        baseline_path = {"multiply": args.baseline_multiply, "loop": args.baseline_loop,
                         "power": folder / "power-baseline.tfpc"}[name]
        expected = {"generated_steps": row["baseline_generated_steps"],
                    "trace_sha256": row["baseline_trace_sha256"]}
        baseline = run(args.benchmark, baseline_path, folder / f"{name}.stdout", expected)
        main_rows.append({"name": name, "optimized": optimized, "baseline": baseline,
                          "median_run_speedup": baseline["median_cpu_ns"] / optimized["median_cpu_ns"],
                          "median_wall_speedup": baseline["median_ns"] / optimized["median_ns"]})
    samples = []
    for fast, rows in ((True, report["sample"]), (False, report["unsupported_sample"])):
        for row in rows:
            stem = f"sample-{row['index']:04d}"
            circuit = folder / f"{stem}.bench.tfpc" if fast else args.baseline_calc
            result = run(args.benchmark, circuit, folder / f"{stem}.stdout", row,
                         expression=None if fast else row["expression"], repeats=1)
            samples.append({"index": row["index"], "path": "compiled" if fast else "original_calc", **result})
    firmware = []
    for row in report["firmware"]:
        name = row["name"]
        stem = "loop-o2" if name == "loop-o2" else "wide-" + name
        output = "loop" if name == "loop-o2" else name
        result = run(args.benchmark, folder / f"{stem}.tfpc", folder / f"{output}.stdout", row, repeats=1)
        firmware.append(dict(name=name, **result))
    result = {"pass": True, "main": main_rows, "sample": samples, "firmware": firmware,
              "timing_scope": "node01 nice19; CPU clock excludes QUIET pauses; run excludes loader/boot; "
                              "five repeats, wall times retained; no GPU claim"}
    (folder / "native-report.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"pass": True, "main": [{"name": row["name"], "speedup": row["median_run_speedup"]}
                                             for row in main_rows], "sample_count": len(samples)}))


if __name__ == "__main__":
    main()
