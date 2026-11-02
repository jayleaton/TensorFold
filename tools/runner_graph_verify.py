#!/usr/bin/env python3
"""Qualify one native graph process against frozen outputs, actual token evidence and negative controls."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time


LAYOUT = (("V", 32768), ("F0", 32768), ("F1", 32768), ("F2", 4),
          ("J", 65536), ("R", 65536), ("BYTE", 512), ("FLT", 8), ("HEX", 16),
          ("QS", 16), ("QD", 128), ("BD", 256), ("NT", 1), ("ST0", 1),
          ("ST1", 1), ("SS0", 1), ("SS1", 1), ("HALT", 1), ("GO", 1), ("SC", 1))
BYTE = sum(size for _, size in LAYOUT[:6])
HALT = sum(size for _, size in LAYOUT[:17])
RUNTIME_FILES = ("runner_graph_bind.zig", "runner_graph_frontend.zig", "runner_graph.zig", "runner_graph_bank.zig", "runner_graph_machine.zig",
                 "runner_graph_binary.zig", "runner_graph_packet.zig", "runner_graph_algorithms.zig",
                 "runner_graph_state_cache.zig", "runner_graph_print.zig",
                 "runner_graph_factor.zig", "runner_graph_lazy.zig", "runner_graph_f64.zig",
                 "runner_graph_f64_context.zig", "runner_graph_f64_math.zig", "runner_graph_f64_print.zig")


def require(condition, message):
    """Fail even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def read_rows(path):
    """Read plain or compressed JSONL with unique string case IDs."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows),
            f"invalid case IDs in {path.name}")
    require(len({row["id"] for row in rows}) == len(rows), f"duplicate case IDs in {path.name}")
    return rows


def digest_file(path):
    """Hash an existing artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def projection(output):
    """Encode frozen UTF-8 BYTE IDs and final HALT without evaluating arithmetic."""
    return b"".join(struct.pack("<I", BYTE + byte) for byte in output.encode("utf-8")) + struct.pack("<I", HALT)


def structural_audit(source_root, probe_source, manifest, cwd):
    """Record the prepared-only entrypoints and reject direct compiler or JSON circuit-loader calls."""
    paths = [probe_source, *(source_root / name for name in RUNTIME_FILES)]
    sources = {path.name: path.read_text(encoding="utf-8") for path in paths}
    forbidden = ('@import("runner_circuit_load.zig")', '@import("runner_graph_pack.zig")',
                 '@import("runner_circuit.zig")', 'Circuit.load(', 'runner_graph_ir',
                 'runner_arithmetic', 'std.process.spawn', 'std.process.run')
    for name, source in sources.items():
        code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
        require(not any(word in code for word in forbidden), f"runtime compilation path in {name}")
    probe = sources[probe_source.name]
    require(probe.count("graph.Bank.loadWithMode(") == 1, "probe must load its bank exactly once")
    require("m.trace = &trace" in probe and "graph.Machine.boot(" in probe, "missing actual machine trace hook")
    bank = sources["runner_graph_bank.zig"]
    require("binary.loadPinned(" in bank, "bank must load offline prepared packs")
    templates = manifest.get("templates", [])
    require(manifest.get("version") == 1 and templates, "manifest needs prepared templates")
    packs = []
    for template in templates:
        path = cwd / template["path"]
        require(path.suffix == ".tfgp", "runtime template is not an offline prepared pack")
        packs.append({"name": template["name"], "width": template["width"], "sha256": digest_file(path)})
    frozen = {f"steps_v1/runner_circuit_{name}.zig": digest_file(source_root / "steps_v1" / f"runner_circuit_{name}.zig")
              for name in ("model", "load", "state", "eval")}
    return {"pass": True, "sources": {path.name: digest_file(path) for path in paths}, "packs": packs,
            "pinned_evaluator_sources": frozen,
            "scope": "direct runtime entrypoints; model loader methods are unused, not dynamically instrumented"}


def prepare_cases(cases, goldens, manifest, allow_generic):
    """Use the old machine's actual final_output, rejecting caller-supplied replacement answers."""
    old = {row["id"]: row for row in goldens}
    plans = {plan["name"]: plan for plan in manifest.get("plans", [])}
    prepared = []
    for row in cases:
        require(row["id"] in old, f"missing frozen case {row['id']}")
        expected = old[row["id"]].get("final_output")
        require(isinstance(expected, str), f"missing frozen final_output for {row['id']}")
        require(row.get("expected_output") == expected, f"replacement expected output for {row['id']}")
        require(row.get("expected_status", "halted") == "halted" and not row.get("expect_mismatch", False),
                f"production case cannot waive a mismatch: {row['id']}")
        if "final_output_hex" in old[row["id"]]:
            require(expected.encode("utf-8").hex() == old[row["id"]]["final_output_hex"],
                    f"inconsistent frozen UTF-8 for {row['id']}")
        packet = row.get("packet")
        require(isinstance(packet, dict) and packet.get("version") == 1, f"invalid packet for {row['id']}")
        values = packet.get("values")
        require(isinstance(values, list) and all(isinstance(value, str) for value in values),
                f"invalid literal values for {row['id']}")
        if packet.get("plan") is not None:
            require(not packet.get("ops", []) and packet.get("result", 0) == 0,
                    f"named packet overrides boot plan: {row['id']}")
            require(packet["plan"] in plans, f"unloaded named plan: {row['id']}")
            require(len(plans[packet["plan"]]["ops"]) <= 4096, f"plan exceeds operation cap: {row['id']}")
        else:
            require(allow_generic, f"main corpus requires named packets: {row['id']}")
            require(len(packet.get("ops", [])) <= 4096, f"packet exceeds operation cap: {row['id']}")
        prepared.append({"id": row["id"], "packet": packet, "expected_output": expected})
    require(prepared, "empty corpus")
    return prepared


def negative_controls(cases, manifest):
    """Corrupt a real operand and inject invalid packets while keeping the golden answer unchanged."""
    plans = {plan["name"]: plan for plan in manifest.get("plans", [])}
    candidates = []
    for row in cases:
        plan = plans.get(row["packet"].get("plan"), {})
        ops = plan.get("ops", [])
        if (len(ops) == 1 and ops[0].get("op") == "add" and plan.get("inputs") == 2
                and row["expected_output"] != "error\n" and row["packet"]["values"]):
            candidates.append(row)
    require(candidates, "negative operand control needs one successful named addition case")
    seed = candidates[0]
    bad_operand = copy.deepcopy(seed["packet"])
    text = bad_operand["values"][0]
    require(text and text[-1] in "0123456789", "control seed is not signed decimal")
    bad_operand["values"][0] = text[:-1] + ("1" if text[-1] == "0" else "0")
    bad_literal = copy.deepcopy(seed["packet"])
    bad_literal["values"][0] = "0" * 257
    override = copy.deepcopy(seed["packet"])
    override["ops"] = [{"op": "add", "width": 4, "a": 0, "b": 1}]
    missing = "__graph_verify_missing_plan__"
    require(missing not in plans, "negative control plan unexpectedly exists")
    controls = [
        {"id": "control::operand-mutation", "packet": bad_operand, "expected_output": seed["expected_output"],
         "expect_mismatch": True},
        {"id": "control::unknown-plan", "packet": {"version": 1, "plan": missing, "values": []},
         "expected_status": "error:GraphPlan", "expected_output": ""},
        {"id": "control::literal-width", "packet": bad_literal,
         "expected_status": "error:GraphLiteral", "expected_output": ""},
        {"id": "control::template-width", "packet": {"version": 1, "values": ["1", "1"],
         "ops": [{"op": "add", "width": 63, "a": 0, "b": 1}], "result": 2},
         "expected_status": "error:GraphTemplate", "expected_output": ""},
        {"id": "control::named-override", "packet": override,
         "expected_status": "error:GraphNamedPlanOverride", "expected_output": ""},
    ]
    require(not ({row["id"] for row in cases} & {row["id"] for row in controls}), "reserved control case ID")
    return controls


def check_report(report, expected, returncode, boot_limit):
    """Check every reported case and independently hash the frozen visible projection."""
    require(isinstance(report, dict), "native summary must be one JSON object")
    require(returncode == 0 and report.get("pass") is True, "native process reported failure")
    require(0 < report["boot_wall_ns"] < boot_limit, "cold-process bank boot must be below one second")
    require(report["boot_process_cpu_ns"] >= 0, "invalid boot CPU timing")
    require(report["case_count"] == len(expected) and report["failed_cases"] == 0, "case coverage mismatch")
    named = sum(case["packet"].get("plan") is not None for case in expected)
    require(report["named_requests"] == named and report["generic_requests"] == len(expected) - named,
            "named/generic request accounting differs")
    require(report["request_compiled_graphs"] == 0, "request compiled a graph")
    require(report["lifecycle"]["pass"] is True and report["lifecycle"]["checks"] == 9,
            "backend lifecycle checks did not all pass")
    rows = report["cases"]
    require([row["id"] for row in rows] == [case["id"] for case in expected], "case IDs/order differ")
    aggregate = hashlib.sha256()
    for row, case in zip(rows, expected):
        require(row["pass"] is True, f"native case failed: {case['id']}")
        if case.get("expect_mismatch", False):
            require(row["detected_mismatch"] and row["trace_valid"] and row["status"] == "halted",
                    "operand mutation did not fail its intended output gate")
            require(row["actual_output"] != case["expected_output"], "operand mutation retained golden output")
            require(row["expected_canonical_sha256"] != row["actual_canonical_sha256"],
                    "operand mutation did not change actual token hash")
        elif case.get("expected_status", "halted").startswith("error:"):
            require(row["status"] == case["expected_status"] and row["steps"] == 0,
                    f"invalid packet was not refused before execution: {case['id']}")
            continue
        else:
            require(row["actual_output"] == case["expected_output"], f"stdout differs: {case['id']}")
            require(row["status"] == "halted" and row["trace_valid"], f"invalid actual trace: {case['id']}")
        raw = projection(row["actual_output"])
        digest = hashlib.sha256(raw).hexdigest()
        require(row["actual_canonical_sha256"] == digest, f"actual token projection differs: {case['id']}")
        if not case.get("expect_mismatch", False):
            require(row["expected_canonical_sha256"] == digest, f"frozen token projection differs: {case['id']}")
        require(row["steps"] == row["raw_tokens"] == row["alu_halts"] + len(raw) // 4,
                f"HALT/step accounting differs: {case['id']}")
        require(len(row["raw_trace_sha256"]) == 64, f"missing raw audit digest: {case['id']}")
        aggregate.update(raw)
    require(aggregate.hexdigest() == report["actual_canonical_sha256"], "aggregate projection hash differs")
    require(sum(row["steps"] for row in rows) == report["steps"], "aggregate step count differs")
    require(len(report["raw_audit_trace_sha256"]) == 64, "missing aggregate raw trace hash")


def pack_controls(binary, manifest_path, cases, cwd):
    """Reject corrupted payloads, stale schemas and correctly checksummed wrong-width template substitutions."""
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["templates"]
    target = next(row for row in rows if row["name"] == "add" and row["width"] == 4)
    small = next(row for row in rows if row["name"] == "add" and row["width"] == 1)
    original = (manifest_path.parent / target["path"]).read_bytes()
    results = []
    with tempfile.TemporaryDirectory(prefix="runner-graph-pack-controls-") as temporary:
        root = Path(temporary)
        for name, expected, offset in (("checksum", "GraphPackChecksum", -1),
                                       ("schema", "GraphPackSchema", 16),
                                       ("identity", "GraphPackIdentity", None)):
            altered = copy.deepcopy(manifest)
            for row in altered["templates"]:
                row["path"] = str((manifest_path.parent / row["path"]).resolve())
            victim = next(row for row in altered["templates"] if row["name"] == "add" and row["width"] == 4)
            if offset is None:
                victim["path"] = str((manifest_path.parent / small["path"]).resolve())
            else:
                damaged = bytearray(original)
                damaged[offset] ^= 1
                path = root / "damaged.tfgp"
                path.write_bytes(damaged)
                victim["path"] = str(path)
            path = root / "bank.json"
            path.write_text(json.dumps(altered))
            child = subprocess.run([str(binary), str(path), str(cases), "--eager-bank"], cwd=cwd,
                                   capture_output=True, text=True, timeout=60, check=False)
            failure = json.loads(child.stdout)
            require(child.returncode != 0 and failure.get("fatal_error") == expected,
                    f"pack control {name} failed its intended boot gate: {child.stdout[:200]}")
            results.append({"name": name, "pass": True, "error": expected})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--goldens", type=Path, required=True)
    parser.add_argument("--runtime-source-root", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, default=Path(__file__).with_name("runner-graph-probe.zig"))
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-generic-ssa", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=7200)
    args = parser.parse_args()
    require(args.start >= 0 and (args.count is None or args.count > 0), "invalid corpus slice")
    require(args.budget > 0 and args.timeout > 0, "invalid budget or timeout")
    cwd = args.cwd.resolve()
    binary, manifest_path = args.probe.resolve(), args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = structural_audit(args.runtime_source_root.resolve(), args.probe_source.resolve(), manifest, manifest_path.parent)
    original = read_rows(args.cases)
    selected = original[args.start:None if args.count is None else args.start + args.count]
    prepared = prepare_cases(selected, read_rows(args.goldens), manifest, args.allow_generic_ssa)
    controls = negative_controls(prepared, manifest)
    expected = prepared + controls
    receipt = {"pass": False, "production_cases": len(prepared), "negative_controls": len(controls),
               "corpus_slice": {"start": args.start, "count": len(prepared), "total": len(original)},
               "structural_audit": audit, "probe_sha256": digest_file(binary),
               "manifest_sha256": digest_file(manifest_path), "cases_sha256": digest_file(args.cases),
               "goldens_sha256": digest_file(args.goldens), "request_processes": 1}
    with tempfile.TemporaryDirectory(prefix="runner-graph-verify-") as temporary:
        packets = Path(temporary) / "cases.jsonl"
        packets.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in expected), encoding="utf-8")
        command = [str(binary), str(manifest_path), str(packets), "--budget", str(args.budget), "--lifecycle"]
        began = time.monotonic_ns()
        child = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, timeout=args.timeout, check=False)
        receipt["process_wall_ns"] = time.monotonic_ns() - began
        receipt["pack_controls"] = pack_controls(binary, manifest_path, packets, cwd)
    receipt["returncode"] = child.returncode
    try:
        report = json.loads(child.stdout)
        receipt["native"] = report
        check_report(report, expected, child.returncode, 1_000_000_000)
        receipt["pass"] = True
    except (ValueError, KeyError, TypeError) as exc:
        receipt["error"] = str(exc)
        if "native" not in receipt:
            receipt["native_stdout"] = child.stdout
    args.report.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": receipt["pass"], "production_cases": len(prepared), "negative_controls": len(controls),
                      "report": str(args.report)}))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"pass": False, "error": str(error)}))
        sys.exit(1)
