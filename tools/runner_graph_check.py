#!/usr/bin/env python3
"""Check reusable ALU templates against independent arithmetic and upstream numeric contracts."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

from runner_export_check import predict, restore
from runner_graph_ir import build_template
from runner_trace import load_upstream


def bind(doc, values):
    doc = copy.deepcopy(doc)
    nodes = {n["name"]: n for n in doc["nodes"] if n["kind"] == "head"}
    for slot in doc["program"]["input_slots"]:
        value = values[slot["index"]]
        text = str(abs(value))
        digits = [int(d) for d in reversed(text)]
        width = slot["width"]
        if len(digits) > width:
            raise ValueError("input exceeds template width")
        sparse = [[0, 16777216.0], [1, 16777216.0]]
        for i in range(slot["digits"]):
            at = slot["offset"] + i
            digit = digits[at] if at < len(digits) else 0
            sparse += [[2 + 4 * i + bit, 128.0] for bit in range(4) if digit >> bit & 1]
        if slot["offset"] == 0 and value < 0:
            sparse.append([126, 128.0])
        nodes[slot["head"]]["value_heads"][0]["init"] = [[0, sparse]]
    return doc


def outputs(doc, matrix):
    nodes = {n["name"]: n for n in doc["nodes"] if n["kind"] == "head"}
    result = {}
    for slot in doc["program"]["output_slots"]:
        row = {col: float(matrix[-1, dim]) for col, dim, _ in nodes[slot["head"]]["value_heads"][0]["out"]}
        if any(v not in (0, 1) for v in row.values()):
            raise ValueError("nonbinary output tensor")
        value, error = result.get(slot["index"], (0, False))
        for i in range(slot["digits"]):
            digit = sum(int(row[2 + 4 * i + bit]) << bit for bit in range(4))
            if digit > 9:
                raise ValueError("nondecimal output tensor")
            value += digit * 10 ** (slot["offset"] + i)
        if slot["offset"] == 0:
            error = bool(row[127])
        result[slot["index"]] = value, error
    for index in result:
        first = next(s for s in doc["program"]["output_slots"] if s["index"] == index and s["offset"] == 0)
        dim = next(dim for col, dim, _ in nodes[first["head"]]["value_heads"][0]["out"] if col == 126)
        value, error = result[index]
        if value == 0 and matrix[-1, dim]:
            raise ValueError("negative zero result")
        result[index] = (-value if matrix[-1, dim] else value), error
    return [result[i] for i in range(len(result))]


def run(root, widths, rounds):
    load_upstream(root.resolve())
    import numpy as np
    from chipformer.q35 import ir
    from chipformer.rv import spec

    rows, rng = [], random.Random(0)
    for width in widths:
        limit = 10 ** width - 1
        for op in ("add", "sub", "mul", "divstep", "divfinal", "emit"):
            doc, _ = build_template(root, op, width)
            examples = [(0, 0), (1, 1), (-1, 1), (9, -3), (-9, 3), (limit, 1)]
            examples += [(rng.randint(-limit, limit), rng.randint(-limit, limit)) for _ in range(rounds)]
            for a, b in examples:
                if op == "divstep":
                    divisor = max(1, abs(b))
                    remainder = abs(a) % divisor
                    digit = rng.randrange(10)
                    values = [remainder, divisor, digit]
                    want = [(remainder * 10 + digit) % divisor, (remainder * 10 + digit) // divisor]
                elif op == "divfinal":
                    if b == 0:
                        b = 1
                    q, r = divmod(abs(a), abs(b))
                    values, want = [q, r, a, b], [a // b, a % b]
                elif op == "emit":
                    values, want = [a], [a]
                else:
                    values = [a, b]
                    want = [{"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[op]()]
                bound = bind(doc, values)
                if op == "emit":
                    stream = str(a).encode() + b"\n"
                    tokens = [spec.tok("GO")] + [spec.tok("BYTE", byte) for byte in stream]
                    expected_tokens = tokens[1:] + [spec.tok("HALT")]
                else:
                    tokens, expected_tokens = [spec.tok("GO")], [spec.tok("HALT")]
                matrix = ir.evaluate(restore(bound, ir, np), tokens, check=True)
                if predict(bound, matrix) != expected_tokens:
                    raise ValueError(f"{op}/{width} {values}: token mismatch")
                got = outputs(bound, matrix)
                for actual, expected in zip(got, want):
                    overflow = abs(expected) > limit
                    if actual[1] != overflow or not overflow and actual[0] != expected:
                        raise ValueError(f"{op}/{width} {values}: got {got}, expected {want}")
            rows.append({"op": op, "width": width, "cases": len(examples), "stats": doc["stats"]})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", required=True, type=Path)
    parser.add_argument("--widths", default="1,4")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    rows = run(args.qwenputer, [int(v) for v in args.widths.split(",")], args.rounds)
    print(json.dumps({"pass": True, "cases": sum(r["cases"] for r in rows)}))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
