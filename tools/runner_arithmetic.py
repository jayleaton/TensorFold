#!/usr/bin/env python3
"""Compile unsigned arithmetic into exact decimal FFNs, with snapshot operand bits."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from runner_export import HEADER, encode_node, features
from runner_trace import load_upstream


MAGIC = b"TFARITH1"
LIMIT = 62


class Compiler:
    def __init__(self, core, ir, np):
        self.c, self.ir, self.np = core.RvCore(), ir, np
        self.errors = []
        self.serial = 0
        self.literal_count = 0
        self.input_slots = []

    def node(self, terms, name):
        self.serial += 1
        return self.c.node(terms, f"ar.{self.serial}.{name}")

    def ge(self, value, threshold, maximum):
        return self.node(self.c.t_ge(value, threshold, 0, maximum), "ge")

    def literal(self, value):
        text = str(value)
        if value < 0 or len(text) > LIMIT:
            raise ValueError("literals must be unsigned integers of at most 62 digits")
        digits = list(map(int, reversed(text)))
        self.literal_count += 1
        result = []
        for offset in range(0, len(digits), 31):
            group = digits[offset:offset + 31]
            name = f"operand{self.literal_count}.{offset}"
            self.input_slots.append({"head": name, "row": 0, "digit_offset": offset, "digits": len(group),
                                     "first_col": 2, "bits_per_digit": 4, "order": "lsd_first"})
            cols = list(range(2, 2 + 4 * len(group)))
            bits = self.c.head(name, {0: 128 * self.c.fam["GO"]}, {}, {},
                               0, {col: 128 * self.c.one for col in cols}, [(col, 128) for col in cols],
                               tiny=2.0**-90)
            row = self.np.zeros(128)
            row[0] = row[1] = self.ir.A_REF
            for i, digit in enumerate(group):
                for bit in range(4):
                    row[2 + 4 * i + bit] = self.ir.A_BIT * ((digit >> bit) & 1)
                result.append(sum((bits[4 * i + bit] * (1 << bit) for bit in range(4)), self.c.one * 0))
            self.c.nodes[-1].init = [{0: row}]
        return result

    def trim(self, digits):
        if len(digits) > LIMIT:
            self.errors.extend(self.ge(d, 1, 9) for d in digits[LIMIT:])
        return digits[:LIMIT]

    def add(self, a, b, trim=True):
        result, carry = [], self.c.one * 0
        for i in range(max(len(a), len(b))):
            value = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) + carry
            carry = self.ge(value, 10, 19)
            result.append(self.node([(1, self.c.one, value - 10 * carry)], "digit"))
        result.append(carry)
        return self.trim(result) if trim else result

    def multiply(self, a, b):
        zero = self.c.one * 0
        rows = []
        for j, bj in enumerate(b):
            row, carry = [zero] * j, zero
            for ai in a:
                product = self.node([(1, ai, bj)], "product")
                tens = self.node([term for k in range(10, 81, 10)
                                  for term in self.c.t_ge(product, k, 0, 81)], "tens")
                ones = product - 10 * tens
                value = ones + carry
                next_carry = self.ge(value, 10, 17)
                row.append(self.node([(1, self.c.one, value - 10 * next_carry)], "digit"))
                carry = tens + next_carry
            row.append(carry)
            rows.append(row)
        while len(rows) > 1:
            rows = [self.add(rows[i], rows[i + 1], trim=False) if i + 1 < len(rows) else rows[i]
                    for i in range(0, len(rows), 2)]
        return self.trim(rows[0])

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return self.literal(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.expression(node.operand)
        if not isinstance(node, ast.BinOp):
            raise ValueError("supports unsigned decimal literals, +, *, and literal **")
        a = self.expression(node.left)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int:
                raise ValueError("power requires a literal exponent")
            exponent = node.right.value
            if not 0 <= exponent <= 999:
                raise ValueError("exponent must be in [0,999]")
            result = None
            while exponent:
                if exponent & 1:
                    result = a if result is None else self.multiply(result, a)
                exponent >>= 1
                if exponent:
                    a = self.multiply(a, a)
            return [self.c.one] if result is None else result
        b = self.expression(node.right)
        if isinstance(node.op, ast.Add):
            return self.add(a, b)
        if isinstance(node.op, ast.Mult):
            return self.multiply(a, b)
        raise ValueError("unsupported arithmetic operator")

    def output(self, digits, core):
        from chipformer.q35.prims import thead

        c, one = self.c, self.c.one
        flags = self.errors[:]
        while len(flags) > 1:
            flags = [self.ge(flags[i] + flags[i + 1], 1, 2) if i + 1 < len(flags) else flags[i]
                     for i in range(0, len(flags), 2)]
        error = flags[0] if flags else one * 0
        stored = []
        for offset in range(0, len(digits), 31):
            group = digits[offset:offset + 31]
            data = {}
            for i, digit in enumerate(group):
                for bit, terms in enumerate(c.fn_bits(digit, 0, 9, lambda d: d, 4, "result")):
                    data[2 + 4 * i + bit] = self.node(terms, "result-bit")
            data[126] = error
            (bits,) = thead(c, f"result{offset}", {0: c.fam["GO"]}, {0: one},
                            [{"data": data, "ref": one}])
            for i in range(len(group)):
                stored.append(sum((bits[4 * i + bit] * (1 << bit) for bit in range(4)), one * 0))
            if offset == 0:
                stored_error = bits[-1]
        digits, error = stored, stored_error
        occupied = [self.ge(d, 1, 9) for d in digits]
        high, length = one * 0, one
        for i in reversed(range(1, len(digits))):
            highest = self.node([(1, occupied[i], one - high)], "highest")
            length += i * highest
            high += highest
        length = self.node([(1, one, length)], "length")
        count = c.counter("output.cursor", c.fam["GO"], c.fam["BYTE"])
        thermo = c.thermo(count, c.fam["GO"], LIMIT + 3, "output.cursor")
        n = sum(thermo, one * 0)
        index = length - 1 - n
        selected = []
        for i, digit in enumerate(digits):
            eq = self.node(c.t_eq(index, i, -LIMIT - 3, LIMIT), "position")
            selected.append((1, digit, eq))
        digit = self.node(selected, "selected")
        newline = self.node(c.t_eq(n - length, 0, -LIMIT, LIMIT + 3), "newline")
        done = self.node(c.t_ge(n - length, 1, -LIMIT, LIMIT + 3), "done")
        digit_bits = [self.node(ts, "ascii") for ts in c.fn_bits(digit, 0, 9, lambda d: 48 + d, 8, "ascii")]
        err_done = self.ge(n, 6, LIMIT + 3)
        error_bits = []
        for bit in range(8):
            terms = []
            for i, byte in enumerate(b"error\n"):
                if byte >> bit & 1:
                    terms += c.t_eq(n, i, 0, LIMIT + 3)
            error_bits.append(self.node(terms, "error-byte"))
        halt = self.node([(1, one - error, done), (1, error, err_done)], "halt")
        payload = []
        for bit, value in enumerate(digit_bits):
            normal = self.node([(1, one - newline, value), ((10 >> bit) & 1, one, newline)], "byte")
            payload.append(self.node([(1, one - error, normal), (1, error, error_bits[bit])], "byte"))
        family = [((core.FAM_ID["BYTE"] >> i) & 1) * (one - halt)
                  + ((core.FAM_ID["HALT"] >> i) & 1) * halt for i in range(core.NFAM)]
        payload = [self.node([(1, one - halt, bit)], "halt-payload") for bit in payload]
        bits = family + payload + [one * 0] * (core.NPAY - 8)
        exit_level = max(node.level for node in c.nodes)
        for i, (code, bit) in enumerate(zip(c.code, bits)):
            c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)
            c.ffn(c.t_const(-core.KAPPA) + [(2 * core.KAPPA, one, bit)],
                  f"exit.code{i}", into=code, min_level=exit_level)


def build(root, expression=None, loop_bound=None):
    root = root.resolve()
    load_upstream(root.resolve())
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip()
    if dirty:
        raise ValueError("Qwenputer tracked files have local modifications")
    import numpy as np
    from chipformer.q35 import ir
    from chipformer.rv import core, spec

    compiler = Compiler(core, ir, np)
    if loop_bound is None:
        if len(expression) > 255:
            raise ValueError("expression exceeds 255 characters")
        if not re.fullmatch(r"[0-9+*()\s]+", expression):
            raise ValueError("only decimal unsigned arithmetic is supported")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("expression exceeds 128 AST nodes")
        digits = compiler.expression(tree.body)
    else:
        if not 0 <= loop_bound <= 32:
            raise ValueError("loop bound must be in [0,32]")
        digits = [compiler.c.one * 0]
        for i in range(loop_bound):
            operand = compiler.literal(i)
            digits = compiler.add(digits, compiler.multiply(operand, operand))
    compiler.output(digits, core)
    c = compiler.c
    # Only this new single-assignment graph is scheduled by dependency level; original P2 stays in upstream order.
    c.nodes.sort(key=lambda node: node.level)
    families = [{"name": name, "base": int(spec.FAMILIES[name][0]), "size": int(spec.FAMILIES[name][1])}
                if name in spec.FAMILIES else {"name": name, "base": spec.CONTROLS[name], "size": 1}
                for name in core.OUT_NAMES[:-1]]
    return {"format": "tensorfold.runner.arithmetic", "version": 1,
            "source": {"repository": "https://github.com/divagr18/qwenputer", "commit":
                       "bec1a0c9654ac8151b6f36ed0154a19c53141384", "license": "MIT",
                       "credit": "Divyansh Agrawal", "license_text": (root / "LICENSE").read_text()},
            "stats": c.stats(), "constants": {"one_dim": 0, "head_dim": ir.HD, "a_ref": float(ir.A_REF),
                                               "a_bit": float(ir.A_BIT), "ref_cols": list(ir.REF_COLS),
                                               "swiglu_k": float(ir.K), "kappa": float(core.KAPPA)},
            "dims": [{"name": d.name, "kind": d.kind, "level": d.level} for d in c.dims],
            "features": features(c, ir, core, spec), "nodes": [encode_node(n, ir) for n in c.nodes],
            "tokens": {"vocab": spec.VOCAB, "families": families, "go": spec.tok("GO"),
                       "halt": spec.tok("HALT"), "byte_family": core.FAM_ID["BYTE"]},
            "output": {"family_bits": core.NFAM, "payload_bits": core.NPAY, "families": families,
                       "code_dims": [next(iter(e.terms)) for e in c.code]},
            "program": {"kind": "compiled_arithmetic", "expression": expression, "loop_bound": loop_bound,
                        "input_slots": compiler.input_slots,
                        "max_digits": LIMIT, "input": "immutable operand-bit DeltaNet snapshots",
                        "result_latch": "GO writes result bits; BYTE positions read the retained DeltaNet rows",
                        "trace": "GO, BYTE stdout bytes, HALT; no RV32 internal tokens"}}


def write(doc, path):
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    data = HEADER.pack(MAGIC, 1, HEADER.size, len(payload), 0, 0, hashlib.sha256(payload).digest()) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return {"sha256": digest, "bytes": len(data), "stats": doc["stats"]}


def read(path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated arithmetic header")
    magic, version, size, length, flags, reserved, checksum = HEADER.unpack(data[:HEADER.size])
    payload = data[HEADER.size:]
    if (magic, version, size, flags, reserved) != (MAGIC, 1, HEADER.size, 0, 0):
        raise ValueError("unsupported experimental arithmetic container")
    if len(payload) != length or hashlib.sha256(payload).digest() != checksum:
        raise ValueError("arithmetic length/checksum mismatch")
    doc = json.loads(payload)
    if doc["format"] != "tensorfold.runner.arithmetic" or doc["version"] != 1:
        raise ValueError("unsupported arithmetic payload")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwenputer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expr")
    group.add_argument("--loop-bound", type=int)
    args = parser.parse_args()
    print(json.dumps(write(build(args.qwenputer, args.expr, args.loop_bound), args.output), sort_keys=True))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
