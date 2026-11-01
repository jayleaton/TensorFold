#!/usr/bin/env python3
"""Replay a serialized P2 graph against upstream reference tokens and original IR residuals."""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

from runner_circuit import read_circuit
from runner_export import feature_value


def restore(doc, ir, np):
    from chipformer.dsl.graph import Expr

    def expr(e):
        return Expr(dict(e))

    def cs(items):
        return {i: ir.Chan(expr(c["expr"]), tuple(c["taps"])) for i, c in items}

    g = ir.Graph()
    g.dims = [ir.Dim(**d) for d in doc["dims"]]
    families = doc["tokens"]["families"]

    def feature(f, token):
        for i, fam in enumerate(families):
            if fam["base"] <= token < fam["base"] + fam["size"]:
                return feature_value(f, i, token - fam["base"])
        raise ValueError(f"invalid input token: {token}")

    g.token_fns = {f["dim"]: (lambda t, f=f: feature(f, t)) for f in doc["features"]}
    for d, dim in enumerate(g.dims):
        if dim.kind == "token" and d not in g.token_fns:
            g.token_fns[d] = lambda t: 0
    g.full_fns = set(g.token_fns)
    for n in doc["nodes"]:
        if n["kind"] == "ffn":
            g.nodes.append(ir.FFN(n["name"], [(w, expr(a), expr(b)) for w, a, b in n["terms"]],
                                  n["out"], n["level"], n["into"]))
            continue
        vhs, init = [], []
        for vh in n["value_heads"]:
            rows = {}
            for row, sparse in vh["init"]:
                data = np.zeros(128)
                for col, v in sparse:
                    data[col] = v
                rows[row] = data
            init.append(rows)
            vhs.append(ir.VHead(cs(vh["v"]), expr(vh["a"]), {col: expr(e) for col, e in vh["z"]},
                               [tuple(o) for o in vh["out"]],
                               None if vh["beta"] is None else expr(vh["beta"]), vh["into"]))
        first = vhs[0]
        g.nodes.append(ir.Head(name=n["name"], q=cs(n["q"]), k=cs(n["k"]), v=first.v, a=first.a,
                               z=first.z, out=first.out, level=n["level"], n_cols=n["n_cols"],
                               tiny=n["tiny"], beta=first.beta, into=first.into, vx=vhs[1:],
                               init=init, levels=n["levels"]))
    return g


def predict(doc, mat):
    codes = mat[:, doc["output"]["code_dims"]] > 0
    fb = doc["output"]["family_bits"]
    families = doc["output"]["families"]
    out = []
    for row in codes:
        fid = sum(int(v) << i for i, v in enumerate(row[:fb]))
        payload = sum(int(v) << i for i, v in enumerate(row[fb:]))
        if fid >= len(families) or payload >= families[fid]["size"]:
            out.append(-1)
        else:
            out.append(families[fid]["base"] + payload)
    return out


def compare_initial_states(original, restored, ir, np):
    if len(original.nodes) != len(restored.nodes):
        raise ValueError("serialized node count differs")
    zero = np.zeros(128)
    checked = 0
    for a, b in zip(original.nodes, restored.nodes):
        if type(a) is not type(b):
            raise ValueError("serialized node types differ")
        if not isinstance(a, ir.Head):
            continue
        if len(a.vheads()) != len(b.vheads()):
            raise ValueError("serialized value-head count differs")
        for i in range(len(a.vheads())):
            ar = a.init[i] if a.init and i < len(a.init) else {}
            br = b.init[i] if b.init and i < len(b.init) else {}
            for row in range(128):
                av = np.asarray(ar.get(row, zero), dtype=np.float64)
                bv = np.asarray(br.get(row, zero), dtype=np.float64)
                if not np.array_equal(av.view(np.uint64), bv.view(np.uint64)):
                    raise ValueError(f"initial head state differs: {a.name}, value head {i}, row {row}")
            checked += 1
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("circuit", type=Path)
    ap.add_argument("--qwenputer", required=True, type=Path)
    ap.add_argument("--elf", type=Path)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()
    if not 1 <= args.chunk <= 2048:
        ap.error("--chunk must be in [1,2048]")
    sys.path.insert(0, str(args.qwenputer.resolve() / "experimental/riscv"))
    import numpy as np
    from chipformer.q35 import ir
    from chipformer.rv import calc, core, elf, ref

    doc = read_circuit(args.circuit)
    if doc["program"]["kind"] == "calc":
        expr = doc["program"]["expression"] or ""
        m = calc.machine(expr)
        expected = calc.expected(expr)
        path = calc.ELF
    else:
        if args.elf is None:
            ap.error("--elf required to verify an ELF circuit")
        path = args.elf
        m = ref.Machine.from_program(elf.program(path))
        expected = None
    if hashlib.sha256(path.read_bytes()).hexdigest() != doc["program"]["elf_sha256"]:
        raise ValueError("reference ELF checksum differs")
    program = doc["program"]
    images = {"rom_le_hex": struct.pack(f"<{len(m.rom)}I", *m.rom).hex(),
              "ram_hex": struct.pack(f"<{len(m.ram)}I", *m.ram).hex(), "drom_hex": m.drom.hex()}
    if program["entry"] != m.pc or any(program[k] != value for k, value in images.items()):
        raise ValueError("serialized program image differs from reference boot state")
    original = core.build()
    core.snapshot(original, m)
    restored = restore(doc, ir, np)
    initial_matrices = compare_initial_states(original, restored, ir, np)
    toks = m.run(10**7)
    if not m.halted or m.fault is not None:
        raise ValueError("reference did not halt cleanly")
    limited = m.out == b"error (too complex: split it)\n"
    if expected is not None and m.out.decode() != expected and not limited:
        raise ValueError("reference stdout differs from arithmetic oracle")
    if expected is not None and m.exit_code != int(limited or expected == "error\n"):
        raise ValueError("reference exit code differs from firmware semantics")
    carry_a, carry_b = {}, {}
    for start in range(0, len(toks), args.chunk):
        chunk = toks[start:start + args.chunk]
        a = ir.evaluate(original, chunk, carry=carry_a)
        b = ir.evaluate(restored, chunk, carry=carry_b)
        if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            where = np.argwhere(a.view(np.uint64) != b.view(np.uint64))[0]
            raise ValueError(f"serialized residual mismatch at token {start + where[0]}, dim {where[1]}")
        got = predict(doc, b)
        want = toks[start + 1:start + len(chunk) + 1]
        if got[:len(want)] != want:
            j = next(i for i, (x, y) in enumerate(zip(got, want)) if x != y)
            raise ValueError(f"next-token mismatch at {start + j}: {got[j]} != {want[j]}")
    token_bytes = struct.pack(f"<{len(toks)}I", *toks)
    print(json.dumps({"pass": True, "tokens": len(toks), "instructions": m.steps,
                      "exit_code": m.exit_code,
                      "initial_matrices": initial_matrices,
                      "outcome": "machine_limit" if limited else "ok",
                      "stdout": m.out.decode(), "trace_sha256": hashlib.sha256(token_bytes).hexdigest(),
                      "residuals": "all dimensions byte-equal to upstream IR", "chunk": args.chunk}))


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
