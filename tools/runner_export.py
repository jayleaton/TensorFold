#!/usr/bin/env python3
"""Export Qwenputer's compiled P2 circuit and program snapshot for TensorFold's runner."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

from runner_isolate import build_isolatable, isolate_blocks

MAGIC = b"TFIMR001"
HEADER = struct.Struct("<8sIIQII32s")
UPSTREAM = "bec1a0c9654ac8151b6f36ed0154a19c53141384"


def expression(e):
    return [[int(d), float(w)] for d, w in e.terms.items()]


def channel(c):
    return {"expr": expression(c.expr), "taps": [float(w) for w in c.taps]}


def channels(cs):
    return [[int(i), channel(c)] for i, c in cs.items()]


def feature_value(f, family, payload):
    kind = f["kind"]
    if kind == "family":
        return float(family == f["family"])
    if kind == "payload_bit":
        return float((payload >> f["bit"] & 1) if family == f["family"] else 0)
    if kind == "next_digit":
        k = f["digit"]
        return float((family == f["qs_family"] and payload == k) or
                     (family == f["qd_family"] and payload >> 4 == k - 1))
    if kind == "code_bit":
        code = family | payload << f["family_bits"]
        return f["amplitude"] * (2 * (code >> f["bit"] & 1) - 1)
    raise ValueError(f"unsupported feature: {kind}")


def features(c, ir, core, spec):
    out = []
    names = core.OUT_NAMES[:-1]
    by_name = {name: i for i, name in enumerate(names)}
    samples = []
    for name in names:
        base, size = spec.FAMILIES.get(name, (spec.CONTROLS.get(name), 1))
        payloads = {0, size - 1, size // 2}
        payloads.update(1 << b for b in range((size - 1).bit_length()))
        payloads.update((1 << b) - 1 for b in range((size - 1).bit_length() + 1))
        samples += [(base + p, by_name[name], p) for p in sorted(payloads) if p < size]
    for d in c.token_fns:
        name = c.dims[d].name
        f = {"dim": d}
        if name.startswith("is") and name[2:] in by_name:
            f.update(kind="family", family=by_name[name[2:]])
        elif (m := re.fullmatch(r"(\w+)\.b(\d+)", name)) and m[1] in by_name:
            f.update(kind="payload_bit", family=by_name[m[1]], bit=int(m[2]))
        elif name.startswith("digit=="):
            f.update(kind="next_digit", digit=int(name[7:]), qs_family=by_name["QS"],
                     qd_family=by_name["QD"])
        elif re.fullmatch(r"code\d+", name):
            f.update(kind="code_bit", bit=int(name[4:]), family_bits=core.NFAM,
                     amplitude=float(core.CODE_AMP))
        else:
            if any(float(c.feature_value(d, t)) != 0 for t, _, _ in samples):
                raise ValueError(f"unknown nonzero input feature: {name}")
            continue
        for t, family, payload in samples:
            if feature_value(f, family, payload) != float(c.feature_value(d, t)):
                raise ValueError(f"input encoding differs: {name}, token {t}")
        out.append(f)
    return out


def encode_node(n, ir):
    common = {"name": n.name, "level": n.level}
    if isinstance(n, ir.FFN):
        return dict(common, kind="ffn", out=n.out, into=n.into,
                    terms=[[float(w), expression(a), expression(b)] for w, a, b in n.terms])
    heads = []
    for i, vh in enumerate(n.vheads()):
        init = (n.init[i] if n.init and i < len(n.init) else {})
        rows = [[int(r), [[j, float(v)] for j, v in enumerate(row) if v != 0]]
                for r, row in sorted(init.items()) if any(row)]
        heads.append({"v": channels(vh.v), "a": expression(vh.a),
                      "beta": None if vh.beta is None else expression(vh.beta),
                      "z": [[int(col), expression(e)] for col, e in vh.z.items()],
                      "out": [[int(col), int(d), float(w)] for col, d, w in vh.out],
                      "into": vh.into, "init": rows})
    return dict(common, kind="head", n_cols=n.n_cols, levels=n.levels, tiny=float(n.tiny),
                q=channels(n.q), k=channels(n.k), value_heads=heads)


def check_elf(path, spec):
    data = path.read_bytes()
    if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("expected an ELF32 little-endian image")
    machine = struct.unpack_from("<H", data, 18)[0]
    if machine != 243 or struct.unpack_from("<H", data, 16)[0] != 2:
        raise ValueError("expected a static RISC-V executable")
    if struct.unpack_from("<I", data, 36)[0] & 15:
        raise ValueError("P2 requires ILP32 and uncompressed instructions")
    entry, phoff = struct.unpack_from("<II", data, 24)
    phentsize, phnum = struct.unpack_from("<HH", data, 42)
    if phentsize != 32 or phoff + phnum * phentsize > len(data):
        raise ValueError("invalid ELF program headers")
    ranges = []
    for i in range(phnum):
        typ, off, addr, _, filesz, memsz, flags, _ = struct.unpack_from("<8I", data, phoff + 32 * i)
        if typ in (2, 3):
            raise ValueError("dynamic ELF images are unsupported")
        if typ != 1:
            continue
        if filesz > memsz or off + filesz > len(data):
            raise ValueError("invalid ELF segment sizes")
        if not memsz:
            continue
        regions = ((spec.ROM_BASE, 4 * spec.ROM_WORDS),
                   (spec.DROM_BASE, 4 * spec.DROM_WORDS), (spec.RAM_BASE, 4 * spec.RAM_WORDS))
        if not any(base <= addr and addr + memsz <= base + size for base, size in regions):
            raise ValueError("ELF segment outside P2 memory")
        if any(addr < end and start < addr + memsz for start, end in ranges):
            raise ValueError("overlapping ELF segments")
        ranges.append((addr, addr + memsz))
        if spec.ROM_BASE <= addr < spec.ROM_BASE + 4 * spec.ROM_WORDS and (addr % 4 or filesz % 4):
            raise ValueError("unaligned ROM segment")
        if flags & 1 and not spec.ROM_BASE <= addr < spec.ROM_BASE + 4 * spec.ROM_WORDS:
            raise ValueError("executable segment outside ROM")
    if entry % 4 or not any(a <= entry < b for a, b in ranges) or not (
            spec.ROM_BASE <= entry < spec.ROM_BASE + 4 * spec.ROM_WORDS):
        raise ValueError("entry must address an aligned ROM instruction")


def build_document(qwenputer, elf_path=None, expr=None):
    root = qwenputer.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    expected = subprocess.check_output(["git", "-C", str(root), "rev-parse", UPSTREAM], text=True).strip()
    if revision != expected:
        raise ValueError(f"Qwenputer must be at {UPSTREAM}, found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip()
    if dirty:
        raise ValueError("Qwenputer tracked files have local modifications")
    sys.path.insert(0, str(root / "experimental" / "riscv"))
    from chipformer.q35 import ir
    from chipformer.rv import calc, core, elf, ref
    from chipformer.rv import spec as S

    path = Path(elf_path) if elf_path else calc.ELF
    if elf_path and expr is not None:
        raise ValueError("--expr is only valid for calc firmware")
    check_elf(path, S)
    m = ref.Machine.from_program(elf.program(path))
    if elf_path is None:
        text = (expr or "").encode("ascii")
        if len(text) + 1 > 256 or b"\0" in text:
            raise ValueError("calc expression must be NUL-free and at most 255 ASCII bytes")
        for i, b in enumerate(text + b"\0"):
            addr = calc.CALC_IN + i
            wi, shift = addr >> 2, 8 * (addr & 3)
            m.ram[wi] = (m.ram[wi] & ~(255 << shift)) | b << shift
    c = build_isolatable(core, ir)
    core.snapshot(c, m)
    isolate_blocks(c, ir)
    families = [{"name": n, "base": int(S.FAMILIES[n][0]), "size": int(S.FAMILIES[n][1])}
                if n in S.FAMILIES else {"name": n, "base": S.CONTROLS[n], "size": 1}
                for n in core.OUT_NAMES[:-1]]
    return {
        "format": "tensorfold.runner.p2", "version": 1,
        "source": {"repository": "https://github.com/divagr18/qwenputer", "commit": revision,
                   "license": "MIT", "credit": "Divyansh Agrawal",
                   "license_text": (root / "LICENSE").read_text(encoding="utf-8")},
        "stats": c.stats(),
        "constants": {"one_dim": 0, "head_dim": ir.HD, "a_ref": float(ir.A_REF),
                      "a_bit": float(ir.A_BIT), "ref_cols": list(ir.REF_COLS),
                      "swiglu_k": float(ir.K), "kappa": float(core.KAPPA)},
        "dims": [{"name": d.name, "kind": d.kind, "level": d.level} for d in c.dims],
        "features": features(c, ir, core, S), "nodes": [encode_node(n, ir) for n in c.nodes],
        "tokens": {"vocab": S.VOCAB, "families": families, "go": S.tok("GO"),
                   "halt": S.tok("HALT"), "byte_family": core.FAM_ID["BYTE"]},
        "output": {"family_bits": core.NFAM, "payload_bits": core.NPAY,
                   "code_dims": [next(iter(e.terms)) for e in c.code], "families": families},
        "program": {"kind": "elf" if elf_path else "calc",
                    "elf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "entry": m.pc,
                    "rom_base": S.ROM_BASE, "rom_words": S.ROM_WORDS,
                    "rom_le_hex": struct.pack(f"<{len(m.rom)}I", *m.rom).hex(),
                    "drom_base": S.DROM_BASE, "drom_hex": m.drom.hex(),
                    "ram_base": S.RAM_BASE, "ram_bytes": 4 * S.RAM_WORDS,
                    "ram_hex": struct.pack(f"<{len(m.ram)}I", *m.ram).hex(),
                    "input_address": calc.CALC_IN, "input_capacity": 256,
                    "expression": expr if elf_path is None else None},
    }


def write_circuit(document, output):
    from runner_circuit import validate

    validate(document)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    header = HEADER.pack(MAGIC, 1, HEADER.size, len(payload), 0, 0, digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_bytes(header + payload)
    tmp.replace(output)
    file_hash = hashlib.sha256(header + payload).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{file_hash}  {output.name}\n", encoding="ascii")
    return {"file": output.name, "bytes": len(header) + len(payload), "sha256": file_hash,
            "payload_sha256": digest.hex(), "stats": document["stats"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwenputer", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--elf", type=Path)
    ap.add_argument("--expr")
    args = ap.parse_args()
    try:
        document = build_document(args.qwenputer, args.elf, args.expr)
        print(json.dumps(write_circuit(document, args.output), sort_keys=True))
    except (ValueError, UnicodeError, OSError, subprocess.CalledProcessError) as e:
        ap.exit(1, f"export failed: {e}\n")


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, PYTHONHASHSEED="0"))
    main()
