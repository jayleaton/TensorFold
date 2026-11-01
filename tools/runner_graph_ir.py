#!/usr/bin/env python3
"""Reusable signed decimal FFN templates on Qwenputer's P2 token bus."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess

from runner_export import HEADER, UPSTREAM, encode_node, features
from runner_trace import load_upstream


BUCKETS = (1, 2, 4, 8, 16, 32, 62)


@dataclass
class Value:
    digits: list
    sign: object
    error: object


class Builder:
    def __init__(self, width, core, ir, np):
        if not 1 <= width <= 63:
            raise ValueError("decimal width must be in [1,63]")
        self.width, self.core, self.ir, self.np = width, core, ir, np
        self.c = core.RvCore()
        self.one, self.zero = self.c.one, self.c.one * 0
        self.serial = 0
        self.input_slots, self.output_slots = [], []
        self.inputs = {}

    def node(self, terms, name="value"):
        self.serial += 1
        return self.c.node(terms, f"alu.{self.serial}.{name}")

    def ge(self, z, t, lo, hi):
        return self.node(self.c.t_ge(z, t, lo, hi), "ge")

    def eq(self, z, t, lo, hi):
        return self.node(self.c.t_eq(z, t, lo, hi), "eq")

    def choose(self, flag, a, b):
        return self.node([(1, a, flag), (1, b, self.one - flag)], "choose")

    def any(self, flags):
        flags = list(flags)
        while len(flags) > 1:
            flags = [self.ge(flags[i] + flags[i + 1], 1, 0, 2) if i + 1 < len(flags) else flags[i]
                     for i in range(0, len(flags), 2)]
        return flags[0] if flags else self.zero

    def nz(self, digits):
        return self.any(self.ge(d, 1, 0, 9) for d in digits)

    def _prefix(self, generate, propagate):
        distance = 1
        while distance < len(generate):
            old_g, old_p = generate, propagate
            generate, propagate = old_g[:], old_p[:]
            for i in range(distance, len(generate)):
                generate[i] = self.node([(1, old_g[i], self.one),
                                         (1, old_g[i - distance], old_p[i])], "prefix.generate")
                propagate[i] = self.node([(1, old_p[i], old_p[i - distance])], "prefix.propagate")
            distance *= 2
        return generate

    def _pad(self, digits, width):
        return list(digits[:width]) + [self.zero] * max(0, width - len(digits))

    def add(self, a, b):
        width = max(len(a), len(b))
        if not width:
            return [], self.zero
        a, b = self._pad(a, width), self._pad(b, width)
        sums = [x + y for x, y in zip(a, b)]
        carry = self._prefix([self.ge(z, 10, 0, 18) for z in sums],
                             [self.eq(z, 9, 0, 18) for z in sums])
        digits = [self.node([(1, z + (carry[i - 1] if i else self.zero) - 10 * carry[i], self.one)],
                            "add.digit") for i, z in enumerate(sums)]
        return digits, carry[-1]

    def sub(self, a, b):
        width = max(len(a), len(b))
        if not width:
            return [], self.zero
        a, b = self._pad(a, width), self._pad(b, width)
        deltas = [x - y for x, y in zip(a, b)]
        borrow = self._prefix([self.one - self.ge(z, 0, -9, 9) for z in deltas],
                              [self.eq(z, 0, -9, 9) for z in deltas])
        digits = [self.node([(1, z - (borrow[i - 1] if i else self.zero) + 10 * borrow[i], self.one)],
                            "sub.digit") for i, z in enumerate(deltas)]
        return digits, borrow[-1]

    def mul(self, a, b, width):
        if width < 1:
            raise ValueError("product width must be positive")
        if not a or not b:
            return [self.zero] * width, self.zero
        full_width = max(width, len(a) + len(b))
        rows = []
        errors = []
        for j, bj in enumerate(b):
            ones, tens = [self.zero] * full_width, [self.zero] * full_width
            for i, ai in enumerate(a):
                product = self.node([(1, ai, bj)], "mul.product")
                ten = self.node([term for threshold in range(10, 81, 10)
                                 for term in self.c.t_ge(product, threshold, 0, 81)], "mul.tens")
                ones[i + j] = self.node([(1, product - 10 * ten, self.one)], "mul.ones")
                tens[i + j + 1] = ten
            rows.extend((ones, tens))
        while len(rows) > 2:
            merged = []
            for i in range(0, len(rows), 3):
                if i + 2 >= len(rows):
                    merged.extend(rows[i:])
                    break
                low, high = [], [self.zero]
                for digits in zip(*rows[i:i + 3]):
                    total = sum(digits, self.zero)
                    carry = self.node(self.c.t_ge(total, 10, 0, 27)
                                      + self.c.t_ge(total, 20, 0, 27), "mul.compress.carry")
                    low.append(self.node([(1, total - 10 * carry, self.one)], "mul.compress.digit"))
                    high.append(carry)
                errors.append(self.ge(high[-1], 1, 0, 2))
                merged.extend((low, high[:-1]))
            rows = merged
        digits, carry = self.add(rows[0], rows[1])
        return digits[:width], self.any([*errors, carry, self.nz(digits[width:])])

    def normalize(self, value):
        sign = self.node([(1, value.sign, self.nz(value.digits))], "normalize.sign")
        return Value(value.digits, sign, value.error)

    def signed_add(self, a, b):
        summed, carry = self.add(a.digits, b.digits)
        ab, borrow = self.sub(a.digits, b.digits)
        ba, _ = self.sub(b.digits, a.digits)
        same = self.eq(a.sign - b.sign, 0, -1, 1)
        difference = [self.choose(borrow, y, x) for x, y in zip(ab, ba)]
        digits = [self.choose(same, x, y) for x, y in zip(summed, difference)]
        sign = self.choose(same, a.sign, self.choose(borrow, b.sign, a.sign))
        overflow = self.node([(1, carry, same)], "add.overflow")
        return self.normalize(Value(digits, sign, self.any([a.error, b.error, overflow])))

    def signed_sub(self, a, b):
        return self.signed_add(a, Value(b.digits, self.one - b.sign, b.error))

    def signed_mul(self, a, b):
        digits, overflow = self.mul(a.digits, b.digits, self.width)
        sign = self.one - self.eq(a.sign - b.sign, 0, -1, 1)
        return self.normalize(Value(digits, sign, self.any([a.error, b.error, overflow])))

    def select(self, flag, a, b):
        width = max(len(a.digits), len(b.digits))
        digits = [self.choose(flag, x, y)
                  for x, y in zip(self._pad(a.digits, width), self._pad(b.digits, width))]
        return self.normalize(Value(digits, self.choose(flag, a.sign, b.sign), self.any([a.error, b.error])))

    def _slot(self, index, offset, width, head):
        return {"index": index, "value_id": index, "head": head, "width": self.width,
                "offset": offset, "digit_offset": offset, "digits": width, "row": 0,
                "cols": list(range(2, 2 + 4 * width)), "first_col": 2, "bits_per_digit": 4,
                "order": "lsd_first", "sign_col": 126 if offset == 0 else None,
                "error_col": 127 if offset == 0 else None}

    def input(self, index):
        if index in self.inputs:
            return self.inputs[index]
        digits, sign, error = [], self.zero, self.zero
        invalid = []
        for offset in range(0, self.width, 31):
            count = min(31, self.width - offset)
            head = f"in{index}.{offset}"
            slot = self._slot(index, offset, count, head)
            self.input_slots.append(slot)
            cols = slot["cols"] + ([126, 127] if offset == 0 else [])
            bits = self.c.head(head, {0: 128 * self.c.fam["GO"]}, {}, {}, 0,
                               {col: 128 * self.one for col in cols}, [(col, 128) for col in cols],
                               tiny=2.0**-90)
            row = self.np.zeros(128)
            row[0] = row[1] = self.ir.A_REF
            self.c.nodes[-1].init = [{0: row}]
            for i in range(count):
                digit = sum((bits[4 * i + bit] * (1 << bit) for bit in range(4)), self.zero)
                bad = self.ge(digit, 10, 0, 15)
                invalid.append(bad)
                digits.append(self.choose(bad, self.zero, digit))
            if offset == 0:
                sign, error = bits[-2:]
        value = self.normalize(Value(digits, sign, self.any([error, *invalid])))
        self.inputs[index] = value
        return value

    def _latch(self, index, value):
        from chipformer.q35.prims import thead

        overflow = self.nz(value.digits[self.width:])
        value = self.normalize(Value(self._pad(value.digits, self.width), value.sign,
                                     self.any([value.error, overflow])))
        stored, sign, error = [], self.zero, self.zero
        for offset in range(0, self.width, 31):
            group = value.digits[offset:offset + 31]
            data = {}
            for i, digit in enumerate(group):
                for bit, terms in enumerate(self.c.fn_bits(digit, 0, 9, lambda d: d, 4, "out")):
                    data[2 + 4 * i + bit] = self.node(terms, "out.bit")
            if offset == 0:
                data[126], data[127] = value.sign, value.error
            head = f"out{index}.{offset}"
            (bits,) = thead(self.c, head, {0: self.c.fam["GO"]}, {0: self.one},
                            [{"data": data, "ref": self.one}])
            self.output_slots.append(self._slot(index, offset, len(group), head))
            for i in range(len(group)):
                stored.append(sum((bits[4 * i + bit] * (1 << bit) for bit in range(4)), self.zero))
            if offset == 0:
                sign, error = bits[-2:]
        return Value(stored, sign, error)

    def _exit(self, halt, payload):
        core, c = self.core, self.c
        family = [((core.FAM_ID["BYTE"] >> i) & 1) * (self.one - halt)
                  + ((core.FAM_ID["HALT"] >> i) & 1) * halt for i in range(core.NFAM)]
        payload = [self.node([(1, bit, self.one - halt)], "exit.payload") for bit in payload]
        bits = family + payload + [self.zero] * (core.NPAY - len(payload))
        level = max(n.level for n in c.nodes)
        for i, (code, bit) in enumerate(zip(c.code, bits)):
            c.ffn([(-1, self.one, code), (1, self.one, -code)], "code.clear", into=code)
            c.ffn(c.t_const(-core.KAPPA) + [(2 * core.KAPPA, self.one, bit)],
                  f"exit.code{i}", into=code, min_level=level)

    def _emit(self, value):
        c, one, width = self.c, self.one, self.width
        occupied = [self.ge(d, 1, 0, 9) for d in value.digits]
        lengths = [(occupied[i], (i + 1) * one) for i in range(width)]
        while len(lengths) > 1:
            merged = []
            for i in range(0, len(lengths), 2):
                if i + 1 == len(lengths):
                    merged.append(lengths[i])
                else:
                    low, high = lengths[i:i + 2]
                    merged.append((self.any([low[0], high[0]]), self.choose(high[0], high[1], low[1])))
            lengths = merged
        length = self.choose(lengths[0][0], lengths[0][1], one)
        limit = max(width + 3, 6)
        count = c.counter("output.cursor", c.fam["GO"], c.fam["BYTE"])
        n = sum(c.thermo(count, c.fam["GO"], limit, "output.cursor"), self.zero)
        index = length - one - n + value.sign
        digit = self.node([(1, d, self.eq(index, i, -limit, width))
                           for i, d in enumerate(value.digits)], "emit.digit")
        minus = self.node([(1, value.sign, self.eq(n, 0, 0, limit))], "emit.minus")
        newline = self.eq(n - length - value.sign, 0, -width - 1, limit)
        done = self.ge(n - length - value.sign, 1, -width - 1, limit)
        err_done = self.ge(n, 6, 0, limit)
        halt = self.choose(value.error, err_done, done)
        payload = []
        for bit, terms in enumerate(c.fn_bits(digit, 0, 9, lambda d: 48 + d, 8, "ascii")):
            normal = self.choose(newline, ((10 >> bit) & 1) * one, self.node(terms, "emit.ascii"))
            normal = self.choose(minus, ((45 >> bit) & 1) * one, normal)
            error = self.node([term for i, byte in enumerate(b"error\n") if (byte >> bit) & 1
                               for term in c.t_eq(n, i, 0, limit)], "emit.error")
            payload.append(self.choose(value.error, error, normal))
        self._exit(halt, payload)

    def finish(self, outputs, emit=False):
        if isinstance(outputs, Value):
            outputs = [outputs]
        if not outputs or (emit and len(outputs) != 1):
            raise ValueError("finish requires outputs; emit requires exactly one Value")
        latched = [self._latch(index, value) for index, value in enumerate(outputs)]
        if emit:
            self._emit(latched[0])
        else:
            self._exit(self.one, [])
        self.c.nodes.sort(key=lambda node: node.level)
        return {"width": self.width, "input_slots": self.input_slots, "output_slots": self.output_slots}


def build_template(root, op, width):
    if width not in BUCKETS:
        raise ValueError(f"template width must be one of {BUCKETS}")
    if op not in ("add", "sub", "mul", "select", "emit", "divstep", "divfinal"):
        raise ValueError(f"unsupported ALU operation: {op}")
    root = Path(root).resolve()
    load_upstream(root)
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip()
    if dirty:
        raise ValueError("Qwenputer tracked files have local modifications")
    import numpy as np
    from chipformer.q35 import ir
    from chipformer.rv import core, spec

    builder = Builder(width, core, ir, np)
    if op.startswith("div"):
        module = importlib.import_module("runner_graph_div")
        arity = 3 if op == "divstep" else 4
        outputs = getattr(module, f"build_{op}")(builder, [builder.input(i) for i in range(arity)])
    elif op == "emit":
        outputs = builder.input(0)
    else:
        a, b = builder.input(0), builder.input(1)
        if op == "select":
            condition = builder.input(2)
            outputs = builder.select(builder.nz(condition.digits), a, b)
            outputs.error = builder.any([outputs.error, condition.error])
        else:
            outputs = getattr(builder, f"signed_{op}")(a, b)
    metadata = builder.finish(outputs, emit=op == "emit")
    metadata.update(op=op, input_arity=len(builder.inputs))
    c = builder.c
    families = [{"name": name, "base": int(spec.FAMILIES[name][0]), "size": int(spec.FAMILIES[name][1])}
                if name in spec.FAMILIES else {"name": name, "base": spec.CONTROLS[name], "size": 1}
                for name in core.OUT_NAMES[:-1]]
    doc = {"format": "tensorfold.runner.arithmetic", "version": 1,
           "source": {"repository": "https://github.com/divagr18/qwenputer", "commit": UPSTREAM,
                      "license": "MIT", "credit": "Divyansh Agrawal", "license_text": (root / "LICENSE").read_text()},
           "stats": c.stats(), "constants": {"one_dim": 0, "head_dim": ir.HD, "a_ref": float(ir.A_REF),
                                              "a_bit": float(ir.A_BIT), "ref_cols": list(ir.REF_COLS),
                                              "swiglu_k": float(ir.K), "kappa": float(core.KAPPA)},
           "dims": [{"name": d.name, "kind": d.kind, "level": d.level} for d in c.dims],
           "features": features(c, ir, core, spec), "nodes": [encode_node(n, ir) for n in c.nodes],
           "tokens": {"vocab": spec.VOCAB, "families": families, "go": spec.tok("GO"),
                      "halt": spec.tok("HALT"), "byte_family": core.FAM_ID["BYTE"]},
           "output": {"family_bits": core.NFAM, "payload_bits": core.NPAY, "families": families,
                      "code_dims": [next(iter(e.terms)) for e in c.code]},
           "program": {"kind": "compiled_alu_template", "max_digits": width, **metadata}}
    return doc, metadata
