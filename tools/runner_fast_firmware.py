#!/usr/bin/env python3
"""Emit direct arithmetic RV32 firmware without evaluating expression results on the host."""

import argparse
import ast
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


MAX_DIGITS = 62
MAX_LIMBS = 16
ENV = Path("experimental/riscv/chipformer/rv/env")

RUNTIME = r'''typedef unsigned int u32;
_Static_assert(sizeof(u32) == 4, "requires 32-bit unsigned int");

static inline void put(u32 byte) {
    register u32 a0 __asm__("a0") = byte;
    register u32 a7 __asm__("a7") = 1;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

__attribute__((noreturn)) void exit(int code) {
    register u32 a0 __asm__("a0") = (u32)code;
    register u32 a7 __asm__("a7") = 93;
    for (;;) __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static int fail(void) {
    put('e'); put('r'); put('r'); put('o'); put('r'); put('\n');
    return 1;
}

static inline u32 print_group(u32 v, u32 started, u32 last) {
    u32 q = v / 1000u;
    if (started || q) { put('0' + q); started = 1; }
    v -= q * 1000u;
    q = v / 100u;
    if (started || q) { put('0' + q); started = 1; }
    v -= q * 100u;
    q = v / 10u;
    if (started || q) { put('0' + q); started = 1; }
    v -= q * 10u;
    if (started || v || last) { put('0' + v); started = 1; }
    return started;
}
'''


@dataclass(frozen=True)
class Value:
    name: str
    digits: int

    @property
    def limbs(self):
        return (self.digits + 3) // 4

    def at(self, i):
        return f"{self.name}_{i}" if i < self.limbs else "0u"


class Generator:
    def __init__(self, expr):
        self.expr = expr.strip()
        self.data = []
        self.body = []
        self.serial = 0
        self.products = 0
        self.columns = 0

    def fresh(self, digits):
        value = Value(f"v{self.serial}", min(digits, MAX_DIGITS))
        self.serial += 1
        return value

    def literal(self, digits):
        digits = digits.lstrip("0") or "0"
        if len(digits) > MAX_DIGITS:
            raise ValueError("integer literal exceeds 62 digits")
        value = self.fresh(len(digits))
        chunks = [digits[max(0, i - 4):i] for i in range(len(digits), 0, -4)]
        initializer = ", ".join(f"{int(chunk)}u" for chunk in chunks)
        self.data.append(f"static volatile u32 input_{value.name}[] = {{{initializer}}};")
        for i in range(value.limbs):
            self.body.append(f"    u32 {value.at(i)} = input_{value.name}[{i}];")
        return value

    def finish_columns(self, result, count):
        overflow = [f"{result.name}_{i}" for i in range(result.limbs, count)]
        if result.limbs == MAX_LIMBS:
            overflow.append(f"({result.at(15)} >= 100u)")
        if overflow:
            self.body.append(f"    if ({' | '.join(overflow)}) return fail();")

    def add_sub(self, a, b, subtract):
        result = self.fresh(max(a.digits, b.digits) + (not subtract))
        count = max(a.limbs, b.limbs) + (not subtract)
        carry = f"c{result.name}"
        self.body.append(f"    u32 {carry} = 0u;")
        for i in range(count):
            term = f"t{result.name}_{i}"
            if subtract:
                self.body.append(f"    u32 {term} = 10000u + {a.at(i)} - {b.at(i)} - {carry};")
                self.body.append(f"    {carry} = {term} < 10000u;")
                self.body.append(f"    u32 {result.name}_{i} = {term} + {carry} * 10000u - 10000u;")
            else:
                self.body.append(f"    u32 {term} = {a.at(i)} + {b.at(i)} + {carry};")
                self.body.append(f"    {carry} = {term} / 10000u;")
                self.body.append(f"    u32 {result.name}_{i} = {term} - {carry} * 10000u;")
        if subtract:
            self.body.append(f"    if ({carry}) return fail();")
        self.finish_columns(result, count)
        return result

    def multiply(self, a, b):
        result = self.fresh(a.digits + b.digits)
        carry = f"c{result.name}"
        self.body.append(f"    u32 {carry} = 0u;")
        count = a.limbs + b.limbs
        for k in range(count):
            terms = [f"{a.at(i)} * {b.at(k - i)}" for i in range(a.limbs) if 0 <= k - i < b.limbs]
            self.products += len(terms)
            self.columns += 1
            term = f"t{result.name}_{k}"
            self.body.append(f"    u32 {term} = {' + '.join([carry] + terms)};")
            self.body.append(f"    {carry} = {term} / 10000u;")
            self.body.append(f"    u32 {result.name}_{k} = {term} - {carry} * 10000u;")
        self.finish_columns(result, count)
        return result

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            digits = ast.get_source_segment(self.expr, node)
            if not digits or not re.fullmatch(r"[0-9]+", digits):
                raise ValueError("only decimal integer literals are supported")
            return self.literal(digits)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self.expression(node.operand)
        if not isinstance(node, ast.BinOp):
            raise ValueError("supported expressions use nonnegative integers, +, -, *, ** and parentheses")
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if not (isinstance(exponent, ast.Constant) and type(exponent.value) is int
                    and 0 <= exponent.value <= 999):
                raise ValueError("exponent must be a literal integer from 0 through 999")
            base = self.expression(node.left)
            power = exponent.value
            result = None
            while power:
                if power & 1:
                    result = base if result is None else self.multiply(result, base)
                power >>= 1
                if power:
                    base = self.multiply(base, base)
            return self.literal("1") if result is None else result
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            raise ValueError("unsupported operator")
        a = self.expression(node.left)
        b = self.expression(node.right)
        if isinstance(node.op, ast.Mult):
            return self.multiply(a, b)
        return self.add_sub(a, b, isinstance(node.op, ast.Sub))

    def generate(self):
        if len(self.expr) > 1024 or not re.fullmatch(r"[0-9+*()\s-]+", self.expr):
            raise ValueError("expression must contain at most 1024 characters of decimal arithmetic")
        tree = ast.parse(self.expr, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("expression exceeds 128 AST nodes")
        result = self.expression(tree.body)
        self.body.append("    u32 started = 0u;")
        for i in reversed(range(result.limbs)):
            self.body.append(f"    started = print_group({result.at(i)}, started, {int(i == 0)}u);")
        self.body.extend(["    put('\\n');", "    return 0;", "}"])
        metadata = json.dumps(self.expr, ensure_ascii=True)
        header = ["// Direct arithmetic firmware; input operands are volatile RAM, never host-computed results.",
                  f"const char runner_expression[] __attribute__((used)) = {metadata};"]
        source = "\n".join(header + [RUNTIME] + self.data + ["int main(void) {"] + self.body) + "\n"
        if len(source.splitlines()) > 600:
            raise ValueError("generated firmware exceeds the 600-line bound; split the expression")
        return source


def compile_command(args):
    if args.qwenputer is None:
        raise ValueError("--qwenputer is required with --compile")
    env = args.qwenputer / ENV
    for name in ("crt0.S", "rom.ld"):
        if not (env / name).is_file():
            raise ValueError(f"upstream runtime is missing {name}")
    return [str(args.compiler), "cc", "-target", "riscv32-freestanding", "-mcpu=generic_rv32+m+zba",
            "-mabi=ilp32", f"-{args.optimize}", "-std=c11", "-ffreestanding", "-fno-builtin",
            "-fno-sanitize=all", "-fno-stack-protector", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables",
            "-nostdlib", f"-Wl,-T,{env / 'rom.ld'}", str(env / "crt0.S"),
            str(args.output_c), "-o", str(args.output_elf)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expr", required=True)
    parser.add_argument("--output-c", required=True, type=Path)
    parser.add_argument("--qwenputer", type=Path, help="upstream checkout containing the P2 runtime")
    parser.add_argument("--compile", action="store_true", help="invoke Zig cc after emitting C")
    parser.add_argument("--output-elf", type=Path, help="default: output C path with .elf suffix")
    parser.add_argument("--compiler", default="zig", help="Zig executable path")
    parser.add_argument("--optimize", choices=("O2", "O3"), default="O2")
    args = parser.parse_args()
    if args.output_elf is None:
        args.output_elf = args.output_c.with_suffix(".elf")
    if args.output_c.resolve() == args.output_elf.resolve():
        parser.error("C and ELF output paths must differ")
    try:
        generator = Generator(args.expr)
        source = generator.generate()
        command = compile_command(args) if args.compile else None
        args.output_c.parent.mkdir(parents=True, exist_ok=True)
        args.output_c.write_text(source, encoding="utf-8")
        print(json.dumps({"c": str(args.output_c), "limb_products": generator.products,
                          "multiply_columns": generator.columns, "measured": False}))
        if command:
            args.output_elf.parent.mkdir(parents=True, exist_ok=True)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
    except (ValueError, SyntaxError, RecursionError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"firmware generation failed: {error}\n")


if __name__ == "__main__":
    main()
