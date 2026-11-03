#!/usr/bin/env python3
"""Offline CPython Fraction references for exact inline rational arithmetic and rendering requests."""
import argparse
import ast
import hashlib
import json
import platform
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

LIVE = "(3.14159/2.71828)*10^(51-21)"
SOURCES = [
    LIVE, "fraction(" + LIVE + ")",
    "1/2", "1/3", "fraction(1/3)", "fraction(1/2)",
    "fraction(112/29)", "fraction(7/12)", "fraction(11/18*4/4)", "fraction(-1/6)",
    "(1.25/.5)*10^(6-3)", "(3.14159/2.5)*10^(5-3)",
    "fraction((3.14159/2.71828)*10^(12-4))",
    "fraction((3.14/2.71)*10^(8-3))", "fraction((1/3)*10^(6-3))",
    "fraction(0.1/3)", "fraction((0.1/3)*3)", "(0.1/3)*3",
    "fraction((1/6)+(1/3))", "fraction(1.25)", "fraction(-0.0/7)",
    "0.5^(4-2)", "(1.25/2.5)^(4/2)", "(-2.5)^(5-2)",
    "fraction((1.5/2.5)^(6-3))", "fraction((0.1*0.1)^(3-1))",
    "0.^(6-2)", "1.^(999-0)", "fraction(1^(999-0))",
    "fraction(0.000000000000000001*0.000000000000000001)",
    "(0.000000000000000001*0.1)*10^(2-1)",
    "fraction(2^(10-3))", "2^(3^2)", "fraction(0.2^3)",
]


def normalize(code):
    for old, new in (("×", "*"), ("÷", "/"), ("−", "-"), ("^", "**")):
        code = code.replace(old, new)
    return code


def value(node, source):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(ast.get_source_segment(source, node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        x = value(node.operand, source)
        return -x if isinstance(node.op, ast.USub) else x
    if isinstance(node, ast.BinOp):
        x, y = value(node.left, source), value(node.right, source)
        if isinstance(node.op, ast.Add): return x + y
        if isinstance(node.op, ast.Sub): return x - y
        if isinstance(node.op, ast.Mult): return x * y
        if isinstance(node.op, ast.Div): return x / y
        if isinstance(node.op, ast.Pow):
            if y.denominator != 1 or not 0 <= y <= 999: raise ValueError("exponent domain")
            result = x ** int(y)
            if max(len(str(abs(result.numerator))), len(str(result.denominator))) > 62:
                raise ValueError("declared bounded power capacity")
            return result
    raise TypeError("outside exact source grammar")


def reference(code):
    source = normalize(code)
    root = ast.parse(source, mode="eval").body
    fraction = isinstance(root, ast.Call) and isinstance(root.func, ast.Name) and root.func.id == "fraction"
    if fraction:
        if len(root.args) != 1 or root.keywords: raise TypeError("format request")
        root = root.args[0]
    result = value(root, source)
    if fraction: return str(result) + "\n"
    rest = result.denominator
    for factor in (2, 5):
        while rest % factor == 0: rest //= factor
    if rest != 1: raise ValueError("nonterminating decimal")
    with localcontext() as ctx:
        ctx.prec = 2048
        text = format(Decimal(result.numerator) / Decimal(result.denominator), "f")
    if "." in text: text = text.rstrip("0").rstrip(".")
    if "." in text and len(text.split('.')[1]) > 18: raise ValueError("decimal output capacity")
    return text + "\n"


def generate():
    if platform.python_implementation() != "CPython": raise ValueError("CPython required")
    rows = []
    for i, code in enumerate(SOURCES):
        base = dict(id=f"rational-{i}", cell=dict(form="inline_expr", lang="python", code=code, prior=[]))
        try: expected = reference(code)
        except (ValueError, ZeroDivisionError):
            rows.append({**base, "expected_status": "error:GraphArithmetic", "expected_output": ""})
            expected = None
        else: rows.append({**base, "expected_status": "halted", "expected_output": expected})
        source = normalize(code)
        leaves = [n for n in ast.walk(ast.parse(source, mode="eval")) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
        # Mutate every original decimal operand, base and exponent operand independently.
        for li, leaf in enumerate(leaves):
            token = ast.get_source_segment(source, leaf)
            for delta in (-1, 1):
                replacement = str(Fraction(token) + delta)
                changed = source[:leaf.col_offset] + "(" + replacement + ")" + source[leaf.end_col_offset:]
                try: want = reference(changed)
                except (ValueError, ZeroDivisionError):
                    rows.append({**base, "id": f"rational-{i}-operand-{li}-{delta}",
                                 "cell": {**base['cell'], "code": changed}, "expected_status": "error:GraphArithmetic", "expected_output": ""})
                    continue
                row = {**base, "id": f"rational-{i}-operand-{li}-{delta}", "cell": {**base['cell'], "code": changed}, "expected_status": "halted", "expected_output": want}
                rows.append(row)
                if expected is not None and want != expected:
                    rows.append({**row, "id": row['id'] + "-stale", "expected_output": expected, "expect_mismatch": True})
    n, d = "9"*60, "1"+"0"*58+"7"
    wide = f"fraction(({n}/{d})*({n}/{d})*({d}/{n})*({d}/{n}))"
    for i, code in enumerate([wide, wide.replace(n, "9"*59+"8", 1), f"({n}*{d})/1", f"{n}*{d}.5"]):
        rows.append(dict(id=f"rational-capacity-{i}", cell=dict(form="inline_expr", lang="python", code=code, prior=[]), expected_status="halted", expected_output=reference(code)))
    for i, code in enumerate([
        "0.1^(1-3)", "fraction(2^(3/2))", "fraction(1^(1001-1))",
        "fraction((1/0)^(2-2))", "fraction(0^((1/0)))", "fraction(10^(64-1))",
        "0.1/3", "fraction(1/0)", "0.1^(20-0)",
    ]):
        rows.append(dict(id=f"rational-domain-{i}", cell=dict(form="inline_expr", lang="python", code=code, prior=[]), expected_status="error:GraphArithmetic", expected_output=""))
    for i, code in enumerate([
        "fraction()", "fraction(1,3)", "fraction(1/3)+1", "fractional(1/3)",
        "Fraction(0.1)", "float(1/3)", "round(1/3,2)", "1e-2^(3-1)",
        "fraction(" + "9"*63 + "/2)", "." + "1"*19 + "^(2-1)",
    ]):
        rows.append(dict(id=f"rational-syntax-{i}", cell=dict(form="inline_expr", lang="python", code=code, prior=[]), expected_status="unsupported", expected_output=""))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    args = p.parse_args()
    rows = generate()
    raw = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()
    with open(args.output, 'xb') as stream: stream.write(raw)
    print(json.dumps(dict(python=sys.version, cases=len(rows), stale=sum(bool(r.get('expect_mismatch')) for r in rows), sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == "__main__": main()
