#!/usr/bin/env python3
"""Offline CPython Fraction/Decimal references for the exact inline graph route; never called by the engine."""
import argparse
import ast
import hashlib
import json
import platform
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

CORPUS = [
    "8613/9", "112 /29", "(-23777868243178464)/(-41832496)",
    "392764052 ÷ 116", "8.3168*5.4828", "0.752 × 22.17",
    "7/12", "11/18 *4/4", "5/24 *3/3", "(1 -1/2)",
    "-1/4", "-1/6", "-1/8", "37*19.95", "18*8.40",
]
BOUNDARIES = [
    "0/3", "-0.00*2.0", "-2^2/3", "(-2)^2/3", "1/-3", "-1/-3",
    "0.1+0.2", "999.99+0.01", "1.0/4", "0.000000000000000001*1",
    "1.5-2.75", "(1/2)+(1/3)", "(1/2)/(3/4)", "(2/3)^3",
    "(0.1/2)+(0.3/2)", "889.35*0.075", "889.35*0.925", "822.64875*1.06",
    "80. /2", "134709035556. /1000", ".5*2", "5.*2", ".75/.25",
    "-0.*2.0", "1.+2.", "5./4", "-.125*8", "(.5+.25)*4",
]


def normalize(source):
    for old, new in (("×", "*"), ("÷", "/"), ("−", "-"), ("^", "**")):
        source = source.replace(old, new)
    return source


def evaluate(node, source):
    if isinstance(node, ast.Expression):
        return evaluate(node.body, source)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(ast.get_source_segment(source, node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate(node.operand, source)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        x, y = evaluate(node.left, source), evaluate(node.right, source)
        if isinstance(node.op, ast.Add): return x + y
        if isinstance(node.op, ast.Sub): return x - y
        if isinstance(node.op, ast.Mult): return x * y
        if isinstance(node.op, ast.Div): return x / y
        if isinstance(node.op, ast.Pow) and y.denominator == 1 and 0 <= y <= 999:
            return x ** int(y)
    raise ValueError("outside exact reference grammar")


def reference(code):
    source = normalize(code)
    tree = ast.parse(source, mode="eval")
    value = evaluate(tree, source)
    rest = value.denominator
    for factor in (2, 5):
        while rest % factor == 0: rest //= factor
    if rest != 1: raise ValueError("nonterminating exact decimal")
    with localcontext() as context:
        context.prec = 1024
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if "." in text: text = text.rstrip("0").rstrip(".")
    return text + "\n"


def generate():
    if platform.python_implementation() != "CPython": raise ValueError("CPython required")
    rows = []
    for index, code in enumerate(CORPUS + BOUNDARIES):
        try: expected = reference(code)
        except (ValueError, ZeroDivisionError):
            rows.append({"id": f"exact-{index}", "cell": {"form": "inline_expr", "lang": "python", "code": code, "prior": []},
                         "expected_output": "", "expected_status": "error:GraphArithmetic"})
            expected = None
        base = {"id": f"exact-{index}", "cell": {"form": "inline_expr", "lang": "python", "code": code, "prior": []},
                "expected_output": expected, "expected_status": "halted"}
        if expected is not None: rows.append(base)
        # Change each original operand independently; do not replace an expression with its answer.
        source = normalize(code)
        tree = ast.parse(source, mode="eval")
        leaves = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
        for li, node in enumerate(leaves):
            token = ast.get_source_segment(source, node)
            for delta in (-1, 1):
                value = Fraction(token) + delta
                is_exponent = any(isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Pow) and parent.right is node for parent in ast.walk(tree))
                if is_exponent and (value.denominator != 1 or not 0 <= value <= 999): continue
                replacement = str(value) if is_exponent else "(" + str(value) + ")"
                expression = source[:node.col_offset] + replacement + source[node.end_col_offset:]
                try: changed = reference(expression)
                except (ValueError, ZeroDivisionError):
                    rows.append({**base, "id": f"exact-{index}-leaf-{li}-{delta}",
                                 "cell": {**base["cell"], "code": expression}, "expected_status": "error:GraphArithmetic", "expected_output": ""})
                    continue
                mutation = {**base, "id": f"exact-{index}-leaf-{li}-{delta}",
                            "cell": {**base["cell"], "code": expression}, "expected_output": changed}
                rows.append(mutation)
                if expected is not None and changed != expected:
                    rows.append({**mutation, "id": mutation["id"] + "-stale-answer", "expected_output": expected, "expect_mismatch": True})
    for index, code in enumerate(("1/0", "1/(1/0)", "(1/0)^0", "0.1/3", "1.0/3")):
        rows.append({"id": f"exact-domain-{index}", "cell": {"form": "inline_expr", "lang": "python", "code": code, "prior": []},
                     "expected_output": "", "expected_status": "error:GraphArithmetic"})
    for index, code in enumerate(("1e-5", ".*2", "5..*2", "x/2", "2**2**2/3", "2**-2/3", "9"*63 + "/2", "0."+"0"*18+"1")):
        rows.append({"id": f"exact-width-syntax-{index}", "cell": {"form": "inline_expr", "lang": "python", "code": code, "prior": []},
                     "expected_output": "", "expected_status": "unsupported"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    rows = generate()
    with open(args.output, "x", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"python": sys.version, "cases": len(rows), "sha256": hashlib.sha256(open(args.output, "rb").read()).hexdigest()}))


if __name__ == "__main__": main()
