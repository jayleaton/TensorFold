#!/usr/bin/env python3
"""Lower offline integer syntax into reusable ALU invocations without computing answers."""

import argparse
import ast
from collections import Counter
import gzip
import json
from pathlib import Path


BUCKETS = (1, 2, 4, 8, 16, 32, 62)
OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.FloorDiv: "div", ast.Mod: "mod"}
MAX_CALLS = 4096
MAX_VALUES = 8192
MAX_ITERATIONS = 10000
LOOP_SQUARES = "s = 0\nfor i in range(10):\n    s += i*i\nprint(s)"


class Unsupported(ValueError):
    """The source is outside the bounded offline plan grammar."""


def bucket(width):
    return next(n for n in BUCKETS if n >= min(width, 62))


def signed_literal(node):
    """Return literal text, accepting unary signs but never evaluating an expression."""
    sign = False
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign ^= isinstance(node.op, ast.USub)
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        return None
    digits = str(node.value)
    return ("-" if sign and digits != "0" else "") + digits


def parse_expression(source):
    if not source or len(source) > 4096:
        raise Unsupported("expression length is outside [1,4096]")
    try:
        tree = ast.parse(source.replace("^", "**"), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Unsupported("invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise Unsupported("expression exceeds 512 syntax nodes")
    return tree


def single_binary(source):
    """Detect the six simple signed-literal binary families by syntax alone."""
    try:
        node = parse_expression(source)
    except Unsupported:
        return None
    if not isinstance(node, ast.BinOp):
        return None
    a, b = signed_literal(node.left), signed_literal(node.right)
    if a is None or b is None:
        return None
    op = "pow" if isinstance(node.op, ast.Pow) else OPS.get(type(node.op))
    if op is None or max(len(a.lstrip("-")), len(b.lstrip("-"))) > 62:
        return None
    if op == "pow" and (b.startswith("-") or len(b) > 3):
        return None
    return {"op": op, "a": a, "b": b}


def power_schedule(exponent):
    """Return shared-mul SSA control steps for one literal exponent, without operand values."""
    if type(exponent) is not int or not 0 <= exponent <= 999:
        raise Unsupported("power exponent must be a control integer in [0,999]")
    calls = []
    result, base = "one", "base"
    while exponent:
        if exponent & 1:
            calls.append({"op": "mul", "a": result, "b": base})
            result = len(calls) - 1
        exponent >>= 1
        if exponent:
            calls.append({"op": "mul", "a": base, "b": base})
            base = len(calls) - 1
    return {"calls": calls, "result": result}


class Lowerer:
    def __init__(self):
        self.literals = []
        self.nodes = []
        self.widths = {}
        self.names = {}
        self.controls = {}
        self.iterations = 0
        self.literal_ids = {}

    def literal(self, text):
        sign = int(text.startswith("-"))
        digits = text.lstrip("+-").lstrip("0") or "0"
        if not digits.isascii() or not digits.isdigit() or len(digits) > 62:
            raise Unsupported("literal exceeds the decimal register")
        text = ("-" if sign and digits != "0" else "") + digits
        if text not in self.literal_ids:
            if len(self.literals) + len(self.nodes) >= MAX_VALUES:
                raise Unsupported("too many bound values")
            self.literal_ids[text] = len(self.literals)
            self.literals.append({"digits": [int(d) for d in reversed(digits)],
                                  "sign": int(text.startswith("-")), "error": 0})
        ref = {"literal": self.literal_ids[text]}
        self.widths[self.key(ref)] = len(digits)
        return ref

    @staticmethod
    def key(ref):
        return next(iter(ref.items()))

    def call(self, op, a, b):
        if isinstance(a, list) or isinstance(b, list):
            raise Unsupported("arithmetic requires scalar operands")
        wa, wb = self.widths[self.key(a)], self.widths[self.key(b)]
        width = wa + wb if op == "mul" else max(wa, wb) + (op in ("add", "sub"))
        width = min(width, 62)
        if len(self.nodes) >= MAX_CALLS or len(self.literals) + len(self.nodes) >= MAX_VALUES:
            raise Unsupported("invocation plan exceeds native bounds")
        ref = {"node": len(self.nodes)}
        self.nodes.append({"op": op, "a": a, "b": b, "width": bucket(width)})
        self.widths[self.key(ref)] = width
        return ref

    def control(self, node, seen=None):
        """Evaluate only bounded schedule metadata such as range bounds, indices or exponents."""
        seen = set() if seen is None else seen
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.Name) and node.id in self.controls and node.id not in seen:
            value = self.control(self.controls[node.id], seen | {node.id})
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.control(node.operand, seen)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = self.control(node.left, seen), self.control(node.right, seen)
            value = a + b if isinstance(node.op, ast.Add) else a - b if isinstance(node.op, ast.Sub) else a * b
        else:
            raise Unsupported("control metadata must be statically bounded")
        if not -MAX_ITERATIONS <= value <= MAX_ITERATIONS:
            raise Unsupported("control metadata exceeds the bounded schedule")
        return value

    def range_values(self, node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
                and 1 <= len(node.args) <= 3 and not node.keywords):
            raise Unsupported("only bounded range iteration is supported")
        args = [self.control(arg) for arg in node.args]
        if len(args) == 3 and args[2] == 0:
            raise Unsupported("range step cannot be zero")
        values = range(*args)
        self.iterations += len(values)
        if self.iterations > MAX_ITERATIONS:
            raise Unsupported("schedule exceeds 10000 iterations")
        return values

    def expression(self, node):
        text = signed_literal(node)
        if text is not None:
            return self.literal(text)
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.names[node.id]
        if isinstance(node, ast.List):
            if len(node.elts) > 32:
                raise Unsupported("literal list exceeds 32 elements")
            values = [self.expression(item) for item in node.elts]
            if any(isinstance(value, list) for value in values):
                raise Unsupported("nested lists are unsupported")
            return values
        if isinstance(node, ast.Subscript):
            values = self.expression(node.value)
            index = self.control(node.slice)
            if not isinstance(values, list) or not 0 <= index < len(values):
                raise Unsupported("list index is outside the static list")
            return values[index]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.expression(node.operand)
            if isinstance(value, list):
                raise Unsupported("unary arithmetic requires a scalar")
            return self.call("sub", self.literal("0"), value) if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                exponent = self.control(node.right)
                schedule = power_schedule(exponent)
                bindings = {"base": self.expression(node.left), "one": self.literal("1")}
                if not exponent:
                    checked_zero = self.call("mul", bindings["base"], self.literal("0"))
                    return self.call("add", checked_zero, bindings["one"])
                for i, step in enumerate(schedule["calls"]):
                    bindings[i] = self.call("mul", bindings[step["a"]], bindings[step["b"]])
                return bindings[schedule["result"]]
            if type(node.op) in OPS:
                return self.call(OPS[type(node.op)], self.expression(node.left), self.expression(node.right))
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            if len(node.generators) != 1:
                raise Unsupported("only single-level comprehensions are supported")
            gen = node.generators[0]
            if gen.is_async or gen.ifs or not isinstance(gen.target, ast.Name):
                raise Unsupported("filtered or async comprehensions are unsupported")
            names, controls = dict(self.names), dict(self.controls)
            values = []
            for i in self.range_values(gen.iter):
                self.names[gen.target.id] = self.literal(str(i))
                self.controls[gen.target.id] = ast.Constant(value=i)
                values.append(self.expression(node.elt))
            self.names, self.controls = names, controls
            return values
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum"
                and len(node.args) == 1 and not node.keywords):
            values = self.expression(node.args[0])
            if not isinstance(values, list):
                raise Unsupported("sum requires a static sequence")
            result = self.literal("0")
            for value in values:
                result = self.call("add", result, value)
            return result
        raise Unsupported("expression outside integer ALU grammar")

    def assign(self, target, value):
        if isinstance(target, ast.Name) and target.id not in ("print", "sum", "range"):
            self.names[target.id] = value
            self.controls.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, list) and len(target.elts) == len(value):
            for item, bound in zip(target.elts, value):
                self.assign(item, bound)
        else:
            raise Unsupported("unsupported assignment target")

    def statements(self, statements, depth=0):
        result = None
        for node in statements:
            if result is not None:
                raise Unsupported("print must be the final statement")
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                rhs = node.value
                value = ([self.expression(item) for item in rhs.elts] if isinstance(rhs, ast.Tuple)
                         else self.expression(rhs))
                target = node.targets[0]
                snapshot = dict(self.controls)
                self.assign(target, value)
                if isinstance(target, ast.Name):
                    self.controls[target.id] = self.capture_control(rhs, snapshot)
                elif isinstance(rhs, (ast.Tuple, ast.List)):
                    for item, expression in zip(target.elts, rhs.elts):
                        if isinstance(item, ast.Name):
                            self.controls[item.id] = self.capture_control(expression, snapshot)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id not in self.names:
                    raise Unsupported("augmented target is undefined")
                rhs = ast.BinOp(left=ast.Name(id=node.target.id), op=node.op, right=node.value)
                value = self.expression(rhs)
                control = self.capture_control(rhs, dict(self.controls))
                self.assign(node.target, value)
                self.controls[node.target.id] = control
            elif isinstance(node, ast.For) and not depth and isinstance(node.target, ast.Name) and not node.orelse:
                for i in self.range_values(node.iter):
                    self.assign(node.target, self.literal(str(i)))
                    self.controls[node.target.id] = ast.Constant(value=i)
                    if self.statements(node.body, depth + 1) is not None:
                        raise Unsupported("print inside a loop is unsupported")
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
                  and len(node.value.args) == 1 and not node.value.keywords and not depth):
                result = self.expression(node.value.args[0])
                if isinstance(result, list):
                    raise Unsupported("print requires an integer")
            else:
                raise Unsupported("statement outside bounded SSA grammar")
        return result

    def capture_control(self, node, snapshot):
        """Capture syntactic control aliases before assignment without evaluating operand results."""
        if isinstance(node, ast.Name):
            return snapshot.get(node.id, node)
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=node.op, operand=self.capture_control(node.operand, snapshot))
        if isinstance(node, ast.BinOp):
            return ast.BinOp(left=self.capture_control(node.left, snapshot), op=node.op,
                             right=self.capture_control(node.right, snapshot))
        return node

    def finish(self, root):
        return {"format": "tensorfold.runner.graph-plan", "version": 1,
                "literals": self.literals, "nodes": self.nodes, "rootref": root}


def compile_expression(source):
    lowerer = Lowerer()
    root = lowerer.expression(parse_expression(source))
    if isinstance(root, list):
        raise Unsupported("result must be an integer")
    return lowerer.finish(root)


def compile_program(source):
    if not source or len(source) > 4096:
        raise Unsupported("program length is outside [1,4096]")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise Unsupported("invalid program syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 2048:
        raise Unsupported("program exceeds 2048 syntax nodes")
    lowerer = Lowerer()
    root = lowerer.statements(tree.body)
    if root is None:
        raise Unsupported("program requires a final print")
    return lowerer.finish(root)


def native_packet(plan):
    """Encode the fixture plan as the native graph Program packet without computing results."""
    values = []
    for value in plan["literals"]:
        text = "".join(str(d) for d in reversed(value["digits"]))
        values.append(("-" if value["sign"] else "") + text)

    if any(node["op"] in ("div", "mod") for node in plan["nodes"]) and "0" not in values:
        values.append("0")
    ops, mapped, produced = [], {}, 0

    def operand(value):
        return value["literal"] if "literal" in value else mapped[value["node"]]

    for i, node in enumerate(plan["nodes"]):
        a, b = operand(node["a"]), operand(node["b"])
        if node["op"] in ("div", "mod"):
            subplan = binary_plan(node["op"], node["width"])
            bindings = [a, b, values.index("0")]
            for call in subplan["ops"]:
                copied = dict(call)
                for field in ("a", "b"):
                    if field in copied:
                        copied[field] = bindings[copied[field]]
                if "args" in copied:
                    copied["args"] = [bindings[ref] for ref in copied["args"]]
                if "copies" in copied:
                    copied["copies"] = [{**part, "ref": bindings[part["ref"]]} for part in copied["copies"]]
                if "flags" in copied:
                    copied["flags"] = bindings[copied["flags"]]
                ops.append(copied)
                count = call.get("outputs", 1)
                bindings += list(range(len(values) + produced, len(values) + produced + count))
                produced += count
            mapped[i] = bindings[subplan["result"]]
        else:
            mapped[i] = len(values) + produced
            ops.append({"op": node["op"], "a": a, "b": b, "width": node["width"]})
            produced += 1
    if len(ops) > MAX_CALLS or len(values) + produced > MAX_VALUES:
        raise Unsupported("expanded packet exceeds native bounds")
    root = plan["rootref"]
    return {"version": 1, "values": values, "ops": ops,
            "result": operand(root)}


def plan_id(op, width, exponent=None):
    """Select a stable named plan using syntax and operand width bounds only."""
    if width not in BUCKETS or op not in (*OPS.values(), "pow"):
        raise Unsupported("unsupported binary family or width")
    if op == "pow":
        power_schedule(exponent)
        return f"pow-{exponent}-62"
    return f"{op}-{width}"


def binary_plan(op, width, exponent=None):
    """Build one reusable named binary schedule with inputs followed by its literal pool."""
    name = plan_id(op, width, exponent)
    if op in ("add", "sub", "mul"):
        return {"name": name, "inputs": 2, "literals": [],
                "ops": [{"op": op, "a": 0, "b": 1, "width": width}], "result": 2}
    if op == "pow":
        steps = power_schedule(exponent)
        bindings, ops = {"base": 0, "one": 1}, []
        if exponent == 0:
            return {"name": name, "inputs": 1, "literals": ["1", "0"],
                    "ops": [{"op": "mul", "a": 0, "b": 2, "width": 62},
                            {"op": "add", "a": 3, "b": 1, "width": 62}], "result": 4}
        for i, step in enumerate(steps["calls"]):
            ops.append({"op": "mul", "a": bindings[step["a"]], "b": bindings[step["b"]], "width": 62})
            bindings[i] = 2 + i
        return {"name": name, "inputs": 1, "literals": ["1"], "ops": ops,
                "result": bindings[steps["result"]]}
    ops, nextref, remainder, quotient = [], 3, 2, 2
    for digit in reversed(range(width)):
        ops.append({"op": "bind", "copies": [{"ref": 0, "from": digit, "to": 0, "digits": 1}], "flags": 0})
        extracted = nextref
        ops.append({"op": "divstep", "args": [remainder, 1, extracted], "width": width, "outputs": 2})
        remainder, qdigit = nextref + 1, nextref + 2
        ops.append({"op": "bind", "copies": [{"ref": quotient, "from": 0, "to": 0, "digits": width},
                                               {"ref": qdigit, "from": 0, "to": digit, "digits": 1}],
                    "flags": qdigit})
        quotient = nextref + 3
        nextref += 4
    ops.append({"op": "divfinal", "args": [quotient, remainder, 0, 1], "width": width, "outputs": 2})
    return {"name": name, "inputs": 2, "literals": ["0"], "ops": ops,
            "result": nextref + (op == "mod")}


def binary_request(source):
    """Bind only the operands of a simple six-family expression to a preloaded named plan."""
    detected = single_binary(source)
    if detected is None:
        raise Unsupported("request is not a simple signed binary expression")
    op, a, b = detected["op"], detected["a"], detected["b"]
    wa, wb = len(a.lstrip("-")), len(b.lstrip("-"))
    width = bucket(wa + wb if op == "mul" else max(wa, wb) + (op in ("add", "sub")))
    exponent = int(b) if op == "pow" else None
    return {"version": 1, "plan": plan_id(op, width, exponent), "values": [a] if op == "pow" else [a, b]}


def loop_plan(shape, bound, width=62):
    """Unroll loop control into shared ALU calls, binding all seeds as offline literal data."""
    if width not in BUCKETS or type(bound) is not int or not 0 <= bound <= 128:
        raise Unsupported("loop width or bound is outside the preloaded catalog")
    aliases = {"sum": "sum", "sums": "sum", "squares": "squares", "sum_squares": "squares",
               "factorial": "factorial", "fib": "fib", "fibonacci": "fib", "list_sum": "list_sum"}
    shape = aliases.get(shape)
    if shape is None:
        raise Unsupported("unsupported loop shape")
    inputs = bound if shape == "list_sum" else 0
    literals, ops = ["0", "1"], []
    nextref = inputs + 2

    def call(op, a, b):
        nonlocal nextref
        ops.append({"op": op, "a": a, "b": b, "width": width})
        nextref += 1
        return nextref - 1

    if shape in ("sum", "squares", "factorial"):
        literals += [str(i) for i in range(1, bound + 1)] if shape == "factorial" else [str(i) for i in range(bound)]
        nextref = inputs + len(literals)
        result = inputs + (shape == "factorial")
        for i in range(bound):
            value = inputs + 2 + i
            if shape == "squares":
                value = call("mul", value, value)
            result = call("mul" if shape == "factorial" else "add", result, value)
    elif shape == "fib":
        result, other = inputs, inputs + 1
        for _ in range(bound):
            result, other = other, call("add", result, other)
    else:
        result = inputs
        for i in range(bound):
            result = call("add", result, i)
    return {"name": f"loop-{shape}-{bound}-{width}", "inputs": inputs,
            "literals": literals, "ops": ops, "result": result}


def plan_catalog():
    plans = [binary_plan(op, width) for op in OPS.values() for width in BUCKETS]
    plans += [binary_plan("pow", 62, exponent) for exponent in range(1000)]
    plans += [loop_plan(shape, bound) for shape in ("sum", "squares", "factorial", "fib", "list_sum")
              for bound in range(129)]
    return plans


def route_corpus(rows):
    """Route every source row by syntax before examining any reference output or outcome."""
    routed = []
    for index, row in enumerate(rows):
        expression = row.get("input", {}).get("expression")
        try:
            if expression is not None:
                plan = compile_expression(expression)
                simple = single_binary(expression)
            elif row.get("id") == "loop-squares" and row.get("operation") == "sum_squares":
                plan, simple = compile_program(LOOP_SQUARES), None
            else:
                raise Unsupported("non-expression input has no known static plan")
            packet = native_packet(plan)
            mode, legacy_backend = "named", "graph"
            if simple:
                request = binary_request(expression)
                selected = binary_plan(simple["op"], int(request["plan"].rsplit("-", 1)[1]),
                                       int(simple["b"]) if simple["op"] == "pow" else None)
            elif row.get("id") == "loop-squares":
                selected = loop_plan("squares", 10)
                request = {"version": 1, "plan": selected["name"], "values": []}
            else:
                selected, request = None, packet
                mode, legacy_backend = "generic_ssa", "calc"
            routed.append({"index": index, "id": row["id"], "routed": True, "simple_binary": simple,
                           "mode": mode, "legacy_backend": legacy_backend,
                           "plan": plan, "packet": packet, "named_plan": selected, "request": request})
        except (Unsupported, RecursionError) as exc:
            routed.append({"index": index, "id": row["id"], "routed": False, "mode": "unsupported",
                           "legacy_backend": "calc", "reason": str(exc)})
    return routed


def corpus_document(directory, sample_count=20):
    with gzip.open(directory / "cases.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    routed = route_corpus(rows)
    requests = []
    for route in routed:
        row = rows[route["index"]]
        route.update({key: row[key] for key in ("operation", "input", "outcome", "expected_output", "final_output",
                                               "final_output_hex", "exit_code") if key in row})
        if route["routed"]:
            requests.append(route)
    manifest = json.loads((directory / "manifest.json").read_text())
    return {"format": "tensorfold.runner.graph-corpus", "version": 1,
            "source": manifest["source"], "upstream_commit": manifest["upstream_commit"],
            "total_cases": len(rows), "routed_cases": len(requests),
            "mode_counts": dict(Counter(row["mode"] for row in routed)),
            "legacy_backend_counts": dict(Counter(row["legacy_backend"] for row in routed)),
            "syntax_counts": dict(Counter(row["operation"] for row in requests)),
            "simple_binary_counts": dict(Counter(row["simple_binary"]["op"] for row in requests
                                                 if row["simple_binary"] is not None)),
            "machine_limit_cases": sum(row.get("outcome") == "machine_limit" for row in requests),
            "baseline_machine_limit_cases": sum(row.get("outcome") == "machine_limit" for row in routed),
            "machine_limit_by_mode": dict(Counter(row["mode"] for row in routed
                                                   if row.get("outcome") == "machine_limit")),
            "routing": [{k: v for k, v in row.items() if k in ("index", "id", "routed", "reason", "mode",
                                                              "legacy_backend", "outcome")}
                        for row in routed], "requests": requests, "samples": requests[:sample_count]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--expr")
    modes.add_argument("--program", type=Path)
    modes.add_argument("--corpus", type=Path)
    modes.add_argument("--power-table", action="store_true")
    modes.add_argument("--catalog", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.samples <= 10002:
        parser.error("samples must be in [0,10002]")
    if args.corpus:
        doc = corpus_document(args.corpus, args.samples)
    elif args.power_table:
        doc = {"version": 1, "powers": [power_schedule(i) for i in range(1000)]}
    elif args.catalog:
        doc = {"version": 1, "plans": plan_catalog()}
    else:
        plan = compile_expression(args.expr) if args.expr is not None else compile_program(args.program.read_text())
        doc = {"plan": plan, "packet": native_packet(plan)}
    args.output.write_text(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
