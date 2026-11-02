"""Offline qualification only: 4096 source chars, 2048 AST nodes, 4096 FFN ops, 8192 values, 128 prints; no runtime."""

import ast
import re
from dataclasses import dataclass

try:
    from .runner_graph_plans import Unsupported, bucket, power_schedule, signed_literal
except ImportError:
    from runner_graph_plans import Unsupported, bucket, power_schedule, signed_literal


RESERVED = frozenset(("print", "abs", "float", "divmod", "min", "max", "round", "format",
                      "math", "Fraction", "Decimal", "range", "sum"))
DECIMAL = re.compile(r"([+-]?)([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")


@dataclass(frozen=True)
class Number:
    ref: int
    digits: int
    kind: str = "integer"
    scale: int = 0
    second: object = None


class Binder:
    """Keep constants and produced slots separate until the entire literal pool is known."""

    def __init__(self):
        self.values = []
        self.ids = {}
        self.ops = []
        self.produced = 0
        self.prints = []
        self.names = {}
        self.imports = set()
        self.calls = {}

    def limits(self):
        if len(self.ops) > 4096 or len(self.values) + self.produced > 8192 or len(self.prints) > 128:
            raise Unsupported("program exceeds native packet bounds")

    def literal(self, text, kind="integer", scale=0):
        negative = text.startswith("-")
        digits = text.lstrip("+-").lstrip("0") or "0"
        if not digits.isascii() or not digits.isdigit() or len(digits) > 128:
            raise Unsupported("literal exceeds 128 decimal digits")
        text = ("-" if negative and (digits != "0" or kind == "fixed") else "") + digits
        if text not in self.ids:
            self.ids[text] = len(self.values)
            self.values.append(text)
            self.limits()
        return Number(self.ids[text], len(digits), kind, scale)

    @staticmethod
    def integer(value):
        if not isinstance(value, Number) or value.kind != "integer":
            raise Unsupported("operation requires integer operands")
        return value

    def call(self, op, a, b=None, outputs=1, precision=0):
        storage = max(a.digits, b.digits if b is not None else 1)
        needed = storage
        if op in ("mul", "decimal_mul"):
            needed = a.digits + b.digits
        elif op in ("add", "sub"):
            needed += 1
        elif op == "fixed_div":
            needed = max(a.digits + precision + 1, b.digits)
        elif op == "scale10":
            needed += 1
        width = bucket(min(128, needed))
        key = (op, a.ref, b.ref if b is not None else None, width, outputs, precision, min(128, needed))
        if key in self.calls:
            return self.calls[key]
        row = {"op": op, "width": width, "a": a.ref, "outputs": outputs,
               "precision": precision, "checked": True}
        if b is not None:
            row["b"] = b.ref
        self.ops.append(row)
        result = tuple(Number(-(self.produced + i + 1), min(128, needed)) for i in range(outputs))
        self.produced += outputs
        self.limits()
        value = result[0] if outputs == 1 else result
        self.calls[key] = value
        return value

    def fraction(self, a, b):
        a, b = self.integer(a), self.integer(b)
        n, d = self.call("fraction", a, b, outputs=2)
        return Number(n.ref, n.digits, "fraction", second=d)

    def ratio(self, value):
        if value.kind == "fraction":
            return Number(value.ref, value.digits), value.second
        return self.integer(value), self.literal("1")

    def floating(self, a, b=None):
        """Abstain dynamically on fractional division or over 15 digits; do not emulate IEEE arithmetic."""
        value = self.call("divexact", a, b) if b is not None else a
        value = self.call("float_exact", value)
        return Number(value.ref, value.digits, "integral_float")

    def decimal(self, node):
        """Bind fixed-notation Decimal coefficients with scale at most 6 and default precision at most 28."""
        if not isinstance(node, ast.Constant) or type(node.value) is not str:
            raise Unsupported("Decimal requires a lexical decimal string")
        match = DECIMAL.fullmatch(node.value)
        if match is None:
            raise Unsupported("Decimal string is outside fixed decimal grammar")
        sign, body = match.groups()
        whole, dot, fraction = body.partition(".")
        scale = len(fraction) if dot else 0
        if scale > 6:
            raise Unsupported("Decimal scale may require exponent notation")
        return self.literal(sign + whole + fraction, "fixed", scale)

    def binary(self, node):
        if isinstance(node.op, ast.Pow):
            exponent = signed_literal(node.right)
            if exponent is None or exponent.startswith("-") or len(exponent) > 3:
                raise Unsupported("power requires a literal exponent in [0,999]")
            schedule = power_schedule(int(exponent))
            a = self.integer(self.expression(node.left))
            bindings = {"base": a, "one": self.literal("1")}
            if exponent == "0":
                return self.call("add", self.call("mul", a, self.literal("0")), bindings["one"])
            for i, step in enumerate(schedule["calls"]):
                bindings[i] = self.call("mul", bindings[step["a"]], bindings[step["b"]])
            return bindings[schedule["result"]]
        a, b = self.expression(node.left), self.expression(node.right)
        if not isinstance(a, Number) or not isinstance(b, Number):
            raise Unsupported("arithmetic requires numeric scalars")
        if a.kind == "fixed" or b.kind == "fixed":
            if a.kind != "fixed" or b.kind != "fixed" or not isinstance(node.op, ast.Mult):
                raise Unsupported("Decimal supports Decimal multiplication only")
            if a.digits + b.digits > 28 or a.scale + b.scale > 6:
                raise Unsupported("Decimal product exceeds exact default precision or scale")
            result = self.call("decimal_mul", a, b)
            return Number(result.ref, result.digits, "fixed", a.scale + b.scale)
        if a.kind == "fraction" or b.kind == "fraction":
            an, ad = self.ratio(a)
            bn, bd = self.ratio(b)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                numerator = self.call("add" if isinstance(node.op, ast.Add) else "sub",
                                      self.call("mul", an, bd), self.call("mul", bn, ad))
                denominator = self.call("mul", ad, bd)
            elif isinstance(node.op, (ast.Mult, ast.Div)):
                numerator = self.call("mul", an, bd if isinstance(node.op, ast.Div) else bn)
                denominator = self.call("mul", ad, bn if isinstance(node.op, ast.Div) else bd)
            else:
                raise Unsupported("unsupported Fraction arithmetic")
            return self.fraction(numerator, denominator)
        a, b = self.integer(a), self.integer(b)
        op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(node.op))
        if op:
            return self.call(op, a, b)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            pair = self.call("divmod", a, b, outputs=2)
            return pair[isinstance(node.op, ast.Mod)]
        if isinstance(node.op, ast.Div):
            return self.floating(a, b)
        raise Unsupported("unsupported arithmetic operator")

    def comparison(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise Unsupported("chained comparisons are unsupported")
        a = self.integer(self.expression(node.left))
        b = self.integer(self.expression(node.comparators[0]))
        order = self.call("compare", a, b)
        zero, one = self.literal("0"), self.literal("1")
        operator = node.ops[0]
        if isinstance(operator, (ast.Lt, ast.GtE)):
            result = self.call("sub", zero, self.call("min", order, zero))
        elif isinstance(operator, (ast.Gt, ast.LtE)):
            result = self.call("max", order, zero)
        elif isinstance(operator, (ast.Eq, ast.NotEq)):
            result = self.call("max", order, self.call("sub", zero, order))
        else:
            raise Unsupported("unsupported comparison")
        if isinstance(operator, (ast.GtE, ast.LtE, ast.Eq)):
            result = self.call("sub", one, result)
        return Number(result.ref, result.digits, "boolean")

    def invocation(self, node):
        if node.keywords:
            raise Unsupported("numeric calls do not accept keywords")
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "math" and "math" in self.imports and func.attr == "gcd"):
            name = "gcd"
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            raise Unsupported("call is outside the numeric allowlist")
        if name == "Decimal" and name in self.imports and len(node.args) == 1:
            return self.decimal(node.args[0])
        if name == "Fraction" and name in self.imports and 1 <= len(node.args) <= 2:
            values = [self.expression(arg) for arg in node.args]
            if any(not isinstance(value, Number) for value in values):
                raise Unsupported("Fraction requires numeric scalars")
            if len(values) == 1:
                if values[0].kind == "fraction":
                    return values[0]
                values.append(self.literal("1"))
            return self.fraction(*values)
        if name in ("abs", "float") and len(node.args) == 1:
            value = self.expression(node.args[0])
            if not isinstance(value, Number):
                raise Unsupported("numeric conversion requires a scalar")
            if name == "float":
                if value.kind == "fraction":
                    return self.floating(*self.ratio(value))
                return self.floating(self.integer(value))
            value = self.integer(value)
            return self.call("max", value, self.call("sub", self.literal("0"), value))
        if name in ("divmod", "gcd", "min", "max") and len(node.args) == 2:
            if name == "gcd" and not isinstance(func, ast.Attribute):
                raise Unsupported("gcd requires imported math.gcd")
            a, b = (self.integer(self.expression(arg)) for arg in node.args)
            return self.call(name, a, b, outputs=2 if name == "divmod" else 1)
        raise Unsupported("call is outside the numeric allowlist")

    def expression(self, node):
        text = signed_literal(node)
        if text is not None:
            return self.literal(text)
        if isinstance(node, ast.Name) and node.id in self.names:
            return self.names[node.id]
        if isinstance(node, ast.BinOp):
            return self.binary(node)
        if isinstance(node, ast.Compare):
            return self.comparison(node)
        if isinstance(node, ast.Call):
            return self.invocation(node)
        if isinstance(node, ast.Attribute) and node.attr in ("numerator", "denominator"):
            value = self.expression(node.value)
            if isinstance(value, Number) and value.kind == "fraction":
                return Number(value.ref, value.digits) if node.attr == "numerator" else value.second
            raise Unsupported("attributes require a Fraction")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.integer(self.expression(node.operand))
            return value if isinstance(node.op, ast.UAdd) else self.call("sub", self.literal("0"), value)
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            return tuple(self.integer(self.expression(item)) for item in node.elts)
        raise Unsupported("expression is outside the numeric allowlist")

    def formatted(self, node):
        if not isinstance(node, ast.JoinedStr) or len(node.values) != 1:
            return self.expression(node), None, 0
        field = node.values[0]
        if not isinstance(field, ast.FormattedValue) or field.conversion != -1:
            raise Unsupported("formatted output requires one numeric field without conversion")
        spec = field.format_spec
        if (not isinstance(spec, ast.JoinedStr) or len(spec.values) != 1
                or not isinstance(spec.values[0], ast.Constant) or type(spec.values[0].value) is not str):
            raise Unsupported("format specification must be literal")
        text = spec.values[0].value
        if text in (",", ",d", "d"):
            return self.integer(self.expression(field.value)), "comma" if text != "d" else "integer", 0
        match = re.fullmatch(r"\.([0-9]{1,2})f", text)
        if match is None or int(match[1]) > 18:
            raise Unsupported("unsupported numeric format")
        places = int(match[1])
        value = self.expression(field.value)
        if not isinstance(value, Number) or value.kind not in ("integer", "integral_float"):
            raise Unsupported("fixed formatting requires an integer or guarded exact float")
        if value.kind == "integer":
            value = self.call("float_exact", value)
        for _ in range(places):
            value = self.call("scale10", value)
        return value, "fixed", places

    def output(self, node):
        if not node.args:
            raise Unsupported("print requires numeric arguments")
        if len(node.keywords) > 1:
            raise Unsupported("print accepts only one end keyword")
        ending = "newline"
        for keyword in node.keywords:
            if (keyword.arg != "end" or not isinstance(keyword.value, ast.Constant)
                    or type(keyword.value.value) is not str or keyword.value.value not in ("\n", " ", "", ", ")):
                raise Unsupported("print supports only a bounded literal end")
            ending = {"\n": "newline", " ": "space", "": "empty", ", ": "comma_space"}[keyword.value.value]
        for i, node_arg in enumerate(node.args):
            value, style, precision = self.formatted(node_arg)
            row = {"ending": ending if i + 1 == len(node.args) else "space", "prefix": False}
            if isinstance(value, tuple):
                row.update(ref=value[0].ref, second=value[1].ref, style="tuple", precision=0)
            else:
                row.update(ref=value.ref, style=style or value.kind, precision=precision if style else value.scale)
                if value.kind == "fraction":
                    row["second"] = value.second.ref
            self.prints.append(row)
            self.limits()

    def assign(self, target, value):
        if isinstance(target, ast.Name) and target.id not in RESERVED and not target.id.startswith("__"):
            self.names[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) and len(target.elts) == len(value):
            for item, part in zip(target.elts, value):
                self.assign(item, part)
        else:
            raise Unsupported("assignment target is reserved or unsupported")

    def statement(self, node):
        if isinstance(node, ast.Import):
            if len(node.names) != 1 or node.names[0].name != "math" or node.names[0].asname is not None:
                raise Unsupported("only import math is allowed")
            self.imports.add("math")
        elif isinstance(node, ast.ImportFrom):
            allowed = {"fractions": "Fraction", "decimal": "Decimal"}
            if (node.level or node.module not in allowed or len(node.names) != 1
                    or node.names[0].name != allowed[node.module] or node.names[0].asname is not None):
                raise Unsupported("only Fraction and Decimal imports are allowed")
            self.imports.add(node.names[0].name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            self.assign(node.targets[0], self.expression(node.value))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            expression = ast.BinOp(left=ast.Name(id=node.target.id), op=node.op, right=node.value)
            self.assign(node.target, self.expression(expression))
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"):
            self.output(node.value)
        else:
            raise Unsupported("statement is outside the offline numeric grammar")

    def finish(self):
        if not self.prints:
            raise Unsupported("program requires numeric output")
        offset = len(self.values)

        def reference(ref):
            return ref if ref >= 0 else offset - ref - 1

        ops = []
        for original in self.ops:
            row = dict(original)
            for key in ("a", "b"):
                if key in row:
                    row[key] = reference(row[key])
            ops.append(row)
        prints = []
        for original in self.prints:
            row = dict(original)
            for key in ("ref", "second"):
                if key in row:
                    row[key] = reference(row[key])
            prints.append(row)
        return {"version": 1, "values": self.values, "ops": ops, "prints": prints}


def compile_program(source):
    """Compile fresh source into a native v1 packet; unsupported syntax raises Unsupported without execution."""
    if not isinstance(source, str) or not source or len(source) > 4096:
        raise Unsupported("program length is outside [1,4096]")
    try:
        tree = ast.parse(source, mode="exec")
        if sum(1 for _ in ast.walk(tree)) > 2048:
            raise Unsupported("program exceeds 2048 syntax nodes")
        binder = Binder()
        for statement in tree.body:
            binder.statement(statement)
        return binder.finish()
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, Unsupported):
            raise
        raise Unsupported("invalid or unsupported numeric program") from exc
