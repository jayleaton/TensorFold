"""Whole-cell decimal kernels compiled exclusively from Builder FFN operations."""

from runner_graph_ir import Value


OPERATIONS = ("round_div", "decimal_mul", "compare", "min", "max", "scale10", "float_exact", "reverse_decimal", "assert_true")


def build(b, op):
    arity = 4 if op == "round_div" else 1 if op in ("scale10", "float_exact", "reverse_decimal", "assert_true") else 2
    inputs = [b.input(i, preserve_sign=True) for i in range(arity)]
    error = b.any(value.error for value in inputs)
    if op == "assert_true":
        value = inputs[0]
        return Value(value.digits, value.sign, b.any([error, b.one - b.nz(value.digits)]))
    if op == "reverse_decimal":
        a = b.normalize(inputs[0])
        suffix = [b.zero] * (b.width + 1)
        for i in reversed(range(b.width)):
            suffix[i] = b.any([b.ge(a.digits[i], 1, 0, 9), suffix[i + 1]])
        lengths = [b.node([(1, suffix[i], b.one - suffix[i + 1])], "reverse.length")
                   for i in range(b.width)]
        digits = [b.node([(1, a.digits[end - i], lengths[end]) for end in range(i, b.width)],
                         "reverse.digit") for i in range(b.width)]
        return Value(digits, b.zero, b.any([error, a.sign]))
    if op == "float_exact":
        a = inputs[0]
        return Value(a.digits, a.sign, b.any([error, b.nz(a.digits[15:])]))
    if op == "scale10":
        a = inputs[0]
        return Value([b.zero] + a.digits[:-1], a.sign, b.any([error, b.nz(a.digits[-1:])]))
    if op == "round_div":
        return round_div(b, inputs)
    a, c = inputs
    if op == "decimal_mul":
        digits, overflow = b.mul(a.digits, c.digits, b.width)
        sign = b.one - b.eq(a.sign - c.sign, 0, -1, 1)
        return Value(digits, sign, b.any([error, overflow]))
    a, c = b.normalize(a), b.normalize(c)
    _, magnitude_less = b.sub(a.digits, c.digits)
    equal = b.one - b.any(b.one - b.eq(x - y, 0, -9, 9) for x, y in zip(a.digits, c.digits))
    same_sign = b.eq(a.sign - c.sign, 0, -1, 1)
    negative_less = b.one - b.any([magnitude_less, equal])
    less = b.choose(same_sign, b.choose(a.sign, negative_less, magnitude_less), a.sign)
    if op == "compare":
        same = b.node([(1, equal, same_sign)], "compare.equal")
        return Value([b.one - same] + [b.zero] * (b.width - 1), less, error)
    if op in ("min", "max"):
        return b.select(less if op == "min" else b.one - less, a, c)
    raise ValueError(f"unsupported whole-cell kernel: {op}")


def round_div(b, inputs):
    """Round unsigned Q/R half-even and preserve numerator XOR denominator sign, including zero."""
    quotient, remainder, numerator, denominator = inputs
    twice, high = b.add(remainder.digits, remainder.digits)
    twice.append(high)
    divisor = denominator.digits + [b.zero]
    _, below = b.sub(twice, divisor)
    _, above = b.sub(divisor, twice)
    tie = b.one - b.any([below, above])
    odd = b.any(b.eq(quotient.digits[0], digit, 0, 9) for digit in (1, 3, 5, 7, 9))
    increment = b.any([above, b.node([(1, tie, odd)], "round.tie.odd")])
    digits, overflow = b.add(quotient.digits, [increment] + [b.zero] * (b.width - 1))
    _, valid_remainder = b.sub(remainder.digits, denominator.digits)
    error = b.any([*(value.error for value in inputs), quotient.sign, remainder.sign,
                   b.one - b.nz(denominator.digits), b.one - valid_remainder, overflow])
    sign = b.one - b.eq(numerator.sign - denominator.sign, 0, -1, 1)
    return Value(digits, sign, error)


def plans(widths):
    """Offline compare-exchange schedules compute sum(abs(a[i]-a[j])) for lists of up to 16 integers."""
    result = []
    for width in widths:
        for size in range(17):
            literals = [str(i) for i in range(max(1, size))]
            zero = size
            values = list(range(size))
            ops = []

            def call(op, a, c):
                ref = size + len(literals) + len(ops)
                ops.append({"op": op, "a": a, "b": c, "width": width, "checked": True})
                return ref

            for i in range(1, size):
                for j in range(i, 0, -1):
                    a, c = values[j - 1:j + 1]
                    values[j - 1] = call("min", a, c)
                    values[j] = call("max", a, c)
            prefix = answer = zero
            for i, value in enumerate(values):
                product = call("mul", value, size + i)
                difference = call("sub", product, prefix)
                answer = call("add", answer, difference)
                prefix = call("add", prefix, value)
            result.append({"name": f"loop-sorted_distance-{size}-{width}", "inputs": size,
                           "literals": literals, "ops": ops, "result": answer})
    return result
