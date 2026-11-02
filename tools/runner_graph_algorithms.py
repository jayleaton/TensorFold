"""Offline integer algorithm kernels; every arithmetic value and branch predicate is an FFN output."""

from runner_graph_ir import Value


OPERATIONS = ("meta", "half", "digit_sum", "mulmodstep")
ALGORITHMS = {"powmod": 3, "gcd": 2, "isqrt": 1, "range_product": 2,
              "factorial": 1, "digit_sum": 1, "divmod": 2, "square_offset": 2,
              "fraction": 2, "divexact": 2}


def build(b, op):
    zero = b.zero
    width = b.width
    arity = {"meta": 2, "half": 1, "digit_sum": 1, "mulmodstep": 4}[op]
    inputs = [b.input(i) for i in range(arity)]
    error = b.any(value.error for value in inputs)
    if op == "mulmodstep":
        remainder, a, digit, modulus = inputs
        product, overflow = b.mul(a.digits, digit.digits[:1], width + 1)
        total, carry = b.add([zero] + remainder.digits, product)
        total += [carry]
        multiple = b._pad(modulus.digits, width + 2)
        multiples = [multiple]
        for _ in range(4):
            multiple, _ = b.add(multiple, multiple)
            multiples.append(multiple)
        for multiple in reversed(multiples):
            difference, borrow = b.sub(total, multiple)
            total = [b.choose(borrow, old, new) for old, new in zip(total, difference)]
        _, r_less = b.sub(remainder.digits, modulus.digits)
        _, a_less = b.sub(a.digits, modulus.digits)
        error = b.any([error, *(value.sign for value in inputs), overflow,
                       b.one - b.nz(modulus.digits), b.one - r_less, b.one - a_less,
                       b.nz(digit.digits[1:]), b.nz(total[width:])])
        return Value(total[:width], zero, error)
    if op == "digit_sum":
        rows = [[digit] for digit in inputs[0].digits]
        while len(rows) > 1:
            merged = []
            for i in range(0, len(rows), 2):
                if i + 1 == len(rows):
                    merged.append(rows[i])
                else:
                    digits, carry = b.add(rows[i], rows[i + 1])
                    merged.append(digits + [carry])
            rows = merged
        return Value(rows[0], zero, error)
    if op == "meta":
        a, c = inputs
        _, less = b.sub(a.digits, c.digits)
        equal = b.one - b.any(b.one - b.eq(x - y, 0, -9, 9)
                                 for x, y in zip(a.digits, c.digits))
        flags = [b.nz(a.digits), b.nz(c.digits), a.sign, c.sign, less, equal]
        return [Value(a.digits, zero, error), Value(c.digits, zero, error),
                Value(flags, zero, error)]
    if op == "half":
        a = inputs[0]
        halves, parity = [], []
        for digit in a.digits:
            halves.append(b.node([term for threshold in (2, 4, 6, 8)
                                  for term in b.c.t_ge(digit, threshold, 0, 9)], "half.digit"))
            parity.append(b.node([(1, digit - 2 * halves[-1], b.one)], "half.parity"))
        result = [b.node([(1, half + 5 * (parity[i + 1] if i + 1 < width else zero), b.one)],
                         "half.quotient") for i, half in enumerate(halves)]
        error = b.any([error, a.sign])
        return [Value(result, zero, error), Value([parity[0]], zero, error)]
    raise ValueError(f"unsupported integer kernel: {op}")


def catalog(widths):
    plans = [{"name": f"{name}-{width}", "inputs": arity, "literals": [], "ops": [],
             "result": 0, "algorithm": name, "width": width}
            for name, arity in ALGORITHMS.items() for width in widths]
    plans += [{"name": f"fixed_div-{places}-{width}", "inputs": 2, "literals": [], "ops": [],
               "result": 0, "algorithm": "fixed_div", "width": width, "precision": places}
              for width in widths for places in range(19)]
    return plans
