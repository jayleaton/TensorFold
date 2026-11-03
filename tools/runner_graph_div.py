"""Build reusable unsigned division steps and signed Python floor correction graphs."""

from runner_graph_ir import Value


def optional_plans(width):
    """Build checked wide division schedules without changing the default plan catalog."""
    from runner_graph_ir import OPTIONAL_BUCKETS

    if width not in OPTIONAL_BUCKETS:
        raise ValueError("division extension requires an optional width")
    ops, nextref, remainder, quotient = [], 3, 2, 2
    for digit in reversed(range(width)):
        ops.append({"op": "bind", "copies": [{"ref": 0, "from": digit, "to": 0, "digits": 1}], "flags": 0})
        extracted = nextref
        ops.append({"op": "divstep", "args": [remainder, 1, extracted],
                    "width": width, "outputs": 2, "checked": True})
        remainder, qdigit = nextref + 1, nextref + 2
        ops.append({"op": "bind", "copies": [{"ref": quotient, "from": 0, "to": 0, "digits": width},
                                               {"ref": qdigit, "from": 0, "to": digit, "digits": 1}],
                    "flags": qdigit})
        quotient = nextref + 3
        nextref += 4
    ops.append({"op": "divfinal", "args": [quotient, remainder, 0, 1],
                "width": width, "outputs": 2, "checked": True})
    plans = [{"name": f"{op}-{width}", "inputs": 2, "literals": ["0"], "ops": ops,
              "result": nextref + (op == "mod")} for op in ("div", "mod")]
    plans.append({"name": f"divmod-{width}", "inputs": 2, "literals": [], "ops": [],
                  "result": 0, "algorithm": "divmod", "width": width})
    return plans


def build(b, op):
    """Create division inputs and return outputs using the reusable Builder template hook."""
    if op not in ("divstep", "divfinal"):
        raise ValueError(f"unsupported division template: {op}")
    inputs = [b.input(i) for i in range(3 if op == "divstep" else 4)]
    return build_divstep(b, inputs) if op == "divstep" else build_divfinal(b, inputs)


def build_divstep(b, inputs):
    """Consume one dividend digit and return the unsigned remainder and quotient digit."""
    remainder, divisor, nextdigit = inputs
    width = len(remainder.digits)
    zero = b.one * 0
    shifted = [nextdigit.digits[0]] + list(remainder.digits)
    padded = list(divisor.digits) + [zero]
    quotient = zero
    for _ in range(9):
        difference, borrow = b.sub(shifted, padded)
        take = b.one - borrow
        shifted = [b.choose(take, a, c) for a, c in zip(difference, shifted)]
        quotient += take
    quotient = b.node([(1, b.one, quotient)], "division-digit")
    error = b.any([value.error for value in inputs] + [b.one - b.nz(divisor.digits), b.nz(shifted[width:])])
    return [Value(shifted[:width], zero, error),
            Value([quotient] + [zero] * (width - 1), zero, error)]


def build_divfinal(b, inputs):
    """Correct unsigned Q/R for Python signed floor division and modulo semantics."""
    quotient, remainder, dividend, divisor = inputs
    zero = b.one * 0
    width = len(quotient.digits)
    different = b.eq(dividend.sign + divisor.sign, 1, 0, 2)
    correct = b.choose(different, b.nz(remainder.digits), zero)
    increment = [correct] + [zero] * (width - 1)
    increased, carry = b.add(quotient.digits, increment)
    complement, borrow = b.sub(divisor.digits, remainder.digits)
    corrected = [b.choose(correct, a, c) for a, c in zip(complement, remainder.digits)]
    qsign = b.choose(b.nz(increased), different, zero)
    rsign = b.choose(b.nz(corrected), divisor.sign, zero)
    error = b.any([value.error for value in inputs] + [b.one - b.nz(divisor.digits), carry,
                  b.choose(correct, borrow, zero)])
    return [Value(increased, qsign, error), Value(corrected, rsign, error)]
