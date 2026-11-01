"""Build reusable unsigned division steps and signed Python floor correction graphs."""

from runner_graph_ir import Value


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
