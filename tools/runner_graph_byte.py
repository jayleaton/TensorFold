"""Checked numeric-byte output compiled into FFNs; native scheduling never constructs answer bytes."""

from runner_graph_ir import Value


def check(b, value):
    magnitude = sum((10 ** i * digit for i, digit in enumerate(value.digits[:3])), b.zero)
    error = b.any([value.error, value.sign, b.nz(value.digits[3:]), b.ge(magnitude, 256, 0, 999)])
    return Value(value.digits, b.zero, error)


def finish(b, value):
    value = b._latch(0, check(b, value))
    magnitude = sum((10 ** i * digit for i, digit in enumerate(value.digits[:3])), b.zero)
    count = b.c.counter("output.cursor", b.c.fam["GO"], b.c.fam["BYTE"])
    n = sum(b.c.thermo(count, b.c.fam["GO"], 7, "output.cursor"), b.zero)
    halt = b.choose(value.error, b.ge(n, 6, 0, 7), b.ge(n, 1, 0, 7))
    payload = []
    for bit, terms in enumerate(b.c.fn_bits(magnitude, 0, 255, lambda d: d, 8, "byte.ascii")):
        normal = b.node(terms, "byte.ascii")
        error = b.node([term for i, byte in enumerate(b"error\n") if (byte >> bit) & 1
                        for term in b.c.t_eq(n, i, 0, 7)], "byte.error")
        payload.append(b.choose(value.error, error, normal))
    b._exit(halt, payload)
    b.c.nodes.sort(key=lambda node: node.level)
    return {"width": b.width, "input_slots": b.input_slots, "output_slots": b.output_slots}
