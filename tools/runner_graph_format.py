"""Generic runtime-controlled decimal ASCII formatting compiled entirely into FFN weights."""

ENDINGS = (b"\n", b" ", b"", b", ", b")\n", b"/", b") ", b")", b"), ")


def _length(b, digits):
    occupied = [b.ge(digit, 1, 0, 9) for digit in digits]
    lengths = [(flag, (i + 1) * b.one) for i, flag in enumerate(occupied)]
    while len(lengths) > 1:
        merged = []
        for i in range(0, len(lengths), 2):
            if i + 1 == len(lengths):
                merged.append(lengths[i])
            else:
                low, high = lengths[i:i + 2]
                merged.append((b.any([low[0], high[0]]), b.choose(high[0], high[1], low[1])))
        lengths = merged
    return b.choose(lengths[0][0], lengths[0][1], b.one)


def _cursor(b, limit):
    """Keep each counter's thermometer within the exact FP32 64/65 split used by the emitter."""
    c = b.c
    reset, step = c.fam["GO"], c.fam["BYTE"]
    pieces = []
    for offset in range(0, limit, 64):
        name = f"format.cursor.{offset}"
        count = c.counter(name, reset, step)
        size = min(65, limit - offset)
        flags = c.thermo(count, reset, size, name)
        pieces.extend(flags[:64])
        if offset + 64 < limit:
            reset = b.one - flags[-1]
            step = b.node([(1, c.fam["BYTE"], flags[-1])], "format.cursor.step")
    return b.node([(1, sum(pieces, b.zero), b.one)], "format.cursor")


def _controls(b, value, control, max_precision=18):
    digits = control.digits
    precision = b.node([(1, digits[0] + 10 * digits[1], b.one)], "format.precision")
    modes = [b.eq(digits[2], mode, 0, 9) for mode in range(5)]
    endings = [b.eq(digits[3], ending, 0, 9) for ending in range(len(ENDINGS))]
    prefix = digits[4]
    precision_nz = b.ge(precision, 1, 0, 99)
    fixed = b.node([(1, modes[1], precision_nz)], "format.fixed")
    invalid_precision = b.node([(1, b.one - modes[1], precision_nz)], "format.precision.invalid")
    invalid_bool = b.node([(1, modes[4], b.any([b.nz(value.digits[1:]),
                           b.ge(value.digits[0], 2, 0, 9),
                           b.node([(1, value.sign, b.nz(value.digits))], "format.bool.negative")]))],
                          "format.bool.invalid")
    error = b.any([value.error, control.error, control.sign, b.nz(digits[5:]),
                   b.ge(precision, max_precision + 1, 0, 99), b.ge(digits[2], 5, 0, 9),
                   b.ge(digits[3], len(ENDINGS), 0, 9), b.ge(prefix, 2, 0, 9), invalid_precision, invalid_bool])
    if b.width > 128:
        precision = b.choose(error, b.zero, precision)
        prefix = b.choose(error, b.zero, prefix)
        fixed = b.choose(error, b.zero, fixed)
    return precision, modes, endings, prefix, fixed, error


def _layout(b, value, precision, modes, fixed):
    width = b.width
    length = _length(b, value.digits)
    padded = b.choose(b.ge(precision - length, 0, -width, 98), precision + b.one, length)
    coefficient_length = b.choose(modes[1], padded, length)
    groups = b.node([(1, b.ge(length, 3 * i + 1, 1, width), b.one)
                     for i in range(1, (width - 1) // 3 + 1)], "format.groups")
    numeric_length = b.choose(modes[2], length + groups, coefficient_length + fixed)
    numeric_length = b.choose(modes[3], length + 2 * b.one, numeric_length)
    truth = b.ge(value.digits[0], 1, 0, 9)
    body_length = b.choose(modes[4], 5 * b.one - truth, numeric_length)
    preserve = b.any([b.nz(value.digits), modes[1], modes[3]])
    sign = b.node([(1, value.sign, preserve)], "format.sign.nonzero")
    sign = b.node([(1, sign, b.one - modes[4])], "format.sign")
    return length, coefficient_length, groups, truth, body_length, sign


def _numeric(b, value, n, precision, modes, fixed, layout, prefix, limit):
    length, coefficient_length, groups, truth, body_length, sign = layout
    position = n - prefix - sign
    point = coefficient_length - precision
    after_point = b.ge(position - point, 1, -limit, limit)
    index = coefficient_length - b.one - position
    index = b.choose(fixed, index + after_point, index)
    reverse = length + groups - b.one - position
    digits = []
    for i, digit in enumerate(value.digits):
        normal = b.eq(index, i, -limit, limit)
        grouped = b.eq(reverse, i + i // 3, -limit, limit)
        digits.append((1, digit, b.choose(modes[2], grouped, normal)))
    digit = b.node(digits, "format.digit")
    comma = b.node([(1, modes[2], b.any(b.eq(reverse, 4 * i - 1, -limit, limit)
                                       for i in range(1, (b.width - 1) // 3 + 1)))], "format.comma")
    dot = b.node([(1, fixed, b.eq(position - point, 0, -limit, limit)),
                  (1, modes[3], b.eq(position - length, 0, -limit, limit))], "format.dot")
    float_zero = b.node([(1, modes[3], b.eq(position - length, 1, -limit, limit))], "format.float.zero")
    minus = b.node([(1, sign, b.eq(n - prefix, 0, -limit, limit))], "format.minus")
    return digit, comma, dot, float_zero, minus, position


def _emit(b, value, control, max_precision=18):
    precision, modes, endings, prefix, fixed, error = _controls(b, value, control, max_precision)
    layout = _layout(b, value, precision, modes, fixed)
    length, coefficient_length, groups, truth, body_length, sign = layout
    limit = b.width + (b.width - 1) // 3 + 8
    n = _cursor(b, limit)
    digit, comma, dot, float_zero, minus, position = _numeric(
        b, value, n, precision, modes, fixed, layout, prefix, limit)
    tail = n - prefix - sign - body_length
    suffix_length = b.node([(len(suffix), flag, b.one) for suffix, flag in zip(ENDINGS, endings)],
                           "format.ending.length")
    done = b.ge(tail - suffix_length, 0, -limit, limit)
    halt = b.choose(error, b.ge(n, 6, 0, limit), done)
    opening = b.node([(1, b.eq(prefix, 1, 0, 9), b.eq(n, 0, 0, limit))], "format.opening")
    ascii_bits = b.c.fn_bits(digit, 0, 9, lambda d: 48 + d, 8, "format.ascii")
    payload = []
    for bit, terms in enumerate(ascii_bits):
        normal = b.node(terms, "format.ascii")
        for flag, byte in ((comma, 44), (dot, 46), (float_zero, 48), (minus, 45)):
            normal = b.choose(flag, ((byte >> bit) & 1) * b.one, normal)
        boolean = b.node([(1, flag, b.eq(position, i, -limit, limit))
                          for text, flag in ((b"True", truth), (b"False", b.one - truth))
                          for i, byte in enumerate(text) if (byte >> bit) & 1], "format.bool")
        normal = b.choose(modes[4], boolean, normal)
        normal = b.choose(opening, ((40 >> bit) & 1) * b.one, normal)
        for suffix, flag in zip(ENDINGS, endings):
            for i, byte in enumerate(suffix):
                at = b.node([(1, flag, b.eq(tail, i, -limit, limit))], "format.ending.at")
                normal = b.choose(at, ((byte >> bit) & 1) * b.one, normal)
        err = b.node([term for i, byte in enumerate(b"error\n") if (byte >> bit) & 1
                      for term in b.c.t_eq(n, i, 0, limit)], "format.error")
        payload.append(b.choose(error, err, normal))
    b._exit(halt, payload)


def finish(builder, inputs, max_precision=18):
    """Latch coefficient and control as two output Values before emitting any BYTE steps."""
    if builder.width not in (62, 128) or len(inputs) != 2:
        raise ValueError("format requires width 62 or 128 and two input Values")
    latched = [builder._latch(i, value, preserve_sign=True) for i, value in enumerate(inputs)]
    _emit(builder, *latched, max_precision=max_precision)
    builder.c.nodes.sort(key=lambda node: node.level)
    return {"width": builder.width, "input_slots": builder.input_slots, "output_slots": builder.output_slots}
