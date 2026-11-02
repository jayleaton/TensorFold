"""Isolate selected P2 ALU and memory blocks using recorded upstream builder contracts."""

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from types import FunctionType


CORE_IR_SHA256 = "013975703c945d07e88dc237405d6fd06ab8640d820cdfa3dc857171f1e609b6"
BLOCKS = (("muldiv", ("div.", "mul."), ()),
          ("shift", ("shL", "shR", "shA", "shift.", "zba."), ()),
          ("logic", ("logic.",), ()), ("load", ("load.",), ()),
          ("merge", ("merge.",), ()), ("btgt", ("btgt.",), ()),
          ("ram", (), ("ram",)), ("drom", ("drom.",), ("drom",)))


def expression_key(e):
    return tuple(sorted(e.terms.items()))


def terms_key(terms):
    return tuple((w, expression_key(a), expression_key(b)) for w, a, b in terms)


def graph_key(c, ir):
    """Canonical static IR identity, excluding variable head snapshots and Python callback identities."""
    def canonical(value):
        if isinstance(value, Fraction):
            return {"fraction": [value.numerator, value.denominator]}
        if isinstance(value, (list, tuple)):
            return tuple(canonical(item) for item in value)
        return value

    nodes = []
    for n in c.nodes:
        if isinstance(n, ir.FFN):
            body = (n.out, n.into, tuple((w, expression_key(a), expression_key(b)) for w, a, b in n.terms))
        elif isinstance(n, ir.Head):
            cs = lambda channels: tuple((i, expression_key(v.expr), tuple(v.taps)) for i, v in channels.items())
            heads = tuple((cs(h.v), expression_key(h.a), None if h.beta is None else expression_key(h.beta),
                           tuple((i, expression_key(e)) for i, e in h.z.items()), tuple(h.out), h.into)
                          for h in n.vheads())
            body = (cs(n.q), cs(n.k), heads, n.n_cols, n.tiny, n.levels)
        else:
            raise ValueError("operand isolation: unsupported node type")
        nodes.append((type(n).__name__, n.name, n.level, body))
    return canonical((tuple((d.name, d.kind, d.level) for d in c.dims), tuple(nodes), tuple(c.token_fns),
                      tuple(sorted(c.full_fns)), tuple(expression_key(e) for e in c.code), tuple(c.units.items())))


def core_ir_digest(c, ir):
    payload = json.dumps(graph_key(c, ir), separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_isolatable(core, ir):
    """Bind recorded arithmetic lemmas to the source-reviewed pinned IR, rather than arbitrary lo/hi claims."""
    from chipformer.dsl.graph import as_expr

    class TaggedCore(core.RvCore):
        def __init__(self, *args, **kwargs):
            self._isolation_gates = defaultdict(list)
            self._isolation_bounds = {}
            self._isolation_support = {}
            self._isolation_zero = set()
            self._isolation_opcodes = {}
            self._isolation_equalities = []
            self._isolation_arithmetic = {}
            self._isolation_bytes = set()
            super().__init__(*args, **kwargs)

        def bound(self, e, lo, hi):
            e = as_expr(e)
            key = expression_key(e)
            old_lo, old_hi = self._isolation_bounds.get(key, (lo, hi))
            self._isolation_bounds[key] = (max(lo, old_lo), min(hi, old_hi))

        def record(self, terms, constant, sources):
            expected = constant * self.one
            for scale, e, _, _ in sources:
                expected = expected + scale * e
            for _, e, lo, hi in sources:
                self.bound(e, lo, hi)
            key = expression_key(expected)
            if any(expression_key(gate) == key for _, _, gate in terms):
                self._isolation_gates[key].append((constant, tuple(sources)))

        def t_and(self, lits, coef=1, a=None):
            lits = [as_expr(e) for e in lits]
            terms = super().t_and(lits, coef, a)
            self.record(terms, 2 - 4 * len(lits), [(4, e, 0, 1) for e in lits])
            return terms

        def t_ge(self, z, theta, lo, hi, coef=1, lits=()):
            z, lits = as_expr(z), [as_expr(e) for e in lits]
            terms = super().t_ge(z, theta, lo, hi, coef, lits)
            if lo < theta <= hi and hi - theta + 1 <= 256:
                m = 1 << (hi - lo).bit_length()
                sources = [(1, z, lo, hi)] + [(m, e, 0, 1) for e in lits]
                for offset in (1, 0):
                    self.record(terms, -m * len(lits) - theta + offset, sources)
            return terms

        def t_lt(self, z, theta, lo, hi, coef=1, lits=()):
            z, lits = as_expr(z), [as_expr(e) for e in lits]
            terms = super().t_lt(z, theta, lo, hi, coef, lits)
            if lo < theta <= hi and theta - lo <= 256:
                m = 1 << (hi - lo).bit_length()
                sources = [(-1, z, lo, hi)] + [(m, e, 0, 1) for e in lits]
                for offset in (0, -1):
                    self.record(terms, theta - m * len(lits) + offset, sources)
            return terms

        def zeros(self, n, name):
            result = super().zeros(n, name)
            self._isolation_zero.update(d for e in result for d in e.terms)
            return result

        def t_eq(self, z, value, lo, hi, coef=1, lits=()):
            z, lits = as_expr(z), list(lits)
            result = super().t_eq(z, value, lo, hi, coef, lits)
            self._isolation_equalities.append((expression_key(z), value, lo, hi, coef, lits, terms_key(result)))
            return result

        def node(self, terms, name):
            result = super().node(terms, name)
            if name.startswith("uop="):
                code = core.RF.U.get(name[4:])
                records = self._isolation_equalities[-2:]
                if code is None or len(records) != 2 or not 0 <= code < 128:
                    raise ValueError("operand isolation: invalid opcode decoder")
                a, b = records
                if (a[:6] != (b[0], code, 0, 255, 1, []) or
                        b[1:6] != (code + 128, 0, 255, 1, []) or terms_key(terms) != a[6] + b[6]):
                    raise ValueError("operand isolation: opcode equality formulas differ")
                self._isolation_opcodes[next(iter(result.terms))] = (code, a[0])
            return result

        def table(self, name, queries, keys, vheads, levels="bit", tiny=0.0):
            result = super().table(name, queries, keys, vheads, levels, tiny)
            node = self.nodes[-1]
            if levels == "byte" and not node.k and all(not h.v and not h.a.terms for h in node.vheads()):
                self._isolation_bytes.update(output_dims(node, ir))
            return result

    # Inject the instrumented factory into a private function namespace; upstream globals stay unchanged.
    build = core.build

    def stage1(c, sums, name, hi):
        result = core.stage1_digits(c, sums, name, hi)
        enables = set()
        partitions = []
        for forms in sums:
            partition = []
            for _, lits in forms:
                modes = [set(as_expr(e).terms) for e in lits if pure_opcodes(c, as_expr(e))]
                if len(lits) != 1 or len(modes) != 1 or enables.intersection(modes[0]):
                    partition = []
                    break
                partition.append(frozenset(modes[0]))
                enables.update(modes[0])
            partitions.append(tuple(partition))
            enables = set()
        if hi not in (30, 31) or not partitions or not partitions[0] or any(p != partitions[0] for p in partitions):
            return result
        enables = set().union(*partitions[0])
        for e in flatten(result):
            c.bound(e, 0, 1)
            if len(e.terms) == 1:
                d, weight = next(iter(e.terms.items()))
                if weight == 1:
                    node = next(n for n in reversed(c.nodes) if isinstance(n, ir.FFN) and n.out == d)
                    if not all(opcode_gate(gate, enables, c._isolation_gates) for _, _, gate in node.terms):
                        raise ValueError(f"operand isolation: unproved arithmetic support: {name}")
                    c._isolation_support[d] = frozenset(enables)
        c._isolation_arithmetic[tuple(expression_key(e) for e in flatten(result))] = len(sums)
        return result

    def stage2(c, *args):
        result = core.adder_stage2(c, *args)
        key = tuple(expression_key(e) for e in flatten(args[:3]))
        if c._isolation_arithmetic.get(key) == args[3]:
            for e in flatten(result):
                c.bound(e, 0, 1)
        return result

    namespace = dict(build.__globals__, RvCore=TaggedCore, stage1_digits=stage1, adder_stage2=stage2)
    function = core.add_nibbles
    namespace["add_nibbles"] = FunctionType(function.__code__, namespace, function.__name__,
                                           function.__defaults__, function.__closure__)
    wrapped = FunctionType(build.__code__, namespace, build.__name__, build.__defaults__, build.__closure__)
    c = wrapped()
    original = graph_key(c, ir)
    if core_ir_digest(c, ir) != CORE_IR_SHA256:
        raise ValueError("operand isolation: static core IR differs from its pinned construction certificate")
    decoded = list(c._isolation_opcodes.values())
    if (len(decoded) != len(core.RF.U) or len({code for code, _ in decoded}) != len(decoded) or
            len({byte for _, byte in decoded}) != 1):
        raise ValueError("operand isolation: opcode decoder is not a common-byte one-hot partition")
    c._isolation_original = original
    c._isolation_tokens = tuple(c.token_fns.items())
    return c


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def pure_opcodes(c, e):
    return bool(e.terms) and all(d in c._isolation_opcodes and w == 1 for d, w in e.terms.items())


def head_expressions(node):
    for channels in (node.q, node.k):
        yield from (channel.expr for channel in channels.values())
    for head in node.vheads():
        yield from (channel.expr for channel in head.v.values())
        yield head.a
        if head.beta is not None:
            yield head.beta
        yield from head.z.values()


def opcode_gate(gate, enables, contracts):
    active = {d: weight for d, weight in gate.terms.items() if d != 0 and weight != 0}
    if active and all(d in enables and weight > 0 for d, weight in active.items()):
        return gate.terms.get(0, 0) <= 0
    for constant, sources in contracts.get(expression_key(gate), ()):
        upper = constant
        has_enable = False
        for scale, e, lo, hi in sources:
            terms = {d: w for d, w in e.terms.items() if w != 0}
            if terms and all(d in enables and w > 0 for d, w in terms.items()):
                lo = hi = 0
                has_enable = True
            upper += scale * (hi if scale >= 0 else lo)
        if has_enable and upper <= 0:
            return True
    return False


def output_dims(n, ir):
    return {n.out} if isinstance(n, ir.FFN) else {d for h in n.vheads() for _, d, _ in h.out}


def dimension_bounds(c):
    bounds = {0: (1, 1)}
    for key, (lo, hi) in c._isolation_bounds.items():
        variables = [(d, w) for d, w in key if d != 0]
        if len(variables) != 1:
            continue
        d, weight = variables[0]
        constant = dict(key).get(0, 0)
        ends = (Fraction(lo - constant) / weight, Fraction(hi - constant) / weight)
        low, high = min(ends), max(ends)
        old_low, old_high = bounds.get(d, (low, high))
        bounds[d] = max(low, old_low), min(high, old_high)
        if bounds[d][0] > bounds[d][1]:
            raise ValueError("operand isolation: contradictory builder bounds")
    return bounds


def arithmetic_support(c, ir):
    support = dict(c._isolation_support)
    for n in c.nodes:
        if not isinstance(n, ir.FFN) or n.into or n.out in support:
            continue
        inputs = {d for _, a, b in n.terms for e in (a, b) for d in e.terms if d != 0}
        if inputs and all(d in support for d in inputs):
            zero = sum(w * a.terms.get(0, 0) * max(0, b.terms.get(0, 0)) for w, a, b in n.terms)
            if zero == 0:
                support[n.out] = frozenset().union(*(support[d] for d in inputs))
    return support


def key_pairs(c, heads, bounds, support):
    vectors = defaultdict(list)
    channels = [ch for n in heads for ch in n.k.values()]
    for i, ch in enumerate(channels):
        for d, w in ch.expr.terms.items():
            if d != 0 and d not in c._isolation_opcodes:
                vectors[d].append((i, w))
    groups = defaultdict(list)
    for d, vector in vectors.items():
        groups[tuple(vector)].append(d)
    pairs = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        seen = set()
        for d in group:
            if (c.dims[d].kind != "ffn" or bounds.get(d) != (0, 1) or
                    not support.get(d) or seen.intersection(support[d])):
                raise ValueError(f"operand isolation: unproved paired address: {c.dims[d].name}")
            seen.update(support[d])
        for d in group:
            pairs[d] = tuple(group)
    return pairs


def prove_key(c, channel, enables, pairs, bounds):
    upper, seen = channel.expr.terms.get(0, 0), set()
    for d, weight in channel.expr.terms.items():
        if d == 0 or d in enables:
            continue
        group = pairs.get(d, (d,))
        if group in seen:
            continue
        seen.add(group)
        if len(group) > 1:
            lo, hi = 0, 1
        elif d in c._isolation_opcodes:
            lo, hi = 0, 1
        elif d in bounds:
            lo, hi = bounds[d]
        else:
            raise ValueError(f"operand isolation: missing key input bounds: {c.dims[d].name}")
        upper += weight * (hi if weight >= 0 else lo)
    if upper > 0:
        raise ValueError(f"operand isolation: key can fire off-opcode (upper bound {upper})")


def lags(channel):
    return {i for i, weight in enumerate(channel.taps) if weight != 0}


def head_inputs(heads):
    key_lags = {lag for n in heads for ch in n.k.values() for lag in lags(ch)}
    readonly = not any(n.k for n in heads)
    if not readonly and len(key_lags) != 1:
        raise ValueError("operand isolation: expected one key lag")
    lag = None if readonly else next(iter(key_lags))
    keep = {0}
    for n in heads:
        for ch in n.q.values():
            if lags(ch) != {0} or ch.taps[0] != 1:
                raise ValueError(f"operand isolation: query has non-current taps: {n.name}")
        for ch in n.k.values():
            if lags(ch) != {lag} or ch.taps[lag] != 1:
                raise ValueError(f"operand isolation: unsupported key taps: {n.name}")
        for h in n.vheads():
            if h.a.terms or (readonly and h.v):
                raise ValueError(f"operand isolation: unsupported decay or read-only value writes: {n.name}")
            keep.update(h.a.terms)
            if h.beta is not None:
                keep.update(h.beta.terms)
            for ch in h.v.values():
                if lags(ch) != {lag}:
                    keep.update(ch.expr.terms)
    external = set()
    for n in heads:
        for ch in (*n.q.values(), *n.k.values()):
            external.update(set(ch.expr.terms) - keep)
        for h in n.vheads():
            for e in h.z.values():
                external.update(set(e.terms) - keep)
            for ch in h.v.values():
                if lags(ch) == {lag}:
                    external.update(set(ch.expr.terms) - keep)
    return external, lag


def plan_block(c, ir, spec, writers, bounds, support):
    name, fprefix, hprefix = spec
    ffns = {i for i, n in enumerate(c.nodes) if isinstance(n, ir.FFN) and n.name.startswith(fprefix)}
    head_ids = {i for i, n in enumerate(c.nodes) if isinstance(n, ir.Head) and n.name.startswith(hprefix)}
    members = ffns | head_ids
    if not members or (hprefix and not head_ids):
        raise ValueError(f"operand isolation: missing selected block: {name}")
    if any(c.nodes[i].into for i in ffns):
        raise ValueError(f"operand isolation: selected FFN adds into a dimension: {name}")
    outputs = set().union(*(output_dims(c.nodes[i], ir) for i in members))
    if 0 in outputs or any(outputs.intersection(e.terms) for e in c.code):
        raise ValueError(f"operand isolation: block writes constants or output code: {name}")
    for d in outputs:
        if any(i not in members for i in writers[d]) or (d in c.token_fns and d not in c._isolation_zero):
            raise ValueError(f"operand isolation: output has an external writer: {name}/{c.dims[d].name}")
    consumers, enables = [], set()
    for i, n in enumerate(c.nodes):
        if i in members:
            continue
        if isinstance(n, ir.Head):
            if any(outputs.intersection(e.terms) for e in head_expressions(n)):
                raise ValueError(f"operand isolation: outside head reads {name}: {n.name}")
        else:
            for _, a, b in n.terms:
                if outputs.intersection(a.terms) or outputs.intersection(b.terms):
                    consumers.append((n.name, b))
                    enables.update(d for d in b.terms if d in c._isolation_opcodes)
    heads = [c.nodes[i] for i in sorted(head_ids)]
    if any(outputs.intersection(e.terms) for n in heads for e in head_expressions(n)):
        raise ValueError(f"operand isolation: selected head reads its block outputs: {name}")
    external, lag = head_inputs(heads) if heads else (set(), None)
    for n in heads:
        for ch in n.k.values():
            key_enables = {d for d, w in ch.expr.terms.items() if w > 0 and d in c._isolation_opcodes}
            if not key_enables:
                raise ValueError(f"operand isolation: key has no positive opcode: {n.name}")
            enables.update(key_enables)
    if not enables:
        raise ValueError(f"operand isolation: no derived enables: {name}")
    for consumer, gate in consumers:
        if not opcode_gate(gate, enables, c._isolation_gates):
            raise ValueError(f"operand isolation: unproved external gate: {name}/{consumer}")
    pairs = key_pairs(c, heads, bounds, support)
    for n in heads:
        for ch in n.k.values():
            prove_key(c, ch, enables, pairs, bounds)
    external.update(d for i in ffns for _, a, b in c.nodes[i].terms for e in (a, b) for d in e.terms)
    external.difference_update(outputs | {0})
    first = min(members)
    for d in external | enables:
        if any(i >= first for i in writers[d]):
            raise ValueError(f"operand isolation: late input writer: {name}/{c.dims[d].name}")
        if not writers[d] and d not in c.token_fns:
            raise ValueError(f"operand isolation: input has no producer: {name}/{c.dims[d].name}")
    return dict(name=name, ffns=ffns, heads=head_ids, first=first, inputs=sorted(external),
                enables=sorted(enables), lag=lag, pairs=len(set(pairs.values())))


def rewrite_head(n, lag, ir, rewrite):
    def channels(items):
        return {i: ir.Chan(rewrite(ch.expr), ch.taps) for i, ch in items.items()}

    q, k = channels(n.q), channels(n.k)
    heads = []
    for h in n.vheads():
        v = {i: ir.Chan(rewrite(ch.expr) if lags(ch) == {lag} else ch.expr, ch.taps)
             for i, ch in h.v.items()}
        heads.append(ir.VHead(v, h.a, {i: rewrite(e) for i, e in h.z.items()}, list(h.out), h.beta, h.into))
    return q, k, heads


def isolate_blocks(c, ir):
    """Plan every block on the untouched graph, then apply all plans in a fixed deterministic order."""
    from chipformer.dsl.graph import Expr

    if graph_key(c, ir) != c._isolation_original or tuple(c.token_fns.items()) != c._isolation_tokens:
        raise ValueError("operand isolation: graph changed after recording builder contracts")
    if any(d.name.startswith(("iso:", "iso.")) for d in c.dims):
        raise ValueError("operand isolation: graph already contains isolation dimensions")
    writers = defaultdict(list)
    for i, n in enumerate(c.nodes):
        for d in output_dims(n, ir):
            writers[d].append(i)
    for d, (_, byte) in c._isolation_opcodes.items():
        if (len(writers[d]) != 1 or not isinstance(c.nodes[writers[d][0]], ir.FFN) or
                c.nodes[writers[d][0]].into or c.dims[d].kind != "ffn"):
            raise ValueError("operand isolation: opcode has an additional or incompatible writer")
        if len(byte) != 1 or byte[0][1] != 256 or byte[0][0] not in c._isolation_bytes:
            raise ValueError("operand isolation: opcode source is not the pinned read-only byte construction")
        source = byte[0][0]
        if source not in c._isolation_zero or c.dims[source].kind != "token":
            raise ValueError("operand isolation: decoder byte lacks its zero-initialized accumulator")
        for i in writers[source]:
            n = c.nodes[i]
            if not isinstance(n, ir.Head) or n.levels != "byte" or n.k:
                raise ValueError("operand isolation: decoder byte has an unsupported producer")
    bounds, support = dimension_bounds(c), arithmetic_support(c, ir)
    plans = [plan_block(c, ir, spec, writers, bounds, support) for spec in BLOCKS]
    selected = set()
    for p in plans:
        members = p["ffns"] | p["heads"]
        if selected.intersection(members):
            raise ValueError("operand isolation: overlapping selected blocks")
        selected.update(members)
    dimensions, inserts, ffn_changes, head_changes = [], defaultdict(list), {}, {}
    units = dict(c.units)
    for p in plans:
        replacements = {}
        # Execution follows stored node order; synthetic kind/level metadata is descriptive, not physical placement.
        level = c.nodes[p["first"]].level
        level += level % 2
        for d in p["inputs"]:
            original = c.dims[d]
            isolated = len(c.dims) + len(dimensions)
            replacements[d] = isolated
            dimensions.append(ir.Dim(f"iso.{p['name']}:{original.name}", original.kind, original.level))
            terms = [(1, Expr({d: 1}), Expr({u: 1 for u in p["enables"]}))]
            inserts[p["first"]].append(ir.FFN(f"iso.{p['name']}.{original.name}", terms, isolated, level, False))
        units[level] = units.get(level, 0) + len(p["inputs"])

        def rewrite(e):
            return Expr({replacements.get(d, d): weight for d, weight in e.terms.items()})

        for i in p["ffns"]:
            ffn_changes[i] = [(w, rewrite(a), rewrite(b)) for w, a, b in c.nodes[i].terms]
        for i in p["heads"]:
            head_changes[i] = rewrite_head(c.nodes[i], p["lag"], ir, rewrite)
    for i, terms in ffn_changes.items():
        c.nodes[i].terms = terms
    for i, (q, k, heads) in head_changes.items():
        n, first = c.nodes[i], heads[0]
        n.q, n.k, n.v, n.z, n.vx = q, k, first.v, first.z, heads[1:]
    c.dims.extend(dimensions)
    c.nodes = [node for i, n in enumerate(c.nodes) for node in (*inserts[i], n)]
    c.units = units
    report = [dict(block=p["name"], ffns=len(p["ffns"]), heads=len(p["heads"]), inputs=len(p["inputs"]),
                   enables=[c.dims[d].name for d in p["enables"]], key_lag=p["lag"], address_pairs=p["pairs"])
              for p in plans]
    attributes = ("gates", "bounds", "support", "zero", "original", "opcodes", "equalities",
                  "arithmetic", "bytes", "tokens")
    for attribute in attributes:
        delattr(c, "_isolation_" + attribute)
    return report
