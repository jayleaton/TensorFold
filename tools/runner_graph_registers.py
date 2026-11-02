"""Compact generated integer templates in place; callers select the optional 256-digit exports."""

import heapq
import math


def _require(ok, message):
    if not ok:
        raise ValueError(f"register compaction: {message}")


def _array(value, size=None):
    _require(type(value) is list and (size is None or len(value) == size), "invalid array shape")
    return value


def _object(value, keys):
    _require(type(value) is dict and set(value) == set(keys.split()), "unknown object shape")


def _uint(value, limit=1 << 32):
    _require(type(value) is int and 0 <= value < limit, "invalid index or count")
    return value


def _real(value):
    _require(type(value) in (int, float) and math.isfinite(value), "invalid numeric weight")


def _expr(expr, dims):
    seen = set()
    for pair in _array(expr):
        dim, weight = _array(pair, 2)
        _uint(dim, dims)
        _real(weight)
        _require(dim not in seen, "duplicate expression dimension")
        seen.add(dim)
    return seen


def _channels(channels, dims):
    seen = set()
    for pair in _array(channels):
        col, channel = _array(pair, 2)
        _uint(col, 128)
        _require(col not in seen, "duplicate channel")
        seen.add(col)
        _object(channel, "expr taps")
        _expr(channel["expr"], dims)
        for tap in _array(channel["taps"], 4):
            _real(tap)


def _validate_node(node, dims):
    _require(type(node) is dict, "node must be an object")
    kind = node.get("kind")
    _require(kind in ("ffn", "head"), "unknown node kind")
    _uint(node["level"])
    _require(type(node["name"]) is str, "invalid node name")
    if kind == "ffn":
        _object(node, "kind name level out into terms")
        _uint(node["out"], dims)
        _require(node["out"] != 0 and type(node["into"]) is bool, "invalid FFN output")
        for term in _array(node["terms"]):
            coef, up, gate = _array(term, 3)
            _real(coef)
            _expr(up, dims)
            _expr(gate, dims)
        return
    _object(node, "kind name level n_cols levels tiny q k value_heads")
    _require(node["n_cols"] == 128 and node["levels"] in ("bit", "byte", "decay"), "invalid head geometry")
    _real(node["tiny"])
    _channels(node["q"], dims)
    _channels(node["k"], dims)
    _require(bool(_array(node["value_heads"])), "empty value heads")
    for vh in node["value_heads"]:
        _object(vh, "v a beta z out into init")
        _require(type(vh["into"]) is bool, "invalid head output mode")
        _channels(vh["v"], dims)
        _expr(vh["a"], dims)
        if vh["beta"] is not None:
            _expr(vh["beta"], dims)
        cols = set()
        for pair in _array(vh["z"]):
            col, expr = _array(pair, 2)
            _uint(col, 128)
            _require(col not in cols, "duplicate z column")
            cols.add(col)
            _expr(expr, dims)
        for out in _array(vh["out"]):
            col, dim, weight = _array(out, 3)
            _uint(col, 128)
            _uint(dim, dims)
            _require(dim != 0, "head writes ONE")
            _real(weight)
        rows = set()
        for pair in _array(vh["init"]):
            row, values = _array(pair, 2)
            _uint(row, 128)
            _require(row not in rows, "duplicate state row")
            rows.add(row)
            cols = set()
            for entry in _array(values):
                col, value = _array(entry, 2)
                _uint(col, 128)
                _require(col not in cols, "duplicate state column")
                cols.add(col)
                _real(value)


def _expressions(node):
    if node["kind"] == "ffn":
        for _, up, gate in node["terms"]:
            yield up
            yield gate
        return
    for key in ("q", "k"):
        for _, channel in node[key]:
            yield channel["expr"]
    for vh in node["value_heads"]:
        for _, channel in vh["v"]:
            yield channel["expr"]
        yield vh["a"]
        if vh["beta"] is not None:
            yield vh["beta"]
        for _, expr in vh["z"]:
            yield expr


def _writes(node):
    if node["kind"] == "ffn":
        yield node["out"], node["into"]
    else:
        for vh in node["value_heads"]:
            for _, dim, _ in vh["out"]:
                yield dim, vh["into"]


def _reads(node):
    for expr in _expressions(node):
        for dim, _ in expr:
            yield dim
    for dim, into in _writes(node):
        if into:
            yield dim


def _analyse(doc):
    _require(type(doc) is dict, "document must be an object")
    _require(doc["format"] in ("tensorfold.runner.arithmetic", "tensorfold.runner.p2"), "unknown format")
    _require(doc["version"] == 1, "unknown version")
    dims = _array(doc["dims"])
    count = len(dims)
    _uint(count)
    _require(count > 0 and doc["stats"]["dims"] == count, "dimension count mismatch")
    _require(doc["constants"]["one_dim"] == 0, "ONE must be dimension zero")
    _require(dims[0] == {"name": "one", "kind": "one", "level": 0}, "invalid ONE metadata")
    for dim in dims:
        _object(dim, "name kind level")
        _require(type(dim["name"]) is str and type(dim["kind"]) is str, "invalid dimension metadata")
        _uint(dim["level"])
    pins = {0}
    feature_dims = set()
    for feature in _array(doc["features"]):
        _require(type(feature) is dict and feature.get("kind") in
                 ("family", "payload_bit", "next_digit", "code_bit"), "unknown feature shape")
        fields = {"family": "family", "payload_bit": "family bit",
                  "next_digit": "digit qs_family qd_family", "code_bit": "bit family_bits amplitude"}
        kind = feature["kind"]
        _object(feature, "dim kind " + fields[kind])
        dim = _uint(feature["dim"], count)
        _require(dim != 0 and dim not in feature_dims, "duplicate or constant feature dimension")
        feature_dims.add(dim)
        for field in fields[kind].split():
            if field == "amplitude":
                _real(feature[field])
            else:
                _uint(feature[field])
        pins.add(dim)
    code_dims = set()
    for dim in _array(doc["output"]["code_dims"]):
        _uint(dim, count)
        _require(dim != 0 and dim not in code_dims, "duplicate or constant code dimension")
        code_dims.add(dim)
        pins.add(dim)
    first = [-1] * count
    last = [-1] * count
    units = {}
    for index, node in enumerate(_array(doc["nodes"])):
        _validate_node(node, count)
        for dim in _reads(node):
            if first[dim] == -1:
                pins.add(dim)
            last[dim] = index
        for dim, into in _writes(node):
            if into:
                pins.add(dim)
            if first[dim] == -1:
                first[dim] = index
            last[dim] = index
        if node["kind"] == "ffn":
            level = node["level"]
            units[level] = units.get(level, 0) + len(node["terms"])
    pins.update(dim for dim, start in enumerate(first) if start == -1)
    return pins, last, units


class _Registers:
    def __init__(self, dims, pins, last):
        self.mapping = [-1] * len(dims)
        self.physical = []
        self.owners = []
        self.active = []
        self.free = []
        self.last = last
        self.dims = dims
        self.reuse = 0
        self.barrier = None
        for dim in sorted(pins):
            self.mapping[dim] = self._fresh(dim)
        _require(self.mapping[0] == 0, "ONE mapping changed")

    def _fresh(self, dim):
        physical = len(self.physical)
        self.physical.append(self.dims[dim])
        self.owners.append(dim)
        return physical

    def expire(self, index):
        while self.active and self.active[0][0] < index:
            _, dim = heapq.heappop(self.active)
            physical = self.mapping[dim]
            _require(self.owners[physical] == dim, "expired register has a live alias")
            self.owners[physical] = None
            heapq.heappush(self.free, physical)

    def checked(self, dim):
        physical = self.mapping[dim]
        _require(physical >= 0 and self.owners[physical] == dim, "read or write aliases a live register")
        return physical

    def allocate(self, dim):
        if self.mapping[dim] >= 0:
            return self.checked(dim), False
        reused = bool(self.free)
        if reused:
            physical = heapq.heappop(self.free)
            _require(self.owners[physical] is None, "free register still owned")
            self.owners[physical] = dim
            self.reuse += 1
        else:
            physical = self._fresh(dim)
        self.mapping[dim] = physical
        heapq.heappush(self.active, (self.last[dim], dim))
        return physical, reused

    def barrier_node(self, reused, level):
        if self.barrier is None:
            self.barrier = len(self.physical)
            self.physical.append({"name": "register.barrier", "kind": "ffn", "level": 0})
            self.owners.append("barrier")
        return {"kind": "ffn", "name": "register.barrier", "level": level - level % 2,
                "out": self.barrier, "into": False,
                "terms": [[0.0, [[dim, 0.0] for dim in sorted(reused)], [[0, 1.0]]]]}


def _remap(node, registers):
    for expr in _expressions(node):
        seen = set()
        for pair in expr:
            pair[0] = registers.checked(pair[0])
            _require(pair[0] not in seen, "remapped expression contains an alias")
            seen.add(pair[0])
    if node["kind"] == "ffn":
        node["out"] = registers.checked(node["out"])
    else:
        for vh in node["value_heads"]:
            for out in vh["out"]:
                out[1] = registers.checked(out[1])


def compact(doc):
    """Reuse dead registers without scheduling nodes; return counts, never a full mapping or runtime claim."""
    pins, last, units = _analyse(doc)
    registers = _Registers(doc["dims"], pins, last)
    original_dims = len(doc["dims"])
    nodes = doc["nodes"]
    result = []
    zero_deps = barriers = 0
    for index, node in enumerate(nodes):
        registers.expire(index)
        inputs = {registers.checked(dim) for dim in _reads(node)}
        reused = set()
        for dim, _ in _writes(node):
            physical, recycled = registers.allocate(dim)
            if recycled:
                _require(physical not in inputs, "reused output aliases a current input")
                reused.add(physical)
        _remap(node, registers)
        zero_deps += len(reused)
        if reused and node["kind"] == "ffn" and node["terms"]:
            up = node["terms"][0][1]
            _require(not reused.intersection(pair[0] for pair in up), "zero dependency duplicates an input")
            up.extend([physical, 0.0] for physical in sorted(reused))
        elif reused:
            barrier = registers.barrier_node(reused, node["level"])
            result.append(barrier)
            level = barrier["level"]
            units[level] = units.get(level, 0) + 1
            barriers += 1
        result.append(node)
    for feature in doc["features"]:
        feature["dim"] = registers.checked(feature["dim"])
    code = doc["output"]["code_dims"]
    for index, dim in enumerate(code):
        code[index] = registers.checked(dim)
    nodes[:] = result
    doc["dims"][:] = registers.physical
    stats = doc["stats"]
    stats["dims"] = len(registers.physical)
    stats["ffn_units"] = sum(units.values())
    stats["max_units"] = max(units.values(), default=0)
    return {"original_dims": original_dims, "physical_dims": stats["dims"],
            "pins": len(pins) + int(registers.barrier is not None), "reuse": registers.reuse,
            "zero_deps": zero_deps, "barriers": barriers}
