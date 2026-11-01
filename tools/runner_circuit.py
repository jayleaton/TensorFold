"""Read and inspect the checksummed TensorFold P2 circuit container."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


HEADER = struct.Struct("<8sIIQII32s")
P2_LAYOUT = (("V", 32768), ("F0", 32768), ("F1", 32768), ("F2", 4),
             ("J", 65536), ("R", 65536), ("BYTE", 512), ("FLT", 8), ("HEX", 16),
             ("QS", 16), ("QD", 128), ("BD", 256), ("NT", 1), ("ST0", 1), ("ST1", 1),
             ("SS0", 1), ("SS1", 1), ("HALT", 1), ("GO", 1), ("SC", 1))
CONSTANTS = {"one_dim": 0, "head_dim": 128, "a_ref": 16777216.0, "a_bit": 128.0,
             "ref_cols": [0, 1], "swiglu_k": 128.0, "kappa": 2.0 ** -20}


def read_circuit(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated circuit header")
    magic, version, size, length, flags, reserved, digest = HEADER.unpack_from(data)
    if (magic, version, size, flags, reserved) != (b"TFIMR001", 1, 64, 0, 0):
        raise ValueError("unsupported circuit header")
    if length != len(data) - size:
        raise ValueError("circuit payload length mismatch")
    payload = data[size:]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("circuit payload checksum mismatch")

    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError(f"duplicate object key: {k}")
            obj[k] = v
        return obj

    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")

    doc = json.loads(payload.decode("utf-8"), parse_constant=invalid, object_pairs_hook=pairs)
    validate(doc)
    return doc


def validate(doc):
    if doc["format"] != "tensorfold.runner.p2" or doc["version"] != 1:
        raise ValueError("unsupported circuit payload")
    dims = len(doc["dims"])
    if dims != doc["stats"]["dims"] or not dims:
        raise ValueError("dimension count mismatch")
    if doc.get("constants") != CONSTANTS:
        raise ValueError("unsupported or missing execution constants")
    if doc["dims"][0] != {"name": "one", "kind": "one", "level": 0}:
        raise ValueError("invalid constant dimension")

    def index(v, limit):
        if type(v) is not int or not 0 <= v < limit:
            raise ValueError(f"index {v} outside [0,{limit})")

    def real(v):
        if type(v) not in (int, float) or not math.isfinite(v):
            raise ValueError("nonfinite or invalid weight")

    def expr(e):
        seen = set()
        for d, w in e:
            index(d, dims)
            real(w)
            if d in seen:
                raise ValueError("duplicate expression dimension")
            seen.add(d)

    def cs(channels):
        seen = set()
        for col, channel in channels:
            index(col, 128)
            if col in seen or len(channel["taps"]) != 4:
                raise ValueError("invalid channel")
            seen.add(col)
            expr(channel["expr"])
            for tap in channel["taps"]:
                real(tap)

    families = doc["tokens"]["families"]
    if families != doc["output"]["families"]:
        raise ValueError("inconsistent family codes")
    end = 0
    if len(families) != len(P2_LAYOUT):
        raise ValueError("unsupported P2 families")
    for f, (name, size) in zip(families, P2_LAYOUT):
        if f["base"] != end or type(f["size"]) is not int or not 0 < f["size"] <= 65536:
            raise ValueError("invalid family interval")
        if f["name"] != name or f["size"] != size:
            raise ValueError("unsupported P2 family encoding")
        end += f["size"]
    if end != doc["tokens"]["vocab"]:
        raise ValueError("vocabulary size mismatch")
    token_ids = {f["name"]: f["base"] for f in families}
    if (doc["tokens"]["go"], doc["tokens"]["halt"], doc["tokens"]["byte_family"]) != (
            token_ids["GO"], token_ids["HALT"], 6):
        raise ValueError("invalid control token identities")
    bits = doc["output"]["family_bits"]
    if bits != 5 or doc["output"]["payload_bits"] != 16:
        raise ValueError("invalid output code width")
    if len(doc["output"]["code_dims"]) != bits + 16:
        raise ValueError("output dimensions mismatch")
    for d in doc["output"]["code_dims"]:
        index(d, dims)
    if 0 in doc["output"]["code_dims"] or len(set(doc["output"]["code_dims"])) != bits + 16:
        raise ValueError("output code dimensions must be unique and nonconstant")
    seen = set()
    for f in doc["features"]:
        index(f["dim"], dims)
        if f["dim"] == 0 or f["dim"] in seen:
            raise ValueError("duplicate or constant input feature")
        seen.add(f["dim"])
        kind = f["kind"]
        if kind in ("family", "payload_bit"):
            index(f["family"], len(families))
            if kind == "payload_bit":
                index(f["bit"], 16)
        elif kind == "next_digit":
            index(f["digit"], 8)
            index(f["qs_family"], len(families))
            index(f["qd_family"], len(families))
        elif kind == "code_bit":
            index(f["bit"], bits + 16)
            if f["family_bits"] != bits or f["amplitude"] != 128:
                raise ValueError("invalid token code feature")
        else:
            raise ValueError("unsupported input feature")
    for n in doc["nodes"]:
        if n["kind"] == "ffn":
            if type(n["into"]) is not bool:
                raise ValueError("invalid FFN output mode")
            index(n["out"], dims)
            for w, a, b in n["terms"]:
                real(w)
                expr(a)
                expr(b)
        elif n["kind"] == "head":
            if n["n_cols"] != 128 or n["levels"] not in ("bit", "byte", "decay"):
                raise ValueError("invalid head geometry")
            if not isinstance(n["value_heads"], list) or not n["value_heads"]:
                raise ValueError("head must have at least one value head")
            real(n["tiny"])
            cs(n["q"])
            cs(n["k"])
            for vh in n["value_heads"]:
                if type(vh["into"]) is not bool:
                    raise ValueError("invalid head output mode")
                cs(vh["v"])
                expr(vh["a"])
                if vh["beta"] is not None:
                    expr(vh["beta"])
                for col, e in vh["z"]:
                    index(col, 128)
                    expr(e)
                for col, d, w in vh["out"]:
                    index(col, 128)
                    index(d, dims)
                    real(w)
                rows = set()
                for row, values in vh["init"]:
                    index(row, 128)
                    if row in rows:
                        raise ValueError("duplicate state row")
                    rows.add(row)
                    cols = set()
                    for col, value in values:
                        index(col, 128)
                        if col in cols:
                            raise ValueError("duplicate state column")
                        cols.add(col)
                        real(value)
        else:
            raise ValueError("unsupported circuit node")
    p = doc["program"]
    if p["kind"] not in ("calc", "elf") or p["entry"] % 4 or not 0x40000 <= p["entry"] < 0x80000:
        raise ValueError("unsupported program kind or entry")
    if (p["input_address"], p["input_capacity"]) != (256, 256):
        raise ValueError("invalid calc input buffer")
    if (p["rom_base"], p["rom_words"], p["drom_base"], p["ram_base"], p["ram_bytes"]) != (
            0x40000, 65536, 0x20000, 0, 32768):
        raise ValueError("unsupported P2 address map")
    if len(bytes.fromhex(p["rom_le_hex"])) != 262144 or len(bytes.fromhex(p["ram_hex"])) != 32768:
        raise ValueError("invalid program image size")
    if len(bytes.fromhex(p["drom_hex"])) > 131072:
        raise ValueError("DROM image too large")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("circuit", type=Path)
    args = ap.parse_args()
    try:
        doc = read_circuit(args.circuit)
    except (ValueError, KeyError, TypeError, OSError) as e:
        ap.exit(1, f"invalid circuit: {e}\n")
    print(json.dumps({"stats": doc["stats"], "source": doc["source"],
                      "program": {k: v for k, v in doc["program"].items() if not k.endswith("hex")}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
