"""The P2 container rejects corrupted and malformed weight artifacts before evaluation."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from runner_circuit import CONSTANTS, P2_LAYOUT, read_circuit
from runner_export import check_elf, write_circuit


def document():
    families, base = [], 0
    for name, size in P2_LAYOUT:
        families.append({"name": name, "base": base, "size": size})
        base += size
    ids = {f["name"]: f["base"] for f in families}
    return {"format": "tensorfold.runner.p2", "version": 1, "stats": {"dims": 22},
            "constants": copy.deepcopy(CONSTANTS),
            "dims": [{"name": "one", "kind": "one", "level": 0}] * 22,
            "tokens": {"families": families, "vocab": base, "go": ids["GO"],
                       "halt": ids["HALT"], "byte_family": 6},
            "output": {"families": families, "family_bits": 5, "payload_bits": 16,
                       "code_dims": list(range(1, 22))}, "features": [],
            "nodes": [{"kind": "ffn", "out": 1, "into": False,
                       "terms": [[1.0, [[0, 1.0]], [[0, 1.0]]]]}],
            "program": {"rom_base": 0x40000, "rom_words": 65536, "drom_base": 0x20000,
                        "kind": "calc", "entry": 0x40000, "input_address": 256, "input_capacity": 256,
                        "ram_base": 0, "ram_bytes": 32768, "rom_le_hex": "00" * 262144,
                        "ram_hex": "00" * 32768, "drom_hex": ""}}


class CircuitContainer(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "circuit.tfpc"
        self.doc = document()
        write_circuit(self.doc, self.path)
        self.data = self.path.read_bytes()

    def rewrite_payload(self, doc):
        payload = json.dumps(doc, separators=(",", ":")).encode()
        header = struct.pack("<8sIIQII32s", b"TFIMR001", 1, 64, len(payload), 0, 0,
                             hashlib.sha256(payload).digest())
        self.path.write_bytes(header + payload)

    def test_deterministic_file_and_sidecar(self):
        result = write_circuit(self.doc, self.path)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertEqual(read_circuit(self.path), self.doc)
        self.assertEqual(hashlib.sha256(self.data).hexdigest(), result["sha256"])
        self.assertEqual(self.path.with_suffix(".tfpc.sha256").read_text(),
                         f"{result['sha256']}  circuit.tfpc\n")

    def test_byte_corruption_is_not_a_valid_weight_change(self):
        changed = bytearray(self.data)
        changed[-2] ^= 1
        self.path.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_circuit(self.path)

    def test_truncation_and_trailing_bytes(self):
        for data in (self.data[:63], self.data[:-1], self.data + b"\0"):
            with self.subTest(length=len(data)):
                self.path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_circuit(self.path)

    def test_version_flags_and_endian(self):
        for offset, replacement in ((8, struct.pack("<I", 2)), (24, struct.pack("<I", 1)),
                                    (28, struct.pack("<I", 1)), (8, struct.pack(">I", 1))):
            with self.subTest(offset=offset, replacement=replacement):
                data = bytearray(self.data)
                data[offset:offset + 4] = replacement
                self.path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "header"):
                    read_circuit(self.path)

    def test_valid_checksum_cannot_hide_invalid_dimensions(self):
        for dim in (-1, 22, True):
            with self.subTest(dim=dim):
                doc = copy.deepcopy(self.doc)
                doc["nodes"][0]["terms"][0][1] = [[dim, 1.0]]
                self.rewrite_payload(doc)
                with self.assertRaisesRegex(ValueError, "index"):
                    read_circuit(self.path)

    def test_valid_checksum_cannot_hide_nonfinite_weights(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][0]["terms"][0][0] = float("inf")
        self.rewrite_payload(doc)
        with self.assertRaisesRegex(ValueError, "number"):
            read_circuit(self.path)

    def test_duplicate_expression_dimensions(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"][0]["terms"][0][1] = [[0, 1.0], [0, 2.0]]
        self.rewrite_payload(doc)
        with self.assertRaisesRegex(ValueError, "duplicate expression"):
            read_circuit(self.path)

    def test_unknown_features(self):
        doc = copy.deepcopy(self.doc)
        doc["features"] = [{"dim": 1, "kind": "execute_instruction"}]
        self.rewrite_payload(doc)
        with self.assertRaisesRegex(ValueError, "unsupported input feature"):
            read_circuit(self.path)

    def test_execution_constants_and_control_ids(self):
        for mutate in (lambda doc: doc.pop("constants"),
                       lambda doc: doc["tokens"].update(go=0),
                       lambda doc: doc["constants"].update(swiglu_k=64.0)):
            doc = copy.deepcopy(self.doc)
            mutate(doc)
            self.rewrite_payload(doc)
            with self.assertRaisesRegex(ValueError, "constants|identities"):
                read_circuit(self.path)

    def test_compressed_or_float_abi_elf_is_refused(self):
        for flags in (1, 2, 4, 6, 8):
            with self.subTest(flags=flags):
                data = bytearray(52)
                data[:7] = b"\x7fELF\x01\x01\x01"
                struct.pack_into("<HH", data, 16, 2, 243)
                struct.pack_into("<I", data, 36, flags)
                path = Path(self.directory.name) / "unsupported.elf"
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "ILP32 and uncompressed"):
                    check_elf(path, object())

    def test_empty_value_head_array_is_refused(self):
        doc = copy.deepcopy(self.doc)
        doc["nodes"] = [{"kind": "head", "n_cols": 128, "levels": "bit", "tiny": 0.0,
                         "q": [], "k": [], "value_heads": []}]
        self.rewrite_payload(doc)
        with self.assertRaisesRegex(ValueError, "at least one value head"):
            read_circuit(self.path)


if __name__ == "__main__":
    unittest.main()
