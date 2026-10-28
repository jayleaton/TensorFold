"""Native frozen-wire controls for issue 50; the retired Spark fixture is not imported."""
from pathlib import Path
import importlib.util
import json
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_wire", ROOT / "zig/tests/server/wire.py")
wire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wire)


class HealthWireTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / "zig/tests/server/golden/status/get-health.json"
        self.reply = json.loads(path.read_text())["replies"][0]
        self.reply["body"] = self.reply["body"].encode()
        self.reply["interim"] = b""

    def changed(self, body):
        reply = dict(self.reply, body=json.dumps(body).encode())
        reply["headers"] = [(k, str(len(reply["body"])) if k.lower() == "content-length" else v)
                            for k, v in self.reply["headers"]]
        return wire.normal(reply)

    def test_native_health_fixture_has_no_historical_request_totals(self):
        body = json.loads(self.reply["body"])
        counters = {"requests", "requests_total", "prompt_tokens_total", "completion_tokens_total",
                    "cached_tokens_total", "rounds_total", "drafted_total", "accepted_total"}
        self.assertFalse(counters.intersection(body))
        self.assertEqual(self.changed(body), wire.normal(self.reply))

    def test_missing_malformed_and_nonvolatile_health_changes_remain_red(self):
        original = json.loads(self.reply["body"])
        for key, value in (("model", "wrong-model"), ("max_batch_size", "4"), ("live", None)):
            with self.subTest(key=key):
                changed = dict(original, **{key: value})
                self.assertNotEqual(self.changed(changed), wire.normal(self.reply))
        missing = dict(original)
        del missing["status"]
        self.assertNotEqual(self.changed(missing), wire.normal(self.reply))

    def test_normalization_does_not_erase_usage_or_counting_regressions(self):
        one = dict(self.reply, body=b'{"usage": {"prompt_tokens": 10}, "requests_total": 1}')
        two = dict(self.reply, body=b'{"usage": {"prompt_tokens": 11}, "requests_total": 2}')
        self.assertNotEqual(wire.normal(one)["body"], wire.normal(two)["body"])


if __name__ == "__main__":
    unittest.main()
