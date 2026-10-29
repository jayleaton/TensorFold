"""Read only the selected completed benchmark receipts; never contact the serving endpoint."""
import json
from pathlib import Path

root = Path.home() / "work/dsv41-bench/out/20261029T102349Z-2501898"
for arm in ("01-A", "02-C"):
    path = root / arm / "decode.json"
    doc = json.loads(path.read_text())
    print(arm, "greedy cells")
    for c in doc["cells"]:
        if c["temp"] == 0 and c["streams"] == 1:
            print({k: c.get(k) for k in ("work", "rep", "mean_tok_s", "mean_tpr", "tokens", "prompt_ids")})
    for stem in ("public-blackfold-code-0-r0", "public-blackfold-prose-0-r0", "public-bertholomus-code-0-r0"):
        path = root / arm / (stem + ".json")
        if path.exists():
            doc = json.loads(path.read_text())
            m = doc.get("measurement", {})
            print(stem, {k: m.get(k) for k in ("tokens", "decode_tok_s", "tpr", "ttft_s")})
