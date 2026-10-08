#!/usr/bin/env bash
# The single-pass window's served checks (quant repo results/zig-window-plan.md, step 3), one server on URL, in order:
#   1. replies: diverge.py ask --bench (prompts.txt, thinking on, + the bench's 8 prompts) -> replies.jsonl; for a label
#      other than prod: diverge.py compare against out/bench-prod/replies.jsonl (ids: return_token_ids), must be 16/16
#   2. speed: code and prose, T0 and T0.7, 1 and 4 streams, 2 reps; one bench_http.py run a cell, phases.json copied
#      after each (cumulative: phases/<n>-<cell>.json)
#   3. long prompts: 32K and 131K, one stream each (prompt tok/s, TTFT, decode, reply sha, tokens a round)
#   4. one 4-stream long-context check: a 131K prompt while 4 code streams run (no error, both finish)
#   5. a label other than prod: --compare tables against bench-prod and the verdict (Zig >= Python on every cell, every
#      T0 reply equal, replies 16/16) in verdict.txt
# Run on dgx-01 from the kit dir:
#   ./zig-bench.sh prod      before the window, Python prod up (the reference replies and cells)
#   ./zig-bench.sh zig       in the window, the Zig server up with TF_DSV41_PHASES=1 TF_DSV41_PHASES_OUT=/state/phases.json
# Env: URL (http://127.0.0.1:8000), PHASES (the host path of /state/phases.json: /home/dgx01/ai/dsv41-tf/phases.json),
# REPS (2), LONG (32768,131072), MIX_LONG (131072). Exit 0 always unless the server is unreachable: the verdict is data.
set -u
LABEL=${1:?label: prod | zig}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
URL=${URL:-http://127.0.0.1:8000}; PH=${PHASES:-/home/dgx01/ai/dsv41-tf/phases.json}
OUT=$KIT/out/bench-$LABEL; PROD=$KIT/out/bench-prod
mkdir -p "$OUT/cells" "$OUT/phases"
B="python3 $KIT/bench_http.py --url $URL --label $LABEL"
D="python3 $KIT/tools/zig/dsv41_diverge/diverge.py"
log() { echo "[zig-bench $(date +%T)] $*" | tee -a "$OUT/bench.log"; }
n=0
phases() { n=$((n + 1)); [[ -f "$PH" ]] && cp "$PH" "$OUT/phases/$(printf %02d $n)-$1.json"; }
curl -sf -m 10 "$URL/v1/models" > /dev/null || { log "no server on $URL"; exit 2; }
t0=$(date +%s)

log "1. replies (diverge.py ask --bench)"
$D ask --url "$URL" --prompts "$KIT/prompts.txt" --bench --out "$OUT/replies.jsonl" >> "$OUT/bench.log" 2>&1 || log "ask failed"
phases replies
if [[ "$LABEL" != prod ]]; then
    if [[ -f "$PROD/replies.jsonl" ]]; then
        $D compare "$PROD/replies.jsonl" "$OUT/replies.jsonl" > "$OUT/compare.txt" 2>&1
        log "replies vs prod: $(tail -1 "$OUT/compare.txt")"
    else log "no $PROD/replies.jsonl: run ./zig-bench.sh prod before the window"; fi
fi

log "2. speed cells"
for work in code prose; do for temp in 0 0.7; do for s in 1 4; do
    c="$work-t$temp-s$s"
    $B --work "$work" --temps "$temp" --streams "$s" --reps "${REPS:-2}" --out "$OUT/cells/$c.json" > /dev/null 2>> "$OUT/bench.log" \
        || log "cell $c failed"
    phases "$c"
    log "  $c: $(python3 -c "import json,sys;print([c['mean_tok_s'] if c['streams']==1 else c['aggregate_tok_s'] for c in json.load(open(sys.argv[1]))['cells']])" "$OUT/cells/$c.json" 2>/dev/null)"
done; done; done
python3 - "$OUT/cells" "$OUT/cells.json" "$LABEL" <<'PY'
import glob, json, sys
cells = [c for f in sorted(glob.glob(sys.argv[1] + "/*.json")) for c in json.load(open(f))["cells"]]
json.dump({"label": sys.argv[3], "cells": cells}, open(sys.argv[2], "w"), indent=1)
PY

log "3. long prompts (${LONG:-32768,131072})"
$B --long "${LONG:-32768,131072}" --out "$OUT/long.json" > /dev/null 2>> "$OUT/bench.log" || log "long cells failed"
phases long

log "4. 4-stream long-context check (${MIX_LONG:-131072} + 4 code streams)"
$B --long "${MIX_LONG:-131072}" --out "$OUT/mix-long.json" > /dev/null 2>> "$OUT/bench.log" &
pl=$!; sleep 20
$B --work code --temps 0 --streams 4 --reps 1 --out "$OUT/mix-short.json" > /dev/null 2>> "$OUT/bench.log"; rs=$?
wait "$pl"; rl=$?
log "  long rc $rl, 4 streams rc $rs"
phases mix

if [[ "$LABEL" != prod && -f "$PROD/cells.json" ]]; then
    log "5. against prod"
    { python3 "$KIT/bench_http.py" --compare "$PROD/cells.json" "$OUT/cells.json"; echo
      [[ -f "$PROD/long.json" ]] && python3 "$KIT/bench_http.py" --compare "$PROD/long.json" "$OUT/long.json"; } > "$OUT/compare.md" 2>&1
    python3 - "$PROD" "$OUT" > "$OUT/verdict.txt" <<'PY'
import json, os, sys
p, z = sys.argv[1:]
def best(f):
    out = {}
    for c in json.load(open(f))["cells"]:
        k = (c["work"], c["temp"], c["streams"])
        v = c["mean_tok_s"] if c["streams"] == 1 else (c["aggregate_tok_s"] or c["mean_tok_s"])
        out[k] = max(out.get(k, 0.0), v or 0.0)
    return out
def shas(f):
    return {(c["work"], c["temp"], c["streams"], c.get("rep")): c.get("sha") for c in json.load(open(f))["cells"] if c["temp"] == 0}
worse, rows = [], []
for f in ("cells.json", "long.json"):
    if not (os.path.exists(f"{p}/{f}") and os.path.exists(f"{z}/{f}")):
        continue
    a, b = best(f"{p}/{f}"), best(f"{z}/{f}")
    for k in sorted(a):
        if k in b:
            rows.append(f"{k}: prod {a[k]:.1f} zig {b[k]:.1f}")
            if b[k] < a[k]:
                worse.append(k)
    if f == "long.json":   # the prompt's tok/s too
        pa = {c["work"]: c.get("prompt_tok_s") or 0 for c in json.load(open(f"{p}/{f}"))["cells"]}
        pb = {c["work"]: c.get("prompt_tok_s") or 0 for c in json.load(open(f"{z}/{f}"))["cells"]}
        for k in pa:
            if k in pb:
                rows.append(f"{k} prompt: prod {pa[k]} zig {pb[k]}")
                if pb[k] < pa[k]:
                    worse.append((k, "prompt"))
sa, sb = shas(f"{p}/cells.json"), shas(f"{z}/cells.json")
diff = [k for k in sa if k in sb and sa[k] != sb[k]]
cmp = open(f"{z}/compare.txt").read().strip().splitlines()[-1] if os.path.exists(f"{z}/compare.txt") else "no compare"
print("\n".join(rows))
print(f"replies: {cmp}")
print(f"T0 cells with a differing reply: {len(diff)} {diff[:8]}")
print(f"cells where Zig is slower: {len(worse)} {worse}")
ok = not worse and not diff and cmp.startswith("16/16")
print("VERDICT: Zig beats Python on every cell, replies equal" if ok else "VERDICT: Zig does NOT beat Python on every cell (or replies differ): restore perf1 unless the user says otherwise")
PY
    log "$(tail -1 "$OUT/verdict.txt")"
fi
log "done in $(( ($(date +%s) - t0) / 60 )) min: $OUT"
