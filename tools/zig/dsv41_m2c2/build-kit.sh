#!/usr/bin/env bash
# Builds the Spark kit (dev box, no Spark): tf-dsv41-m1 and tensorfold-dsv41 for aarch64 with the sm_121 fatbins
# embedded, the TP mailbox / RoCE fatbin, the Python tools, run_host.sh (M2c-2), run_m4.sh + requests.sh + prompts.txt
# (M4), RUNBOOK.md (the combined window), RUNBOOK-M4.md, RUNBOOK-WINDOW.md + bench_http.py + the slots reference (the full
# window), SHA256SUMS. Run it from a clean archive of the tree
# (KIT_COMMIT=<hash>): the shared worktree may hold other agents' uncommitted files.
#   FATBINS=<dir of sm_121 dsv41_*.fatbin> TP_FATBIN=<tp_mailbox.fatbin (sm_121)> bash build-kit.sh KITDIR
set -euo pipefail
# the dev box's shared build lock (spark-work/heavy.sh: one heavy job at a time, 8 GiB, -j4), when it exists
H=$HOME/Documents/Projects/spark-work/heavy.sh
[[ -n "${HEAVY_HELD:-}" || ! -x "$H" ]] || exec env HEAVY_HELD=1 "$H" bash "$0" "$@"
KIT=${1:?kit dir}; FB=${FATBINS:?sm_121 fatbin dir}; TPF=${TP_FATBIN:?tp_mailbox.fatbin}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); TREE=$(cd "$HERE/../../.." && pwd)
export PATH=$HOME/.local/opt/zig:$PATH
rm -rf "$KIT" && mkdir -p "$KIT/tools/zig/dsv41_m1"
cd "$TREE"
zig build -Dtarget=aarch64-linux-gnu -Dfatbins="$FB" -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix "$KIT/.build" install
cp "$KIT/.build/bin/tf-dsv41-m1" "$KIT/.build/bin/tensorfold-dsv41" "$KIT/" && rm -rf "$KIT/.build"
cp "$TPF" "$KIT/tp_mailbox.fatbin"
cp tools/zig/dsv41_m2b_ref.py tools/zig/dsv41_m1_capture.py tools/zig/triton_aot_manifest.py tools/zig/dsv41_engram_host.py \
    tools/zig/dsv41_aot_merge.py "$KIT/tools/zig/"
cp tools/zig/dsv41_m1/prod-knobs.env tools/zig/dsv41_m1/r1c-knobs.env "$KIT/tools/zig/dsv41_m1/"
cp "$HERE/run_host.sh" "$HERE/RUNBOOK.md" "$KIT/"
# M4: the Zig server on rank 0, the CLI replay, DSpark's acceptance (tools/zig/dsv41_m4)
cp tools/zig/dsv41_m4/run_host.sh "$KIT/run_m4.sh"
cp tools/zig/dsv41_m4/requests.sh tools/zig/dsv41_m4/prompts.txt "$KIT/"
cp tools/zig/dsv41_m4/RUNBOOK.md "$KIT/RUNBOOK-M4.md"
# the full window (RUNBOOK-WINDOW.md): the speed client against prod, the slots reference for the row-mode AOT set
cp tools/zig/dsv41_m4/RUNBOOK-WINDOW.md "$HERE/bench_http.py" "$KIT/"
mkdir -p "$KIT/tools/zig/dsv41_slots" && cp tools/zig/dsv41_slots/ref.py "$KIT/tools/zig/dsv41_slots/"
chmod +x "$KIT/run_host.sh" "$KIT/run_m4.sh" "$KIT/requests.sh" "$KIT/tf-dsv41-m1" "$KIT/tensorfold-dsv41"
git rev-parse HEAD > "$KIT/COMMIT" 2>/dev/null || echo "${KIT_COMMIT:?KIT_COMMIT: the commit of an archived tree (no .git)}" > "$KIT/COMMIT"
(cd "$KIT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
file "$KIT/tf-dsv41-m1"; du -sh "$KIT"; echo "kit at $KIT (tree $(head -c 12 "$KIT/COMMIT"))"
