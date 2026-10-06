#!/usr/bin/env bash
# Builds the M2c-2 Spark kit (dev box, no Spark): tf-dsv41-m1 for aarch64 with the sm_121 fatbins embedded, the TP
# mailbox / RoCE fatbin, the Python tools, run_host.sh, RUNBOOK.md, SHA256SUMS.
#   FATBINS=<dir of sm_121 dsv41_*.fatbin> TP_FATBIN=<tp_mailbox.fatbin (sm_121)> bash build-kit.sh KITDIR
set -euo pipefail
KIT=${1:?kit dir}; FB=${FATBINS:?sm_121 fatbin dir}; TPF=${TP_FATBIN:?tp_mailbox.fatbin}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); TREE=$(cd "$HERE/../../.." && pwd)
export PATH=$HOME/.local/opt/zig:$PATH
rm -rf "$KIT" && mkdir -p "$KIT/tools/zig/dsv41_m1"
cd "$TREE"
zig build -Dtarget=aarch64-linux-gnu -Dfatbins="$FB" -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix "$KIT/.build" install
cp "$KIT/.build/bin/tf-dsv41-m1" "$KIT/" && rm -rf "$KIT/.build"
cp "$TPF" "$KIT/tp_mailbox.fatbin"
cp tools/zig/dsv41_m2b_ref.py tools/zig/dsv41_m1_capture.py tools/zig/triton_aot_manifest.py tools/zig/dsv41_engram_host.py "$KIT/tools/zig/"
cp tools/zig/dsv41_m1/prod-knobs.env "$KIT/tools/zig/dsv41_m1/"
cp "$HERE/run_host.sh" "$HERE/RUNBOOK.md" "$KIT/"
chmod +x "$KIT/run_host.sh" "$KIT/tf-dsv41-m1"
git rev-parse HEAD > "$KIT/COMMIT"
(cd "$KIT" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
file "$KIT/tf-dsv41-m1"; du -sh "$KIT"; echo "kit at $KIT (tree $(head -c 12 "$KIT/COMMIT"))"
