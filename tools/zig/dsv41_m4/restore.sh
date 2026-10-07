#!/usr/bin/env bash
# Restore on failure, one host: stop every Zig / reference container this kit started; on dgx-01 (rank 0) then bring
# prod back on perf1 and show its status. Run on dgx-02 first, then on dgx-01 (or on dgx-01 alone: rank 1's containers
# end on their own once rank 0's are gone; `docker ps` on dgx-02 confirms).
#   ./restore.sh 1     (dgx-02)
#   ./restore.sh 0     (dgx-01: also scripts/v2-rollback.sh perf1)
set -u
RANK=${1:?rank}
ids=$(docker ps -q --filter name=dsv41-zig)
[[ -n "$ids" ]] && { echo "stopping: $(docker ps --format '{{.Names}}' --filter name=dsv41-zig | tr '\n' ' ')"; docker stop $ids > /dev/null; }
docker ps --format '{{.Names}}' | grep dsv41 || echo "no dsv41 container left on $(hostname)"
if [[ "$RANK" == 0 ]]; then
    cd ~/ai/src/dsv41-tensorfold-spark || exit 1
    scripts/v2-rollback.sh perf1 && scripts/serve.sh status
fi
