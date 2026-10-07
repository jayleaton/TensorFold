#!/usr/bin/env bash
# M4 on the two DGX Sparks, one host's part (RUNBOOK.md): the Zig server on rank 0 (tensorfold-dsv41) with rank 1
# following (tf-dsv41-m1 follow), then the CLI replay of the server's recorded replies (tf-dsv41-m1 generate) with
# rank 1 following again. Prod must be stopped first; start both hosts within about a minute for each phase.
#   dgx-01:  ./run_host.sh 0 OUTDIR server     then (after the requests)  ./run_host.sh 0 OUTDIR generate
#   dgx-02:  ./run_host.sh 1 OUTDIR follow     then                        ./run_host.sh 1 OUTDIR follow
# Knobs: ASSETS (a dir with rope.bin, engram-host.bin, aot/: an M2c-2 seed's ref/ dir has all three; DSpark's Triton
# variants come from a set P capture merged in, else run DRAFTS=0), PORT (8091), DRAFTS (1), BACKEND (roce | nccl).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}; PHASE=${3:?server|generate|follow}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MASTER=${MASTER:-192.168.100.10}
IMAGE=${IMAGE:-tensorfold-dsv41:e1e2-8474f31}
if [[ "$RANK" == 0 ]]; then MODEL=/home/dgx01/ai/models/dsv41-q28-v2; else MODEL=/home/root/.cache/dsv41-q28-v2/model; fi
MODEL=${PACK:-$MODEL}; ASSETS=${ASSETS:?ASSETS: rope.bin, engram-host.bin, aot/}; PORT=${PORT:-8091}
DRAFTS=${DRAFTS:-1}; BACKEND=${BACKEND:-roce}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ARGS=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$ASSETS:/assets:ro" -v "$OUT:/out"
      -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" -e TF_TP_PORT=29851
      -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
      -e TF_DSV41_ASSETS=/assets -e TF_DSV41_EXPERT_TOPP=0.85 -e TF_DSV41_R1="${R1:-0}" -e TF_DSV41_DRAFTS="$DRAFTS"
      -e TF_DSV41_PLAN_TIMEOUT=1800 -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
      -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1})
NCCL='export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2)'
case "$PHASE" in
    server)
        nd=(); [[ "$DRAFTS" == 0 ]] && nd=(--no-drafts)
        docker run "${ARGS[@]}" -e TF_DSV41_TRACE_TOKENS=/out/trace.jsonl --name dsv41-zig-m4-r0 --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tensorfold-dsv41 serve /model --port $PORT --parallel 1 ${nd[*]}" 2>&1 | tee "$OUT/server.log" ;;
    generate)
        docker run "${ARGS[@]}" --name dsv41-zig-m4-gen --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 generate /model /assets /out/trace.jsonl" 2>&1 | tee "$OUT/generate.log" ;;
    follow)
        docker run "${ARGS[@]}" --name dsv41-zig-m4-r$RANK --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 follow /model /assets" 2>&1 | tee -a "$OUT/follow.log" ;;
    *) echo "phase: server | generate | follow"; exit 2 ;;
esac
