#!/usr/bin/env bash
# M2c-2 on the two DGX Sparks, one host's rank: the Python engine's prod tree (8474f31, prod's image and cache volume)
# at TP=2 over the whole backbone + head as the reference (a 1,536-token prompt, then greedy decoding at expert top-p
# 0.85 with M1's prod kernel knobs: the launches block.zig emits), then our forward at TP=2 from its state over the
# hybrid collective (RoCE small exchanges, NCCL large), every token / logits digest / first-window block compared.
# Start it on both hosts within about a minute (each phase's rendezvous waits up to 30 min for the other host):
#   dgx-01:  ./run_host.sh 0 OUTDIR
#   dgx-02:  ./run_host.sh 1 OUTDIR
# Prod must be stopped first (RUNBOOK.md). Knobs: STEPS (512), SEEDS ("4101"), PROMPT (1536), BACKEND (roce | nccl).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MASTER=${MASTER:-192.168.100.10}
IMAGE=${IMAGE:-tensorfold-dsv41:e1e2-8474f31}
CACHE_VOL=${CACHE_VOL:-dsv41-perf1-cache}
COMMIT=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
if [[ "$RANK" == 0 ]]; then TF=/home/dgx01/ai/src/dsv41-prod/$COMMIT MODEL=/home/dgx01/ai/models/dsv41-q28-v2
else TF=/home/root/ai/src/dsv41-prod/$COMMIT MODEL=/home/root/.cache/dsv41-q28-v2/model; fi
TF=${TF_TREE:-$TF}; MODEL=${PACK:-$MODEL}
STEPS=${STEPS:-512}; SEEDS=${SEEDS:-4101}; PROMPT=${PROMPT:-1536}; BACKEND=${BACKEND:-roce}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
log() { echo "[m2c2 r$RANK $(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
drop_caches() { sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || log "drop_caches refused"; }

# preflight: nothing else on the GPU, the kit intact, the tree and pack where prod keeps them
if docker ps --format '{{.Names}}' | grep -q 'dsv41'; then log "a dsv41 container is running: stop prod first (RUNBOOK.md)"; exit 2; fi
(cd "$KIT" && sha256sum -c --quiet SHA256SUMS) || { log "kit checksum mismatch"; exit 2; }
[[ -d "$TF/src" && -f "$MODEL/config.json" ]] || { log "missing $TF/src or $MODEL/config.json"; exit 2; }
ulimit -l unlimited 2> /dev/null || log "warning: memlock $(ulimit -l) KiB"
{ hostname; nvidia-smi -L; free -g; } > "$OUT/host.txt" 2>&1
ARGS=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$CACHE_VOL:/cache" -v "$KIT:/kit:ro" -v "$OUT:/out"
      -e PYTHONPATH=/dsv41-tf/src:/cache/pylib -e PYTHONDONTWRITEBYTECODE=1 -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions
      -e TRITON_CACHE_DIR=/cache/triton -e CUDA_CACHE_PATH=/cache/nv/ComputeCache
      -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
      -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1})
KNOBS=(); while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KNOBS+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/prod-knobs.env"

port=29811
for seed in $SEEDS; do
    R=/out/seed$seed
    mkdir -p "$OUT/seed$seed"
    # 1. the reference: Python at TP=2 (this host's rank), state + tokens + digests + the AOT set + rope tables
    drop_caches
    log "seed $seed: Python reference (layers 0-39, prompt $PROMPT, $STEPS greedy steps, top-p 0.85)"
    t0=$(date +%s)
    docker run "${ARGS[@]}" "${KNOBS[@]}" -e TF_DSV41_EXPERT_TOPP=0.85 --name "dsv41-zig-ref-r$RANK" \
        --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m2b_ref.py --pack /model --out "$R/ref" \
        --layers 0-39 --prompt "$PROMPT" --steps "$STEPS" --seed "$seed" --rank "$RANK" --master "$MASTER" --port $port \
        > "$OUT/seed$seed/ref.log" 2>&1
    rc=$?; log "seed $seed: reference rc $rc in $(( $(date +%s) - t0 )) s ($(tail -1 "$OUT/seed$seed/ref.log"))"
    (( rc == 0 )) || exit 1
    docker run "${ARGS[@]}" --entrypoint python "$IMAGE" -B /kit/tools/zig/dsv41_engram_host.py --py-src /dsv41-tf/src \
        --config /model/config.json --tokenizer /model/tokenizer.json --out "$R/ref/engram-host.bin" > "$OUT/seed$seed/engram-host.log" 2>&1 \
        || { log "engram host tables failed"; exit 1; }
    # 2. ours: the same rank from the reference's state, over the hybrid collective
    drop_caches
    log "seed $seed: our forward (backend $BACKEND)"
    t0=$(date +%s)
    docker run "${ARGS[@]}" -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" \
        -e TF_TP_PORT=$((port + 10)) -e TF_DSV41_PLAN_TIMEOUT=1800 -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error \
        -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin -e M2B_TOPP=0.85 -e TF_DSV41_FAILFAST=0 --name "dsv41-zig-m2c2-r$RANK" \
        --entrypoint bash "$IMAGE" -c 'export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2); exec /kit/tf-dsv41-m1 m2b /model '"$R/ref"' '"$R/zig" \
        > "$OUT/seed$seed/zig.log" 2>&1
    rc=$?; log "seed $seed: ours rc $rc in $(( $(date +%s) - t0 )) s: $(grep -h '^PASS\|^FAIL' "$OUT/seed$seed/zig.log" | tail -1)"
    port=$((port + 20))
done
drop_caches
log "done: $(grep -h '^PASS\|^FAIL' "$OUT"/seed*/zig.log | tr '\n' ' ')"
