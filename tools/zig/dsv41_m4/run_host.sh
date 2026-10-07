#!/usr/bin/env bash
# M4 on the two DGX Sparks, one host's part (RUNBOOK.md): the Zig server on rank 0 (tensorfold-dsv41) with rank 1
# following (tf-dsv41-m1 follow), then the CLI replay of the server's recorded replies (tf-dsv41-m1 generate) with
# rank 1 following again. Prod must be stopped first; start both hosts within about a minute for each phase.
#   dgx-01:  ./run_host.sh 0 OUTDIR server     then (after the requests)  ./run_host.sh 0 OUTDIR generate
#   dgx-02:  ./run_host.sh 1 OUTDIR follow     then                        ./run_host.sh 1 OUTDIR follow
#   both hosts once before the server (drafts): ./run_host.sh R OUTDIR dspark-aot (ASSETS/aot gains DSpark's variants;
#   ASSETS a writable copy: cp -r OUTDIR/seedN/ref/{rope.bin,engram-host.bin,aot} OUTDIR/m4-assets/)
# Knobs: ASSETS (a dir with rope.bin, engram-host.bin, aot/: an M2c-2 seed's ref/ dir has all three; DSpark's Triton
# variants come from a set P capture merged in, else run DRAFTS=0), PORT (8091), DRAFTS (1), BACKEND (roce | nccl),
# OWN_PREFILL (1: prompts through the Zig prefill; 0: 16-row decode windows; both ranks alike), GRAPHS (1: decode
# windows through CUDA graphs, both ranks alike; 0: eager), DRAFT_TIMED (0: drafts sized from the cost tables, the
# same for a request whatever the load, so HTTP == CLI; 1: the measured round times).
# The full-model window's knobs (both ranks alike; RUNBOOK-WINDOW.md):
#   SAMPLING (1: keyed sampling, T > 0 served), SLOTS (1; N > 1: N live slots in row mode, the server's --parallel N;
#   needs POOL_TOKENS and the rows-aot phase's variants; drafts off unless SLOT_DRAFTS=1: DSpark per slot with a
#   batched verify, as prod's batcher drafts), POOL_TOKENS (the paged pool;
#   unset: the contiguous slot), KV_SPLIT (1: comp rows sharded over the ranks, needs POOL_TOKENS), INDEX_KV (prod: fp8),
#   ENGRAM (1: R3, the node's packed Engram shards as prod reads them: ENGRAM_DIR, prod-perf1.env's HEAD_ENGRAM /
#   WORKER_ENGRAM), CALIB (real: the drafts priced by prod's stored table, CACHE_VOL read-only; "": the built-in costs),
#   TREE (DSpark draft trees, e.g. 3; "": chains), PHASES (1: per-round phase times into OUTDIR/phases.json).
# rows-aot (both hosts, once, before SLOTS > 1): the slots reference (tools/zig/dsv41_slots/ref.py) at TP=2 over the
# two hosts on prod's tree, its row-mode Triton variants merged into ASSETS/aot (about 8 min).
set -u
RANK=${1:?rank}; OUT=${2:?outdir}; PHASE=${3:?server|generate|follow}
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MASTER=${MASTER:-192.168.100.10}
IMAGE=${IMAGE:-tensorfold-dsv41:e1e2-8474f31}
if [[ "$RANK" == 0 ]]; then MODEL=/home/dgx01/ai/models/dsv41-q28-v2; else MODEL=/home/root/.cache/dsv41-q28-v2/model; fi
MODEL=${PACK:-$MODEL}; ASSETS=${ASSETS:?ASSETS: rope.bin, engram-host.bin, aot/ (writable: dspark-aot merges into it)}; PORT=${PORT:-8091}
DRAFTS=${DRAFTS:-1}; BACKEND=${BACKEND:-roce}; SLOTS=${SLOTS:-1}; CACHE_VOL=${CACHE_VOL:-dsv41-perf1-cache}
# several live slots: drafts only with SLOT_DRAFTS=1 (TF_DSV41_SLOT_DRAFTS: per-slot DSpark, batched verify); else off
(( SLOTS > 1 )) && [[ "${SLOT_DRAFTS:-0}" != 1 ]] && DRAFTS=0
if [[ "$RANK" == 0 ]]; then ENGRAM_DIR=${ENGRAM_DIR:-/home/dgx01/dsv41-engram}; else ENGRAM_DIR=${ENGRAM_DIR:-/home/root/dsv41-engram}; fi
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
ARGS=(--rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$ASSETS:/assets:ro" -v "$OUT:/out"
      -e TF_TP_WORLD=2 -e TF_TP_RANK="$RANK" -e TF_TP_DEVICE=0 -e TF_TP_MASTER="$MASTER" -e TF_TP_PORT=29851
      -e TF_COMM_BACKEND="$BACKEND" -e TF_TP_ROCE_FALLBACK=error -e TF_TP_KERNEL_IMAGE=/kit/tp_mailbox.fatbin
      -e TF_DSV41_ASSETS=/assets -e TF_DSV41_EXPERT_TOPP=0.85 -e TF_DSV41_R1="${R1:-1}" -e TF_DSV41_DRAFTS="$DRAFTS" -e TF_DSV41_OWN_PREFILL="${OWN_PREFILL:-1}"
      -e TF_DSV41_GRAPHS="${GRAPHS:-1}" -e TF_DSV41_DRAFT_TIMED="${DRAFT_TIMED:-0}"
      -e TF_DSV41_PLAN_TIMEOUT=1800 -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
      -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
      -e TF_DSV41_SAMPLING="${SAMPLING:-1}" -e TF_DSV41_SLOTS="$SLOTS")
[[ -n "${POOL_TOKENS:-}" ]] && ARGS+=(-e TF_DSV41_POOL_TOKENS="$POOL_TOKENS")
[[ "${KV_SPLIT:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_KV_SPLIT=1)
[[ -n "${INDEX_KV:-}" ]] && ARGS+=(-e TF_DSV41_INDEX_KV="$INDEX_KV")
[[ "${ENGRAM:-1}" == 1 ]] && ARGS+=(-v "$ENGRAM_DIR:/engram:ro" -e TF_DSV41_ENGRAM_DIR=/engram)
[[ "${CALIB-real}" == real ]] && ARGS+=(-v "$CACHE_VOL:/cache:ro" -e TF_DSV41_CALIB=real -e TF_DSV41_CALIB_DIR=/cache/dsv41-calib)
[[ -n "${TREE:-}" ]] && ARGS+=(-e TF_DSV41_TREE="$TREE")
[[ "${SLOT_DRAFTS:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_SLOT_DRAFTS=1)
[[ "${PHASES:-0}" == 1 ]] && ARGS+=(-e TF_DSV41_PHASES=1 -e TF_DSV41_PHASES_OUT=/out/phases.json)
NCCL='export TF_NCCL_LIB=$(python -c "import nvidia.nccl, os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), \"lib\", \"libnccl.so.2\"))" 2>/dev/null || echo libnccl.so.2)'
case "$PHASE" in
    server)
        nd=(); [[ "$DRAFTS" == 0 ]] && nd=(--no-drafts)
        docker run "${ARGS[@]}" -e TF_DSV41_TRACE_TOKENS=/out/trace.jsonl --name dsv41-zig-m4-r0 --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tensorfold-dsv41 serve /model --port $PORT --parallel $SLOTS ${nd[*]}" 2>&1 | tee "$OUT/server.log" ;;
    generate)
        # GEN_STREAMS=N: N replies in flight (several slots: batched rows, as served)
        docker run "${ARGS[@]}" -e TF_DSV41_GEN_STREAMS="${GEN_STREAMS:-1}" --name dsv41-zig-m4-gen --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 generate /model /assets /out/trace.jsonl" 2>&1 | tee "$OUT/generate.log" ;;
    dspark-aot)
        # DSpark's Triton variants on this GPU (the M2c-2 reference never drafts): a set P capture (one GPU, ~3 min) on
        # prod's tree and knobs (+ R1), its AOT set merged into ASSETS/aot
        CAP=$OUT/capture-p; rm -rf "$CAP"
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        if [[ "$RANK" == 0 ]]; then TF=/home/dgx01/ai/src/dsv41-prod/$C; else TF=/home/root/ai/src/dsv41-prod/$C; fi
        docker run --rm --gpus all --ipc=host -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" \
            -v "${CACHE_VOL:-dsv41-perf1-cache}:/cache" "${KN[@]}" -e TF_DSV41_GM_V2=1 -e M1_PREFILL_ROWS=0 \
            -e PYTHONPATH=/dsv41-tf/src:/cache/pylib -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_m1_capture.py --pack /model --out /out/capture-p --sets P \
            > "$OUT/capture-p.log" 2>&1 && rm -rf "$CAP/blobs" \
            && python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$CAP/aot" | tee -a "$OUT/capture-p.log" ;;
    rows-aot)
        # the row-mode Triton variants (ROWS, int64 POS, SL / PTS) of every bucket up to 16 on this GPU: Python's slots
        # engine (prod's tree: rowtab / rowmode / rowgraphs) at TP=2 over the two hosts, 4 slots, layers 0-24 (every
        # layer kind), then this host's AOT set merged into ASSETS/aot
        C=8474f3164ea60cb9c9f8e9bd5a4493fdf13eaa47
        if [[ "$RANK" == 0 ]]; then TF=/home/dgx01/ai/src/dsv41-prod/$C; else TF=/home/root/ai/src/dsv41-prod/$C; fi
        KN=(); for f in prod-knobs.env r1c-knobs.env; do while read -r kv; do [[ "$kv" =~ ^TF_ ]] && KN+=(-e "$kv"); done < "$KIT/tools/zig/dsv41_m1/$f"; done
        RA=$OUT/rows-aot; rm -rf "$RA"
        docker run --rm --gpus all --ipc=host --network host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
            -v "$TF:/dsv41-tf:ro" -v "$MODEL:/model:ro" -v "$KIT:/kit:ro" -v "$OUT:/out" -v "$CACHE_VOL:/cache" "${KN[@]}" \
            -e TF_DSV41_GM_V2=1 -e TF_DSV41_EXPERT_TOPP=0.85 -e PYTHONPATH=/dsv41-tf/src:/cache/pylib \
            -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
            -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1} \
            -e GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1} --name "dsv41-zig-rows-r$RANK" \
            --entrypoint python "$IMAGE" -u -B /kit/tools/zig/dsv41_slots/ref.py --pack /model --out /out/rows-aot \
            --layers 0-24 --slots 4 --steps "${ROWS_STEPS:-8}" --rank "$RANK" --master "$MASTER" --port 29821 \
            > "$OUT/rows-aot.log" 2>&1 \
            && python3 "$KIT/tools/zig/dsv41_aot_merge.py" "$ASSETS/aot" "$RA/aot" | tee -a "$OUT/rows-aot.log" ;;
    follow)
        docker run "${ARGS[@]}" --name dsv41-zig-m4-r$RANK --entrypoint bash "$IMAGE" \
            -c "$NCCL; exec /kit/tf-dsv41-m1 follow /model /assets" 2>&1 | tee -a "$OUT/follow.log" ;;
    *) echo "phase: server | generate | follow | dspark-aot | rows-aot"; exit 2 ;;
esac
