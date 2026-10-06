#!/usr/bin/env bash
# DeepSeek-V4.1's M1 GPU job on ONE GPU of the Python twin's architecture (RTX PRO 6000, sm_120), in the e1e2 runtime
# (the Python engine at /dsv41-tf with its prebuilt extensions; the quant repo's runpod job `zigm1` unpacks it):
#   1. kernels: the kernel agent's exactness job (zig/tests/cuda/deepseek_v41/job.sh: seeded inputs, no checkpoint)
#   2. capture: tools/zig/dsv41_m1_capture.py, the Python engine's blocks 1-3 and 20-24 on our pack, every launch's
#      buffers before / after (local NVMe)
#   3. replay: tf-dsv41-m1 replay, our loader + every launch re-issued and compared bit for bit
# Each step has its own timeout and log in $RES; a failing step does not stop the next. Exit 0 = every step passed.
#
#   Z=<staged tree: bin/ fatbin/ tools/ zig/> PACK=<pack dir> RES=<results dir> LOCAL=<scratch> bash job.sh
set -u
Z=${Z:?staged tree}; PACK=${PACK:?pack}; RES=${RES:?results}; LOCAL=${LOCAL:?scratch}
mkdir -p "$RES" "$LOCAL"
export PYTHONPATH=/dsv41-tf/src TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/opt/e1e2/ext} \
    TRITON_CACHE_DIR=$LOCAL/triton CUDA_CACHE_PATH=$LOCAL/nv PYTHONDONTWRITEBYTECODE=1
rc=0
step() { # NAME MINUTES CMD...: output in $RES/NAME.log, rc and seconds in $RES/steps.txt
    local name=$1 min=$2 t0 r; shift 2
    t0=$(date +%s); echo "=== $(date -u +%H:%M:%S) $name: $*"
    timeout --kill-after=30 "$(( min * 60 ))" "$@" > "$RES/$name.log" 2>&1; r=$?
    echo "$name rc=$r s=$(( $(date +%s) - t0 ))" | tee -a "$RES/steps.txt"
    tail -5 "$RES/$name.log"
    (( r == 0 )) || rc=1
    return $r
}
{ cat "$Z/COMMIT" 2>/dev/null; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
  python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
  free -g | head -2; df -h "$LOCAL" | tail -1; } > "$RES/env.txt" 2>&1
cat "$RES/env.txt"

TF_DSV41_PY_SRC=/dsv41-tf TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
    step kernels 25 bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels"

CAP=$LOCAL/capture
rm -rf "$CAP"
if step capture "${M1_CAPTURE_MIN:-35}" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$PACK" --out "$CAP" \
        --sets "${M1_SETS:-A,B,C}"; then
    du -sh "$CAP" "$CAP/blobs" | tee "$RES/capture-size.txt"
    step replay "${M1_REPLAY_MIN:-30}" "$Z/bin/tf-dsv41-m1" replay "$PACK" "$CAP" "$RES/m1"
fi
# the capture's small files come back (the blobs stay on the pod: GBs); ops.jsonl compressed
mkdir -p "$RES/capture"
for f in meta.json weights.json manifest.json jit.json; do [[ -f "$CAP/$f" ]] && cp "$CAP/$f" "$RES/capture/"; done
[[ -f "$CAP/aot/aot.json" ]] && cp "$CAP/aot/aot.json" "$RES/capture/"
[[ -f "$CAP/ops.jsonl" ]] && gzip -c "$CAP/ops.jsonl" > "$RES/capture/ops.jsonl.gz"
[[ -f "$CAP/launches.json" ]] && gzip -c "$CAP/launches.json" > "$RES/capture/launches.json.gz"
echo "=== rc=$rc"
exit $rc
