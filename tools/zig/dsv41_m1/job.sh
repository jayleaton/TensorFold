#!/usr/bin/env bash
# DeepSeek-V4.1's M1 GPU job on ONE GPU of the Python twin's architecture (RTX PRO 6000, sm_120), in the e1e2 runtime
# (the Python engine at /dsv41-tf with its prebuilt extensions; the quant repo's runpod job `zigm1` unpacks it):
#   (order: the capture and our gates first, then the kernel job in the budget left, then the optional r1c step)
#   1. kernels: the kernel agent's exactness job (zig/tests/cuda/deepseek_v41/job.sh: oracle fixtures + the twin's own GPU
#      suites recorded and replayed), on the decode1 twin when r1c.tar.gz is staged
#   2. capture: tools/zig/dsv41_m1_capture.py, the Python engine's blocks 1-3 and 20-24 on our pack, every launch's
#      buffers before / after (local NVMe)
#   3. m1: tf-dsv41-m1 both, our loader + every launch re-issued and compared bit for bit (the replay), then block.zig's
#      calls for every captured decode window on our own buffers, structure on the host then bits on the GPU (the
#      block forward; M1_MODE=replay|forward runs one)
#   4. r1c (optional, M1_R1C_MIN > 0 with r1c.tar.gz staged): a second capture with prod + R1c knobs (r1c-knobs.env) on
#      the decode1 twin, set B by default, replayed bit for bit, when M1_R1C_MIN minutes of the budget remain
# Each step has its own timeout and log in $RES; a failing step does not stop the next. Exit 0 = every step passed.
# Budget: M1_BUDGET_MIN (default 52) of a 60-minute hard max, the rest for the runtime unpack and packing: capture 18
# + m1 14 min at most, the kernel job gets what is left.
#
#   Z=<staged tree: bin/ fatbin/ tools/ zig/> PACK=<pack dir> RES=<results dir> LOCAL=<scratch> bash job.sh
set -u
Z=${Z:?staged tree}; PACK=${PACK:?pack}; RES=${RES:?results}; LOCAL=${LOCAL:?scratch}
mkdir -p "$RES" "$LOCAL"
export PYTHONPATH=/dsv41-tf/src TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/opt/e1e2/ext} \
    TRITON_CACHE_DIR=$LOCAL/triton CUDA_CACHE_PATH=$LOCAL/nv PYTHONDONTWRITEBYTECODE=1
rc=0
T0=$(date +%s)
step() { # NAME MINUTES CMD...: output in $RES/NAME.log, rc and seconds in $RES/steps.txt
    local name=$1 min=$2 t0 r; shift 2
    t0=$(date +%s); echo "=== $(date -u +%H:%M:%S) $name: $*"
    timeout --kill-after=30 "$(( min * 60 ))" "$@" > "$RES/$name.log" 2>&1; r=$?
    echo "$name rc=$r s=$(( $(date +%s) - t0 ))" | tee -a "$RES/steps.txt"
    tail -5 "$RES/$name.log"
    (( r == 0 )) || rc=1
    return $r
}
# M1_KNOBS=prod: the prod config's kernel knobs (prod-knobs.env) for the capture; default: the engine's code defaults
if [[ "${M1_KNOBS:-}" == prod ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$Z/tools/zig/dsv41_m1/prod-knobs.env"
    set +a
    echo "knobs: prod ($(grep -c '^TF_' "$Z/tools/zig/dsv41_m1/prod-knobs.env") TF_DSV41_* settings)"
fi
{ cat "$Z/COMMIT" 2>/dev/null; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
  python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
  free -g | head -2; df -h "$LOCAL" | tail -1; } > "$RES/env.txt" 2>&1
cat "$RES/env.txt"

# the decode1 twin (r1c.tar.gz: /opt/r1c/{src,tests,ext}, its extensions prebuilt for sm_120 at these absolute paths, so
# torch's JIT loads them without building): the kernel job's Python twin, and the optional R1c capture's
PYTWIN=/dsv41-tf PYEXT=$TORCH_EXTENSIONS_DIR
if [[ -f "$Z/r1c.tar.gz" ]] && tar -xzf "$Z/r1c.tar.gz" -C /; then PYTWIN=/opt/r1c PYEXT=/opt/r1c/ext; fi
echo "kernel job twin: $PYTWIN" | tee -a "$RES/env.txt"
left() { echo $(( ${M1_BUDGET_MIN:-52} - ($(date +%s) - T0) / 60 )); }

# M1_JOB=m2a: M2a instead of M1 (the kernel job first: glue.cu's first GPU run; then the Python reference over layers
# M2A_LAYERS (0-24) with its slot state, rope tables and AOT set; the Engram host tables; tf-dsv41-m1 m2a)
if [[ "${M1_JOB:-m1}" == m2a ]]; then
    (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
     export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
            TORCH_EXTENSIONS_DIR=$PYEXT TF_DSV41_CAPTURE_MIN=6
     step kernels 10 bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels") || rc=1
    REF=$LOCAL/m2a-ref
    rm -rf "$REF"
    if step m2a-ref "${M2A_REF_MIN:-22}" python -B "$Z/tools/zig/dsv41_m2a_digests.py" --pack "$PACK" --out "$REF" \
            --layers "${M2A_LAYERS:-0-24}" \
       && step engram-host 3 python -B "$Z/tools/zig/dsv41_engram_host.py" --py-src /dsv41-tf/src \
            --config "$PACK/config.json" --tokenizer "$PACK/tokenizer.json" --out "$REF/engram-host.bin"; then
        step m2a "$(( $(left) > 4 ? $(left) - 2 : 2 ))" "$Z/bin/tf-dsv41-m1" m2a "$PACK" "$REF" "$RES/m2a"
    else rc=1; fi
    mkdir -p "$RES/m2a-ref"
    for f in digests.json manifest.json; do [[ -f "$REF/$f" ]] && cp "$REF/$f" "$RES/m2a-ref/"; done
    [[ -f "$REF/state/state.json" ]] && cp "$REF/state/state.json" "$RES/m2a-ref/"
    [[ -d "$REF/aot" ]] && tar -czf "$RES/m2a-ref/aot.tar.gz" -C "$REF" aot
    echo "=== rc=$rc"
    exit $rc
fi

# 1-2. the prod capture and our gates first (the replay + block.zig's forward on one load)
CAP=$LOCAL/capture
rm -rf "$CAP"
if step capture "${M1_CAPTURE_MIN:-18}" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$PACK" --out "$CAP" \
        --sets "${M1_SETS:-A,B,C,D}"; then
    du -sh "$CAP" "$CAP/blobs" | tee "$RES/capture-size.txt"
    step m1 "${M1_REPLAY_MIN:-14}" "$Z/bin/tf-dsv41-m1" "${M1_MODE:-both}" "$PACK" "$CAP" "$RES/m1"
fi

# 3. the kernel agent's exactness job in what is left (its pytest capture gets 4 minutes less), every TF_DSV41_ knob
# unset: the twin's suites pick their own parametrizations
kmin=$(( $(left) - ${M1_R1C_MIN:-0} ))
if (( kmin >= 6 )); then
    (for v in $(compgen -e | grep '^TF_DSV41_'); do unset "$v"; done
     export TF_DSV41_PY_SRC=$PYTWIN TF_DSV41_TEST=$Z/bin/tf-dsv41-test TF_DSV41_FATBINS=$Z/fatbin \
            TORCH_EXTENSIONS_DIR=$PYEXT TF_DSV41_CAPTURE_MIN=$(( kmin - 4 ))
     step kernels "$kmin" bash "$Z/zig/tests/cuda/deepseek_v41/job.sh" "$RES/kernels") || rc=1
else
    echo "kernels skipped: $(left) min of the budget left" | tee -a "$RES/steps.txt"; rc=1
fi

# 4. optional (M1_R1C_MIN > 0): a capture with prod + R1c knobs on the decode1 twin, replayed bit for bit
if (( ${M1_R1C_MIN:-0} > 0 )) && [[ "$PYTWIN" == /opt/r1c ]]; then
    l=$(left)
    if (( l >= M1_R1C_MIN )); then
        CAP2=$LOCAL/capture-r1c
        rm -rf "$CAP2"
        half=$(( (l - 1) / 2 ))
        if (set -a; source "$Z/tools/zig/dsv41_m1/prod-knobs.env"; source "$Z/tools/zig/dsv41_m1/r1c-knobs.env"; set +a
            export PYTHONPATH=/opt/r1c/src TORCH_EXTENSIONS_DIR=/opt/r1c/ext
            step r1c-capture "$half" python -B "$Z/tools/zig/dsv41_m1_capture.py" --pack "$PACK" --out "$CAP2" \
                --sets "${M1_R1C_SETS:-B}"); then
            step r1c-replay "$half" "$Z/bin/tf-dsv41-m1" replay "$PACK" "$CAP2" "$RES/m1-r1c"
        else rc=1; fi
        mkdir -p "$RES/capture-r1c"
        for f in meta.json manifest.json; do [[ -f "$CAP2/$f" ]] && cp "$CAP2/$f" "$RES/capture-r1c/"; done
        [[ -f "$CAP2/ops.jsonl" ]] && gzip -c "$CAP2/ops.jsonl" > "$RES/capture-r1c/ops.jsonl.gz"
    else
        echo "r1c skipped: $l min of the budget left" | tee -a "$RES/steps.txt"
    fi
fi

# the capture's small files come back (the blobs stay on the pod: GBs); ops.jsonl compressed
mkdir -p "$RES/capture"
for f in meta.json weights.json manifest.json jit.json; do [[ -f "$CAP/$f" ]] && cp "$CAP/$f" "$RES/capture/"; done
# the AOT set whole (aot.json + cubins, a few MB): M2 launches Triton through it (zig/src/cuda/aot.zig)
[[ -d "$CAP/aot" ]] && tar -czf "$RES/capture/aot.tar.gz" -C "$CAP" aot
[[ -f "$CAP/ops.jsonl" ]] && gzip -c "$CAP/ops.jsonl" > "$RES/capture/ops.jsonl.gz"
[[ -f "$CAP/launches.json" ]] && gzip -c "$CAP/launches.json" > "$RES/capture/launches.json.gz"
echo "=== rc=$rc"
exit $rc
